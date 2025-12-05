"""
Training SVM dengan MULTIPLE FEATURES untuk deteksi fake review
- Text features (TF-IDF)
- Numeric features (stars, reviewer info, image, time patterns, dll)
- Pattern detection untuk fake review
- Text quality features (gibberish detection)
"""

import numpy as np
import pandas as pd
import pickle
from collections import Counter
from datetime import datetime
from text_quality_detector import extract_text_quality_features, get_text_quality_feature_names

# ---------------------------
# === FEATURE ENGINEERING
# ---------------------------

def extract_features(df):
    """
    Extract semua fitur penting untuk deteksi fake review

    Features yang diextract:
    1. Text features (TF-IDF) - dari text_preprocessed
    2. Stars (rating bintang)
    3. Review length (panjang review)
    4. Has image (ada foto atau tidak)
    5. Reviewer number of reviews (jumlah review dari reviewer)
    6. Is local guide
    7. Detail rating all 5s (apakah detail rating semua 5)
    8. Same day count (jumlah review di hari yang sama)
    9. Short review with 5 stars (pattern: bintang 5 tapi review pendek)
    10. No photo with 5 stars (pattern: bintang 5 tapi tidak ada foto)
    """

    print("\n[Feature Engineering] Extracting features...")

    # 1. Text length
    df['text_length'] = df['text_preprocessed'].fillna('').apply(lambda x: len(str(x).split()))
    print(f"  ✓ Text length (mean: {df['text_length'].mean():.2f} words)")

    # 2. Stars (normalize to 0-1)
    df['stars_norm'] = df['stars'].fillna(0) / 5.0
    print(f"  ✓ Stars normalized")

    # 3. Has image
    df['has_image'] = df['reviewImageUrls'].apply(
        lambda x: 0 if pd.isna(x) or str(x).strip() == '' or str(x).lower() == 'nan' else 1
    )
    print(f"  ✓ Has image ({df['has_image'].sum()} reviews with photos)")

    # 4. Reviewer number of reviews (normalize dengan log)
    df['reviewer_count'] = df['reviewerNumberOfReviews'].fillna(0)
    df['reviewer_count_log'] = np.log1p(df['reviewer_count'])  # log(1+x) untuk handle 0
    print(f"  ✓ Reviewer count (mean: {df['reviewer_count'].mean():.1f} reviews)")

    # 5. Is local guide
    df['is_local_guide'] = df['isLocalGuide'].apply(
        lambda x: 1 if str(x).lower() in ['true', '1', 'yes'] else 0
    )
    print(f"  ✓ Local guide ({df['is_local_guide'].sum()} local guides)")

    # 6. Detail rating all 5s
    def check_all_fives(x):
        if pd.isna(x) or str(x).strip() == '':
            return 0
        # Parse detail rating (format bisa: "5,5,5" atau "Makanan: 5, Layanan: 5")
        try:
            numbers = [float(s) for s in str(x).replace(',', ' ').split() if s.replace('.','').isdigit()]
            if len(numbers) > 0:
                return 1 if all(n == 5.0 for n in numbers) else 0
        except:
            pass
        return 0

    df['detail_rating_all_5'] = df['reviewDetailedRating'].apply(check_all_fives)
    print(f"  ✓ Detail rating all 5s ({df['detail_rating_all_5'].sum()} reviews)")

    # 7. Published date - extract time patterns
    df['published_date'] = pd.to_datetime(df['publishedAtDate'], errors='coerce')
    df['publish_hour'] = df['published_date'].dt.hour.fillna(0).astype(int)
    df['publish_day'] = df['published_date'].dt.date

    # 8. Same day review count
    if df['publish_day'].notna().sum() > 0:
        day_counts = df.groupby('publish_day').size()
        df['same_day_count'] = df['publish_day'].map(day_counts).fillna(1).astype(int)
    else:
        df['same_day_count'] = 1
    print(f"  ✓ Same day count (max: {df['same_day_count'].max()} reviews in one day)")

    # 9. Suspicious patterns
    # Pattern 1: 5 stars + short review (< 5 words)
    df['pattern_5star_short'] = ((df['stars'] == 5) & (df['text_length'] < 5)).astype(int)

    # Pattern 2: 5 stars + no photo
    df['pattern_5star_nophoto'] = ((df['stars'] == 5) & (df['has_image'] == 0)).astype(int)

    # Pattern 3: New reviewer (< 3 reviews)
    df['pattern_new_reviewer'] = (df['reviewer_count'] <= 2).astype(int)

    # Pattern 4: Many reviews same day (> 5)
    df['pattern_same_day'] = (df['same_day_count'] > 5).astype(int)

    print(f"  ✓ Suspicious patterns detected")

    # 10. Text Quality Features (Gibberish Detection)
    print(f"\n  [NEW] Adding text quality features for gibberish detection...")
    df = extract_text_quality_features(df, text_column='text_preprocessed')

    return df


# ---------------------------
# === TF-IDF VECTORIZER
# ---------------------------

class TfidfVectorizer:
    """TF-IDF Vectorizer dari scratch"""

    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocabulary = {}
        self.idf_values = {}

    def fit(self, documents):
        word_doc_count = Counter()
        all_words = Counter()

        for doc in documents:
            words = str(doc).split()
            unique_words = set(words)

            for word in unique_words:
                word_doc_count[word] += 1

            for word in words:
                all_words[word] += 1

        most_common_words = all_words.most_common(self.max_features)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common_words)}

        n_documents = len(documents)
        for word in self.vocabulary:
            self.idf_values[word] = np.log(n_documents / (word_doc_count[word] + 1))

    def transform(self, documents):
        n_documents = len(documents)
        n_features = len(self.vocabulary)
        tfidf_matrix = np.zeros((n_documents, n_features))

        for doc_idx, doc in enumerate(documents):
            words = str(doc).split()
            word_count = Counter(words)
            total_words = len(words)

            if total_words == 0:
                continue

            for word, count in word_count.items():
                if word in self.vocabulary:
                    word_idx = self.vocabulary[word]
                    tf = count / total_words
                    tfidf_matrix[doc_idx, word_idx] = tf * self.idf_values[word]

        return tfidf_matrix

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)


# ---------------------------
# === LINEAR SVM
# ---------------------------

class LinearSVM:
    """Linear SVM dengan Gradient Descent"""

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for iteration in range(self.n_iterations):
            loss = 0

            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * (-y[idx])
                    loss += 1 - y[idx] * (np.dot(x_i, self.w) + self.b)

            loss += self.lambda_param * np.dot(self.w, self.w)
            self.losses.append(loss / n_samples)

            if (iteration + 1) % 100 == 0:
                print(f"  Iterasi {iteration + 1}/{self.n_iterations}, Loss: {self.losses[-1]:.4f}")

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b


# ---------------------------
# === EVALUATION METRICS
# ---------------------------

def calculate_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall, F1-score"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fn = np.sum((y_true == 1) & (y_pred == -1))

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    }


def print_confusion_matrix(cm):
    """Print confusion matrix"""
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              FAKE    REAL")
    print(f"Actual FAKE   {cm['tp']:4d}    {cm['fn']:4d}")
    print(f"       REAL   {cm['fp']:4d}    {cm['tn']:4d}")


# ---------------------------
# === MAIN TRAINING
# ---------------------------

def main():
    print("="*80)
    print("TRAINING SVM - MULTI-FEATURE FAKE REVIEW DETECTION")
    print("="*80)

    # KONFIGURASI
    INPUT_FILE = "dataset_balance/google_review_balanced_combined.csv"
    OUTPUT_DIR = "model_output"

    # Hyperparameters
    MAX_FEATURES_TFIDF = 1000
    LEARNING_RATE = 0.001
    LAMBDA_PARAM = 0.01
    N_ITERATIONS = 1000
    TEST_SIZE = 0.2
    RANDOM_SEED = 42

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # [1/7] LOAD DATA
    print(f"\n[1/7] Membaca dataset dari: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"  Total data: {len(df)} baris")

    # Kolom yang diperlukan
    required_cols = ['text_preprocessed', 'label', 'stars', 'reviewerNumberOfReviews',
                     'reviewImageUrls', 'isLocalGuide', 'publishedAtDate', 'reviewDetailedRating']

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\n⚠ Kolom yang hilang: {missing_cols}")
        print("  Akan menggunakan nilai default untuk kolom yang hilang")
        for col in missing_cols:
            df[col] = np.nan

    # Filter data kosong
    df = df[df['text_preprocessed'].notna()]
    df = df[df['text_preprocessed'].str.strip() != '']
    print(f"  Data valid: {len(df)} baris")

    # Distribusi label
    print(f"\n  Distribusi label:")
    for label, count in df['label'].value_counts().items():
        print(f"    - {label}: {count} ({count/len(df)*100:.2f}%)")

    # [2/7] FEATURE ENGINEERING
    print(f"\n[2/7] Feature Engineering")
    df = extract_features(df)

    # [3/7] SPLIT DATA
    print(f"\n[3/7] Split data (train: 80%, test: 20%)")
    np.random.seed(RANDOM_SEED)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    split_idx = int(len(df) * (1 - TEST_SIZE))
    df_train = df[:split_idx]
    df_test = df[split_idx:]

    print(f"  Data training: {len(df_train)} baris")
    print(f"  Data testing: {len(df_test)} baris")

    # [4/7] PREPARE FEATURES
    print(f"\n[4/7] Prepare Features")

    # Text features
    X_train_text = df_train['text_preprocessed'].fillna('').tolist()
    X_test_text = df_test['text_preprocessed'].fillna('').tolist()

    # Numeric features (including text quality features)
    text_quality_features = get_text_quality_feature_names()

    numeric_features = [
        'text_length', 'stars_norm', 'has_image', 'reviewer_count_log',
        'is_local_guide', 'detail_rating_all_5', 'same_day_count',
        'pattern_5star_short', 'pattern_5star_nophoto', 'pattern_new_reviewer', 'pattern_same_day'
    ] + text_quality_features

    X_train_numeric = df_train[numeric_features].fillna(0).values
    X_test_numeric = df_test[numeric_features].fillna(0).values

    print(f"  Numeric features: {len(numeric_features)}")
    for feat in numeric_features:
        print(f"    - {feat}")

    # TF-IDF features
    print(f"\n  TF-IDF Vectorization (max_features={MAX_FEATURES_TFIDF})...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES_TFIDF)
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    # Combine features
    X_train = np.hstack([X_train_tfidf, X_train_numeric])
    X_test = np.hstack([X_test_tfidf, X_test_numeric])

    print(f"\n  Combined features shape:")
    print(f"    - X_train: {X_train.shape}")
    print(f"    - X_test: {X_test.shape}")

    # Labels
    label_mapping = {'FAKE': 1, 'REAL': -1}
    y_train = df_train['label'].map(label_mapping).values
    y_test = df_test['label'].map(label_mapping).values

    # [5/7] TRAINING
    print(f"\n[5/7] Training Linear SVM")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Lambda: {LAMBDA_PARAM}")
    print(f"  Iterations: {N_ITERATIONS}")

    svm = LinearSVM(
        learning_rate=LEARNING_RATE,
        lambda_param=LAMBDA_PARAM,
        n_iterations=N_ITERATIONS
    )

    svm.fit(X_train, y_train)

    # [6/7] EVALUATION
    print(f"\n[6/7] Evaluasi Model")

    print("\n  Training Set:")
    y_train_pred = svm.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_train_pred)
    print(f"    Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"    Precision: {train_metrics['precision']:.4f}")
    print(f"    Recall:    {train_metrics['recall']:.4f}")
    print(f"    F1-Score:  {train_metrics['f1_score']:.4f}")
    print_confusion_matrix(train_metrics['confusion_matrix'])

    print("\n  Test Set:")
    y_test_pred = svm.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    print(f"    Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"    Precision: {test_metrics['precision']:.4f}")
    print(f"    Recall:    {test_metrics['recall']:.4f}")
    print(f"    F1-Score:  {test_metrics['f1_score']:.4f}")
    print_confusion_matrix(test_metrics['confusion_matrix'])

    # [7/7] SAVE MODEL
    print(f"\n[7/7] Menyimpan Model")

    model_data = {
        'svm': svm,
        'vectorizer': vectorizer,
        'label_mapping': label_mapping,
        'numeric_features': numeric_features,
        'hyperparameters': {
            'max_features_tfidf': MAX_FEATURES_TFIDF,
            'learning_rate': LEARNING_RATE,
            'lambda_param': LAMBDA_PARAM,
            'n_iterations': N_ITERATIONS
        },
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }

    model_file = os.path.join(OUTPUT_DIR, "svm_multifeature_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"  ✓ Model disimpan: {model_file}")

    # Save metrics
    metrics_df = pd.DataFrame({
        'Set': ['Training', 'Test'],
        'Accuracy': [train_metrics['accuracy'], test_metrics['accuracy']],
        'Precision': [train_metrics['precision'], test_metrics['precision']],
        'Recall': [train_metrics['recall'], test_metrics['recall']],
        'F1-Score': [train_metrics['f1_score'], test_metrics['f1_score']]
    })

    metrics_file = os.path.join(OUTPUT_DIR, "svm_multifeature_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"  ✓ Metrics disimpan: {metrics_file}")

    print("\n" + "="*80)
    print("TRAINING SELESAI!")
    print("="*80)
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
