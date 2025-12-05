"""
Prediksi fake review dari CSV file menggunakan SVM multi-feature model
- Input: CSV file dengan kolom lengkap (seperti dataset asli)
- Output: CSV baru dengan tambahan kolom 'predicted_label' dan 'confidence'
- Text quality features (gibberish detection)
"""

import numpy as np
import pandas as pd
import pickle
import sys
import os
from collections import Counter
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from text_quality_detector import extract_text_quality_features, get_text_quality_feature_names

# Inisialisasi Sastrawi
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

factory_stopword = StopWordRemoverFactory()
stopword_remover = factory_stopword.create_stop_word_remover()

# Kamus normalisasi
normalization_dict = {
    'gak': 'tidak', 'ga': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak', 'gk': 'tidak',
    'tdk': 'tidak', 'bgt': 'banget', 'bgd': 'banget', 'bngt': 'banget', 'emg': 'memang',
    'udh': 'sudah', 'udah': 'sudah', 'blm': 'belum', 'jg': 'juga', 'dgn': 'dengan',
    'yg': 'yang', 'utk': 'untuk', 'sy': 'saya', 'kl': 'kalau', 'klo': 'kalau',
    'thx': 'terima kasih', 'thanks': 'terima kasih', 'mantul': 'mantap', 'mantep': 'mantap',
    'rekomend': 'rekomendasi', 'rekomen': 'rekomendasi', 'recommended': 'rekomendasi',
}

# ---------------------------
# === PREPROCESSING FUNCTIONS
# ---------------------------

def preprocess_text(text):
    """Preprocessing text untuk prediksi"""
    if pd.isna(text) or str(text).strip() == '':
        return ''

    # Case folding
    text = str(text).lower()

    # Cleansing
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Normalisasi
    words = text.split()
    words = [normalization_dict.get(word, word) for word in words]
    text = ' '.join(words)

    # Stopword removal
    text = stopword_remover.remove(text)

    # Stemming
    text = stemmer.stem(text)

    return text


# ---------------------------
# === TF-IDF VECTORIZER (sama dengan training)
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
# === LINEAR SVM (sama dengan training)
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

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b


# ---------------------------
# === FEATURE ENGINEERING (sama dengan training)
# ---------------------------

def extract_features(df):
    """Extract semua fitur penting untuk deteksi fake review"""

    print("  Extracting features...")

    # 1. Text length
    df['text_length'] = df['text_preprocessed'].fillna('').apply(lambda x: len(str(x).split()))

    # 2. Stars (normalize to 0-1)
    df['stars_norm'] = df['stars'].fillna(0) / 5.0

    # 3. Has image
    df['has_image'] = df['reviewImageUrls'].apply(
        lambda x: 0 if pd.isna(x) or str(x).strip() == '' or str(x).lower() == 'nan' else 1
    )

    # 4. Reviewer number of reviews (normalize dengan log)
    df['reviewer_count'] = df['reviewerNumberOfReviews'].fillna(0)
    df['reviewer_count_log'] = np.log1p(df['reviewer_count'])

    # 5. Is local guide
    df['is_local_guide'] = df['isLocalGuide'].apply(
        lambda x: 1 if str(x).lower() in ['true', '1', 'yes'] else 0
    )

    # 6. Detail rating all 5s
    def check_all_fives(x):
        if pd.isna(x) or str(x).strip() == '':
            return 0
        try:
            numbers = [float(s) for s in str(x).replace(',', ' ').split() if s.replace('.','').isdigit()]
            if len(numbers) > 0:
                return 1 if all(n == 5.0 for n in numbers) else 0
        except:
            pass
        return 0

    df['detail_rating_all_5'] = df['reviewDetailedRating'].apply(check_all_fives)

    # 7. Published date - time patterns
    df['published_date'] = pd.to_datetime(df['publishedAtDate'], errors='coerce')
    df['publish_hour'] = df['published_date'].dt.hour.fillna(0).astype(int)
    df['publish_day'] = df['published_date'].dt.date

    # 8. Same day review count
    if df['publish_day'].notna().sum() > 0:
        day_counts = df.groupby('publish_day').size()
        df['same_day_count'] = df['publish_day'].map(day_counts).fillna(1).astype(int)
    else:
        df['same_day_count'] = 1

    # 9. Suspicious patterns
    df['pattern_5star_short'] = ((df['stars'] == 5) & (df['text_length'] < 5)).astype(int)
    df['pattern_5star_nophoto'] = ((df['stars'] == 5) & (df['has_image'] == 0)).astype(int)
    df['pattern_new_reviewer'] = (df['reviewer_count'] <= 2).astype(int)
    df['pattern_same_day'] = (df['same_day_count'] > 5).astype(int)

    # 10. Text Quality Features (Gibberish Detection)
    print("  [NEW] Adding text quality features...")
    df = extract_text_quality_features(df, text_column='text_preprocessed')

    print("  [OK] Features extracted")

    return df


# ---------------------------
# === LOAD MODEL
# ---------------------------

def load_model(model_path="model_output/svm_multifeature_model.pkl"):
    """Load trained SVM model"""
    print(f"Loading model dari: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    print("[OK] Model berhasil dimuat!")
    print(f"\nModel Info:")
    print(f"  TF-IDF features: {model_data['hyperparameters']['max_features_tfidf']}")
    print(f"  Numeric features: {len(model_data['numeric_features'])}")
    print(f"\nTest Set Performance:")
    test_metrics = model_data['test_metrics']
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")

    return model_data


# ---------------------------
# === PREDICT CSV
# ---------------------------

def predict_csv(input_csv, output_csv, model_data):
    """
    Prediksi fake review dari CSV file

    Parameters:
    - input_csv: path ke file CSV input
    - output_csv: path untuk menyimpan hasil prediksi
    - model_data: dictionary dari load_model()
    """

    print("\n" + "="*80)
    print("PREDIKSI FAKE REVIEW DARI CSV")
    print("="*80)

    # Load data
    print(f"\n[1/4] Membaca data dari: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"  Total data: {len(df)} baris")

    # Backup kolom asli
    df_original = df.copy()

    # Handle text preprocessing
    if 'text_preprocessed' not in df.columns:
        print("\n  ⚠ Kolom 'text_preprocessed' tidak ditemukan.")

        # Cari kolom text yang tersedia
        if 'text' in df.columns:
            text_col = 'text'
        elif 'textTranslated' in df.columns:
            text_col = 'textTranslated'
        else:
            raise ValueError("Tidak ditemukan kolom 'text' atau 'textTranslated' di dataset!")

        print(f"  → Menggunakan kolom '{text_col}' dan melakukan preprocessing...")
        print("     (Case folding → Cleansing → Normalisasi → Stopword removal → Stemming)")

        # Lakukan preprocessing
        df['text_preprocessed'] = df[text_col].apply(preprocess_text)
        print("  [OK] Preprocessing selesai!")

    # Cek kolom yang diperlukan
    required_cols = ['text_preprocessed', 'stars', 'reviewerNumberOfReviews',
                     'reviewImageUrls', 'isLocalGuide', 'publishedAtDate', 'reviewDetailedRating']

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\n⚠ Kolom yang hilang: {missing_cols}")
        print("  Akan menggunakan nilai default untuk kolom yang hilang")
        for col in missing_cols:
            df[col] = np.nan

    # Filter data valid (text yang tidak kosong setelah preprocessing)
    df['text_preprocessed'] = df['text_preprocessed'].fillna('').astype(str)
    valid_mask = df['text_preprocessed'].str.strip().str.len() > 0
    df_valid = df[valid_mask].copy()
    df_invalid = df[~valid_mask].copy()

    print(f"  Data valid: {len(df_valid)} baris")
    if len(df_invalid) > 0:
        print(f"  ⚠ Data invalid (text kosong): {len(df_invalid)} baris")

    # [2/4] FEATURE ENGINEERING
    print(f"\n[2/4] Feature Engineering")
    df_valid = extract_features(df_valid)

    # [3/4] PREPARE FEATURES
    print(f"\n[3/4] Preparing features untuk prediksi...")

    # Text features
    X_text = df_valid['text_preprocessed'].fillna('').tolist()

    # Numeric features
    numeric_features = model_data['numeric_features']
    X_numeric = df_valid[numeric_features].fillna(0).values

    # TF-IDF
    vectorizer = model_data['vectorizer']
    X_tfidf = vectorizer.transform(X_text)

    # Combine features
    X = np.hstack([X_tfidf, X_numeric])

    print(f"  Feature shape: {X.shape}")

    # [4/4] PREDIKSI
    print(f"\n[4/4] Melakukan prediksi...")

    svm = model_data['svm']

    # Prediksi
    y_pred = svm.predict(X)
    decision_scores = svm.decision_function(X)

    # Convert ke label string
    label_mapping_inv = {v: k for k, v in model_data['label_mapping'].items()}
    predicted_labels = [label_mapping_inv[pred] for pred in y_pred]

    # Hitung confidence
    confidences = 1 / (1 + np.exp(-np.abs(decision_scores)))

    # Tambahkan hasil ke dataframe
    df_valid['predicted_label'] = predicted_labels
    df_valid['confidence'] = confidences
    df_valid['decision_score'] = decision_scores

    # POST-PROCESSING: Rule-based override untuk gibberish
    print(f"\n  [POST-PROCESSING] Applying gibberish detection filter...")
    gibberish_threshold = 0.65  # Jika gibberish score > 0.65, override ke FAKE

    # Cari review dengan gibberish score tinggi
    high_gibberish = df_valid['tq_gibberish_score'] > gibberish_threshold

    if high_gibberish.sum() > 0:
        print(f"    Found {high_gibberish.sum()} reviews with high gibberish score (>{gibberish_threshold})")
        print(f"    Overriding predictions to FAKE...")

        # Override prediksi
        df_valid.loc[high_gibberish, 'predicted_label'] = 'FAKE'
        df_valid.loc[high_gibberish, 'confidence'] = df_valid.loc[high_gibberish, 'tq_gibberish_score']
        df_valid.loc[high_gibberish, 'decision_score'] = -1.0  # Negative for FAKE

        print(f"    [OK] {high_gibberish.sum()} predictions overridden based on gibberish detection")
    else:
        print(f"    No reviews with high gibberish score found")

    # Handle data invalid (set sebagai unknown)
    if len(df_invalid) > 0:
        df_invalid['predicted_label'] = 'UNKNOWN'
        df_invalid['confidence'] = 0.0
        df_invalid['decision_score'] = 0.0

    # Combine kembali
    df_result = pd.concat([df_valid, df_invalid]).sort_index()

    # Statistik hasil
    print(f"\n  [OK] Prediksi selesai!")
    print(f"\n  Distribusi prediksi:")
    pred_dist = df_result['predicted_label'].value_counts()
    for label, count in pred_dist.items():
        print(f"    - {label}: {count} ({count/len(df_result)*100:.2f}%)")

    print(f"\n  Confidence statistics:")
    print(f"    - Mean: {df_result['confidence'].mean():.4f}")
    print(f"    - Median: {df_result['confidence'].median():.4f}")
    print(f"    - High confidence (>0.8): {(df_result['confidence'] > 0.8).sum()} reviews")

    # Simpan hasil
    print(f"\n[Save] Menyimpan hasil ke: {output_csv}")
    df_result.to_csv(output_csv, index=False)
    print(f"  [OK] Hasil disimpan!")

    # Summary file
    summary_file = output_csv.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SUMMARY PREDIKSI FAKE REVIEW\n")
        f.write("="*80 + "\n\n")
        f.write(f"Input file: {input_csv}\n")
        f.write(f"Output file: {output_csv}\n")
        f.write(f"Total data: {len(df_result)}\n\n")

        f.write("Distribusi Prediksi:\n")
        for label, count in pred_dist.items():
            f.write(f"  - {label}: {count} ({count/len(df_result)*100:.2f}%)\n")

        f.write(f"\nConfidence Statistics:\n")
        f.write(f"  - Mean: {df_result['confidence'].mean():.4f}\n")
        f.write(f"  - Median: {df_result['confidence'].median():.4f}\n")
        f.write(f"  - Std: {df_result['confidence'].std():.4f}\n")

        f.write(f"\nModel Performance (Test Set):\n")
        test_metrics = model_data['test_metrics']
        f.write(f"  - Accuracy:  {test_metrics['accuracy']:.4f}\n")
        f.write(f"  - Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"  - Recall:    {test_metrics['recall']:.4f}\n")
        f.write(f"  - F1-Score:  {test_metrics['f1_score']:.4f}\n")

    print(f"  [OK] Summary disimpan: {summary_file}")

    print("\n" + "="*80)
    print("PREDIKSI SELESAI!")
    print("="*80)

    return df_result


# ---------------------------
# === MAIN
# ---------------------------

def main():
    print("="*80)
    print("SVM MULTI-FEATURE - PREDIKSI FAKE REVIEW")
    print("="*80)

    # Cek arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python svm_multifeature_predict.py <input_csv> [output_csv]")
        print("\nContoh:")
        print("  python svm_multifeature_predict.py data_baru.csv hasil_prediksi.csv")
        print("\nAtau edit script ini dan set INPUT_CSV & OUTPUT_CSV di bagian konfigurasi")
        print("\n" + "="*80)

        # Default files (untuk testing)
        ini_input = input("Masukan Nama File input csv")
        INPUT_CSV = ini_input
        OUTPUT_CSV = "model_output/hasil_prediksi.csv"

        print(f"\nMenggunakan default files:")
        print(f"  Input:  {INPUT_CSV}")
        print(f"  Output: {OUTPUT_CSV}")

        if not os.path.exists(INPUT_CSV):
            print(f"\n[ERROR] File input tidak ditemukan: {INPUT_CSV}")
            return
    else:
        INPUT_CSV = sys.argv[1]
        OUTPUT_CSV = sys.argv[2] if len(sys.argv) > 2 else INPUT_CSV.replace('.csv', '_predicted.csv')

    # Load model
    try:
        model_data = load_model()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPastikan Anda sudah training model dengan:")
        print("  python svm_multifeature_train.py")
        return
    except Exception as e:
        print(f"\n❌ ERROR loading model: {e}")
        return

    # Predict
    try:
        df_result = predict_csv(INPUT_CSV, OUTPUT_CSV, model_data)
        print(f"\n✓ Selesai! Cek file: {OUTPUT_CSV}")
    except FileNotFoundError:
        print(f"\n❌ ERROR: File input tidak ditemukan: {INPUT_CSV}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
