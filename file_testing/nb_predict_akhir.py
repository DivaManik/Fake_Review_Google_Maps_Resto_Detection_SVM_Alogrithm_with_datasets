"""
Prediksi fake review dari CSV file menggunakan Naive Bayes multi-feature model
- Input: CSV file dengan kolom lengkap (dataset asli)
- Output: CSV baru dengan tambahan kolom 'predicted_label' dan 'confidence'
- Text quality features (gibberish detection)

PENJELASAN WORKFLOW PREDIKSI:
1. Load model yang sudah di-training
2. Preprocessing text (jika belum ada)
3. Extract features (sama dengan training)
4. Transform ke format yang sesuai (TF-IDF + numeric)
5. Prediksi menggunakan Naive Bayes
6. Post-processing dengan rule-based detection
7. Save hasil prediksi
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

factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()  

factory_stopword = StopWordRemoverFactory()
stopword_remover = factory_stopword.create_stop_word_remover() 

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
    if pd.isna(text) or str(text).strip() == '':
        return ''

    # 1. Case folding
    text = str(text).lower()

    # 2. Cleansing
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Buang mention (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Buang hashtag (#hashtag)
    text = re.sub(r'#\w+', '', text)
    
    # Buang punctuation (tanda baca)
    # [^\w\s] = bukan huruf, bukan whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Buang angka
    text = re.sub(r'\d+', '', text)
    
    # Buang whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    # 3. Normalisasi
    words = text.split()
    words = [normalization_dict.get(word, word) for word in words]
    text = ' '.join(words)

    # 4. Stopword removal
    text = stopword_remover.remove(text)

    # 5. Stemming
    text = stemmer.stem(text)

    return text


# ---------------------------
# === TF-IDF VECTORIZER (sama dengan training)
# ---------------------------

class TfidfVectorizer:
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
# === MULTINOMIAL NAIVE BAYES (sama dengan training)
# ---------------------------

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior = {}
        self.feature_log_prob = {}
        self.classes = None

    def fit(self, X, y):
        """Training model"""
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        for class_label in self.classes:
            X_class = X[y == class_label]
            n_samples_class = X_class.shape[0]

            self.class_log_prior[class_label] = np.log(n_samples_class / n_samples)

            word_counts = X_class.sum(axis=0)
            total_count = word_counts.sum()

            self.feature_log_prob[class_label] = np.log(
                (word_counts + self.alpha) / (total_count + self.alpha * n_features)
            )

    def predict_log_proba(self, X):
        """Prediksi log probability"""
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        log_proba = np.zeros((n_samples, n_classes))

        for idx, class_label in enumerate(self.classes):
            log_prior = self.class_log_prior[class_label]
            log_likelihood = X.dot(self.feature_log_prob[class_label])
            log_proba[:, idx] = log_prior + log_likelihood

        return log_proba

    def predict(self, X):
        """Prediksi kelas"""
        log_proba = self.predict_log_proba(X)
        return self.classes[np.argmax(log_proba, axis=1)]


# ---------------------------
# === GAUSSIAN NAIVE BAYES (sama dengan training)
# ---------------------------

class GaussianNB:
    def __init__(self):
        self.class_prior = {}
        self.mean = {}
        self.var = {}
        self.classes = None

    def fit(self, X, y):
        """Training model"""
        self.classes = np.unique(y)
        n_samples = X.shape[0]

        for class_label in self.classes:
            X_class = X[y == class_label]
            n_samples_class = X_class.shape[0]

            self.class_prior[class_label] = n_samples_class / n_samples
            self.mean[class_label] = X_class.mean(axis=0)
            self.var[class_label] = X_class.var(axis=0) + 1e-9

    def _calculate_likelihood(self, X, mean, var):
        """Hitung likelihood menggunakan distribusi Gaussian"""
        exponent = -0.5 * ((X - mean) ** 2) / var
        normalizer = -0.5 * np.log(2 * np.pi * var)
        return (normalizer + exponent).sum(axis=1)

    def predict_log_proba(self, X):
        """Prediksi log probability"""
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        log_proba = np.zeros((n_samples, n_classes))

        for idx, class_label in enumerate(self.classes):
            log_prior = np.log(self.class_prior[class_label])
            log_likelihood = self._calculate_likelihood(
                X, 
                self.mean[class_label], 
                self.var[class_label]
            )
            log_proba[:, idx] = log_prior + log_likelihood

        return log_proba

    def predict(self, X):
        """Prediksi kelas"""
        log_proba = self.predict_log_proba(X)
        return self.classes[np.argmax(log_proba, axis=1)]


# ---------------------------
# === HYBRID NAIVE BAYES (sama dengan training)
# ---------------------------

class HybridNaiveBayes:
    def __init__(self, alpha=1.0, text_weight=0.6, numeric_weight=0.4):
        self.multinomial_nb = MultinomialNB(alpha=alpha)
        self.gaussian_nb = GaussianNB()
        self.text_weight = text_weight
        self.numeric_weight = numeric_weight
        self.classes = None

    def fit(self, X_text, X_numeric, y):
        """Training model"""
        self.multinomial_nb.fit(X_text, y)
        self.gaussian_nb.fit(X_numeric, y)
        self.classes = np.unique(y)

    def predict_proba(self, X_text, X_numeric):
        """
        Prediksi probability dengan gabungkan probability dari Multinomial NB dan Gaussian NB
        dengan weighted combination
        """
        log_proba_text = self.multinomial_nb.predict_log_proba(X_text)
        log_proba_numeric = self.gaussian_nb.predict_log_proba(X_numeric)

        log_proba_combined = (
            self.text_weight * log_proba_text + 
            self.numeric_weight * log_proba_numeric
        )

        # ubah dari log ke probability
        proba = np.exp(log_proba_combined)
        proba = proba / proba.sum(axis=1, keepdims=True)

        return proba

    def predict(self, X_text, X_numeric):
        """Prediksi kelas"""
        proba = self.predict_proba(X_text, X_numeric)
        return self.classes[np.argmax(proba, axis=1)]

    def predict_log_proba(self, X_text, X_numeric):
        """Prediksi log probability"""
        log_proba_text = self.multinomial_nb.predict_log_proba(X_text)
        log_proba_numeric = self.gaussian_nb.predict_log_proba(X_numeric)
        
        return (
            self.text_weight * log_proba_text + 
            self.numeric_weight * log_proba_numeric
        )


# ---------------------------
# === FEATURE ENGINEERING (sama dengan training)
# ---------------------------

def extract_features(df):
    """
    Extract semua fitur untuk prediksi
    """

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

def load_model(model_path="model_output/naive_bayes_multifeature_model.pkl"):
    print(f"Loading model dari: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    print("[OK] Model berhasil dimuat!")
    print(f"\nModel Info:")
    print(f"  TF-IDF features: {model_data['hyperparameters']['max_features_tfidf']}")
    print(f"  Numeric features: {len(model_data['numeric_features'])}")
    print(f"  Text weight: {model_data['hyperparameters']['text_weight']}")
    print(f"  Numeric weight: {model_data['hyperparameters']['numeric_weight']}")
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
    
    workflownya:
    1. Load data dari CSV
    2. Preprocessing text (jika belum ada)
    3. Feature extraction
    4. Transform ke format TF-IDF + numeric
    5. Prediksi menggunakan Naive Bayes
    6. Hitung confidence score
    7. Post-processing dengan rule-based detection
    8. Save hasil ke CSV
    """

    print("\n" + "="*80)
    print("PREDIKSI FAKE REVIEW DARI CSV - NAIVE BAYES")
    print("="*80)

    # [1/4] LOAD DATA
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

    # Filter data valid (text yang tidak kosong)
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

    # TF-IDF transformation
    vectorizer = model_data['vectorizer']
    X_tfidf = vectorizer.transform(X_text)

    print(f"  Feature shapes:")
    print(f"    - Text (TF-IDF): {X_tfidf.shape}")
    print(f"    - Numeric: {X_numeric.shape}")

    # [4/4] PREDIKSI
    print(f"\n[4/4] Melakukan prediksi...")

    # Get model
    nb_model = model_data['nb_model']

    # Prediksi
    y_pred = nb_model.predict(X_tfidf, X_numeric)
    
    #predict_proba() memberikan probability untuk setiap kelas
    proba = nb_model.predict_proba(X_tfidf, X_numeric)

    # Convert ke label string
    label_mapping_inv = {v: k for k, v in model_data['label_mapping'].items()}
    predicted_labels = [label_mapping_inv[pred] for pred in y_pred]

    # Hitung confidence score
    confidences = proba.max(axis=1)

    # Tambahkan hasil ke dataframe
    df_valid['predicted_label'] = predicted_labels
    df_valid['confidence'] = confidences
    
    # Add probability untuk kedua kelas
    df_valid['prob_fake'] = proba[:, 0]   
    df_valid['prob_real'] = proba[:, 1]  

    # POST-PROCESSING: Rule-based override untuk gibberish
    print(f"\n  [POST-PROCESSING] Applying gibberish detection filter...")
    gibberish_threshold = 0.65

    high_gibberish = df_valid['tq_gibberish_score'] > gibberish_threshold

    if high_gibberish.sum() > 0:
        print(f"    Found {high_gibberish.sum()} reviews with high gibberish score (>{gibberish_threshold})")
        print(f"    Overriding predictions to FAKE...")

        # Override prediksi
        df_valid.loc[high_gibberish, 'predicted_label'] = 'FAKE'
        df_valid.loc[high_gibberish, 'confidence'] = df_valid.loc[high_gibberish, 'tq_gibberish_score']
        df_valid.loc[high_gibberish, 'prob_fake'] = 1.0
        df_valid.loc[high_gibberish, 'prob_real'] = 0.0

        print(f"    [OK] {high_gibberish.sum()} predictions overridden based on gibberish detection")
    else:
        print(f"    No reviews with high gibberish score found")

    # Handle data invalid
    if len(df_invalid) > 0:
        df_invalid['predicted_label'] = 'UNKNOWN'
        df_invalid['confidence'] = 0.0
        df_invalid['prob_fake'] = 0.0
        df_invalid['prob_real'] = 0.0

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
        f.write("SUMMARY PREDIKSI FAKE REVIEW - NAIVE BAYES\n")
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
    """
    Main function untuk menjalankan prediksi
    
    PENJELASAN:
    1. Parse command line arguments atau gunakan default
    2. Load model yang sudah di-training
    3. Prediksi CSV file
    4. Simpan hasil
    """
    print("="*80)
    print("NAIVE BAYES MULTI-FEATURE - PREDIKSI FAKE REVIEW")
    print("="*80)

    # Cek arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python naive_bayes_multifeature_predict.py <input_csv> [output_csv]")
        print("\nContoh:")
        print("  python naive_bayes_multifeature_predict.py data_baru.csv hasil_prediksi.csv")
        print("\nAtau edit script ini dan set INPUT_CSV & OUTPUT_CSV di bagian konfigurasi")
        print("\n" + "="*80)

        # Default files (untuk testing)
        ini_input = r"E:\Machine_Learning\Projek Fake Review\projek v2\dataset\tes.csv"
        INPUT_CSV = ini_input
        OUTPUT_CSV = "model_output/hasil_prediksi_nb.csv"

        print(f"\nMenggunakan files:")
        print(f"  Input:  {INPUT_CSV}")
        print(f"  Output: {OUTPUT_CSV}")

        if not os.path.exists(INPUT_CSV):
            print(f"\n[ERROR] File input tidak ditemukan: {INPUT_CSV}")
            return
    else:
        INPUT_CSV = sys.argv[1]
        OUTPUT_CSV = sys.argv[2] if len(sys.argv) > 2 else INPUT_CSV.replace('.csv', '_predicted_nb.csv')

    # Load model
    try:
        model_data = load_model()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPastikan Anda sudah training model dengan:")
        print("  python naive_bayes_multifeature_train.py")
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