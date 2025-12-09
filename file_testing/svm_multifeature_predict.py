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

class TfidfVectorizerManual:
    """TF-IDF Vectorizer dari scratch"""

    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocab = {}
        self.idf = {}

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
        self.vocab = {word: idx for idx, (word, _) in enumerate(most_common_words)}

        n_documents = len(documents)
        for word in self.vocab:
            self.idf[word] = np.log(n_documents / (word_doc_count[word] + 1))

    def transform(self, documents):
        n_documents = len(documents)
        n_features = len(self.vocab)
        tfidf_matrix = np.zeros((n_documents, n_features))

        for doc_idx, doc in enumerate(documents):
            words = str(doc).split()
            word_count = Counter(words)
            total_words = len(words)

            if total_words == 0:
                continue

            for word, count in word_count.items():
                if word in self.vocab:
                    word_idx = self.vocab[word]
                    tf = count / total_words
                    tfidf_matrix[doc_idx, word_idx] = tf * self.idf[word]

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
# === STANDARD SCALER (sama dengan training)
# ---------------------------

class StandardScalerSimple:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ---------------------------
# === HELPER FUNCTIONS
# ---------------------------

def parse_image_field(x):
    """Parse reviewImageUrls field (robust parser)"""
    if pd.isna(x):
        return []
    if isinstance(x, (list, tuple)):
        raw = list(x)
    else:
        s = str(x).strip()
        try:
            import json
            parsed = json.loads(s)
            if isinstance(parsed, (list, tuple)):
                raw = parsed
            else:
                raw = [parsed]
        except Exception:
            try:
                import ast
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    raw = list(parsed)
                else:
                    raw = [parsed]
            except Exception:
                if ',' in s and not s.startswith('http'):
                    raw = [p.strip() for p in s.split(',') if p.strip()!='']
                elif ';' in s:
                    raw = [p.strip() for p in s.split(';') if p.strip()!='']
                elif '|' in s:
                    raw = [p.strip() for p in s.split('|') if p.strip()!='']
                else:
                    if s.lower() in ('', 'nan', 'none', '[]'):
                        raw = []
                    else:
                        raw = [s]
    clean = []
    for item in raw:
        if item is None: continue
        t = str(item).strip()
        if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
            t = t[1:-1].strip()
        if t == '' or t.lower() in ('nan','none','null'): continue
        clean.append(t)
    return clean


# ---------------------------
# === FEATURE ENGINEERING (sama dengan training)
# ---------------------------

def extract_features(df):
    """Extract semua fitur untuk prediksi (SAMA PERSIS dengan training script)"""
    print("  Extracting features...")

    # basic text features
    df['text_word_count'] = df['text_preprocessed'].apply(lambda x: len(str(x).split()))
    df['text_char_count'] = df['text_preprocessed'].apply(lambda x: len(str(x)))
    df['avg_word_length'] = df['text_preprocessed'].apply(lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split())>0 else 0.0)
    df['exclamation_count'] = df['text_preprocessed'].str.count('!').fillna(0).astype(int)
    df['question_count'] = df['text_preprocessed'].str.count(r'\?').fillna(0).astype(int)
    df['punctuation_count'] = df['text_preprocessed'].str.count(r'[.,;:!?"\'()\\-]').fillna(0).astype(int)
    df['uppercase_word_count'] = df['text_preprocessed'].apply(lambda s: sum(1 for w in str(s).split() if w.isupper()))
    df['uppercase_word_ratio'] = df['uppercase_word_count'] / df['text_word_count'].replace(0,1)

    # stars & reviewer
    df['stars'] = df['stars'].fillna(0).astype(float)
    df['stars_norm'] = df['stars'] / 5.0
    df['reviewer_count'] = pd.to_numeric(df['reviewerNumberOfReviews'], errors='coerce').fillna(0).astype(float)
    df['reviewer_count_log'] = np.log1p(df['reviewer_count'])
    df['is_local_guide'] = df['isLocalGuide'].apply(lambda x: 1 if str(x).lower() in ('1','true','yes') else 0)

    # images (using robust parser)
    print("  Parsing reviewImageUrls ...")
    df['review_images_list'] = df['reviewImageUrls'].apply(parse_image_field)
    df['n_images'] = df['review_images_list'].apply(lambda lst: len(lst))
    df['has_image'] = (df['n_images'] > 0).astype(int)

    # detail rating all 5
    def check_all_fives(x):
        try:
            nums = [float(s) for s in str(x).replace(',', ' ').split() if s.replace('.','').isdigit()]
            return 1 if (len(nums)>0 and all(n==5.0 for n in nums)) else 0
        except:
            return 0
    df['detail_rating_all_5'] = df['reviewDetailedRating'].apply(check_all_fives)

    # published date
    df['published_date'] = pd.to_datetime(df['publishedAtDate'], errors='coerce')
    df['publish_hour'] = df['published_date'].dt.hour.fillna(0).astype(int)
    df['publish_day'] = df['published_date'].dt.date
    if df['publish_day'].notna().sum() > 0:
        day_counts = df.groupby('publish_day').size()
        df['same_day_count'] = df['publish_day'].map(day_counts).fillna(1).astype(int)
    else:
        df['same_day_count'] = 1

    # patterns
    df['pattern_5star_short'] = ((df['stars'] == 5) & (df['text_word_count'] < 5)).astype(int)
    df['pattern_5star_nophoto'] = ((df['stars'] == 5) & (df['has_image'] == 0)).astype(int)
    df['pattern_new_reviewer'] = (df['reviewer_count'] <= 2).astype(int)
    df['pattern_same_day'] = (df['same_day_count'] > 5).astype(int)

    # text quality
    print("  Adding text quality features...")
    try:
        df = extract_text_quality_features(df, text_column='text_preprocessed')
    except Exception as e:
        print(f"  Warning: text quality features failed: {e}")
        # fallback: create dummy features
        tq_features = get_text_quality_feature_names()
        for feat in tq_features:
            if feat not in df.columns:
                df[feat] = 0.0

    print("  [OK] Features extracted")
    return df


# ---------------------------
# === LOAD MODEL
# ---------------------------

def load_model(model_path="../file_training/svm_model_output/svm_multifeature_model_unified.pkl"):
    """Load trained SVM model"""
    print(f"Loading model dari: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    print("[OK] Model berhasil dimuat!")
    print(f"\nModel Info:")
    print(f"  TF-IDF features: {model_data['max_features_tfidf']}")
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
    tfidf = model_data['tfidf']
    X_tfidf = tfidf.transform(X_text)

    # Standardize numeric features using saved scaler
    scaler = model_data['scaler']
    X_numeric_scaled = scaler.transform(X_numeric)

    # Combine features
    X = np.hstack([X_tfidf, X_numeric_scaled])

    print(f"  Feature shape: {X.shape}")

    # [4/4] PREDIKSI
    print(f"\n[4/4] Melakukan prediksi...")

    svm = model_data['svm']

    # Prediksi
    y_pred = svm.predict(X)
    decision_scores = svm.decision_function(X)

    # Convert ke label string (mapping: 1 -> 'FAKE', -1 -> 'REAL')
    label_mapping_inv = {1: 'FAKE', -1: 'REAL'}
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

    # Handle data invalid (text kosong = FAKE)
    if len(df_invalid) > 0:
        df_invalid['predicted_label'] = 'FAKE'
        df_invalid['confidence'] = 1.0  # High confidence karena jelas empty text
        df_invalid['decision_score'] = -1.0  # Negative score for FAKE

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

    # Create output directory if not exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"  [OK] Created directory: {output_dir}")

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

    # Default paths
    DEFAULT_INPUT = "../dataset_testing/tes.csv"
    DEFAULT_OUTPUT = "svm_model_output/hasil_prediksi_svm.csv"

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python svm_multifeature_predict.py <input_csv> [output_csv]")
        print("\nContoh:")
        print("  python svm_multifeature_predict.py data_baru.csv hasil_prediksi.csv")
        print("\nMenggunakan default files:")
        print(f"  Input:  {DEFAULT_INPUT}")
        print(f"  Output: {DEFAULT_OUTPUT}")
        print("=" * 80)

        INPUT_CSV = DEFAULT_INPUT
        OUTPUT_CSV = DEFAULT_OUTPUT

        if not os.path.exists(INPUT_CSV):
            print(f"\n[ERROR] File input tidak ditemukan: {INPUT_CSV}")
            return
    else:
        INPUT_CSV = sys.argv[1]
        OUTPUT_CSV = sys.argv[2] if len(sys.argv) > 2 else INPUT_CSV.replace('.csv', '_predicted_svm.csv')

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
