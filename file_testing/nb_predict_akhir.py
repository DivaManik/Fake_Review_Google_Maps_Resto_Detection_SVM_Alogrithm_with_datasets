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

class TfidfVectorizerManual:
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

def parse_image_field(x):
    """Parse reviewImageUrls field (robust parser from training script)"""
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


def extract_features(df):
    """
    Extract semua fitur untuk prediksi (SAMA PERSIS dengan training script)
    """
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

def load_model(model_path="../file_training/nb_model_output/naive_bayes_multifeature_model.pkl"):
    print(f"Loading model dari: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    print("[OK] Model berhasil dimuat!")
    print(f"\nModel Info:")
    print(f"  TF-IDF features: {model_data['max_features_tfidf']}")
    print(f"  Numeric features: {len(model_data['numeric_features'])}")
    nb_model = model_data['nb_model']
    print(f"  Text weight: {nb_model.text_weight}")
    print(f"  Numeric weight: {nb_model.numeric_weight}")
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
    tfidf = model_data['tfidf']
    X_tfidf = tfidf.transform(X_text)

    # Standardize numeric features using saved scaler
    scaler = model_data['scaler']
    X_numeric = scaler.transform(X_numeric)

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

    # Convert ke label string (mapping: 1 -> 'FAKE', -1 -> 'REAL')
    label_mapping_inv = {1: 'FAKE', -1: 'REAL'}
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

    # Handle data invalid (text kosong = FAKE)
    if len(df_invalid) > 0:
        df_invalid['predicted_label'] = 'FAKE'
        df_invalid['confidence'] = 1.0  # High confidence karena jelas empty text
        df_invalid['prob_fake'] = 1.0
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
        INPUT_CSV = "../dataset_testing/tes.csv"
        OUTPUT_CSV = "nb_model_output/hasil_prediksi_nb.csv"

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