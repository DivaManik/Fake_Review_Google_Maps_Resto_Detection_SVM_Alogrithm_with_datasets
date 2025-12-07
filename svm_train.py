"""
Training SVM OPTIMAL untuk deteksi fake review
Gabungan fitur terbaik:
- Preprocessing: Sastrawi (stemming + stopword) + normalisasi singkatan
- TF-IDF: N-grams (1-2) dengan min_df
- SVM: Vectorized batch training + early stopping
- StandardScaler untuk normalisasi fitur numerik
- Numeric features lengkap (17+ fitur)
- Text quality features (gibberish detection)
- Visualisasi lengkap
"""

import os
import numpy as np
import pandas as pd
import pickle
import re
from collections import Counter
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from text_quality_detector import extract_text_quality_features, get_text_quality_feature_names

# ---------------------------
# === SASTRAWI PREPROCESSING
# ---------------------------

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    stemmer = StemmerFactory().create_stemmer()
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    HAS_SASTRAWI = True
    print("[OK] Sastrawi loaded successfully")
except Exception as e:
    print(f"[WARNING] Sastrawi not available: {e}")
    print("  Install via: pip install Sastrawi")
    HAS_SASTRAWI = False
    class DummyStemmer:
        def stem(self, text):
            return text
    class DummyStopwordRemover:
        def remove(self, text):
            return text
    stemmer = DummyStemmer()
    stopword_remover = DummyStopwordRemover()

# Normalisasi singkatan Indonesia
NORMALIZATION_DICT = {
    'gak': 'tidak', 'ga': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak', 'gk': 'tidak',
    'tdk': 'tidak', 'bgt': 'banget', 'bgd': 'banget', 'bngt': 'banget', 'emg': 'memang',
    'udh': 'sudah', 'udah': 'sudah', 'blm': 'belum', 'jg': 'juga', 'dgn': 'dengan',
    'yg': 'yang', 'utk': 'untuk', 'sy': 'saya', 'kl': 'kalau', 'klo': 'kalau',
    'thx': 'terima kasih', 'thanks': 'terima kasih', 'makasih': 'terima kasih',
    'mantul': 'mantap', 'mantep': 'mantap', 'mntap': 'mantap',
    'rekomend': 'rekomendasi', 'rekomen': 'rekomendasi', 'recommended': 'rekomendasi',
    'bsk': 'besok', 'skrg': 'sekarang', 'skg': 'sekarang', 'hrs': 'harus',
    'org': 'orang', 'org2': 'orang orang', 'sm': 'sama', 'dr': 'dari',
    'krn': 'karena', 'karna': 'karena', 'soalnya': 'karena', 'tp': 'tapi',
    'bkn': 'bukan', 'lg': 'lagi', 'aja': 'saja', 'aj': 'saja', 'doang': 'saja',
    'bisa': 'dapat', 'bs': 'dapat', 'gmn': 'bagaimana', 'gimana': 'bagaimana',
    'knp': 'kenapa', 'dmn': 'dimana', 'kmn': 'kemana', 'gt': 'gitu', 'gtu': 'gitu',
    'spy': 'supaya', 'biar': 'supaya', 'pdhl': 'padahal', 'sbnrnya': 'sebenarnya',
    'emang': 'memang', 'emg': 'memang', 'mmg': 'memang',
    'pesen': 'pesan', 'nyoba': 'coba', 'cobain': 'coba',
    'enk': 'enak', 'uenak': 'enak', 'ena': 'enak',
    'bgs': 'bagus', 'baguslah': 'bagus', 'top': 'bagus', 'ok': 'bagus', 'oke': 'bagus',
    'jelek': 'buruk', 'jlk': 'buruk', 'ancur': 'buruk', 'parah': 'buruk',
    'lmyn': 'lumayan', 'lmayan': 'lumayan',
    'dtg': 'datang', 'plg': 'pulang', 'msk': 'masuk',
}


def preprocess_text_full(text):
    """
    Preprocessing lengkap untuk text Indonesia:
    1. Lowercase (case folding)
    2. Remove URLs, mentions, hashtags
    3. Remove special characters & numbers
    4. Normalisasi singkatan
    5. Remove stopwords (Sastrawi)
    6. Stemming (Sastrawi)
    """
    if pd.isna(text) or str(text).strip() == '':
        return ''

    s = str(text).lower()

    # Remove URLs
    s = re.sub(r'http\S+|www\S+|https\S+', '', s)
    # Remove mentions & hashtags
    s = re.sub(r'@\w+', '', s)
    s = re.sub(r'#\w+', '', s)
    # Remove special characters (keep spaces)
    s = re.sub(r'[^\w\s]', ' ', s)
    # Remove numbers
    s = re.sub(r'\d+', ' ', s)
    # Remove extra whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    # Normalisasi singkatan
    words = s.split()
    words = [NORMALIZATION_DICT.get(w, w) for w in words]
    s = ' '.join(words)

    # Stopword removal
    if HAS_SASTRAWI:
        s = stopword_remover.remove(s)

    # Stemming
    if HAS_SASTRAWI:
        s = stemmer.stem(s)

    return s


# ---------------------------
# === FEATURE ENGINEERING
# ---------------------------

def extract_features(df):
    """
    Extract features lengkap untuk deteksi fake review
    """
    print("\n[Feature Engineering] Extracting features...")

    # Basic text-derived features
    df['text_preprocessed'] = df['text_preprocessed'].fillna('').astype(str)
    df['text_word_count'] = df['text_preprocessed'].apply(lambda x: len(x.split()))
    df['text_char_count'] = df['text_preprocessed'].apply(lambda x: len(x))
    df['avg_word_length'] = df['text_preprocessed'].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
    )

    # Punctuation/exclamation/question counts (dari teks asli jika ada)
    if 'text' in df.columns:
        text_col = df['text'].fillna('')
    else:
        text_col = df['text_preprocessed']

    df['exclamation_count'] = text_col.str.count('!')
    df['question_count'] = text_col.str.count(r'\?')
    df['punctuation_count'] = text_col.str.count(r'[.,;:!?"\'()\-]')
    df['uppercase_word_count'] = text_col.apply(lambda s: sum(1 for w in str(s).split() if w.isupper()))
    df['uppercase_word_ratio'] = df['uppercase_word_count'] / df['text_word_count'].replace(0, 1)

    # Stars normalized
    df['stars_norm'] = df['stars'].fillna(0) / 5.0

    # Has image
    df['has_image'] = df['reviewImageUrls'].apply(
        lambda x: 0 if pd.isna(x) or str(x).strip() == '' or str(x).lower() == 'nan' else 1
    )

    # Reviewer count log
    df['reviewer_count'] = df['reviewerNumberOfReviews'].fillna(0).astype(float)
    df['reviewer_count_log'] = np.log1p(df['reviewer_count'])

    # Is local guide
    df['is_local_guide'] = df['isLocalGuide'].apply(
        lambda x: 1 if str(x).lower() in ['true', '1', 'yes'] else 0
    )

    # Detail rating all 5s
    def check_all_fives(x):
        if pd.isna(x) or str(x).strip() == '':
            return 0
        try:
            numbers = [float(s) for s in str(x).replace(',', ' ').split() if s.replace('.', '').isdigit()]
            if len(numbers) > 0:
                return 1 if all(n == 5.0 for n in numbers) else 0
        except:
            pass
        return 0

    df['detail_rating_all_5'] = df['reviewDetailedRating'].apply(check_all_fives)

    # Published date features
    df['published_date'] = pd.to_datetime(df['publishedAtDate'], errors='coerce')
    df['publish_hour'] = df['published_date'].dt.hour.fillna(0).astype(int)
    df['publish_day'] = df['published_date'].dt.date

    if df['publish_day'].notna().sum() > 0:
        day_counts = df.groupby('publish_day').size()
        df['same_day_count'] = df['publish_day'].map(day_counts).fillna(1).astype(int)
    else:
        df['same_day_count'] = 1

    # Suspicious patterns
    df['pattern_5star_short'] = ((df['stars'] == 5) & (df['text_word_count'] < 5)).astype(int)
    df['pattern_5star_nophoto'] = ((df['stars'] == 5) & (df['has_image'] == 0)).astype(int)
    df['pattern_new_reviewer'] = (df['reviewer_count'] <= 2).astype(int)
    df['pattern_same_day'] = (df['same_day_count'] > 5).astype(int)

    # Text quality features (gibberish detection)
    print("  Adding text quality features (gibberish detection)...")
    df = extract_text_quality_features(df, text_column='text_preprocessed')

    print("  [OK] Features extracted")
    return df


# ---------------------------
# === TF-IDF VECTORIZER (dengan N-grams)
# ---------------------------

class TfidfVectorizer:
    """TF-IDF vectorizer dengan ngram_range dan min_df"""

    def __init__(self, max_features=2000, ngram_range=(1, 2), min_df=2, smooth_idf=True):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vocabulary = {}
        self.idf_values = {}
        self.smooth_idf = smooth_idf

    def _generate_ngrams(self, tokens):
        min_n, max_n = self.ngram_range
        ngrams = []
        L = len(tokens)
        for n in range(min_n, max_n + 1):
            for i in range(L - n + 1):
                ngrams.append(' '.join(tokens[i:i + n]))
        return ngrams

    def fit(self, documents):
        doc_count = Counter()
        term_freq = Counter()
        total_docs = 0

        for doc in documents:
            total_docs += 1
            tokens = str(doc).split()
            ngrams = set(self._generate_ngrams(tokens))
            for ng in ngrams:
                doc_count[ng] += 1
            for ng in self._generate_ngrams(tokens):
                term_freq[ng] += 1

        # Filter by min_df
        candidates = [(ng, term_freq[ng]) for ng, cnt in doc_count.items() if cnt >= self.min_df]
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = candidates[:self.max_features]

        self.vocabulary = {ng: idx for idx, (ng, _) in enumerate(selected)}

        # Compute IDF
        for ng in self.vocabulary:
            df = doc_count.get(ng, 0)
            if self.smooth_idf:
                idf = np.log((1 + total_docs) / (1 + df)) + 1.0
            else:
                idf = np.log(total_docs / (df + 1))
            self.idf_values[ng] = idf

    def transform(self, documents):
        n_docs = len(documents)
        n_feats = len(self.vocabulary)
        X = np.zeros((n_docs, n_feats), dtype=float)

        for i, doc in enumerate(documents):
            tokens = str(doc).split()
            ngrams = self._generate_ngrams(tokens)
            if len(ngrams) == 0:
                continue
            counts = Counter(ngrams)
            total = sum(counts.values())
            for ng, cnt in counts.items():
                if ng in self.vocabulary:
                    idx = self.vocabulary[ng]
                    tf = cnt / total
                    X[i, idx] = tf * self.idf_values.get(ng, 0.0)
        return X

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)


# ---------------------------
# === STANDARD SCALER
# ---------------------------

class StandardScaler:
    """Standard scaler untuk normalisasi fitur numerik"""

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
# === LINEAR SVM (Vectorized + Early Stopping)
# ---------------------------

class LinearSVM:
    """
    Linear SVM dengan hinge loss + L2 regularization
    - Vectorized batch gradient descent (cepat)
    - Early stopping untuk mencegah overfitting
    """

    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=2000,
                 tol=1e-4, early_stopping_rounds=50, verbose=True):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.tol = tol
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.w = None
        self.b = 0.0
        self.losses = []

    def _hinge_loss_and_grad(self, X, y):
        """Compute hinge loss and gradient (vectorized)"""
        n = X.shape[0]
        scores = X.dot(self.w) + self.b
        margins = 1 - y * scores
        loss_hinge = np.maximum(0, margins)
        loss = np.mean(loss_hinge) + self.lambda_param * np.dot(self.w, self.w)

        mask = (margins > 0).astype(float)

        if mask.sum() == 0:
            grad_w = 2 * self.lambda_param * self.w
            grad_b = 0.0
        else:
            grad_w = 2 * self.lambda_param * self.w - (1.0 / n) * ((mask * y) @ X)
            grad_b = -(1.0 / n) * np.sum(mask * y)

        return loss, grad_w, grad_b

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

        best_val_loss = np.inf
        no_improve_rounds = 0
        best_w = self.w.copy()
        best_b = self.b

        for it in range(1, self.n_iterations + 1):
            loss, grad_w, grad_b = self._hinge_loss_and_grad(X, y)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            self.losses.append(loss)

            # Early stopping with validation
            if X_val is not None and y_val is not None:
                val_loss, _, _ = self._hinge_loss_and_grad(X_val, y_val)
                if val_loss + self.tol < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_rounds = 0
                    best_w = self.w.copy()
                    best_b = self.b
                else:
                    no_improve_rounds += 1

                if self.verbose and it % 100 == 0:
                    print(f"  Iter {it}/{self.n_iterations} - train_loss: {loss:.6f} - val_loss: {val_loss:.6f}")

                if no_improve_rounds >= self.early_stopping_rounds:
                    print(f"  [Early Stopping] at iter {it} (no improvement for {no_improve_rounds} rounds)")
                    self.w = best_w
                    self.b = best_b
                    break
            else:
                if self.verbose and it % 100 == 0:
                    print(f"  Iter {it}/{self.n_iterations} - train_loss: {loss:.6f}")

    def decision_function(self, X):
        return X.dot(self.w) + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)


# ---------------------------
# === EVALUATION METRICS
# ---------------------------

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == -1) & (y_pred == 1)))
    tn = int(np.sum((y_true == -1) & (y_pred == -1)))
    fn = int(np.sum((y_true == 1) & (y_pred == -1)))

    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    }


def print_confusion_matrix(cm):
    print("\n  Confusion Matrix:")
    print("                  Predicted")
    print("                FAKE    REAL")
    print(f"  Actual FAKE   {cm['tp']:4d}    {cm['fn']:4d}")
    print(f"         REAL   {cm['fp']:4d}    {cm['tn']:4d}")


# ---------------------------
# === PCA (untuk visualisasi)
# ---------------------------

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        Xc = X - self.mean
        cov = np.cov(Xc.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        idx = eigvals.argsort()[::-1]
        self.components = eigvecs[:, idx[:self.n_components]]

    def transform(self, X):
        Xc = X - self.mean
        return Xc.dot(self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ---------------------------
# === VISUALIZATIONS
# ---------------------------

def plot_hyperplane_2d(X, y, svm, output_dir, filename='hyperplane_visualization.png'):
    print(f"\n[Visualization] Creating hyperplane plot...")
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    w2 = svm.w.dot(pca.components) if svm.w is not None else np.zeros(2)

    plt.figure(figsize=(12, 8))
    fake_mask = (y == 1)
    real_mask = (y == -1)
    plt.scatter(X2[fake_mask, 0], X2[fake_mask, 1], c='red', marker='o', alpha=0.6, s=40, label='FAKE')
    plt.scatter(X2[real_mask, 0], X2[real_mask, 1], c='blue', marker='s', alpha=0.6, s=40, label='REAL')

    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1], 200)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = grid.dot(w2) + svm.b
    Z = Z.reshape(XX.shape)
    plt.contour(XX, YY, Z, colors='black', levels=[-1, 0, 1], linestyles=['--', '-', '--'], linewidths=[1.5, 2, 1.5])

    plt.legend(loc='best')
    plt.title('SVM Hyperplane (PCA 2D Projection)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {path}")


def plot_loss_curve(losses, output_dir, filename='training_loss_curve.png'):
    print(f"\n[Visualization] Creating loss curve plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(losses) + 1), losses, '-b', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)

    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {path}")


def plot_confusion_matrix_heatmap(cm, output_dir, filename='confusion_matrix.png'):
    print(f"\n[Visualization] Creating confusion matrix heatmap...")
    arr = np.array([[cm['tp'], cm['fn']], [cm['fp'], cm['tn']]])

    plt.figure(figsize=(8, 6))
    plt.imshow(arr, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar()

    classes = ['FAKE', 'REAL']
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)

    thresh = arr.max() / 2
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(arr[i, j]), ha='center', va='center',
                     color='white' if arr[i, j] > thresh else 'black',
                     fontsize=20, fontweight='bold')

    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {path}")


def plot_performance_metrics(train_metrics, test_metrics, output_dir, filename='performance_metrics.png'):
    print(f"\n[Visualization] Creating performance metrics comparison...")

    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    train_values = [train_metrics['accuracy'], train_metrics['precision'],
                    train_metrics['recall'], train_metrics['f1_score']]
    test_values = [test_metrics['accuracy'], test_metrics['precision'],
                   test_metrics['recall'], test_metrics['f1_score']]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, train_values, width, label='Training', color='skyblue', edgecolor='navy')
    bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='lightcoral', edgecolor='darkred')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance: Training vs Test', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {path}")


# ---------------------------
# === MAIN TRAINING
# ---------------------------

def main():
    print("=" * 80)
    print("TRAINING SVM OPTIMAL - FAKE REVIEW DETECTION")
    print("=" * 80)
    print("\nFitur:")
    print("  - Preprocessing: Sastrawi + Normalisasi singkatan")
    print("  - TF-IDF: N-grams (1-2) dengan min_df")
    print("  - SVM: Vectorized + Early stopping")
    print("  - StandardScaler untuk normalisasi")
    print("  - 17+ numeric features + gibberish detection")

    # KONFIGURASI
    INPUT_FILE = "dataset_balance/google_review_balanced_combined.csv"
    OUTPUT_DIR = "model_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Hyperparameters
    MAX_FEATURES_TFIDF = 2000
    NGRAM_RANGE = (1, 2)
    MIN_DF = 3
    LEARNING_RATE = 0.01
    LAMBDA_PARAM = 0.01
    N_ITERATIONS = 2000
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1  # Fraction of training set for early stopping
    RANDOM_SEED = 42

    # [1/8] LOAD DATA
    print(f"\n[1/8] Loading dataset: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"  Total rows: {len(df)}")

    # Required columns
    required_cols = ['text_preprocessed', 'label', 'stars', 'reviewerNumberOfReviews',
                     'reviewImageUrls', 'isLocalGuide', 'publishedAtDate', 'reviewDetailedRating']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # [2/8] PREPROCESSING
    print(f"\n[2/8] Preprocessing text...")
    print(f"  Sastrawi available: {HAS_SASTRAWI}")

    # Preprocessing jika perlu
    if 'text' in df.columns:
        mask_empty = (df['text_preprocessed'].isna()) | (df['text_preprocessed'].str.strip() == '')
        if mask_empty.sum() > 0:
            print(f"  Preprocessing {mask_empty.sum()} rows from 'text' column...")
            df.loc[mask_empty, 'text_preprocessed'] = df.loc[mask_empty, 'text'].apply(preprocess_text_full)

    # Apply full preprocessing untuk konsistensi
    print(f"  Applying full preprocessing...")
    df['text_preprocessed'] = df['text_preprocessed'].fillna('').apply(preprocess_text_full)

    # Filter empty
    df = df[df['text_preprocessed'].notna()]
    df = df[df['text_preprocessed'].str.strip() != '']
    print(f"  [OK] Valid rows: {len(df)}")

    # Label distribution
    print(f"\n  Label distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"    - {label}: {count} ({count/len(df)*100:.1f}%)")

    # [3/8] FEATURE ENGINEERING
    print(f"\n[3/8] Feature Engineering")
    df = extract_features(df)

    # [4/8] SPLIT DATA
    print(f"\n[4/8] Splitting data (train: 80%, test: 20%)")
    np.random.seed(RANDOM_SEED)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    split_idx = int(len(df) * (1 - TEST_SIZE))
    df_train = df[:split_idx].reset_index(drop=True)
    df_test = df[split_idx:].reset_index(drop=True)

    print(f"  Training: {len(df_train)} rows")
    print(f"  Testing:  {len(df_test)} rows")

    # [5/8] PREPARE FEATURES
    print(f"\n[5/8] Preparing features...")

    # Define numeric features
    text_quality_features = get_text_quality_feature_names()
    numeric_features = [
        'text_word_count', 'text_char_count', 'avg_word_length',
        'exclamation_count', 'question_count', 'punctuation_count', 'uppercase_word_ratio',
        'stars_norm', 'has_image', 'reviewer_count_log', 'is_local_guide',
        'detail_rating_all_5', 'same_day_count',
        'pattern_5star_short', 'pattern_5star_nophoto', 'pattern_new_reviewer', 'pattern_same_day'
    ] + text_quality_features

    # Ensure all features exist
    for f in numeric_features:
        if f not in df_train.columns:
            df_train[f] = 0
            df_test[f] = 0

    print(f"\n  Numeric features ({len(numeric_features)}):")
    for f in numeric_features:
        print(f"    - {f}")

    # TF-IDF vectorization
    print(f"\n  TF-IDF (n-grams={NGRAM_RANGE}, max_features={MAX_FEATURES_TFIDF}, min_df={MIN_DF})")
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES_TFIDF, ngram_range=NGRAM_RANGE, min_df=MIN_DF)
    X_train_text = df_train['text_preprocessed'].fillna('').tolist()
    X_test_text = df_test['text_preprocessed'].fillna('').tolist()
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    print(f"  TF-IDF features: {X_train_tfidf.shape[1]}")

    # Numeric arrays
    X_train_num = df_train[numeric_features].fillna(0).values.astype(float)
    X_test_num = df_test[numeric_features].fillna(0).values.astype(float)

    # Scale numeric features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_test_num = scaler.transform(X_test_num)

    # Combine features
    X_train = np.hstack([X_train_tfidf, X_train_num])
    X_test = np.hstack([X_test_tfidf, X_test_num])
    print(f"\n  Combined features: X_train={X_train.shape}, X_test={X_test.shape}")

    # Labels
    label_map = {'FAKE': 1, 'REAL': -1}
    y_train = df_train['label'].map(label_map).fillna(-1).values.astype(int)
    y_test = df_test['label'].map(label_map).fillna(-1).values.astype(int)

    # Validation split for early stopping
    if 0 < VAL_SIZE < 1.0:
        val_count = max(1, int(len(X_train) * VAL_SIZE))
        X_val = X_train[:val_count]
        y_val = y_train[:val_count]
        X_train_trim = X_train[val_count:]
        y_train_trim = y_train[val_count:]
        print(f"  Validation split: {val_count} rows for early stopping")
    else:
        X_val, y_val = None, None
        X_train_trim, y_train_trim = X_train, y_train

    # [6/8] TRAINING
    print(f"\n[6/8] Training SVM...")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Lambda: {LAMBDA_PARAM}")
    print(f"  Max iterations: {N_ITERATIONS}")
    print(f"  Early stopping rounds: 50")

    svm = LinearSVM(
        learning_rate=LEARNING_RATE,
        lambda_param=LAMBDA_PARAM,
        n_iterations=N_ITERATIONS,
        tol=1e-6,
        early_stopping_rounds=50,
        verbose=True
    )
    svm.fit(X_train_trim, y_train_trim, X_val=X_val, y_val=y_val)

    # [7/8] EVALUATION
    print(f"\n[7/8] Evaluating model...")

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

    # Visualizations
    print(f"\n[7.5/8] Creating visualizations...")
    plot_hyperplane_2d(X_test, y_test, svm, OUTPUT_DIR, 'hyperplane_visualization.png')
    plot_loss_curve(svm.losses, OUTPUT_DIR, 'training_loss_curve.png')
    plot_confusion_matrix_heatmap(test_metrics['confusion_matrix'], OUTPUT_DIR, 'confusion_matrix_test.png')
    plot_performance_metrics(train_metrics, test_metrics, OUTPUT_DIR, 'performance_metrics.png')

    # [8/8] SAVE MODEL
    print(f"\n[8/8] Saving model...")

    model_data = {
        'svm': svm,
        'tfidf': tfidf,
        'scaler': scaler,
        'numeric_features': numeric_features,
        'label_map': label_map,
        'hyperparameters': {
            'max_features_tfidf': MAX_FEATURES_TFIDF,
            'ngram_range': NGRAM_RANGE,
            'min_df': MIN_DF,
            'learning_rate': LEARNING_RATE,
            'lambda_param': LAMBDA_PARAM,
            'n_iterations': N_ITERATIONS
        },
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'preprocessing': {
            'has_sastrawi': HAS_SASTRAWI,
            'normalization_dict': NORMALIZATION_DICT
        }
    }

    model_file = os.path.join(OUTPUT_DIR, "svm_multifeature_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"  [OK] Model saved: {model_file}")

    # Save metrics CSV
    metrics_df = pd.DataFrame({
        'Set': ['Training', 'Test'],
        'Accuracy': [train_metrics['accuracy'], test_metrics['accuracy']],
        'Precision': [train_metrics['precision'], test_metrics['precision']],
        'Recall': [train_metrics['recall'], test_metrics['recall']],
        'F1-Score': [train_metrics['f1_score'], test_metrics['f1_score']]
    })
    metrics_file = os.path.join(OUTPUT_DIR, "svm_multifeature_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"  [OK] Metrics saved: {metrics_file}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
