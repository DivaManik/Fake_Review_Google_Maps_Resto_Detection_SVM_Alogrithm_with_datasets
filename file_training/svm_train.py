#!/usr/bin/env python3
# svm_multifeature_unified.py
"""
SVM training script adjusted to use the same feature set as NB & RF:
- header normalization & auto-mapping
- Sastrawi preprocessing (optional, fallback)
- robust reviewImageUrls parsing
- TF-IDF manual (unigram) max_features=1000 (same as NB/RF)
- numeric/pattern features identical to NB/RF
- StandardScaler for numeric features
- text quality features (imported if available)
- linear SVM (vectorized) with early stopping
"""

import os
import sys
import re
import json
import ast
import math
import pickle
from collections import Counter, defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure parent dir can be used for imports if needed
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------
# Header normalization & auto-mapping
# ---------------------------
from difflib import get_close_matches

def normalize_and_map_columns(df: pd.DataFrame):
    df = df.copy()
    # normalize columns: strip and remove BOM
    new_cols = []
    for c in df.columns:
        if isinstance(c, str):
            c2 = c.replace('\ufeff', '').strip()
        else:
            c2 = str(c).strip()
        new_cols.append(c2)
    df.columns = new_cols

    cols_lower = [c.lower() for c in df.columns]
    expected = {
        'reviewImageUrls': ['image', 'images', 'reviewimage', 'reviewimageurls', 'review_image_urls', 'image_urls', 'imageurl'],
        'reviewDetailedRating': ['detailed', 'detailedrating', 'reviewdetailedrating', 'review_detailed_rating','detailrating'],
        'publishedAtDate': ['published', 'publishedatdate', 'published_date', 'publishdate', 'date','createdat'],
        'reviewerNumberOfReviews': ['reviewernumberofreviews','reviewer_count','reviewer_number_of_reviews','num_reviews','review_count'],
        'isLocalGuide': ['islocalguide','is_local_guide','localguide'],
        'text': ['text','review','content','texttranslated','text_translated','review_text','reviewtext'],
        'stars': ['stars','rating','rating_score','star']
    }

    mapping = {}
    for expected_name, keywords in expected.items():
        found = None
        # exact match
        for c in df.columns:
            if c.strip().lower() == expected_name.lower():
                found = c; break
        if not found:
            # contains keyword
            for kw in keywords:
                for c in df.columns:
                    if kw in c.lower():
                        found = c; break
                if found: break
        if not found:
            # close match by difflib
            cm = get_close_matches(expected_name.lower(), cols_lower, n=1, cutoff=0.8)
            if cm:
                idx = cols_lower.index(cm[0])
                found = df.columns[idx]
        if found:
            mapping[found] = expected_name

    if mapping:
        df = df.rename(columns=mapping)

    return df, mapping

# ---------------------------
# Sastrawi (preprocessing Indonesian)
# ---------------------------
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    stemmer = StemmerFactory().create_stemmer()
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    HAS_SASTRAWI = True
except Exception:
    HAS_SASTRAWI = False
    class Dummy:
        def remove(self, x): return x
        def stem(self, x): return x
    stopword_remover = Dummy()
    stemmer = Dummy()

NORMALIZATION_DICT = {
    'gak': 'tidak', 'ga': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak', 'gk': 'tidak',
    'tdk': 'tidak', 'bgt': 'banget', 'bngt': 'banget', 'emg': 'memang',
    'udh': 'sudah', 'udah': 'sudah', 'blm': 'belum', 'jg': 'juga', 'dgn': 'dengan',
    'yg': 'yang', 'utk': 'untuk', 'sy': 'saya', 'kl': 'kalau', 'klo': 'kalau',
    'thx': 'terima kasih', 'thanks': 'terima kasih', 'mantul': 'mantap', 'mantep': 'mantap',
    'rekomend': 'rekomendasi', 'rekomen': 'rekomendasi', 'recommended': 'rekomendasi',
}

def preprocess_text_full(text):
    if pd.isna(text):
        return ''
    s = str(text).lower()
    s = re.sub(r'http\S+|www\S+|https\S+', '', s)
    s = re.sub(r'@\w+', '', s)
    s = re.sub(r'#\w+', '', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\d+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    words = s.split()
    words = [NORMALIZATION_DICT.get(w, w) for w in words]
    s = ' '.join(words)
    if HAS_SASTRAWI:
        s = stopword_remover.remove(s)
        s = stemmer.stem(s)
    return s

# ---------------------------
# Robust image field parser
# ---------------------------
def parse_image_field(x):
    if pd.isna(x):
        return []
    if isinstance(x, (list, tuple)):
        raw = list(x)
    else:
        s = str(x).strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, (list, tuple)):
                raw = parsed
            else:
                raw = [parsed]
        except Exception:
            try:
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
# Text quality detector (optional)
# ---------------------------
try:
    from file_testing.text_quality_detector import extract_text_quality_features, get_text_quality_feature_names
    HAS_TQ = True
except Exception:
    HAS_TQ = False
    def extract_text_quality_features(df, text_column='text_preprocessed'):
        df['tq_entropy'] = 0.0
        df['tq_valid_word_ratio'] = 0.0
        df['tq_avg_word_length'] = 0.0
        return df
    def get_text_quality_feature_names():
        return ['tq_entropy','tq_valid_word_ratio','tq_avg_word_length']

# ---------------------------
# TF-IDF Vectorizer Manual (unigram to match NB/RF)
# ---------------------------
class TfidfVectorizerManual:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocab = {}
        self.idf = {}

    def fit(self, documents: List[str]):
        word_doc_count = Counter()
        all_words = Counter()
        for doc in documents:
            words = str(doc).split()
            unique = set(words)
            for w in unique:
                word_doc_count[w] += 1
            for w in words:
                all_words[w] += 1
        most_common = all_words.most_common(self.max_features)
        self.vocab = {w: i for i, (w, _) in enumerate(most_common)}
        N = len(documents)
        for w in self.vocab:
            self.idf[w] = math.log((N) / (word_doc_count[w] + 1))

    def transform(self, documents: List[str]):
        X = np.zeros((len(documents), len(self.vocab)), dtype=float)
        for i, doc in enumerate(documents):
            words = str(doc).split()
            wc = Counter(words)
            total = len(words)
            if total == 0:
                continue
            for w, c in wc.items():
                if w in self.vocab:
                    idx = self.vocab[w]
                    tf = c / total
                    X[i, idx] = tf * self.idf[w]
        return X

    def fit_transform(self, documents: List[str]):
        self.fit(documents)
        return self.transform(documents)

# ---------------------------
# Standard Scaler for numeric features
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
# Feature engineering (matching NB & RF)
# ---------------------------
def extract_features(df: pd.DataFrame):
    print("\n[Feature Engineering] Extracting features...")
    df = df.copy()

    # normalize & map
    df, mapping = normalize_and_map_columns(df)
    if mapping:
        print("[FE] Applied column mapping (actual -> expected):")
        for k, v in mapping.items():
            print(f"   {k} -> {v}")

    defaults = {
        "text_preprocessed": "", "text": "", "stars": 0, "reviewImageUrls": None,
        "reviewerNumberOfReviews": 0, "isLocalGuide": 0, "reviewDetailedRating": "", "publishedAtDate": pd.NaT
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
            print(f"[FE] Warning: created missing column '{col}' with default")

    # If text_preprocessed empty, fallback to preprocessing of 'text'
    df['text_preprocessed'] = df['text_preprocessed'].fillna('').astype(str)
    mask_empty = df['text_preprocessed'].str.strip() == ''
    if mask_empty.any():
        df.loc[mask_empty, 'text_preprocessed'] = df.loc[mask_empty, 'text'].fillna('').astype(str).apply(preprocess_text_full)
    # Always run preprocess
    df['text_preprocessed'] = df['text_preprocessed'].apply(preprocess_text_full)

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

    # images
    print("[FE] Parsing reviewImageUrls ...")
    df['review_images_list'] = df['reviewImageUrls'].apply(parse_image_field)
    df['n_images'] = df['review_images_list'].apply(lambda lst: len(lst))
    df['has_image'] = (df['n_images'] > 0).astype(int)
    df['first_image'] = df['review_images_list'].apply(lambda lst: lst[0] if isinstance(lst, list) and len(lst)>0 else np.nan)

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
    try:
        df = extract_text_quality_features(df, text_column='text_preprocessed')
    except Exception:
        for n in get_text_quality_feature_names():
            if n not in df.columns:
                df[n] = 0.0

    print("[FE] Feature engineering done.")
    return df

# ---------------------------
# Metrics & plotting helpers
# ---------------------------
def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == -1) & (y_pred == 1)))
    tn = int(np.sum((y_true == -1) & (y_pred == -1)))
    fn = int(np.sum((y_true == 1) & (y_pred == -1)))

    total = tp+fp+tn+fn if (tp+fp+tn+fn)>0 else 1
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}}

def print_confusion_matrix(cm):
    print("\n  Confusion Matrix:")
    print("                  Predicted")
    print("                FAKE    REAL")
    print(f"  Actual FAKE   {cm['tp']:4d}    {cm['fn']:4d}")
    print(f"         REAL   {cm['fp']:4d}    {cm['tn']:4d}")

# ---------------------------
# Linear SVM (vectorized + early stopping)
# ---------------------------
class LinearSVM:
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
# PCA & visualization helpers (same as before)
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

def plot_hyperplane_2d(X, y, svm, output_dir, filename='hyperplane_visualization.png'):
    print(f"\n[Visualization] Creating hyperplane plot...")
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    w2 = svm.w.dot(pca.components) if svm.w is not None else np.zeros(2)

    plt.figure(figsize=(12, 8))
    fake_mask = (y == 1)
    real_mask = (y == -1)
    plt.scatter(X2[fake_mask, 0], X2[fake_mask, 1], marker='o', alpha=0.6, s=40, label='FAKE')
    plt.scatter(X2[real_mask, 0], X2[real_mask, 1], marker='s', alpha=0.6, s=40, label='REAL')

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
    bars1 = ax.bar(x - width/2, train_values, width, label='Training')
    bars2 = ax.bar(x + width/2, test_values, width, label='Test')

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
# MAIN training pipeline
# ---------------------------
def main():
    print("=" * 80)
    print("TRAINING SVM (UNIFIED FEATURES - MATCH NB & RF)")
    print("=" * 80)

    INPUT_FILE = "../dataset_balance/google_review_balanced_undersampling.csv"
    OUTPUT_DIR = "svm_model_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # hyperparams (aligned with NB/RF)
    MAX_FEATURES_TFIDF = 1000   # match NB & RF
    ALPHA = 1.0
    LEARNING_RATE = 0.01
    LAMBDA_PARAM = 0.01
    N_ITERATIONS = 2000
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    RANDOM_SEED = 42

    print("Loading dataset:", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print("Rows:", len(df))

    # feature engineering (this adds all unified numeric/features)
    df = extract_features(df)

    # TF-IDF (unigram manual)
    texts = df['text_preprocessed'].fillna('').astype(str).tolist()
    tfidf = TfidfVectorizerManual(max_features=MAX_FEATURES_TFIDF)
    print("Fitting TF-IDF (unigram, max_features=%d) ..." % MAX_FEATURES_TFIDF)
    X_text = tfidf.fit_transform(texts)

    # numeric features exactly matching RF & NB scripts
    text_quality_features = get_text_quality_feature_names()
    numeric_features = [
        'text_word_count','text_char_count','avg_word_length',
        'exclamation_count','question_count','punctuation_count','uppercase_word_ratio',
        'stars_norm','has_image','n_images','reviewer_count_log','is_local_guide',
        'detail_rating_all_5','same_day_count',
        'pattern_5star_short','pattern_5star_nophoto','pattern_new_reviewer','pattern_same_day'
    ] + text_quality_features

    # ensure numeric columns exist
    for f in numeric_features:
        if f not in df.columns:
            df[f] = 0.0

    X_num = df[numeric_features].fillna(0).values.astype(float)

    # Standard scale numeric features (important for SVM)
    scaler = StandardScalerSimple()
    X_num_scaled = scaler.fit_transform(X_num)

    # Shuffle + split
    y = df['label'].map({'FAKE':1, 'REAL':-1}).values
    rng = np.random.RandomState(RANDOM_SEED)
    perm = rng.permutation(len(X_text))
    X_text = X_text[perm]
    X_num_scaled = X_num_scaled[perm]
    y = y[perm]

    split_idx = int(len(X_text)*(1-TEST_SIZE))
    X_train_text, X_test_text = X_text[:split_idx], X_text[split_idx:]
    X_train_num, X_test_num = X_num_scaled[:split_idx], X_num_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print("Train rows:", len(y_train), "Test rows:", len(y_test))

    # Combine features for SVM
    X_train = np.hstack([X_train_text, X_train_num])
    X_test = np.hstack([X_test_text, X_test_num])
    print("Combined feature shape:", X_train.shape)

    # Validation split for early stopping
    if 0 < VAL_SIZE < 1.0:
        val_count = max(1, int(len(X_train) * VAL_SIZE))
        X_val = X_train[:val_count]
        y_val = y_train[:val_count]
        X_train_trim = X_train[val_count:]
        y_train_trim = y_train[val_count:]
        print(f"Validation split: {val_count} rows for early stopping")
    else:
        X_val, y_val = None, None
        X_train_trim, y_train_trim = X_train, y_train

    # Train SVM
    svm = LinearSVM(learning_rate=LEARNING_RATE, lambda_param=LAMBDA_PARAM,
                    n_iterations=N_ITERATIONS, tol=1e-6, early_stopping_rounds=50, verbose=True)
    print("Training Linear SVM (vectorized) ...")
    svm.fit(X_train_trim, y_train_trim, X_val=X_val, y_val=y_val)

    # Evaluate
    print("\nTRAIN METRICS:")
    y_train_pred = svm.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_train_pred)
    print(train_metrics)
    print_confusion_matrix(train_metrics['confusion_matrix'])

    print("\nTEST METRICS:")
    y_test_pred = svm.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    print(test_metrics)
    print_confusion_matrix(test_metrics['confusion_matrix'])

    # Save model and metadata
    model_data = {
        'svm': svm,
        'tfidf': tfidf,
        'scaler': scaler,
        'numeric_features': numeric_features,
        'max_features_tfidf': MAX_FEATURES_TFIDF,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    model_file = os.path.join(OUTPUT_DIR, "svm_multifeature_model_unified.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)
    print("Model saved to:", model_file)

    # Save metrics CSV & plots
    metrics_df = pd.DataFrame({
        'Set': ['Train','Test'],
        'Accuracy': [train_metrics['accuracy'], test_metrics['accuracy']],
        'Precision':[train_metrics['precision'], test_metrics['precision']],
        'Recall':[train_metrics['recall'], test_metrics['recall']],
        'F1': [train_metrics['f1_score'], test_metrics['f1_score']]
    })
    metrics_file = os.path.join(OUTPUT_DIR, "metrics_summary_unified.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print("Saved metrics summary:", metrics_file)

    # Plots
    metrics_plot = os.path.join(OUTPUT_DIR, "metrics_visualization_unified.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_df['Set']))
    width = 0.2

    bars1 = ax.bar(x - 1.5*width, metrics_df['Accuracy'], width, label='Accuracy', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, metrics_df['Recall'], width, label='Recall', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, metrics_df['F1'], width, label='F1 Score', alpha=0.8)

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('SVM Performance Metrics (Unified Features)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Set'])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(metrics_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved metrics visualization:", metrics_plot)

    # Additional visualizations
    plot_hyperplane_2d(X_test, y_test, svm, OUTPUT_DIR, 'hyperplane_visualization_unified.png')
    plot_loss_curve(svm.losses, OUTPUT_DIR, 'training_loss_curve_unified.png')
    plot_confusion_matrix_heatmap(test_metrics['confusion_matrix'], OUTPUT_DIR, 'confusion_matrix_test_unified.png')
    plot_performance_metrics(train_metrics, test_metrics, OUTPUT_DIR, 'performance_metrics_unified.png')

    print("TRAINING COMPLETE (UNIFIED FEATURES).")

if __name__ == "__main__":
    main()
