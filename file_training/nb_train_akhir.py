#!/usr/bin/env python3
# train_naive_bayes_multifeature.py
"""
Naive Bayes multi-feature training pipeline (manual TF-IDF + Hybrid Naive Bayes)
- Uses robust feature engineering (same as RF & SVM):
  - header normalization & auto-mapping
  - Sastrawi preprocessing (Indonesian)
  - robust reviewImageUrls parsing (lists / JSON / CSV-like)
  - text quality detector integration (optional)
  - TF-IDF manual (max_features)
  - Numeric/pattern features identical to RF/SVM
  - StandardScaler for numeric features
- Saves model and metrics to output dir
"""

import os
import re
import json
import ast
import math
import pickle
from collections import Counter, defaultdict
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    # fallback simple stubs
    class Dummy:
        def remove(self, x): return x
        def stem(self, x): return x
    stopword_remover = Dummy()
    stemmer = Dummy()

normalization_dict = {
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
    words = [normalization_dict.get(w, w) for w in words]
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
        # placeholders
        df['tq_entropy'] = 0.0
        df['tq_valid_word_ratio'] = 0.0
        df['tq_avg_word_length'] = 0.0
        return df
    def get_text_quality_feature_names():
        return ['tq_entropy','tq_valid_word_ratio','tq_avg_word_length']

# ---------------------------
# TF-IDF Vectorizer (manual)
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
# Naive Bayes models (Multinomial for text, Gaussian for numeric, Hybrid combiner)
# ---------------------------
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior = {}
        self.feature_log_prob = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        for class_label in self.classes:
            X_class = X[y == class_label]
            n_samples_class = X_class.shape[0]
            self.class_log_prior[class_label] = np.log(n_samples_class / n_samples) if n_samples>0 else -np.inf
            word_counts = X_class.sum(axis=0)
            total_count = word_counts.sum()
            self.feature_log_prob[class_label] = np.log(
                (word_counts + self.alpha) / (total_count + self.alpha * n_features + 1e-12)
            )

    def predict_log_proba(self, X):
        n_samples = X.shape[0]
        log_proba = np.zeros((n_samples, len(self.classes)))
        for idx, class_label in enumerate(self.classes):
            log_prior = self.class_log_prior[class_label]
            log_likelihood = X.dot(self.feature_log_prob[class_label])
            log_proba[:, idx] = log_prior + log_likelihood
        return log_proba

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        return self.classes[np.argmax(log_proba, axis=1)]

class GaussianNB:
    def __init__(self):
        self.class_prior = {}
        self.mean = {}
        self.var = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        for class_label in self.classes:
            X_class = X[y == class_label]
            n_samples_class = X_class.shape[0]
            self.class_prior[class_label] = n_samples_class / n_samples if n_samples>0 else 0.0
            self.mean[class_label] = X_class.mean(axis=0) if n_samples_class>0 else np.zeros(X.shape[1])
            self.var[class_label] = X_class.var(axis=0) if n_samples_class>0 else np.ones(X.shape[1]) * 1e-6
            self.var[class_label] += 1e-9

    def _calculate_likelihood(self, X, mean, var):
        exponent = -0.5 * ((X - mean) ** 2) / var
        normalizer = -0.5 * np.log(2 * np.pi * var)
        return (normalizer + exponent).sum(axis=1)

    def predict_log_proba(self, X):
        n_samples = X.shape[0]
        log_proba = np.zeros((n_samples, len(self.classes)))
        for idx, class_label in enumerate(self.classes):
            log_prior = np.log(self.class_prior[class_label] + 1e-12)
            log_likelihood = self._calculate_likelihood(X, self.mean[class_label], self.var[class_label])
            log_proba[:, idx] = log_prior + log_likelihood
        return log_proba

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        return self.classes[np.argmax(log_proba, axis=1)]

class HybridNaiveBayes:
    def __init__(self, alpha=1.0, text_weight=0.6, numeric_weight=0.4):
        self.multinomial_nb = MultinomialNB(alpha=alpha)
        self.gaussian_nb = GaussianNB()
        self.text_weight = text_weight
        self.numeric_weight = numeric_weight
        self.classes = None

    def fit(self, X_text, X_numeric, y):
        # fit both components
        self.multinomial_nb.fit(X_text, y)
        self.gaussian_nb.fit(X_numeric, y)
        self.classes = np.unique(y)

    def predict_proba(self, X_text, X_numeric):
        log_proba_text = self.multinomial_nb.predict_log_proba(X_text)
        log_proba_numeric = self.gaussian_nb.predict_log_proba(X_numeric)
        log_proba_combined = self.text_weight * log_proba_text + self.numeric_weight * log_proba_numeric
        proba = np.exp(log_proba_combined - np.max(log_proba_combined, axis=1, keepdims=True))
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X_text, X_numeric):
        proba = self.predict_proba(X_text, X_numeric)
        return self.classes[np.argmax(proba, axis=1)]

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
# Feature engineering (reuse robust)
# ---------------------------
def extract_features(df: pd.DataFrame):
    print("[FE] Running feature engineering...")
    df = df.copy()

    # normalize & map
    df, mapping = normalize_and_map_columns(df)
    if mapping:
        print("[FE] Applied column mapping (actual -> expected):")
        for k, v in mapping.items():
            print(f"   {k} -> {v}")

    # ensure defaults for expected names
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
        # fallback ensure features exist
        for n in get_text_quality_feature_names():
            if n not in df.columns:
                df[n] = 0.0

    print("[FE] Feature engineering done.")
    return df

# ---------------------------
# Metrics & plotting helpers
# ---------------------------
def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true==1) & (y_pred==1)))
    fp = int(np.sum((y_true==-1) & (y_pred==1)))
    tn = int(np.sum((y_true==-1) & (y_pred==-1)))
    fn = int(np.sum((y_true==1) & (y_pred==-1)))
    total = tp+fp+tn+fn if (tp+fp+tn+fn)>0 else 1
    accuracy = (tp+tn)/total
    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    return {'accuracy':accuracy, 'precision':precision, 'recall':recall, 'f1_score':f1,
            'confusion_matrix': {'tp':tp,'fp':fp,'tn':tn,'fn':fn}}

def print_confusion_matrix(cm):
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              FAKE    REAL")
    print(f"Actual FAKE   {cm['tp']:4d}    {cm['fn']:4d}")
    print(f"       REAL   {cm['fp']:4d}    {cm['tn']:4d}")

# ---------------------------
# MAIN training pipeline
# ---------------------------
def main():
    INPUT = "../dataset_balance/google_review_balanced_undersampling.csv"  # ganti sesuai path
    OUTDIR = "nb_model_output"
    os.makedirs(OUTDIR, exist_ok=True)

    # hyperparams
    MAX_FEATURES_TFIDF = 1000
    ALPHA = 1.0
    TEXT_WEIGHT = 0.6
    NUMERIC_WEIGHT = 0.4
    TEST_SIZE = 0.2
    RANDOM_SEED = 42

    print("Loading dataset:", INPUT)
    df = pd.read_csv(INPUT, low_memory=False)
    print("Rows:", len(df))

    # feature engineering
    df = extract_features(df)

    # TF-IDF
    texts = df['text_preprocessed'].fillna('').astype(str).tolist()
    tfidf = TfidfVectorizerManual(max_features=MAX_FEATURES_TFIDF)
    print("Fitting TF-IDF (max_features=%d) ..." % MAX_FEATURES_TFIDF)
    X_text = tfidf.fit_transform(texts)

    # numeric features exactly matching RF & SVM scripts
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

    # Standard scale numeric features (important for Gaussian NB)
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

    # Train Hybrid Naive Bayes
    nb = HybridNaiveBayes(alpha=ALPHA, text_weight=TEXT_WEIGHT, numeric_weight=NUMERIC_WEIGHT)
    print("Training Hybrid Naive Bayes (text + numeric)...")
    nb.fit(X_train_text, X_train_num, y_train)

    # Evaluate
    y_train_pred = nb.predict(X_train_text, X_train_num)
    train_metrics = calculate_metrics(y_train, y_train_pred)
    print("\nTRAIN METRICS:", train_metrics)
    print_confusion_matrix(train_metrics['confusion_matrix'])

    y_test_pred = nb.predict(X_test_text, X_test_num)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    print("\nTEST METRICS:", test_metrics)
    print_confusion_matrix(test_metrics['confusion_matrix'])

    # Save model and metadata
    model_data = {
        'nb_model': nb,
        'tfidf': tfidf,
        'scaler': scaler,
        'numeric_features': numeric_features,
        'max_features_tfidf': MAX_FEATURES_TFIDF,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    model_file = os.path.join(OUTDIR, "naive_bayes_multifeature_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)
    print("Model saved to:", model_file)

    # Save metrics CSV & simple plots
    metrics_df = pd.DataFrame({
        'Set': ['Train','Test'],
        'Accuracy': [train_metrics['accuracy'], test_metrics['accuracy']],
        'Precision':[train_metrics['precision'], test_metrics['precision']],
        'Recall':[train_metrics['recall'], test_metrics['recall']],
        'F1': [train_metrics['f1_score'], test_metrics['f1_score']]
    })
    metrics_file = os.path.join(OUTDIR, "metrics_summary.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print("Saved metrics summary:", metrics_file)

    # Plot metrics visualization (values above bars)
    metrics_plot = os.path.join(OUTDIR, "metrics_visualization.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_df['Set']))
    width = 0.2

    bars1 = ax.bar(x - 1.5*width, metrics_df['Accuracy'], width, label='Accuracy', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x - 0.5*width, metrics_df['Precision'], width, label='Precision', alpha=0.8, color='forestgreen')
    bars3 = ax.bar(x + 0.5*width, metrics_df['Recall'], width, label='Recall', alpha=0.8, color='darkorange')
    bars4 = ax.bar(x + 1.5*width, metrics_df['F1'], width, label='F1 Score', alpha=0.8, color='crimson')

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Naive Bayes Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Set'])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(metrics_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved metrics visualization:", metrics_plot)

    print("TRAINING COMPLETE.")

if __name__ == "__main__":
    main()
