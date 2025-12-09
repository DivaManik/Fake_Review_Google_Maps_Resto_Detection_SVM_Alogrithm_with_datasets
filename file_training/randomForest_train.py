#!/usr/bin/env python3
# train_random_forest_multifeature_full.py
"""
Full training pipeline (manual Random Forest + TF-IDF)
- Normalizes CSV headers (strip/BOM) and maps similar column names to expected names
- Preprocessing (Sastrawi)
- TF-IDF manual (max_features=1000)
- Numeric & pattern features
- Robust parsing for reviewImageUrls (lists / JSON / CSV-like)
- Manual DecisionTree + RandomForest implementation (no sklearn)
- Save model and plots to rf_model_output/
"""

import os
import math
import pickle
import re
import json
import ast
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
    """
    Normalize header (strip, remove BOM) and map close column names to expected names.
    Returns (df_renamed, mapping_actual_to_expected)
    """
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
except Exception as e:
    raise RuntimeError("Sastrawi required. Install via `pip install Sastrawi`. Error: " + str(e))

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
        # try JSON
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
        if item is None:
            continue
        t = str(item).strip()
        if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
            t = t[1:-1].strip()
        if t == '' or t.lower() in ('nan','none','null'):
            continue
        clean.append(t)
    return clean

# ---------------------------
# Text quality stubs (if not available)
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
# Decision Tree (CART) manual
# ---------------------------
class TreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTreeCART:
    def __init__(self, max_depth=12, min_samples_split=2, min_samples_leaf=2, max_features: Optional[int]=None, random_state: Optional[int]=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None
        self.random_state = np.random.RandomState(random_state)

    @staticmethod
    def gini(y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs * probs)

    def _best_split(self, X, y):
        n, m = X.shape
        parent_gini = self.gini(y)
        best_gain = 0.0
        best_feat = None
        best_thresh = None
        feat_indices = np.arange(m)
        if self.max_features is not None and self.max_features < m:
            feat_indices = self.random_state.choice(m, self.max_features, replace=False)
        for feat in feat_indices:
            vals = X[:, feat]
            sorted_idx = np.argsort(vals)
            vals_s = vals[sorted_idx]
            y_s = y[sorted_idx]
            unique_vals = np.unique(vals_s)
            if unique_vals.size == 1:
                continue
            left_counts = defaultdict(int)
            right_counts = Counter(y_s)
            n_left = 0
            n_right = len(y_s)
            for i in range(0, n-1):
                yi = y_s[i]
                left_counts[yi] += 1
                right_counts[yi] -= 1
                n_left += 1
                n_right -= 1
                if vals_s[i] == vals_s[i+1]:
                    continue
                thresh = 0.5 * (vals_s[i] + vals_s[i+1])
                left_probs = np.array(list(left_counts.values())) / n_left if n_left>0 else np.array([0.0])
                g_left = 1.0 - np.sum(left_probs*left_probs) if n_left>0 else 0.0
                right_vals = np.array(list(right_counts.values()))
                if n_right>0:
                    right_probs = right_vals / n_right
                    g_right = 1.0 - np.sum(right_probs * right_probs)
                else:
                    g_right = 0.0
                weighted = (n_left/n)*g_left + (n_right/n)*g_right
                gain = parent_gini - weighted
                if gain > best_gain:
                    if n_left >= self.min_samples_leaf and n_right >= self.min_samples_leaf:
                        best_gain = gain
                        best_feat = feat
                        best_thresh = thresh
        return best_feat, best_thresh, best_gain

    def _build(self, X, y, depth=0):
        n, m = X.shape
        if depth >= self.max_depth or n < self.min_samples_split or len(np.unique(y)) == 1:
            return TreeNode(value=Counter(y).most_common(1)[0][0])
        feat, thresh, gain = self._best_split(X, y)
        if feat is None:
            return TreeNode(value=Counter(y).most_common(1)[0][0])
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return TreeNode(value=Counter(y).most_common(1)[0][0])
        left = self._build(X[left_mask], y[left_mask], depth+1)
        right = self._build(X[right_mask], y[right_mask], depth+1)
        return TreeNode(feature_idx=feat, threshold=thresh, left=left, right=right)

    def fit(self, X, y):
        self.root = self._build(np.asarray(X), np.asarray(y), depth=0)

    def _predict_one(self, x):
        node = self.root
        while not node.is_leaf():
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x) for x in X])

    def _compute_feature_importances(self):
        # naive count of usage
        # find max feature index used
        max_idx = -1
        def traverse(node):
            nonlocal max_idx
            if node is None or node.is_leaf():
                return
            if node.feature_idx is not None:
                max_idx = max(max_idx, node.feature_idx)
            traverse(node.left); traverse(node.right)
        traverse(self.root)
        n_features = max_idx+1 if max_idx>=0 else 0
        imp = np.zeros(n_features, dtype=float) if n_features>0 else np.array([])
        def traverse2(node):
            if node is None or node.is_leaf():
                return
            imp[node.feature_idx] += 1.0
            traverse2(node.left); traverse2(node.right)
        if imp.size>0:
            traverse2(self.root)
            if imp.sum()>0:
                imp = imp / imp.sum()
        return imp

# ---------------------------
# Random Forest manual
# ---------------------------
class RandomForestManual:
    def __init__(self, n_estimators=30, max_depth=12, min_samples_split=2, min_samples_leaf=2,
                 max_features="sqrt", bootstrap=True, random_state: Optional[int]=42, verbose=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = np.random.RandomState(random_state)
        self.trees: List[DecisionTreeCART] = []
        self.verbose = verbose
        self.oob_score_ = None
        self.feature_importances_ = None

    def _resolve_max_features(self, m):
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, m))
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(m)))
        if self.max_features == "log2":
            return max(1, int(math.log2(m)))
        return m

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.trees = []
        max_feat = self._resolve_max_features(n_features)
        if self.bootstrap:
            oob_votes = [defaultdict(int) for _ in range(n_samples)]
        for i in range(self.n_estimators):
            if self.bootstrap:
                indices = self.random_state.randint(0, n_samples, size=n_samples)
            else:
                indices = np.arange(n_samples)
            Xb = X[indices]
            yb = y[indices]
            tree = DecisionTreeCART(max_depth=self.max_depth,
                                    min_samples_split=self.min_samples_split,
                                    min_samples_leaf=self.min_samples_leaf,
                                    max_features=max_feat,
                                    random_state=self.random_state.randint(0,1<<30))
            if self.verbose:
                print(f"[RF] training tree {i+1}/{self.n_estimators}")
            tree.fit(Xb, yb)
            self.trees.append(tree)
            # OOB
            if self.bootstrap:
                mask_in = np.zeros(n_samples, dtype=bool)
                mask_in[indices] = True
                oob_idx = np.where(~mask_in)[0]
                if oob_idx.size > 0:
                    preds_oob = tree.predict(X[oob_idx])
                    for idx_j, p in zip(oob_idx, preds_oob):
                        oob_votes[idx_j][p] += 1
        if self.bootstrap:
            oob_pred = np.array([max(v.items(), key=lambda x: x[1])[0] if v else None for v in oob_votes])
            mask = np.array([bool(v) for v in oob_votes])
            if mask.sum() > 0:
                y_true_oob = y[mask]
                y_pred_oob = oob_pred[mask]
                self.oob_score_ = np.mean(y_true_oob == y_pred_oob)
            else:
                self.oob_score_ = None
        # feature importances average
        importances = np.zeros(n_features, dtype=float)
        for tr in self.trees:
            imp = tr._compute_feature_importances()
            if imp.size == 0:
                continue
            if imp.shape[0] < n_features:
                tmp = np.zeros(n_features, dtype=float)
                tmp[:imp.shape[0]] = imp
                imp = tmp
            importances += imp
        if len(self.trees) > 0:
            importances /= len(self.trees)
        if importances.sum() > 0:
            importances = importances / importances.sum()
        self.feature_importances_ = importances

    def predict(self, X):
        X = np.asarray(X)
        preds = np.zeros((len(self.trees), X.shape[0]), dtype=int)
        for i, tr in enumerate(self.trees):
            preds[i] = tr.predict(X)
        final = []
        for col in preds.T:
            counts = Counter(col)
            most = counts.most_common()
            most.sort(key=lambda x: (-x[1], x[0]))
            final.append(most[0][0])
        return np.array(final)

    def predict_proba(self, X):
        X = np.asarray(X)
        preds = np.zeros((len(self.trees), X.shape[0]), dtype=int)
        for i, tr in enumerate(self.trees):
            preds[i] = tr.predict(X)
        labels = sorted(list({p for row in preds for p in row}))
        proba = []
        for j in range(X.shape[0]):
            col = preds[:, j]
            counts = Counter(col)
            rowp = [counts.get(lbl, 0)/len(self.trees) for lbl in labels]
            proba.append(rowp)
        return np.array(proba), labels

# ---------------------------
# Feature engineering (robust) using normalize_and_map_columns
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
    # Always run preprocess to normalize / stem
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
        print("[FE] text_quality_detector missing or failed; using placeholders")
        for n in get_text_quality_feature_names():
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

def plot_feature_importances(importances, feature_names, outpath):
    idx = np.argsort(importances)[::-1]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]
    plt.figure(figsize=(10,6))
    plt.bar(range(len(vals)), vals, tick_label=names)
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Importances (approx)')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------
# MAIN training pipeline
# ---------------------------
def main():
    INPUT = "../dataset_balance/google_review_balanced_undersampling.csv"
    OUTDIR = "rf_model_output"
    os.makedirs(OUTDIR, exist_ok=True)

    # hyperparams
    MAX_FEATURES_TFIDF = 1000
    N_ESTIMATORS = 30
    MAX_DEPTH = 12
    MIN_SAMPLES_SPLIT = 2
    MIN_SAMPLES_LEAF = 2
    MAX_FEATURES_RF = "sqrt"
    TEST_SIZE = 0.2
    RANDOM_SEED = 42

    print("Loading dataset:", INPUT)
    df = pd.read_csv(INPUT, low_memory=False)
    print("Rows:", len(df))

    # feature engineering (normalize & map occurs inside)
    df = extract_features(df)

    # ========================================================
    # LABEL CORRECTION: Gibberish text -> FAKE
    # Model akan BELAJAR bahwa gibberish = FAKE
    # ========================================================
    if 'tq_is_gibberish' in df.columns and 'tq_gibberish_score' in df.columns:
        print("\n[LABEL CORRECTION] Checking for gibberish text...")
        gibberish_mask = (df['tq_is_gibberish'] == 1) & (df['tq_gibberish_score'] > 0.7)

        # Count berapa yang akan dikoreksi
        before_fake = (df['label'] == 'FAKE').sum()

        # Koreksi label: jika gibberish tapi label REAL, ubah ke FAKE
        corrected_count = ((gibberish_mask) & (df['label'] == 'REAL')).sum()
        df.loc[gibberish_mask, 'label'] = 'FAKE'

        after_fake = (df['label'] == 'FAKE').sum()

        print(f"  Gibberish samples found: {gibberish_mask.sum()}")
        print(f"  Labels corrected REAL -> FAKE: {corrected_count}")
        print(f"  FAKE labels: {before_fake} -> {after_fake}")
    else:
        print("\n[LABEL CORRECTION] Gibberish columns not found, skipping...")

    # prepare TF-IDF
    texts = df['text_preprocessed'].fillna('').astype(str).tolist()
    tfidf = TfidfVectorizerManual(max_features=MAX_FEATURES_TFIDF)
    print("Fitting TF-IDF (max_features=%d) ..." % MAX_FEATURES_TFIDF)
    X_text = tfidf.fit_transform(texts)

    # numeric features list
    text_quality_features = get_text_quality_feature_names()
    numeric_features = [
        'text_word_count','text_char_count','avg_word_length',
        'exclamation_count','question_count','punctuation_count','uppercase_word_ratio',
        'stars_norm','has_image','n_images','reviewer_count_log','is_local_guide',
        'detail_rating_all_5','same_day_count',
        'pattern_5star_short','pattern_5star_nophoto','pattern_new_reviewer','pattern_same_day'
    ] + text_quality_features

    for f in numeric_features:
        if f not in df.columns:
            df[f] = 0.0

    X_num = df[numeric_features].fillna(0).values.astype(float)
    X = np.hstack([X_text, X_num])
    print("Combined feature shape:", X.shape)

    if 'label' not in df.columns:
        raise ValueError("Dataset must contain 'label' column with values 'FAKE' or 'REAL'")

    y = df['label'].map({'FAKE':1, 'REAL':-1}).values
    # shuffle and split
    rng = np.random.RandomState(RANDOM_SEED)
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]
    split_idx = int(len(X)*(1-TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print("Train rows:", len(X_train), "Test rows:", len(X_test))

    # train RF
    rf = RandomForestManual(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                            min_samples_split=MIN_SAMPLES_SPLIT, min_samples_leaf=MIN_SAMPLES_LEAF,
                            max_features=MAX_FEATURES_RF, bootstrap=True, random_state=RANDOM_SEED, verbose=True)
    print("Training Random Forest (manual)...")
    rf.fit(X_train, y_train)

    # evaluate
    y_train_pred = rf.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_train_pred)
    print("\nTRAIN METRICS:", train_metrics)
    print_confusion_matrix(train_metrics['confusion_matrix'])

    y_test_pred = rf.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    print("\nTEST METRICS:", test_metrics)
    print_confusion_matrix(test_metrics['confusion_matrix'])

    print("\nOOB score:", rf.oob_score_)
    print("Feature importances (approx):")
    for name, val in zip(numeric_features, rf.feature_importances_):
        print(f"  {name}: {val:.4f}")

    # save model (includes RF object, tfidf vocab + numeric features)
    model_data = {
        'rf': rf,
        'tfidf': tfidf,
        'numeric_features': numeric_features,
        'max_features_tfidf': MAX_FEATURES_TFIDF,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    model_file = os.path.join(OUTDIR, "random_forest_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)
    print("Model saved to:", model_file)

    # save metrics CSV
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

    # Plot metrics visualization
    metrics_plot = os.path.join(OUTDIR, "metrics_visualization.png")
    _, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_df['Set']))
    width = 0.2

    # create bars and keep handles so we can put values above each bar
    bars1 = ax.bar(x - 1.5*width, metrics_df['Accuracy'], width, label='Accuracy', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, metrics_df['Recall'], width, label='Recall', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, metrics_df['F1'], width, label='F1 Score', alpha=0.8)

    # Tambahkan nilai di atas setiap bar (format 3 desimal)
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Random Forest Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Set'])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(metrics_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved metrics visualization:", metrics_plot)

    print("TRAINING COMPLETE.")

if __name__ == "__main__":
    main()
