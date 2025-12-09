#!/usr/bin/env python3
# rf_manual_predict.py
"""
Prediksi fake review menggunakan RandomForestManual (tanpa sklearn)
- VERSI ORIGINAL (TANPA POST-PROCESSING GIBBERISH)
- Input: CSV dataset (text, stars, reviewer info, dll)
- Output: CSV dengan predicted_label + confidence + decision_score
- Menggunakan feature engineering yang sama seperti training
"""

import os
import math
import numpy as np
import pandas as pd
import pickle
from collections import Counter, defaultdict
from typing import Optional, List

# ========================================================
# === Manual Random Forest Classes (Required for Pickle) ===
# ========================================================

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

# ===== Try import text_quality_detector =====
try:
    from file_testing.text_quality_detector import extract_text_quality_features, get_text_quality_feature_names
    HAS_TEXT_QUALITY = True
except Exception:
    HAS_TEXT_QUALITY = False
    def extract_text_quality_features(df, text_column='text_preprocessed'):
        # fallback if not available
        df["tq_entropy"] = 0.0
        df["tq_valid_word_ratio"] = 0.0
        df["tq_avg_word_length"] = 0.0
        return df

    def get_text_quality_feature_names():
        return ["tq_entropy", "tq_valid_word_ratio", "tq_avg_word_length"]

# ========================================================
# === Feature Engineering (Identical to Training Script) ===
# ========================================================

def extract_features(df):
    print("[FEATURES] building features...")

    # Pastikan text_preprocessed ada
    if "text_preprocessed" not in df.columns:
        if "text" in df.columns:
            df["text_preprocessed"] = df["text"].fillna("").astype(str)
        else:
            df["text_preprocessed"] = ""

    df["text_preprocessed"] = df["text_preprocessed"].fillna("").astype(str)

    # Basic text features
    df["text_word_count"] = df["text_preprocessed"].apply(lambda x: len(str(x).split()))
    df["text_char_count"] = df["text_preprocessed"].apply(lambda x: len(str(x)))
    df["avg_word_length"] = df["text_preprocessed"].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0.0
    )
    df["exclamation_count"] = df["text_preprocessed"].str.count("!").fillna(0).astype(int)
    df["question_count"] = df["text_preprocessed"].str.count(r"\?").fillna(0).astype(int)
    df["punctuation_count"] = df["text_preprocessed"].str.count(r'[.,;:!?"\'()\\-]').fillna(0).astype(int)
    df["uppercase_word_count"] = df["text_preprocessed"].apply(lambda s: sum(1 for w in str(s).split() if w.isupper()))
    df["uppercase_word_ratio"] = df["uppercase_word_count"] / df["text_word_count"].replace(0, 1)

    # Stars normalized
    df["stars_norm"] = df["stars"].fillna(0).astype(float) / 5.0 if "stars" in df else 0.0

    # Has image
    if "reviewImageUrls" not in df.columns:
        df["reviewImageUrls"] = np.nan
    df["has_image"] = df["reviewImageUrls"].apply(
        lambda x: 0 if pd.isna(x) or str(x).strip()=="" or str(x).lower()=="nan" else 1
    )

    # Reviewer info
    if "reviewerNumberOfReviews" not in df.columns:
        df["reviewerNumberOfReviews"] = 0
    df["reviewer_count"] = df["reviewerNumberOfReviews"].fillna(0).astype(float)
    df["reviewer_count_log"] = np.log1p(df["reviewer_count"])

    # Local guide
    if "isLocalGuide" not in df.columns:
        df["isLocalGuide"] = 0
    df["is_local_guide"] = df["isLocalGuide"].apply(lambda x: 1 if str(x).lower() in ["true","1","yes"] else 0)

    # Detail rating
    def check_all_fives(x):
        if pd.isna(x) or str(x).strip()=="":
            return 0
        try:
            nums = [float(s) for s in str(x).replace(",", " ").split() if s.replace(".","").isdigit()]
            if len(nums)>0:
                return int(all(n==5.0 for n in nums))
        except:
            return 0
        return 0

    if "reviewDetailedRating" not in df.columns:
        df["reviewDetailedRating"] = ""
    df["detail_rating_all_5"] = df["reviewDetailedRating"].apply(check_all_fives)

    # Time features
    df["published_date"] = pd.to_datetime(df.get("publishedAtDate", None), errors="coerce")
    df["publish_hour"] = df["published_date"].dt.hour.fillna(0).astype(int)
    df["publish_day"] = df["published_date"].dt.date

    if df["publish_day"].notna().sum() > 0:
        day_counts = df.groupby("publish_day").size()
        df["same_day_count"] = df["publish_day"].map(day_counts).fillna(1).astype(int)
    else:
        df["same_day_count"] = 1

    # Suspicious patterns
    df["pattern_5star_short"] = ((df["stars"] == 5) & (df["text_word_count"] < 5)).astype(int)
    df["pattern_5star_nophoto"] = ((df["stars"] == 5) & (df["has_image"] == 0)).astype(int)
    df["pattern_new_reviewer"] = (df["reviewer_count"] <= 2).astype(int)
    df["pattern_same_day"] = (df["same_day_count"] > 5).astype(int)

    # Text quality
    df = extract_text_quality_features(df, text_column="text_preprocessed")

    print("[FEATURES] completed.")
    return df

# ========================================================
# === PREDICT FUNCTION
# ========================================================

def predict_csv(input_csv, output_csv, model_path):
    print("="*80)
    print("Predict Fake Review using RandomForestManual")
    print("VERSI ORIGINAL (TANPA POST-PROCESSING)")
    print("="*80)

    # Load model
    print("\n[1] Loading model...")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    rf = model_data["rf"]
    tfidf = model_data["tfidf"]
    numeric_features = model_data["numeric_features"]

    print(f"  Loaded model with {len(rf.trees)} trees.")
    print(f"  TF-IDF vocabulary size: {len(tfidf.vocab)}")

    # Load dataset
    print(f"\n[2] Loading dataset: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"  Total rows: {len(df)}")

    # Preprocessing + feature engineering
    print("\n[3] Extracting features...")
    df = extract_features(df)

    # TF-IDF transformation
    print("\n[3b] Transforming text with TF-IDF...")
    texts = df['text_preprocessed'].fillna('').astype(str).tolist()
    X_text = tfidf.transform(texts)
    print(f"  TF-IDF shape: {X_text.shape}")

    # Ensure numeric features exist
    for feat in numeric_features:
        if feat not in df.columns:
            df[feat] = 0

    X_num = df[numeric_features].fillna(0).values.astype(float)
    print(f"  Numeric features shape: {X_num.shape}")

    # Combine TF-IDF + numeric features
    X = np.hstack([X_text, X_num])
    print(f"  Combined feature shape: {X.shape}")

    # Predict
    print("\n[4] Predicting...")
    preds = rf.predict(X)

    # Confidence = jumlah vote mayoritas / total pohon
    prob, labels_sorted = rf.predict_proba(X)
    # labels_sorted example: [-1, 1] (REAL, FAKE)
    label_index = {lbl:i for i,lbl in enumerate(labels_sorted)}

    # Convert predictions
    predicted_label = []
    decision_scores = []
    confidences = []

    for i, p in enumerate(preds):
        # Voting confidence
        conf = prob[i][label_index[p]]
        confidences.append(conf)

        # Decision score = (votes_FAKE - votes_REAL)
        votes_FAKE = prob[i][label_index.get(1,0)]
        votes_REAL = prob[i][label_index.get(-1,0)]
        decision_scores.append(votes_FAKE - votes_REAL)

        predicted_label.append("FAKE" if p == 1 else "REAL")

    # TIDAK ADA POST-PROCESSING - versi original

    df["predicted_label"] = predicted_label
    df["confidence"] = confidences
    df["decision_score"] = decision_scores

    # Save result
    print(f"\n[Save] Saving result: {output_csv}")
    df.to_csv(output_csv, index=False)
    print("Done!")

    # Summary
    summary_file = output_csv.replace(".csv", "_summary.txt")
    with open(summary_file, "w") as f:
        f.write("Random Forest Manual Prediction Summary\n")
        f.write("VERSI ORIGINAL (TANPA POST-PROCESSING)\n")
        f.write("="*60 + "\n")
        f.write(f"Input: {input_csv}\n")
        f.write(f"Output: {output_csv}\n")
        f.write(f"Rows: {len(df)}\n\n")

        dist = df["predicted_label"].value_counts()
        f.write("Prediction Distribution:\n")
        for label, count in dist.items():
            f.write(f"  {label}: {count}\n")

        f.write("\nConfidence Stats:\n")
        f.write(f"  Mean: {df['confidence'].mean():.4f}\n")
        f.write(f"  Median: {df['confidence'].median():.4f}\n")

    print(f"[Summary saved] {summary_file}")

# ========================================================
# === MAIN ENTRY
# ========================================================

def main():
    # Hardcoded paths - no need for command line arguments
    input_csv = r"E:\Machine_Learning\Projek Fake Review\projek v2\dataset\tes.csv"
    output_csv = r"E:\Machine_Learning\Projek Fake Review\projek v2\rf_model_output\tes_predictions.csv"
    model_path = "rf_model_output/random_forest_model.pkl"

    predict_csv(input_csv, output_csv, model_path)

if __name__ == "__main__":
    main()
