"""
Script lengkap (revisi):
- Baca dataset CSV (set DATA_PATH)
- EDA ringkas + simpan plot
- Heuristic labeling (fake_score -> label FAKE / REAL)
- Tambah fitur numeric untuk analysis
- Opsional: train model jika dipanggil dengan --train

Perbaikan penting:
- Menggunakan foto reviewer: kolom 'reviewImageUrls/*' diprioritaskan untuk deteksi foto.
- Jika tidak ditemukan kolom reviewImageUrls, script fallback ke 'imageUrl' (foto tempat).
- Console output (print) dipertahankan lengkap untuk EDA.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import argparse

# ML imports (only used if training requested)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# Safety check: pastikan ada minimal 2 kelas sebelum split / training
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------
# === UBAH PATH DATA DI SINI
# ---------------------------
DATA_PATH = "dataset_raw/dataset_gabungan.csv"   # <-- Path ke file CSV yang sudah digabung
OUTPUT_DIR = "output_review_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CLI options: default is EDA only (no training) ---
parser = argparse.ArgumentParser(description="EDA and optional modeling for review dataset")
parser.add_argument('--train', action='store_true', help='Run model training after EDA (default: false)')
args = parser.parse_args()

# ---------------------------
# === LOAD DATA
# ---------------------------
print("Membaca dataset dari:", DATA_PATH)
df = pd.read_csv(DATA_PATH, low_memory=False)

print("Kolom tersedia:", df.columns.tolist())
print("Jumlah baris:", len(df))

# --- Tolerant column mapping: map common CSV variants to expected names ---
def _norm(s):
    return ''.join(ch.lower() for ch in str(s) if ch.isalnum())

cols_norm = { _norm(c): c for c in df.columns }

def find_col(*cands):
    """Return first actual column matching any candidate (case/format tolerant), or None."""
    for cand in cands:
        nc = _norm(cand)
        # exact normalized match
        if nc in cols_norm:
            return cols_norm[nc]
        # substring match (some CSVs include prefixes like 'reviewDetailedRating/Makanan')
        for k,v in cols_norm.items():
            if nc in k or k in nc:
                return v
    return None

# Common mappings: expected_name -> list of possible source names
aliases = {
    'nama': ['name', 'nama'],
    'reviewernumberofreview': ['reviewernumberofreviews', 'reviewerNumberOfReviews', 'reviewernumberofreview'],
    'islocalguid': ['islocalguide', 'isLocalGuide', 'islocalguid'],
    'publishAtDate': ['publishedAtDate', 'publishAt', 'publishedatdate', 'publishatdate'],
    'imageUrl': ['imageUrl', 'placeImage', 'placeimage'],
    'text': ['text', 'textTranslated', 'reviewText'],
    'stars': ['stars', 'rating'],
    'title': ['title', 'placeName', 'nameplace', 'restaurantname']
}

for tgt, cands in aliases.items():
    found = find_col(*cands)
    if found and found != tgt:
        df.rename(columns={found: tgt}, inplace=True)

# Construct `reviewDetailRating` by joining any columns that look like detailed rating fields
detail_cols = [c for c in df.columns if _norm('reviewdetailedrating') in _norm(c) or _norm('reviewdetailrating') in _norm(c)]
if detail_cols and 'reviewDetailRating' not in df.columns:
    # join non-empty parts per row into a comma-separated string
    try:
        df['reviewDetailRating'] = df[detail_cols].astype(str).apply(lambda row: ','.join([str(x).strip() for x in row if pd.notna(x) and str(x).strip()!='']), axis=1)
    except Exception:
        # fallback: simply stringify first column
        df['reviewDetailRating'] = df[detail_cols[0]].astype(str)

# Construct `reviewContext` by joining context columns (prefix 'reviewContext' or 'reviewContext/')
context_cols = [c for c in df.columns if 'reviewcontext' in _norm(c)]
if context_cols and 'reviewContext' not in df.columns:
    try:
        df['reviewContext'] = df[context_cols].astype(str).apply(lambda row: ','.join([str(x).strip() for x in row if pd.notna(x) and str(x).strip()!='']), axis=1)
    except Exception:
        df['reviewContext'] = df[context_cols[0]].astype(str)

# ---------------------------
# === PREPROCESSING DASAR
# ---------------------------
# Standarisasi nama kolom (strip)
df.columns = [c.strip() for c in df.columns]

# Pastikan kolom penting ada; jika tidak ada, buat kolom kosong agar tidak error
expected_cols = ['nama','stars','text','reviewernumberofreview','imageUrl','title',
                 'islocalguid','reviewContext','reviewDetailRating','publishAt','publishAtDate']
for c in expected_cols:
    if c not in df.columns:
        print(f"Kolom '{c}' tidak ditemukan. Membuat kolom kosong.")
        df[c] = np.nan

# Normalize beberapa kolom
# convert stars and reviewer counts safely
if 'stars' in df.columns:
    df['stars'] = pd.to_numeric(df['stars'], errors='coerce').fillna(0).astype(int)
else:
    df['stars'] = 0

if 'reviewernumberofreview' in df.columns:
    df['reviewernumberofreview'] = pd.to_numeric(df['reviewernumberofreview'], errors='coerce').fillna(0).astype(int)
else:
    df['reviewernumberofreview'] = 0

# islocalguid mungkin boolean atau string
if 'islocalguid' in df.columns:
    df['islocalguid'] = df['islocalguid'].map({True: True, False: False, 'True': True, 'true': True, 'False': False, 'false': False})
    df['islocalguid'] = df['islocalguid'].fillna(False).astype(bool)
else:
    df['islocalguid'] = False

# PublishAtDate -> datetime (coba beberapa format)
def parse_date_safe(x):
    if pd.isna(x):
        return pd.NaT
    try:
        return pd.to_datetime(x)
    except Exception:
        try:
            return pd.to_datetime(x, format="%Y-%m-%d")
        except Exception:
            return pd.NaT

if 'publishAtDate' in df.columns:
    df['publishAtDate'] = df['publishAtDate'].apply(parse_date_safe)
else:
    df['publishAtDate'] = pd.NaT

if 'publishAt' in df.columns:
    df['publishAt'] = df['publishAt'].apply(parse_date_safe)
else:
    df['publishAt'] = pd.NaT

# ---------------------------
# === DETEKSI KOLom reviewImageUrls (foto reviewer)
# ---------------------------
# cari semua kolom yang mengandung 'reviewimageurls' atau 'reviewimageurl' (case-insensitive)
img_cols = [c for c in df.columns if 'reviewimageurls' in _norm(c) or 'reviewimageurl' in _norm(c) or 'reviewphotos' in _norm(c)]
print("\nDeteksi kolom foto reviewer (reviewImageUrls):", img_cols)
if len(img_cols) == 0:
    print("Tidak ditemukan kolom 'reviewImageUrls/*'. Script akan fallback ke 'imageUrl' (foto tempat).")
    # buat kolom has_review_image False
    df['has_review_image'] = False
else:
    # jika ditemukan, anggap ada foto reviewer jika setidaknya satu kolom non-null per baris
    df['has_review_image'] = df[img_cols].notna().any(axis=1)
    print("Contoh baris dengan review image (index, any True):")
    print(df[df['has_review_image']].head(3)[img_cols + ['text']].to_string(index=False))

# ---------------------------
# === TENTUKAN has_image: prioritas ke review image, fallback ke imageUrl
# ---------------------------
# detect place image column presence
place_image_present = 'imageUrl' in df.columns and df['imageUrl'].notna().any()
if 'has_review_image' in df.columns and df['has_review_image'].any():
    df['has_image'] = df['has_review_image'].astype(bool)
    print("\nMenggunakan 'reviewImageUrls/*' sebagai sumber foto reviewer untuk 'has_image'.")
elif place_image_present:
    df['has_image'] = df['imageUrl'].apply(lambda x: False if pd.isna(x) or str(x).strip()=="" else True)
    print("\nTidak ada kolom reviewImageUrls/*, menggunakan 'imageUrl' (foto tempat) sebagai fallback untuk 'has_image'.")
else:
    df['has_image'] = False
    print("\nTidak ada data foto (reviewImageUrls maupun imageUrl). 'has_image' = False semua.")

# ---------------------------
# === Tampilkan nama restoran unik
# ---------------------------
print("\n--- Nama Restoran Unik dalam Dataset ---")
unique_restaurants = df['title'].dropna().unique()
print(f"Ditemukan {len(unique_restaurants)} nama restoran unik.")
# tampilkan sampai 20 nama saja agar console tidak terlalu panjang
for restaurant in unique_restaurants[:20]:
    print(f"- {restaurant}")
if len(unique_restaurants) > 20:
    print(f"... (total {len(unique_restaurants)} nama, hanya 20 pertama ditampilkan)")

# Cek dan tampilkan nama restoran yang duplikat
print("\n--- Pengecekan Restoran Duplikat ---")
duplicated_restaurants_df = df[df.duplicated('title', keep=False)].dropna(subset=['title'])
if not duplicated_restaurants_df.empty:
    duplicated_titles = duplicated_restaurants_df['title'].unique()
    print(f"Ditemukan {len(duplicated_titles)} nama restoran yang muncul lebih dari sekali:")
    for title in duplicated_titles[:20]:
        count = df[df['title'] == title].shape[0]
        print(f"- {title} (muncul {count} kali)")
    if len(duplicated_titles) > 20:
        print(f"... (hanya 20 pertama ditampilkan)")
else:
    print("Tidak ada nama restoran yang duplikat dalam dataset.")

# ---------------------------
# === EDA RINGKAS (print + simpan plot)
# ---------------------------
print("\n--- EDA ringkas ---")
print("Distribusi stars:")
print(df['stars'].value_counts().sort_index())

# Simpan histogram rating
plt.figure(figsize=(6,4))
df['stars'].value_counts().sort_index().plot(kind='bar')
plt.title("Distribusi Stars")
plt.xlabel("Stars")
plt.ylabel("Jumlah")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stars_distribution.png"))
plt.close()
print("Saved: stars_distribution.png")

# Panjang teks (kata)
df['text'] = df['text'].fillna("").astype(str)
df['text_len_words'] = df['text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(6,4))
plt.hist(df['text_len_words'].clip(0,200), bins=30)
plt.title("Distribusi Panjang Review (kata, truncated 200)")
plt.xlabel("Jumlah kata")
plt.ylabel("Jumlah review")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "text_length_hist.png"))
plt.close()
print("Saved: text_length_hist.png")

# Persentase ada foto (menggunakan sumber prioritas)
print("Persentase review dengan foto (berdasarkan sumber prioritas): {:.2f}%".format(df['has_image'].mean()*100))

# Pengecekan text kosong per stars
print("\nContoh review bintang 5 dengan teks kosong / pendek (sample 5):")
mask_5_short = (df['stars']==5) & (df['text_len_words'] <= 3)
print(df[mask_5_short].head(5)[['title','stars','text','text_len_words','has_image']].to_string(index=False))

# ---------------------------
# === LABELING HEURISTIK (fake_score)
# ---------------------------
def review_detail_all_fives(x):
    if pd.isna(x):
        return False
    s = str(x)
    # cari pattern 5,5,5 atau semua angka 5
    parts = [p.strip() for p in s.replace(';',',').split(',') if p.strip()!='']
    if len(parts) == 0:
        return False
    try:
        return all(int(float(p)) == 5 for p in parts)
    except (ValueError, TypeError):
        return False

def compute_fake_score(row):
    score = 0
    reasons = []
    # A. Stars = 5 + teks pendek atau kosong
    if row['stars'] == 5:
        if row['text_len_words'] == 0:
            score += 3
            reasons.append("Bintang 5 tanpa teks")
        elif row['text_len_words'] < 5:
            score += 2
            reasons.append("Bintang 5 & teks sangat pendek")

    # B. Tidak ada foto (prioritaskan foto reviewer)
    if not row['has_image']:
        score += 1
        reasons.append("Tidak ada foto reviewer")

    # C. reviewDetailRating semua 5
    try:
        if review_detail_all_fives(row.get('reviewDetailRating', None)):
            score += 1
            reasons.append("Detail rating semua 5")
    except Exception:
        pass

    # D. Reviewer hanya punya sedikit review
    if row['reviewernumberofreview'] <= 2:
        score += 2
        reasons.append("Reviewer baru (<=2 review)")

    # E. Bukan local guide
    if not bool(row['islocalguid']):
        score += 1
        reasons.append("Bukan Local Guide")

    # F. ReviewContext kosong
    if pd.isna(row.get('reviewContext')) or str(row.get('reviewContext')).strip() == "":
        score += 1
        reasons.append("Konteks review kosong")

    return score, reasons

# Apply
print("\nMenghitung fake_score untuk setiap review (heuristik)...")
score_reasons = df.apply(compute_fake_score, axis=1)
df['fake_score'] = score_reasons.apply(lambda x: x[0])
df['reasons_list'] = score_reasons.apply(lambda x: x[1])

# ---------------------------
# === DETEKSI POLA WAKTU: banyak review di hari yang sama
# ---------------------------
# Hitung jumlah review per hari (publishAtDate.date)
df['publish_day'] = df['publishAtDate'].dt.date
same_day_counts = df.groupby('publish_day').size().rename("same_day_count")
df = df.merge(same_day_counts, how='left', left_on='publish_day', right_index=True)
df['same_day_count'] = df['same_day_count'].fillna(0).astype(int)

# Jika hari ada > 5 review (threshold bisa disesuaikan), tambahkan poin
df.loc[df['same_day_count'] > 5, 'fake_score'] += 2
mask_sameday = df['same_day_count'] > 5
df.loc[mask_sameday, 'reasons_list'] = df.loc[mask_sameday, 'reasons_list'].apply(lambda r: r + ["Banyak review di hari yg sama (>5)"])

# ---------------------------
# === BUAT LABEL FINAL
# ---------------------------
# Threshold: fake_score >= 4 -> FAKE
THRESHOLD = 4
df['label'] = df['fake_score'].apply(lambda x: 'FAKE' if x >= THRESHOLD else 'REAL')

# Buat kolom 'reason' yang mudah dibaca dari daftar alasan
df['reason'] = df.apply(
    lambda row: "; ".join(row['reasons_list']) if row['label'] == 'FAKE' else "Terlihat asli",
    axis=1
)

print(f"\nTotal FAKE (score >= {THRESHOLD}):", (df['label']=='FAKE').sum())
print("Contoh FAKE (5 baris):")
print(df[df['label']=='FAKE'].head(5)[['title','stars','text_len_words','has_image','fake_score','reason']].to_string(index=False))

print("\nMembuat plot untuk distribusi fake_score dan label...")
plt.figure(figsize=(8, 5))
plt.hist(df['fake_score'], bins=int(df['fake_score'].max() + 1), align='left', rwidth=0.8)
plt.title(f'Distribusi Fake Score (Threshold = {THRESHOLD})')
plt.xlabel('Fake Score')
plt.ylabel('Jumlah Review')
plt.axvline(x=THRESHOLD, color='r', linestyle='--', label=f'Threshold ({THRESHOLD})')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fake_score_distribution.png"))
plt.close()
print("Saved: fake_score_distribution.png")

print(f"\nDistribusi fake_score (threshold = {THRESHOLD}):")
print(df['fake_score'].value_counts().sort_index())

# Simpan dataset berlabel
labeled_path = os.path.join(OUTPUT_DIR, "google_review_labeled.csv")
df.to_csv(labeled_path, index=False)
print(f"\nDataset berlabel disimpan ke: {labeled_path}")

# Plot distribusi label
plt.figure(figsize=(6, 4))
df['label'].value_counts().plot(kind='bar')
plt.title('Distribusi Label Final (REAL vs FAKE)')
plt.xlabel('Label')
plt.ylabel('Jumlah Review')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "label_distribution.png"))
plt.close()
print("Saved: label_distribution.png")
print("Distribusi label:")
print(df['label'].value_counts())

# ---------------------------
# === SIAPKAN FITUR UNTUK MODELING
# ---------------------------
# Fitur teks: 'text'
# Fitur numerik: stars, reviewernumberofreview, text_len_words, has_image (0/1), islocalguid (0/1), same_day_count
df['has_image_i'] = df['has_image'].astype(int)
df['islocalguid_i'] = df['islocalguid'].astype(int)

feature_text = 'text'
numeric_features = ['stars', 'reviewernumberofreview', 'text_len_words', 'has_image_i', 'islocalguid_i', 'same_day_count']

X = df[[feature_text] + numeric_features].copy()
y = df['label'].copy()

# Encode label ke 0/1
le = LabelEncoder()
y_enc = le.fit_transform(y)  # FAKE/REAL -> 0/1 (cek kelas dengan le.classes_)

print("Label encoding:", dict(enumerate(le.classes_)))

# Safety check: pastikan ada minimal 2 kelas sebelum split / training
unique, counts = np.unique(y_enc, return_counts=True)
class_counts = dict(zip(le.inverse_transform(unique), counts))
print("Class counts:", class_counts)
if len(unique) < 2:
    raise ValueError(f"Target contains only one class: {class_counts}. "
                     f"Adjust your labeling heuristic or dataset so there are at least 2 classes.")

# Split (gunakan stratify hanya jika ada >=2 kelas)
stratify_param = y_enc if len(unique) >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=stratify_param)

# ColumnTransformer:
# - TF-IDF pada kolom 'text'
# - scaler pada numeric features
text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2)))
])
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, feature_text),
        ('num', numeric_transformer, numeric_features),
    ]
)

# ---------------------------
# === TRAIN & EVALUATE MODELS (optional)
# Training happens only when script is run with --train
if args.train:
    def train_and_eval(clf, name):
        print("\n=== Training:", name)
        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                               ('clf', clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        print(f"--- Results for {name} ---")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        disp.plot()
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{name}.png"))
        plt.close()
        # simpan model
        model_path = os.path.join(OUTPUT_DIR, f"model_{name}.joblib")
        joblib.dump(pipe, model_path)
        print(f"Model {name} disimpan ke: {model_path}")
        return pipe

    # Linear SVM
    svm_pipe = train_and_eval(LinearSVC(max_iter=5000, dual=False, random_state=42), "LinearSVC")

    # Random Forest
    rf_pipe = train_and_eval(RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), "RandomForest")

    # Multinomial NB (text-only)
    preprocessor_mnb = ColumnTransformer(transformers=[
        ('text', text_transformer, feature_text),
    ], remainder='drop')
    mnb_pipe = Pipeline(steps=[('pre', preprocessor_mnb), ('clf', MultinomialNB())])
    print("\n=== Training: MultinomialNB (text-only) ===")
    mnb_pipe.fit(X_train, y_train)
    y_pred_mnb = mnb_pipe.predict(X_test)
    print(classification_report(y_test, y_pred_mnb, target_names=le.classes_))
    cm = confusion_matrix(y_test, y_pred_mnb)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot()
    plt.title("Confusion Matrix - MultinomialNB")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_MultinomialNB.png"))
    plt.close()
    joblib.dump(mnb_pipe, os.path.join(OUTPUT_DIR, "model_MultinomialNB.joblib"))
    print("Model MultinomialNB disimpan.")
else:
    print("\nModel training skipped (run with --train to enable training).")

# ---------------------------
# === SUMMARY & NEXT STEPS
# ---------------------------
print("\nSelesai. File keluaran ada di folder:", OUTPUT_DIR)
print("- Labeled CSV:", labeled_path)
if args.train:
    print("- Model files: model_*.joblib di folder output")
else:
    print("- Model files: (skipped; run with --train to train & save models)")

print("\nRekomendasi lanjutan:")
print("1) Tweak heuristics (poin/threshold) jika terlalu agresif/kurang deteksi.")
print("2) Tambah fitur: jarak waktu antar review, similarity antar teks (cosine) untuk deteksi copy-paste.")
print("3) Lakukan cross-validation dan hyperparameter tuning untuk tiap model.")
print("4) Pertimbangkan anotasi manual 200-500 sampel untuk validasi heuristics.")

# Jika ingin dump sample hasil label
sample_out_cols = ['nama','stars','text','fake_score','label','reason','has_image']
sample_out_cols = [c for c in sample_out_cols if c in df.columns]
sample_out = df[sample_out_cols].sample(n=min(50, len(df)), random_state=42)
sample_out.to_csv(os.path.join(OUTPUT_DIR, "sample_labeled_50.csv"), index=False)
print("Sample labeled 50 baris disimpan ke output.")
