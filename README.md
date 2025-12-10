# Deteksi Fake Review Google Maps Restaurant

**Anggota Kelompok:**
- Diva Filemon Manik (71220863)
- Anjelita Haninuna (71220925)

**Topik:** Deteksi Fake Review Ulasan Tempat Makan di Yogyakarta dengan Algoritma SVM, Random Forest, dan Naive Bayes

**Sumber Dataset:** Hasil scraping mandiri menggunakan Apify dari Google Maps

---

## Struktur Folder Proyek

```
Fake_Review_Google_Maps_Resto_Detection_SVM_Alogrithm_with_datasets/
│
├── labeling_eda.py                     # Script labeling otomatis (heuristic) + EDA
├── preprocessing.py                    # Pipeline preprocessing teks Indonesia
├── balance_dataset.py                  # Script balancing dataset
├── check_balance.py                    # Verifikasi distribusi dataset
├── debug_gibberish.py                  # Testing deteksi kualitas teks
│
├── dataset_raw/                        # Dataset mentah hasil scraping
│   └── dataset_gabungan.csv            # Data review gabungan dari Apify
│
├── dataset_balance/                    # Dataset yang sudah di-balance
│   ├── google_review_balanced_combined.csv         # Metode hybrid
│   ├── google_review_balanced_oversampling.csv     # Metode oversampling
│   └── google_review_balanced_undersampling.csv    # Metode undersampling
│
├── dataset_testing/                    # Data untuk testing
│   └── tes.csv                         # Sample data test
│
├── file_training/                      # Script training model
│   ├── svm_train.py                    # Training SVM
│   ├── randomForest_train.py           # Training Random Forest
│   ├── nb_train_akhir.py               # Training Naive Bayes
│   │
│   ├── svm_model_output/               # Output model & metrik SVM
│   ├── rf_model_output/                # Output model & metrik Random Forest
│   └── nb_model_output/                # Output model & metrik Naive Bayes
│
├── file_testing/                       # Script prediksi/testing
│   ├── svm_multifeature_predict.py     # Prediksi dengan SVM
│   ├── randomForest_predict.py         # Prediksi dengan Random Forest
│   ├── nb_predict_akhir.py             # Prediksi dengan Naive Bayes
│   ├── text_quality_detector.py        # Modul deteksi gibberish/spam
│   │
│   ├── svm_model_output/               # Hasil prediksi SVM
│   ├── rf_model_output/                # Hasil prediksi Random Forest
│   └── nb_model_output/                # Hasil prediksi Naive Bayes
│
├── label_output/                       # Output labeling & statistik
│   └── balancing_comparison.csv        # Perbandingan metode balancing
│
└── README.md
```

---

## Alur Kerja Sistem

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ALUR KERJA SISTEM                              │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────┐
   │ 1. PENGUMPULAN   │  Scraping data review dari Google Maps
   │    DATA          │  wilayah Yogyakarta menggunakan Apify
   │                  │  Output: dataset_raw/dataset_gabungan.csv
   └────────┬─────────┘
            │
            ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 2. LABELING OTOMATIS (labeling_eda.py)                          │
   │                                                                 │
   │  Menghitung fake_score berdasarkan kriteria:                    │
   │  - Bintang 5 tanpa teks (+3) / teks pendek (+2)                 │
   │  - Tidak ada foto reviewer (+1)                                 │
   │  - Detail rating semua 5 (+1)                                   │
   │  - Reviewer baru ≤2 review (+2)                                 │
   │  - Bukan Local Guide (+1)                                       │
   │  - Konteks review kosong (+1)                                   │
   │  - Banyak review di hari sama >5 (+2)                           │
   │                                                                 │
   │  Label: fake_score >= 4 → FAKE, else → REAL                     │
   │  Output: google_review_labeled.csv                              │
   └────────┬────────────────────────────────────────────────────────┘
            │
            ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ 3. PREPROCESSING (preprocessing.py)                              │
   │                                                                  │
   │  Case Folding → Cleansing → Normalisasi → Tokenizing →           │
   │  Stopword Removal → Stemming                                     │
   │                                                                  │
   │  Output: kolom text_preprocessed                                 │
   └────────┬─────────────────────────────────────────────────────────┘
            │
            ▼
   ┌──────────────────┐
   │ 4. BALANCING     │  Menyeimbangkan distribusi FAKE dan REAL
   │    DATASET       │  dengan 3 metode:
   │                  │  - Undersampling
   │                  │  - Oversampling
   │                  │  - Hybrid (Kombinasi)
   └────────┬─────────┘
            │
            ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 5. FEATURE ENGINEERING                                          │
   │                                                                 │
   │  ┌─────────────────┐   ┌─────────────────┐   ┌───────────────┐  │
   │  │   TF-IDF        │   │ Numeric Features│   │ Text Quality  │  │
   │  │ (1000 fitur)    │   │ (18+ fitur)     │   │ Features      │  │
   │  └────────┬────────┘   └────────┬────────┘   └───────┬───────┘  │
   │           │                     │                    │          │
   │           └─────────────────────┴────────────────────┘          │
   │                                 │                               │
   │                                 ▼                               │
   │                    ┌─────────────────────┐                      │
   │                    │  Combined Features  │                      │
   │                    │    (40+ fitur)      │                      │
   │                    └─────────────────────┘                      │
   └────────┬────────────────────────────────────────────────────────┘
            │
            ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 6. TRAINING MODEL                                               │
   │                                                                 │
   │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
   │  │     SVM       │  │ Random Forest │  │  Naive Bayes  │        │
   │  │  (Linear,     │  │ (30 trees,    │  │   (Hybrid:    │        │
   │  │   manual)     │  │  CART, Gini)  │  │  MNB + GNB)   │        │
   │  └───────────────┘  └───────────────┘  └───────────────┘        │
   └────────┬────────────────────────────────────────────────────────┘
            │
            ▼
   ┌──────────────────┐
   │ 7. EVALUASI      │  Metrik: Accuracy, Precision, Recall, F1-Score
   │                  │  Visualisasi: Confusion Matrix, Performance Chart
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ 8. PREDIKSI      │  Menggunakan model terlatih untuk
   │                  │  memprediksi data baru
   └──────────────────┘
```

---

## Fungsi-Fungsi dalam Proyek

### labeling_eda.py

| Fungsi | Deskripsi |
|--------|-----------|
| `find_col(*cands)` | Mencari kolom dengan toleransi nama (case-insensitive) |
| `parse_date_safe(x)` | Parsing tanggal dengan berbagai format |
| `review_detail_all_fives(x)` | Mengecek apakah semua detail rating bernilai 5 |
| `compute_fake_score(row)` | Menghitung skor fake berdasarkan kriteria heuristik |
| `train_and_eval(clf, name)` | Training dan evaluasi model (opsional dengan --train) |

### preprocessing.py

| Fungsi | Deskripsi |
|--------|-----------|
| `case_folding(text)` | Mengubah semua huruf menjadi lowercase |
| `cleansing(text)` | Menghapus URL, mention, hashtag, emoji, dan angka |
| `normalize_text(text)` | Mengubah kata tidak baku menjadi baku menggunakan kamus normalisasi |
| `tokenizing(text)` | Memecah teks menjadi list kata-kata |
| `stopword_removal(tokens)` | Menghapus stopword bahasa Indonesia menggunakan Sastrawi |
| `stemming(tokens)` | Mengubah kata ke bentuk dasar menggunakan Sastrawi |
| `preprocess_pipeline(text)` | Menjalankan seluruh tahap preprocessing secara berurutan |

### balance_dataset.py

| Fungsi | Deskripsi |
|--------|-----------|
| `undersampling()` | Mengurangi jumlah kelas mayoritas agar seimbang dengan minoritas |
| `oversampling()` | Menduplikasi kelas minoritas agar seimbang dengan mayoritas |
| `hybrid_balancing()` | Kombinasi undersampling dan oversampling |
| `analyze_distribution()` | Menganalisis distribusi label FAKE dan REAL |

### svm_train.py

| Class/Fungsi | Deskripsi |
|--------------|-----------|
| `LinearSVM` | Implementasi manual SVM dengan Hinge Loss dan L2 Regularization |
| `LinearSVM.fit(X, y, X_val, y_val)` | Training model dengan early stopping |
| `LinearSVM.predict(X)` | Memprediksi label dari data input |
| `LinearSVM.decision_function(X)` | Menghitung nilai keputusan (score) |
| `TfidfVectorizerManual` | Implementasi manual TF-IDF vectorization |
| `StandardScalerSimple` | Normalisasi fitur dengan Z-score |
| `PCA` | Principal Component Analysis untuk visualisasi |
| `extract_features(df)` | Mengekstrak semua fitur dari dataframe |
| `calculate_metrics(y_true, y_pred)` | Menghitung Accuracy, Precision, Recall, F1 |

### randomForest_train.py

| Class/Fungsi | Deskripsi |
|--------------|-----------|
| `DecisionTreeCART` | Implementasi manual Decision Tree dengan algoritma CART |
| `DecisionTreeCART.fit(X, y)` | Training tree dengan Gini Impurity |
| `DecisionTreeCART.predict(X)` | Prediksi menggunakan tree |
| `DecisionTreeCART._best_split()` | Mencari split terbaik berdasarkan Gini |
| `RandomForestManual` | Implementasi manual Random Forest (ensemble of trees) |
| `RandomForestManual.fit(X, y)` | Training multiple trees dengan bootstrap sampling |
| `RandomForestManual.predict(X)` | Majority voting dari semua trees |
| `TreeNode` | Class untuk node dalam decision tree |

### nb_train_akhir.py

| Class/Fungsi | Deskripsi |
|--------------|-----------|
| `MultinomialNB` | Naive Bayes untuk fitur TF-IDF (diskrit) |
| `GaussianNB` | Naive Bayes untuk fitur numerik (kontinu) |
| `HybridNaiveBayes` | Kombinasi MultinomialNB (teks) dan GaussianNB (numerik) |
| `HybridNaiveBayes.fit(X_text, X_numeric, y)` | Training kedua model |
| `HybridNaiveBayes.predict(X_text, X_numeric)` | Prediksi dengan weighted combination |

### text_quality_detector.py

| Fungsi | Deskripsi |
|--------|-----------|
| `vowel_consonant_ratio(text)` | Menghitung rasio vokal terhadap konsonan |
| `valid_word_ratio(text)` | Menghitung persentase kata valid dalam kamus |
| `text_entropy(text)` | Menghitung Shannon entropy dari teks |
| `repeated_char_ratio(text)` | Menghitung rasio karakter berulang |
| `detect_gibberish(text)` | Mendeteksi teks gibberish/spam secara keseluruhan |
| `extract_text_quality_features(df, text_column)` | Mengekstrak semua fitur kualitas teks |

---

## Kriteria Labeling Heuristik

Labeling dilakukan secara otomatis menggunakan `labeling_eda.py` dengan menghitung `fake_score`:

| Kriteria | Poin |
|----------|------|
| Bintang 5 tanpa teks | +3 |
| Bintang 5 dengan teks sangat pendek (<5 kata) | +2 |
| Tidak ada foto reviewer | +1 |
| Detail rating semua bernilai 5 | +1 |
| Reviewer baru (≤2 review) | +2 |
| Bukan Local Guide | +1 |
| Konteks review kosong | +1 |
| Banyak review di hari yang sama (>5) | +2 |

**Penentuan Label:**
- `fake_score >= 4` → **FAKE**
- `fake_score < 4` → **REAL**

---

## Fitur-Fitur yang Digunakan

### A. TF-IDF Features (1000 fitur)

Mengubah teks menjadi vektor numerik berdasarkan frekuensi dan kepentingan kata.

### B. Numeric Features (18+ fitur)

| Fitur | Deskripsi |
|-------|-----------|
| `text_word_count` | Jumlah kata dalam review |
| `text_char_count` | Jumlah karakter dalam review |
| `avg_word_length` | Rata-rata panjang kata |
| `exclamation_count` | Jumlah tanda seru (!) |
| `question_count` | Jumlah tanda tanya (?) |
| `punctuation_count` | Total tanda baca |
| `uppercase_word_ratio` | Rasio kata dengan huruf kapital |
| `stars_norm` | Rating bintang dinormalisasi (0-1) |
| `has_image` | Apakah review memiliki foto (0/1) |
| `n_images` | Jumlah foto dalam review |
| `reviewer_count_log` | Log jumlah review dari reviewer |
| `is_local_guide` | Apakah reviewer adalah Local Guide (0/1) |
| `detail_rating_all_5` | Semua detail rating bernilai 5 (0/1) |
| `same_day_count` | Jumlah review di hari yang sama |
| `pattern_5star_short` | Bintang 5 dengan review pendek |
| `pattern_5star_nophoto` | Bintang 5 tanpa foto |
| `pattern_new_reviewer` | Reviewer baru (≤2 review) |
| `pattern_same_day` | Banyak review di hari yang sama |

### C. Text Quality Features

| Fitur | Deskripsi |
|-------|-----------|
| `tq_vowel_ratio` | Rasio vokal vs konsonan |
| `tq_valid_word_ratio` | Persentase kata valid dalam kamus |
| `tq_entropy` | Tingkat keacakan teks (Shannon entropy) |
| `tq_gibberish_score` | Skor gabungan deteksi gibberish |
| `tq_repeated_char_ratio` | Rasio karakter berulang |

---

## Kombinasi Fitur dalam Training Model

### Struktur Fitur yang Digunakan

**KOMBINASI 3 tipe fitur:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMBINED FEATURES (40+ fitur)                │
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────┐             │
│  │  TF-IDF Text Features│  │  Numeric Indicators  │             │
│  │   (1000 fitur)       │  │    (18+ fitur)       │             │
│  │                      │  │                      │             │
│  │ • Frekuensi kata     │  │ • text_word_count    │             │
│  │ • Bobot TF-IDF       │  │ • exclamation_count  │             │
│  │ • Representasi teks  │  │ • stars_norm         │             │
│  │   (vocabulary size   │  │ • has_image          │             │
│  │    = 1000)           │  │ • reviewer_count_log │             │
│  │                      │  │ • Pattern FAKE       │             │
│  │                      │  │ • Date features      │             │
│  └──────────────────────┘  └──────────────────────┘             │
│           │                            │                        │
│           │  ┌──────────────────────┐  │                        │
│           │  │ Text Quality Features│  │                        │
│           │  │  (6 fitur opsional)  │  │                        │
│           │  │                      │  │                        │
│           │  │ • tq_entropy         │  │                        │
│           │  │ • tq_valid_word_ratio│  │                        │
│           │  │ • tq_gibberish_score │  │                        │
│           │  └──────────────────────┘  │                        │
│           └────────────────┬────────────┘                       │
│                            │                                    │
│                    ┌───────▼────────┐                           │
│                    │  Concatenated  │                           │
│                    │  Feature Matrix│                           │
│                    │  untuk Training│                           │
│                    └────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### Rincian Lengkap Fitur Indikator

#### **1. TF-IDF Text Features (1000 fitur)**
Fitur ini menangkap **semantik dan makna teks** secara numerik:
- **Purpose:** Membedakan kata-kata yang sering muncul di fake review vs real review
- **Method:** Term Frequency-Inverse Document Frequency
- **Contoh:** Kata seperti "bagus", "recommended", "enak" punya bobot berbeda di FAKE vs REAL
- **Jumlah Features:** 1000 (max vocabulary size)
- **Type:** Continuous (0.0 - 1.0)

#### **2. Numeric Indicators (18+ fitur)**
Fitur ini menangkap **karakteristik metadata dan pola perilaku**:

**Text Structure Indicators (7 fitur):**
- `text_word_count` - Panjang review (fake lebih pendek)
- `text_char_count` - Jumlah karakter
- `avg_word_length` - Panjang rata-rata kata
- `exclamation_count` - Tanda seru (fake lebih emosional)
- `question_count` - Tanda tanya
- `punctuation_count` - Tanda baca total
- `uppercase_word_ratio` - Huruf kapital (spam lebih banyak)

**Rating & Reviewer Behavior (6 fitur):**
- `stars_norm` - Rating bintang (fake cenderung 5 bintang)
- `reviewer_count_log` - Pengalaman reviewer (fake dari reviewer baru)
- `is_local_guide` - Status Local Guide (real reviewer trusted)
- `has_image` - Ada foto (fake jarang punya foto)
- `n_images` - Jumlah foto (real lebih detail)
- `detail_rating_all_5` - Semua rating detail = 5 (suspiciously perfect)

**Temporal & Frequency Patterns (5 fitur):**
- `publish_hour` - Jam publikasi
- `same_day_count` - Review banyak di hari sama (coordinated attack)
- `pattern_5star_short` - 5 bintang + review pendek (classic fake pattern)
- `pattern_5star_nophoto` - 5 bintang tanpa foto (another fake indicator)
- `pattern_new_reviewer` - Reviewer ≤2 review total (suspicious new account)
- `pattern_same_day` - Banyak review di hari yang sama (fake brigade)

**Fitur Aggregated:**
```
Total Numeric Features = 18 fitur
```

#### **3. Text Quality Features (6 fitur opsional)**
Fitur ini mendeteksi **anomali teks dan spam signals**:
- `tq_entropy` - Shannon entropy (random/gibberish text)
- `tq_valid_word_ratio` - % kata valid dalam kamus (gibberish detection)
- `tq_avg_word_length` - Panjang rata-rata kata (quality check)
- `tq_gibberish_score` - Skor gabungan deteksi gibberish
- `tq_repeat_char_ratio` - Karakter berulang (spam "oooooo", "eeeeee")
- `tq_consonant_cluster_ratio` - Cluster konsonan (gibberish indicator)

### Alasan Menggunakan Kombinasi Fitur

| Fitur Type | Alasan Penggunaan | Contoh Deteksi |
|------------|-------------------|-----------------|
| **TF-IDF (Text)** | Menangkap **makna dan topik** teks | Fake review lebih sering menggunakan kata umum ("bagus", "recommended") |
| **Numeric Indicators** | Menangkap **pola perilaku dan metadata** | Fake review dari akun baru, 5 bintang tanpa foto |
| **Text Quality** | Mendeteksi **spam dan gibberish** | Deteksi review yang ditulis bukan manusia atau spam |

**Kombinasi ketiganya penting karena:**
1. ✅ **TF-IDF saja:** Bisa tertipu dengan review pendek yang valid secara bahasa
2. ✅ **Numeric saja:** Bisa salah classify karena review bintang 5 bisa legitimate (food blogger)
3. ✅ **Ketiganya bersama:** Saling melengkapi untuk deteksi multi-dimensional

### Feature Engineering Pipeline di Training

Setiap model (SVM, Random Forest, Naive Bayes) mengikuti pipeline yang sama:

```python
# 1. Text Preprocessing (Sastrawi)
text_preprocessed = case_folding() 
                    → cleansing() 
                    → normalization() 
                    → stopword_removal() 
                    → stemming()

# 2. TF-IDF Vectorization
X_text = TfidfVectorizer(max_features=1000).fit_transform(text_preprocessed)
# Output shape: (n_samples, 1000)

# 3. Numeric Features Extraction
numeric_features = [
    'text_word_count', 'text_char_count', 'avg_word_length',
    'exclamation_count', 'question_count', 'punctuation_count', 
    'uppercase_word_ratio', 'stars_norm', 'has_image', 'n_images',
    'reviewer_count_log', 'is_local_guide', 'detail_rating_all_5',
    'same_day_count', 'pattern_5star_short', 'pattern_5star_nophoto',
    'pattern_new_reviewer', 'pattern_same_day'
]
X_numeric = df[numeric_features].fillna(0).values
# Output shape: (n_samples, 18)

# 4. Text Quality Features (Optional)
text_quality_features = extract_text_quality_features(df)
# Output: 6 additional features

# 5. Feature Concatenation
X_combined = np.hstack([X_text, X_numeric, X_text_quality])
# Final shape: (n_samples, 1000 + 18 + 6 = 1024 fitur)

# 6. Training Models
model.fit(X_combined, y)
```

### Konsistensi Across Models

✅ **Ketiga model (SVM, RF, NB) menggunakan fitur yang IDENTIK:**
- TF-IDF configuration yang sama (max_features=1000, unigram)
- Numeric features yang sama (18 fitur)
- Text quality features yang sama
- Preprocessing pipeline yang sama

Ini memastikan **fair comparison** antara ketiga algoritma!

---

## Algoritma yang Digunakan

### 1. Support Vector Machine (SVM)

- **Tipe:** Linear SVM dengan implementasi manual
- **Loss Function:** Hinge Loss + L2 Regularization
- **Optimisasi:** Gradient Descent dengan Early Stopping
- **Label:** FAKE (+1), REAL (-1)

**Hyperparameter:**
```
learning_rate = 0.01
lambda_param = 0.01
n_iterations = 2000
early_stopping_rounds = 50
```

### 2. Random Forest

- **Base Learner:** Decision Tree dengan algoritma CART
- **Split Criterion:** Gini Impurity
- **Ensemble Method:** Majority Voting dari 30 trees
- **Bootstrap:** Random sampling with replacement

**Hyperparameter:**
```
n_estimators = 30
max_depth = 12
min_samples_split = 2
min_samples_leaf = 2
max_features = "sqrt"
```

### 3. Naive Bayes (Hybrid)

- **Tipe:** Kombinasi Multinomial NB dan Gaussian NB
- **Multinomial NB:** Untuk fitur TF-IDF (diskrit)
- **Gaussian NB:** Untuk fitur numerik (kontinu)
- **Kombinasi:** Weighted average (60% teks, 40% numerik)

**Hyperparameter:**
```
alpha = 1.0
text_weight = 0.6
numeric_weight = 0.4
max_features_tfidf = 1000
```

---

## Rumus dan Perhitungan Algoritma Training

### 1. TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF adalah metode untuk mengkonversi teks menjadi representasi numerik berdasarkan frekuensi dan kepentingan kata.

#### Rumus Term Frequency (TF):
```
TF(t, d) = jumlah kemunculan kata t dalam dokumen d / total kata dalam dokumen d
```

**Contoh:**
- Dokumen: "makanan enak sekali makanan"
- TF("makanan") = 2/4 = 0.5
- TF("enak") = 1/4 = 0.25

#### Rumus Inverse Document Frequency (IDF):
```
IDF(t) = log(N / (df(t) + 1))

dimana:
- N = total jumlah dokumen
- df(t) = jumlah dokumen yang mengandung kata t
```

**Contoh:**
- Total dokumen (N) = 1000
- Dokumen mengandung "enak" = 100
- IDF("enak") = log(1000 / (100 + 1)) = log(9.9) ≈ 2.29

#### Rumus TF-IDF Final:
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**Implementasi dalam kode:**
```python
# TF-IDF calculation
tf = count_word / total_words_in_doc
idf = math.log(N / (word_doc_count[word] + 1))
tfidf_value = tf * idf
```

---

### 2. Support Vector Machine (SVM) - Linear

SVM adalah algoritma klasifikasi yang mencari hyperplane optimal untuk memisahkan dua kelas.

#### Model Prediksi:
```
f(x) = w · x + b

Prediksi:
- Jika f(x) >= 0 → FAKE (+1)
- Jika f(x) < 0 → REAL (-1)

dimana:
- w = vektor bobot (weight)
- x = vektor fitur input
- b = bias
```

#### Hinge Loss Function:
```
L_hinge(y, f(x)) = max(0, 1 - y × f(x))

dimana:
- y = label sebenarnya (+1 atau -1)
- f(x) = nilai prediksi model
```

**Interpretasi:**
- Jika prediksi benar dan margin > 1: Loss = 0
- Jika prediksi salah atau margin kecil: Loss > 0

#### Total Loss dengan L2 Regularization:
```
L_total = (1/n) × Σᵢ max(0, 1 - yᵢ × (w · xᵢ + b)) + λ × ||w||²

dimana:
- n = jumlah sampel
- λ = parameter regularisasi (lambda_param = 0.01)
- ||w||² = w₁² + w₂² + ... + wₙ² (L2 norm squared)
```

#### Gradient Descent Update:

**Gradient untuk w:**
```
∂L/∂w = 2λw - (1/n) × Σᵢ (yᵢ × xᵢ)   untuk sampel dengan margin < 1

Jika margin ≥ 1 (klasifikasi benar dengan confidence):
∂L/∂w = 2λw
```

**Gradient untuk b:**
```
∂L/∂b = -(1/n) × Σᵢ yᵢ   untuk sampel dengan margin < 1
```

**Update Rule:**
```
w_new = w_old - learning_rate × ∂L/∂w
b_new = b_old - learning_rate × ∂L/∂b
```

**Implementasi dalam kode:**
```python
def _hinge_loss_and_grad(self, X, y):
    scores = X.dot(self.w) + self.b
    margins = 1 - y * scores
    loss_hinge = np.maximum(0, margins)
    loss = np.mean(loss_hinge) + self.lambda_param * np.dot(self.w, self.w)

    mask = (margins > 0).astype(float)  # sampel yang perlu dikoreksi
    grad_w = 2 * self.lambda_param * self.w - (1.0 / n) * ((mask * y) @ X)
    grad_b = -(1.0 / n) * np.sum(mask * y)

    return loss, grad_w, grad_b
```

---

### 3. Random Forest dengan Decision Tree CART

Random Forest adalah ensemble dari banyak Decision Tree yang dilatih pada subset data berbeda.

#### Gini Impurity (Kriteria Split):
```
Gini(S) = 1 - Σᵢ pᵢ²

dimana:
- S = himpunan sampel pada node
- pᵢ = proporsi kelas i dalam S
```

**Contoh:**
- Node dengan 60 FAKE dan 40 REAL (total 100)
- p_FAKE = 60/100 = 0.6
- p_REAL = 40/100 = 0.4
- Gini = 1 - (0.6² + 0.4²) = 1 - (0.36 + 0.16) = 0.48

**Interpretasi Gini:**
- Gini = 0: Node sempurna (semua sampel satu kelas)
- Gini = 0.5: Node sangat tidak murni (50-50 split)

#### Information Gain (Untuk Memilih Split Terbaik):
```
Gain = Gini(parent) - Σⱼ (nⱼ/n) × Gini(childⱼ)

dimana:
- nⱼ = jumlah sampel di child node j
- n = jumlah sampel di parent node
```

**Contoh:**
```
Parent Node: 100 sampel (Gini = 0.48)
Split berdasarkan fitur "stars_norm <= 0.8":
  - Left child: 70 sampel, 50 FAKE, 20 REAL → Gini = 1 - (0.71² + 0.29²) = 0.41
  - Right child: 30 sampel, 10 FAKE, 20 REAL → Gini = 1 - (0.33² + 0.67²) = 0.44

Weighted Gini = (70/100) × 0.41 + (30/100) × 0.44 = 0.287 + 0.132 = 0.419
Information Gain = 0.48 - 0.419 = 0.061
```

#### Algoritma CART (Classification and Regression Trees):

```
Fungsi BUILD_TREE(X, y, depth):
    1. Jika stopping condition (max_depth, min_samples, pure node):
       return LEAF_NODE dengan majority class

    2. Untuk setiap fitur f:
       - Untuk setiap threshold t:
         - Split data: left = X[f <= t], right = X[f > t]
         - Hitung Gini untuk kedua child
         - Hitung Information Gain

    3. Pilih split dengan Information Gain tertinggi

    4. Rekursif:
       left_subtree = BUILD_TREE(X_left, y_left, depth+1)
       right_subtree = BUILD_TREE(X_right, y_right, depth+1)

    5. Return DECISION_NODE(feature, threshold, left_subtree, right_subtree)
```

#### Bootstrap Sampling (Bagging):
```
Untuk setiap tree i dari 1 sampai n_estimators:
    1. Sample n data dengan replacement dari training set
       (beberapa data muncul >1x, beberapa tidak terpilih)
    2. Train Decision Tree pada bootstrap sample
    3. Simpan tree ke dalam forest
```

#### Majority Voting (Prediksi):
```
Untuk input x baru:
    votes = []
    for tree in forest:
        prediction = tree.predict(x)
        votes.append(prediction)

    final_prediction = mode(votes)  # kelas dengan vote terbanyak
```

**Implementasi dalam kode:**
```python
@staticmethod
def gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1.0 - np.sum(probs * probs)

def _best_split(self, X, y):
    parent_gini = self.gini(y)
    best_gain = 0.0
    # ... iterate features and thresholds
    gain = parent_gini - weighted_child_gini
    if gain > best_gain:
        best_gain = gain
        best_feat = feat
        best_thresh = thresh
```

---

### 4. Naive Bayes (Hybrid: Multinomial + Gaussian)

Naive Bayes menggunakan Teorema Bayes dengan asumsi independensi antar fitur.

#### Teorema Bayes:
```
P(C|X) = P(X|C) × P(C) / P(X)

dimana:
- P(C|X) = probabilitas kelas C diberikan fitur X (posterior)
- P(X|C) = probabilitas fitur X diberikan kelas C (likelihood)
- P(C) = probabilitas kelas C (prior)
- P(X) = probabilitas fitur X (evidence)
```

#### Asumsi Naive Bayes (Independensi Fitur):
```
P(X|C) = P(x₁|C) × P(x₂|C) × ... × P(xₙ|C) = Πᵢ P(xᵢ|C)
```

#### A. Multinomial Naive Bayes (untuk TF-IDF):

**Prior Probability:**
```
P(C) = jumlah sampel kelas C / total sampel

log P(C) = log(n_samples_class / n_samples)
```

**Likelihood dengan Laplace Smoothing:**
```
P(word|C) = (count(word, C) + α) / (total_words_C + α × vocabulary_size)

dimana:
- α = smoothing parameter (default = 1.0)
- count(word, C) = jumlah kemunculan word di kelas C
```

**Log Probability untuk Prediksi:**
```
log P(C|X) ∝ log P(C) + Σᵢ xᵢ × log P(wordᵢ|C)
```

**Implementasi dalam kode:**
```python
def fit(self, X, y):
    for class_label in self.classes:
        X_class = X[y == class_label]
        # Prior
        self.class_log_prior[class_label] = np.log(n_samples_class / n_samples)
        # Likelihood dengan Laplace smoothing
        word_counts = X_class.sum(axis=0)
        total_count = word_counts.sum()
        self.feature_log_prob[class_label] = np.log(
            (word_counts + self.alpha) / (total_count + self.alpha * n_features)
        )
```

#### B. Gaussian Naive Bayes (untuk Fitur Numerik):

**Distribusi Gaussian (Normal):**
```
P(xᵢ|C) = (1 / √(2πσ²_C)) × exp(-(xᵢ - μ_C)² / (2σ²_C))

dimana:
- μ_C = mean fitur i untuk kelas C
- σ²_C = variance fitur i untuk kelas C
```

**Log Likelihood:**
```
log P(xᵢ|C) = -0.5 × log(2πσ²_C) - (xᵢ - μ_C)² / (2σ²_C)
```

**Implementasi dalam kode:**
```python
def _calculate_likelihood(self, X, mean, var):
    exponent = -0.5 * ((X - mean) ** 2) / var
    normalizer = -0.5 * np.log(2 * np.pi * var)
    return (normalizer + exponent).sum(axis=1)
```

#### C. Hybrid Naive Bayes (Kombinasi):

```
log P(C|X_text, X_numeric) = w_text × log P(C|X_text) + w_numeric × log P(C|X_numeric)

dimana:
- w_text = 0.6 (bobot untuk fitur teks/TF-IDF)
- w_numeric = 0.4 (bobot untuk fitur numerik)
```

**Normalisasi Probabilitas (Softmax):**
```
P(C|X) = exp(log P(C|X) - max_log) / Σ exp(log P(Cᵢ|X) - max_log)
```

**Implementasi dalam kode:**
```python
def predict_proba(self, X_text, X_numeric):
    log_proba_text = self.multinomial_nb.predict_log_proba(X_text)
    log_proba_numeric = self.gaussian_nb.predict_log_proba(X_numeric)
    log_proba_combined = self.text_weight * log_proba_text + self.numeric_weight * log_proba_numeric
    # Softmax normalization
    proba = np.exp(log_proba_combined - np.max(log_proba_combined, axis=1, keepdims=True))
    proba = proba / proba.sum(axis=1, keepdims=True)
    return proba
```

---

### 5. Standard Scaler (Z-Score Normalization)

Normalisasi fitur numerik agar memiliki mean=0 dan standard deviation=1.

#### Rumus Z-Score:
```
z = (x - μ) / σ

dimana:
- x = nilai asli
- μ = mean dari fitur
- σ = standard deviation dari fitur
```

**Contoh:**
- Fitur: reviewer_count = [1, 5, 10, 100, 500]
- μ = 123.2
- σ = 196.8
- z(1) = (1 - 123.2) / 196.8 = -0.62
- z(100) = (100 - 123.2) / 196.8 = -0.12
- z(500) = (500 - 123.2) / 196.8 = 1.91

**Implementasi dalam kode:**
```python
class StandardScalerSimple:
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0  # prevent division by zero

    def transform(self, X):
        return (X - self.mean_) / self.scale_
```

---

### 6. PCA (Principal Component Analysis)

Digunakan untuk reduksi dimensi dan visualisasi data.

#### Langkah-langkah PCA:

**1. Centering Data:**
```
X_centered = X - μ

dimana μ = mean tiap fitur
```

**2. Covariance Matrix:**
```
C = (1/n) × X_centeredᵀ × X_centered
```

**3. Eigenvalue Decomposition:**
```
C × v = λ × v

dimana:
- λ = eigenvalue (variance explained)
- v = eigenvector (principal component direction)
```

**4. Proyeksi ke Principal Components:**
```
X_reduced = X_centered × V_k

dimana V_k = matrix dari k eigenvectors dengan eigenvalues terbesar
```

**Implementasi dalam kode:**
```python
class PCA:
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        Xc = X - self.mean
        cov = np.cov(Xc.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        idx = eigvals.argsort()[::-1]  # sort descending
        self.components = eigvecs[:, idx[:self.n_components]]

    def transform(self, X):
        Xc = X - self.mean
        return Xc.dot(self.components)
```

---

### 7. Logaritma untuk Fitur Count

Transformasi logaritmik untuk fitur dengan distribusi skewed.

#### Rumus Log1p:
```
x_transformed = log(1 + x)

dimana log adalah natural logarithm (ln)
```

**Kegunaan:**
- Mengurangi skewness pada distribusi
- Menangani outlier (nilai ekstrem)
- Menjaga nilai 0 tetap valid (log(1+0) = 0)

**Contoh:**
- reviewer_count = 1 → log(1+1) = 0.69
- reviewer_count = 10 → log(1+10) = 2.40
- reviewer_count = 100 → log(1+100) = 4.62
- reviewer_count = 1000 → log(1+1000) = 6.91

**Implementasi:**
```python
df['reviewer_count_log'] = np.log1p(df['reviewer_count'])
```

---

## Metrik Evaluasi

| Metrik | Rumus |
|--------|-------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) |

### Penjelasan Confusion Matrix:

```
                        Predicted
                    FAKE        REAL
Actual  FAKE    TP (True     FN (False
                 Positive)    Negative)

        REAL    FP (False    TN (True
                 Positive)    Negative)
```

**Keterangan:**
- **TP (True Positive):** Model prediksi FAKE, kenyataannya FAKE ✓
- **TN (True Negative):** Model prediksi REAL, kenyataannya REAL ✓
- **FP (False Positive):** Model prediksi FAKE, kenyataannya REAL ✗ (Type I Error)
- **FN (False Negative):** Model prediksi REAL, kenyataannya FAKE ✗ (Type II Error)

### Interpretasi Metrik:

| Metrik | Interpretasi | Penting Untuk |
|--------|--------------|---------------|
| **Accuracy** | Persentase prediksi benar secara keseluruhan | Evaluasi umum model |
| **Precision** | Dari semua yang diprediksi FAKE, berapa yang benar FAKE | Menghindari false alarm (FP) |
| **Recall** | Dari semua yang benar FAKE, berapa yang terdeteksi | Mendeteksi semua fake review (menghindari FN) |
| **F1-Score** | Harmonic mean dari Precision dan Recall | Keseimbangan Precision-Recall |

### Contoh Perhitungan:

```
Confusion Matrix:
              Predicted
            FAKE    REAL
Actual FAKE  85      15    (100 actual FAKE)
       REAL  10      90    (100 actual REAL)

Perhitungan:
- TP = 85, TN = 90, FP = 10, FN = 15
- Accuracy = (85 + 90) / (85 + 90 + 10 + 15) = 175/200 = 0.875 = 87.5%
- Precision = 85 / (85 + 10) = 85/95 = 0.895 = 89.5%
- Recall = 85 / (85 + 15) = 85/100 = 0.85 = 85%
- F1-Score = 2 × (0.895 × 0.85) / (0.895 + 0.85) = 1.52 / 1.745 = 0.871 = 87.1%
```

**Implementasi dalam kode:**
```python
def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))   # True Positive
    fp = np.sum((y_true == -1) & (y_pred == 1))  # False Positive
    tn = np.sum((y_true == -1) & (y_pred == -1)) # True Negative
    fn = np.sum((y_true == 1) & (y_pred == -1))  # False Negative

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}
```

---

## Metode Evaluasi pada Setiap Script Training

### 1. Evaluasi pada SVM (`svm_train.py`)

#### A. Train-Test Split
```python
TEST_SIZE = 0.2      # 20% data untuk testing
VAL_SIZE = 0.1       # 10% dari training untuk validation
RANDOM_SEED = 42     # Reproducibility

# Split data
split_idx = int(len(X) * (1 - TEST_SIZE))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

**Pembagian Data:**
```
Total Data: 100%
├── Training Set: 80%
│   ├── Training: 72% (untuk training model)
│   └── Validation: 8% (untuk early stopping)
└── Test Set: 20% (untuk evaluasi final)
```

#### B. Early Stopping dengan Validation Loss
```python
# Proses early stopping
best_val_loss = np.inf
no_improve_rounds = 0
early_stopping_rounds = 50

for iteration in range(n_iterations):
    # Hitung training loss
    train_loss = hinge_loss(X_train, y_train)

    # Hitung validation loss
    val_loss = hinge_loss(X_val, y_val)

    # Cek improvement
    if val_loss < best_val_loss - tolerance:
        best_val_loss = val_loss
        no_improve_rounds = 0
        # Simpan best weights
    else:
        no_improve_rounds += 1

    # Stop jika tidak ada improvement
    if no_improve_rounds >= early_stopping_rounds:
        break  # Early stopping triggered
```

**Kegunaan Early Stopping:**
- Mencegah overfitting
- Menghentikan training saat model mulai overfit ke training data
- Menyimpan bobot terbaik berdasarkan validation loss

#### C. Metrik yang Dihitung
| Metrik | Training Set | Test Set |
|--------|--------------|----------|
| Accuracy | ✓ | ✓ |
| Precision | ✓ | ✓ |
| Recall | ✓ | ✓ |
| F1-Score | ✓ | ✓ |
| Confusion Matrix | ✓ | ✓ |
| Training Loss Curve | ✓ | - |
| Validation Loss | ✓ | - |

#### D. Visualisasi yang Dihasilkan
1. **Confusion Matrix Heatmap** - Distribusi prediksi vs aktual
2. **Performance Metrics Bar Chart** - Perbandingan train vs test
3. **Training Loss Curve** - Loss per iterasi
4. **Hyperplane Visualization** - Proyeksi PCA 2D dengan decision boundary

---

### 2. Evaluasi pada Random Forest (`randomForest_train.py`)

#### A. Train-Test Split
```python
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Shuffle dan split
rng = np.random.RandomState(RANDOM_SEED)
perm = rng.permutation(len(X))
X, y = X[perm], y[perm]

split_idx = int(len(X) * (1 - TEST_SIZE))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

#### B. Out-of-Bag (OOB) Score

OOB Score adalah metode evaluasi unik untuk Random Forest yang memanfaatkan sampel yang tidak terpilih dalam bootstrap sampling.

```python
# Konsep OOB Score
for tree_i in forest:
    # Bootstrap sample (dengan replacement)
    bootstrap_indices = random.choice(n_samples, size=n_samples, replace=True)

    # OOB samples = sampel yang TIDAK terpilih
    oob_indices = samples NOT in bootstrap_indices  # ~37% dari data

    # Prediksi OOB samples menggunakan tree ini
    oob_predictions[oob_indices] = tree_i.predict(X[oob_indices])

# Hitung OOB Score (rata-rata akurasi pada OOB samples)
oob_score = accuracy(y_true[oob_mask], oob_predictions[oob_mask])
```

**Rumus OOB Score:**
```
Probabilitas sampel TIDAK terpilih dalam 1 bootstrap:
P(not selected) = (1 - 1/n)^n ≈ 1/e ≈ 0.368 ≈ 37%

OOB Score = Σ (prediksi benar pada OOB) / Σ (total OOB predictions)
```

**Keunggulan OOB:**
- Tidak memerlukan validation set terpisah
- Estimasi unbiased dari generalization error
- Efisien secara komputasi

**Implementasi dalam kode:**
```python
def fit(self, X, y):
    oob_votes = [defaultdict(int) for _ in range(n_samples)]

    for i in range(self.n_estimators):
        # Bootstrap sampling
        indices = self.random_state.randint(0, n_samples, size=n_samples)

        # Train tree
        tree.fit(X[indices], y[indices])

        # OOB predictions
        mask_in = np.zeros(n_samples, dtype=bool)
        mask_in[indices] = True
        oob_idx = np.where(~mask_in)[0]  # indices NOT in bootstrap

        if oob_idx.size > 0:
            preds_oob = tree.predict(X[oob_idx])
            for idx, pred in zip(oob_idx, preds_oob):
                oob_votes[idx][pred] += 1

    # Calculate OOB score
    oob_pred = [max(votes, key=votes.get) for votes in oob_votes if votes]
    self.oob_score_ = np.mean(y[mask] == oob_pred[mask])
```

#### C. Feature Importances

Random Forest menghitung pentingnya setiap fitur berdasarkan seberapa sering fitur tersebut digunakan untuk split.

```python
# Perhitungan Feature Importance (berdasarkan penggunaan)
def _compute_feature_importances(self):
    importances = np.zeros(n_features)

    def traverse(node):
        if node.is_leaf():
            return
        # Tambah count untuk fitur yang digunakan split
        importances[node.feature_idx] += 1
        traverse(node.left)
        traverse(node.right)

    traverse(self.root)

    # Normalisasi
    return importances / importances.sum()

# Rata-rata importance dari semua trees
final_importances = np.mean([tree.feature_importances_ for tree in forest])
```

#### D. Metrik yang Dihitung
| Metrik | Training Set | Test Set | OOB |
|--------|--------------|----------|-----|
| Accuracy | ✓ | ✓ | ✓ |
| Precision | ✓ | ✓ | - |
| Recall | ✓ | ✓ | - |
| F1-Score | ✓ | ✓ | - |
| Confusion Matrix | ✓ | ✓ | - |
| Feature Importances | ✓ | - | - |
| OOB Score | - | - | ✓ |

#### E. Visualisasi yang Dihasilkan
1. **Metrics Visualization** - Bar chart train vs test
2. **Feature Importances Plot** - Ranking fitur berdasarkan importance

---

### 3. Evaluasi pada Naive Bayes (`nb_train_akhir.py`)

#### A. Train-Test Split
```python
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Shuffle dan split (sama dengan RF)
split_idx = int(len(X) * (1 - TEST_SIZE))
X_train_text, X_test_text = X_text[:split_idx], X_text[split_idx:]
X_train_num, X_test_num = X_num[:split_idx], X_num[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

#### B. Probability Prediction

Naive Bayes memberikan probabilitas untuk setiap kelas, bukan hanya label.

```python
# Prediksi probabilitas
def predict_proba(self, X_text, X_numeric):
    # Log probability dari Multinomial NB (teks)
    log_proba_text = self.multinomial_nb.predict_log_proba(X_text)

    # Log probability dari Gaussian NB (numerik)
    log_proba_numeric = self.gaussian_nb.predict_log_proba(X_numeric)

    # Weighted combination
    log_proba_combined = (self.text_weight * log_proba_text +
                          self.numeric_weight * log_proba_numeric)

    # Softmax untuk convert ke probability
    proba = softmax(log_proba_combined)

    return proba  # [[P(FAKE), P(REAL)], ...]

# Contoh output
# Sample 1: [0.85, 0.15] → 85% FAKE, 15% REAL → Prediksi: FAKE
# Sample 2: [0.20, 0.80] → 20% FAKE, 80% REAL → Prediksi: REAL
```

#### C. Evaluasi Komponen Terpisah

Hybrid NB memungkinkan evaluasi kontribusi masing-masing komponen:

```python
# Evaluasi Multinomial NB saja (fitur teks)
y_pred_text_only = multinomial_nb.predict(X_text)
text_accuracy = accuracy(y_true, y_pred_text_only)

# Evaluasi Gaussian NB saja (fitur numerik)
y_pred_numeric_only = gaussian_nb.predict(X_numeric)
numeric_accuracy = accuracy(y_true, y_pred_numeric_only)

# Evaluasi Hybrid (kombinasi)
y_pred_hybrid = hybrid_nb.predict(X_text, X_numeric)
hybrid_accuracy = accuracy(y_true, y_pred_hybrid)
```

**Contoh Hasil:**
```
Komponen           | Accuracy
-------------------|----------
Multinomial NB     | 78.5%
Gaussian NB        | 72.3%
Hybrid (60:40)     | 82.1%  ← Kombinasi lebih baik
```

#### D. Metrik yang Dihitung
| Metrik | Training Set | Test Set |
|--------|--------------|----------|
| Accuracy | ✓ | ✓ |
| Precision | ✓ | ✓ |
| Recall | ✓ | ✓ |
| F1-Score | ✓ | ✓ |
| Confusion Matrix | ✓ | ✓ |
| Class Probabilities | ✓ | ✓ |

#### E. Visualisasi yang Dihasilkan
1. **Metrics Visualization** - Bar chart dengan warna berbeda per metrik

---

### Perbandingan Metode Evaluasi Antar Model

| Aspek | SVM | Random Forest | Naive Bayes |
|-------|-----|---------------|-------------|
| **Train-Test Split** | 80-20 | 80-20 | 80-20 |
| **Validation Set** | ✓ (10%) | ✗ (pakai OOB) | ✗ |
| **Early Stopping** | ✓ | ✗ | ✗ |
| **OOB Score** | ✗ | ✓ | ✗ |
| **Feature Importance** | Dari weight | ✓ (built-in) | ✗ |
| **Probability Output** | ✗ (score saja) | ✓ | ✓ |
| **Loss Monitoring** | ✓ | ✗ | ✗ |

### Kode Evaluasi Lengkap

```python
# === EVALUASI UMUM (digunakan di semua model) ===

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluasi lengkap untuk semua model"""

    # Prediksi
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Hitung metrik training
    train_metrics = calculate_metrics(y_train, y_train_pred)
    print("=== TRAIN METRICS ===")
    print(f"Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Precision: {train_metrics['precision']:.4f}")
    print(f"Recall:    {train_metrics['recall']:.4f}")
    print(f"F1-Score:  {train_metrics['f1_score']:.4f}")
    print_confusion_matrix(train_metrics['confusion_matrix'])

    # Hitung metrik test
    test_metrics = calculate_metrics(y_test, y_test_pred)
    print("\n=== TEST METRICS ===")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-Score:  {test_metrics['f1_score']:.4f}")
    print_confusion_matrix(test_metrics['confusion_matrix'])

    # Cek overfitting
    accuracy_gap = train_metrics['accuracy'] - test_metrics['accuracy']
    if accuracy_gap > 0.1:
        print(f"\n⚠️ WARNING: Possible overfitting detected!")
        print(f"   Train-Test accuracy gap: {accuracy_gap:.4f}")

    return train_metrics, test_metrics
```

---

## Output yang Dihasilkan

### Model Terlatih (.pkl)
- `svm_multifeature_model_unified.pkl`
- `random_forest_model.pkl`
- `naive_bayes_multifeature_model.pkl`

### Metrik Evaluasi (.csv)
- `metrics_summary_unified.csv` (SVM)
- `metrics_summary.csv` (Random Forest)
- `metrics_summary.csv` (Naive Bayes)

### Visualisasi (.png)
- Confusion Matrix Heatmap
- Performance Metrics Bar Chart
- Training Loss Curve
- Feature Importance Plot

### Hasil Prediksi (.csv)
- `hasil_prediksi_svm.csv`
- `hasil_prediksi_rf.csv`
- `hasil_prediksi_nb.csv`

---

## Library yang Digunakan

| Library | Kegunaan |
|---------|----------|
| **pandas** | Manipulasi data dan I/O CSV |
| **numpy** | Komputasi numerik dan array |
| **matplotlib** | Visualisasi (plot, grafik) |
| **Sastrawi** | Preprocessing teks bahasa Indonesia (stemming, stopword) |
| **scikit-learn** | Digunakan di labeling_eda.py untuk training opsional |

**Catatan:** Algoritma ML di file training (SVM, Random Forest, Naive Bayes, TF-IDF) diimplementasikan secara manual untuk keperluan pembelajaran.

---

## Referensi

1. Sastrawi - Indonesian Stemmer - https://github.com/har07/PySastrawi
2. TF-IDF - https://en.wikipedia.org/wiki/Tf%E2%80%93idf
3. SVM - Cortes & Vapnik (1995)
4. Random Forest - Breiman (2001)
5. Naive Bayes - https://en.wikipedia.org/wiki/Naive_Bayes_classifier

---

Proyek ini dibuat untuk keperluan akademik/pembelajaran.
