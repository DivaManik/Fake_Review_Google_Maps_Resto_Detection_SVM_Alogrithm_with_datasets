Anggota kelompok :
Diva Filemon Manik 71220863
Anjelita Haninuna 71220925

Topik: Deteksi Fake Review Ulasan Tempat Makan di Yogyakarta dengan Algoritma Random Forest, SVM dan Naive Bayes

Sumber Dataset: Hasil scraping mandiri menggunakan Apify

---

# Fake Review Google Maps Restaurant Detection

## Deskripsi Proyek

Proyek ini bertujuan untuk mendeteksi ulasan palsu (fake review) pada restoran di Google Maps wilayah Yogyakarta. Sistem menggunakan 3 algoritma machine learning:
- **Support Vector Machine (SVM)**
- **Random Forest**
- **Naive Bayes (Hybrid)**

---

## Struktur Folder Proyek

```
├── dataset/                          # Dataset mentah
├── dataset_balance/                  # Dataset yang sudah di-balance
├── label_output/                     # Output hasil labeling dan preprocessing
├── model_output/                     # Output model SVM
├── rf_model_output/                  # Output model Random Forest
├── nb_model_output/                  # Output model Naive Bayes
├── file_training/                    # File script training
│   ├── svm_train.py                  # Training SVM
│   ├── randomForest_train.py         # Training Random Forest
│   └── nb_train_akhir.py             # Training Naive Bayes
├── file_testing/                     # File script testing/prediksi
├── preprocessing.py                  # Script preprocessing teks
├── text_quality_detector.py          # Deteksi kualitas teks (gibberish)
├── balance_dataset.py                # Script balancing dataset
└── README.md
```

---

## Alur Kerja Sistem (Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ALUR KERJA SISTEM                                  │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌──────────────┐
   │  1. DATA     │  Scraping data review dari Google Maps
   │  COLLECTION  │  menggunakan Apify
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  2. LABELING │  Memberikan label FAKE atau REAL
   │              │  berdasarkan kriteria tertentu
   └──────┬───────┘
          │
          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  3. PREPROCESSING (Sastrawi - Bahasa Indonesia)              │
   │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
   │  │Case Folding│→ │ Cleansing  │→ │Normalisasi │             │
   │  │(lowercase) │  │(hapus URL, │  │ (singkatan │             │
   │  │            │  │ angka,emoji│  │  ke baku)  │             │
   │  └────────────┘  └────────────┘  └────────────┘             │
   │         │                                                    │
   │         ▼                                                    │
   │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
   │  │ Tokenizing │→ │  Stopword  │→ │  Stemming  │             │
   │  │(pecah kata)│  │  Removal   │  │(akar kata) │             │
   │  └────────────┘  └────────────┘  └────────────┘             │
   └──────────────────────────────────────────────────────────────┘
          │
          ▼
   ┌──────────────┐
   │  4. BALANCING│  Menyeimbangkan jumlah data FAKE dan REAL
   │  DATASET     │  (Undersampling/Oversampling/Kombinasi)
   └──────┬───────┘
          │
          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  5. FEATURE ENGINEERING                                      │
   │  ┌─────────────────┐  ┌─────────────────┐                   │
   │  │   TF-IDF        │  │ Numeric Features│                   │
   │  │ (Text Features) │  │ (17+ fitur)     │                   │
   │  └─────────────────┘  └─────────────────┘                   │
   │           │                    │                             │
   │           └────────┬───────────┘                             │
   │                    ▼                                         │
   │           ┌─────────────────┐                               │
   │           │ Combined Features│                               │
   │           └─────────────────┘                               │
   └──────────────────────────────────────────────────────────────┘
          │
          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  6. TRAINING MODEL                                           │
   │  ┌─────────┐    ┌─────────────┐    ┌─────────────┐          │
   │  │   SVM   │    │Random Forest│    │ Naive Bayes │          │
   │  └─────────┘    └─────────────┘    └─────────────┘          │
   └──────────────────────────────────────────────────────────────┘
          │
          ▼
   ┌──────────────┐
   │  7. EVALUASI │  Accuracy, Precision, Recall, F1-Score
   │              │  Confusion Matrix
   └──────────────┘
```

---

## Tahap Preprocessing

Preprocessing adalah tahap membersihkan dan menyiapkan teks agar bisa diproses oleh model. Proyek ini menggunakan **Sastrawi** library untuk bahasa Indonesia.

### Langkah-langkah Preprocessing:

| No | Tahap | Fungsi | Contoh |
|----|-------|--------|--------|
| 1 | **Case Folding** | Mengubah semua huruf menjadi lowercase | "ENAK Banget!" → "enak banget!" |
| 2 | **Cleansing** | Menghapus URL, mention, hashtag, emoji, angka | "cek di http://xxx @resto #makanan 123" → "cek di resto makanan" |
| 3 | **Normalisasi** | Mengubah kata tidak baku menjadi baku | "gak", "ga", "ngga" → "tidak" |
| 4 | **Tokenizing** | Memecah teks menjadi kata-kata | "makanan enak" → ["makanan", "enak"] |
| 5 | **Stopword Removal** | Menghapus kata-kata umum yang tidak bermakna | "yang", "dan", "di", "ke" dihapus |
| 6 | **Stemming** | Mengubah kata ke bentuk dasar | "makanan" → "makan", "dimakan" → "makan" |

### Contoh Kamus Normalisasi:
```python
'gak' → 'tidak'      'bgt' → 'banget'     'yg' → 'yang'
'udh' → 'sudah'      'jg' → 'juga'        'dgn' → 'dengan'
'mantul' → 'mantap'  'thx' → 'terima kasih'
```

---

## Feature Engineering

### A. TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF adalah metode untuk mengubah teks menjadi angka dengan memberikan bobot pada setiap kata.

**Rumus TF-IDF:**
```
TF-IDF = TF × IDF

dimana:
TF (Term Frequency) = Jumlah kata dalam dokumen / Total kata dalam dokumen
IDF (Inverse Document Frequency) = log(Total dokumen / Jumlah dokumen yang mengandung kata)
```

**Contoh Perhitungan:**
```
Dokumen: "makanan enak makanan lezat"
- TF("makanan") = 2/4 = 0.5
- Misalkan ada 100 dokumen, 20 mengandung kata "makanan"
- IDF("makanan") = log(100/20) = log(5) = 0.699
- TF-IDF("makanan") = 0.5 × 0.699 = 0.3495
```

**Konfigurasi TF-IDF pada proyek ini:**
- `max_features`: 1000-2000 (jumlah kata unik yang digunakan)
- `ngram_range`: (1, 2) artinya menggunakan unigram dan bigram
- `min_df`: 2-3 (kata harus muncul minimal di 2-3 dokumen)

### B. Numeric Features (17+ Fitur)

| No | Fitur | Penjelasan |
|----|-------|------------|
| 1 | `text_word_count` | Jumlah kata dalam review |
| 2 | `text_char_count` | Jumlah karakter dalam review |
| 3 | `avg_word_length` | Rata-rata panjang kata |
| 4 | `exclamation_count` | Jumlah tanda seru (!) |
| 5 | `question_count` | Jumlah tanda tanya (?) |
| 6 | `punctuation_count` | Total tanda baca |
| 7 | `uppercase_word_ratio` | Rasio kata huruf kapital |
| 8 | `stars_norm` | Rating bintang (dinormalisasi 0-1) |
| 9 | `has_image` | Ada foto atau tidak (0/1) |
| 10 | `reviewer_count_log` | Log jumlah review dari user |
| 11 | `is_local_guide` | Apakah Local Guide (0/1) |
| 12 | `detail_rating_all_5` | Semua detail rating = 5 (0/1) |
| 13 | `same_day_count` | Jumlah review di hari yang sama |
| 14 | `pattern_5star_short` | Bintang 5 tapi review pendek |
| 15 | `pattern_5star_nophoto` | Bintang 5 tanpa foto |
| 16 | `pattern_new_reviewer` | Reviewer baru (≤2 review) |
| 17 | `pattern_same_day` | Banyak review di hari yang sama |

### C. Text Quality Features (Deteksi Gibberish)

Fitur tambahan untuk mendeteksi teks yang tidak bermakna:

| Fitur | Penjelasan |
|-------|------------|
| `tq_vowel_ratio` | Rasio vokal vs konsonan |
| `tq_valid_word_ratio` | Persentase kata valid dalam kamus |
| `tq_entropy` | Tingkat keacakan teks |
| `tq_gibberish_score` | Skor gabungan deteksi gibberish |

---

## Algoritma dan Cara Perhitungan

### 1. Support Vector Machine (SVM)

**Konsep Dasar:**
SVM mencari garis pemisah terbaik (hyperplane) antara kelas FAKE dan REAL dengan margin maksimal.

**Rumus Perhitungan:**

```
Fungsi Keputusan:
f(x) = w · x + b

dimana:
- w = vektor bobot
- x = vektor fitur input
- b = bias

Prediksi:
- Jika f(x) ≥ 0 → FAKE
- Jika f(x) < 0 → REAL
```

**Hinge Loss + L2 Regularization:**
```
Loss = (1/n) × Σ max(0, 1 - y × f(x)) + λ × ||w||²

dimana:
- n = jumlah sampel
- y = label sebenarnya (+1 atau -1)
- λ = parameter regularisasi
```

**Gradient Descent Update:**
```
w_baru = w_lama - learning_rate × ∂Loss/∂w
b_baru = b_lama - learning_rate × ∂Loss/∂b
```

**Konfigurasi SVM:**
```python
learning_rate = 0.01      # Seberapa besar langkah update
lambda_param = 0.01       # Regularisasi untuk mencegah overfitting
n_iterations = 2000       # Jumlah iterasi training
early_stopping = 50       # Berhenti jika tidak ada peningkatan
```

---

### 2. Random Forest

**Konsep Dasar:**
Random Forest adalah kumpulan (ensemble) dari banyak Decision Tree. Setiap tree memberikan vote, hasil akhir adalah mayoritas vote.

**Proses Training:**

```
Untuk setiap tree (i = 1 sampai n_estimators):
  1. Ambil sample data dengan Bootstrap (sampling dengan pengembalian)
  2. Pilih subset fitur secara acak (sqrt(total_fitur))
  3. Bangun Decision Tree dengan CART algorithm
  4. Simpan tree
```

**Decision Tree - CART Algorithm:**

Menggunakan **Gini Impurity** untuk menentukan split terbaik:

```
Gini = 1 - Σ(pi)²

dimana:
- pi = proporsi kelas i dalam node

Contoh:
Node dengan 60 FAKE, 40 REAL (total 100)
Gini = 1 - (0.6² + 0.4²) = 1 - (0.36 + 0.16) = 0.48
```

**Information Gain:**
```
Gain = Gini(parent) - (weighted average Gini of children)

Pilih split dengan Gain tertinggi
```

**Prediksi:**
```
Untuk prediksi data baru:
  1. Data melewati semua tree
  2. Setiap tree memberikan prediksi (FAKE atau REAL)
  3. Hasil akhir = label dengan vote terbanyak (majority voting)
```

**Konfigurasi Random Forest:**
```python
n_estimators = 30         # Jumlah pohon dalam forest
max_depth = 12            # Kedalaman maksimal pohon
min_samples_split = 2     # Minimum sampel untuk split
min_samples_leaf = 2      # Minimum sampel di leaf node
max_features = "sqrt"     # Jumlah fitur per split = √total_fitur
```

---

### 3. Naive Bayes (Hybrid)

**Konsep Dasar:**
Naive Bayes menggunakan Teorema Bayes dengan asumsi setiap fitur independen (tidak saling mempengaruhi).

Proyek ini menggunakan **Hybrid Naive Bayes** yang menggabungkan:
- **Multinomial NB**: untuk fitur TF-IDF (teks)
- **Gaussian NB**: untuk fitur numerik

**Teorema Bayes:**
```
P(Class|Features) = P(Features|Class) × P(Class) / P(Features)

Kita bandingkan:
P(FAKE|X) vs P(REAL|X)

Kelas dengan probabilitas lebih tinggi = prediksi
```

**Multinomial Naive Bayes (untuk TF-IDF):**
```
P(kata_i | Class) = (count(kata_i, Class) + α) / (total_kata(Class) + α × n_fitur)

dimana:
- α = smoothing parameter (biasanya 1.0, disebut Laplace smoothing)
- Mencegah probabilitas = 0 untuk kata yang tidak pernah muncul

Log Probability (untuk menghindari underflow):
log P(Class|X) = log P(Class) + Σ log P(kata_i|Class)
```

**Gaussian Naive Bayes (untuk fitur numerik):**
```
P(fitur_i | Class) = (1 / √(2π × σ²)) × exp(-(x - μ)² / (2 × σ²))

dimana:
- μ = mean fitur untuk kelas tersebut
- σ² = variance fitur untuk kelas tersebut
```

**Hybrid Combination:**
```
Final_Score = text_weight × log P(text_features|Class)
            + numeric_weight × log P(numeric_features|Class)

dimana:
- text_weight = 0.6 (bobot untuk fitur teks)
- numeric_weight = 0.4 (bobot untuk fitur numerik)
```

**Konfigurasi Naive Bayes:**
```python
alpha = 1.0               # Laplace smoothing
text_weight = 0.6         # Bobot fitur TF-IDF
numeric_weight = 0.4      # Bobot fitur numerik
max_features_tfidf = 1000 # Jumlah fitur TF-IDF
```

---

## Evaluasi Model

### Metrik Evaluasi:

**Confusion Matrix:**
```
                    Predicted
                 FAKE    REAL
Actual  FAKE     TP      FN
        REAL     FP      TN

TP = True Positive  (prediksi FAKE, sebenarnya FAKE)
TN = True Negative  (prediksi REAL, sebenarnya REAL)
FP = False Positive (prediksi FAKE, sebenarnya REAL)
FN = False Negative (prediksi REAL, sebenarnya FAKE)
```

**Rumus Metrik:**
```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)
```

**Penjelasan Sederhana:**
- **Accuracy**: Seberapa sering model benar (dari semua prediksi)
- **Precision**: Dari yang diprediksi FAKE, berapa yang benar FAKE
- **Recall**: Dari semua FAKE sebenarnya, berapa yang terdeteksi
- **F1-Score**: Keseimbangan antara Precision dan Recall

---

## Cara Hyperparameter Tuning

### 1. Tuning SVM

| Parameter | Rentang | Pengaruh |
|-----------|---------|----------|
| `learning_rate` | 0.001 - 0.1 | Terlalu besar: tidak konvergen. Terlalu kecil: lambat |
| `lambda_param` | 0.001 - 0.1 | Semakin besar: model lebih sederhana (mencegah overfitting) |
| `n_iterations` | 500 - 5000 | Lebih banyak: lebih akurat tapi lebih lama |

### 2. Tuning Random Forest

| Parameter | Rentang | Pengaruh |
|-----------|---------|----------|
| `n_estimators` | 10 - 100 | Lebih banyak tree = lebih akurat tapi lebih lambat |
| `max_depth` | 5 - 20 | Lebih dalam = lebih kompleks (risiko overfitting) |
| `min_samples_split` | 2 - 10 | Lebih besar = tree lebih sederhana |
| `min_samples_leaf` | 1 - 5 | Lebih besar = mencegah overfitting |
| `max_features` | "sqrt", "log2", int | Variasi fitur per split |

### 3. Tuning Naive Bayes

| Parameter | Rentang | Pengaruh |
|-----------|---------|----------|
| `alpha` | 0.1 - 2.0 | Smoothing untuk kata langka |
| `text_weight` | 0.3 - 0.8 | Bobot fitur teks vs numerik |
| `numeric_weight` | 0.2 - 0.7 | Bobot fitur numerik vs teks |
| `max_features_tfidf` | 500 - 2000 | Jumlah kata dalam vocabulary |

---

## Cara Menjalankan Training

### Langkah 1: Persiapan Environment

```bash
# Install dependencies
pip install pandas numpy matplotlib Sastrawi scikit-learn
```

### Langkah 2: Preprocessing Data

```bash
python preprocessing.py
```

### Langkah 3: Balancing Dataset

```bash
python balance_dataset.py
```

### Langkah 4: Training Model

```bash
# Training SVM
cd file_training
python svm_train.py

# Training Random Forest
python randomForest_train.py

# Training Naive Bayes
python nb_train_akhir.py
```

### Langkah 5: Prediksi Data Baru

```bash
# Prediksi dengan SVM
cd file_testing
python svm_multifeature_predict.py

# Prediksi dengan Random Forest
python randomForest_predict.py

# Prediksi dengan Naive Bayes
python nb_predict_akhir.py
```

---

## Output Model

Setelah training, model akan disimpan dalam format pickle (.pkl):

```
model_output/
├── svm_multifeature_model.pkl     # Model SVM
├── svm_multifeature_metrics.csv   # Metrik evaluasi SVM
├── confusion_matrix_test.png      # Confusion matrix
└── performance_metrics.png        # Grafik performa

rf_model_output/
├── random_forest_model.pkl        # Model Random Forest
├── metrics_summary.csv            # Metrik evaluasi RF
└── feature_importances.png        # Kepentingan fitur

nb_model_output/
├── naive_bayes_multifeature_model.pkl  # Model Naive Bayes
├── metrics_summary.csv            # Metrik evaluasi NB
└── metrics_plot.png               # Grafik performa
```

---

## Referensi

1. Scikit-learn Documentation - https://scikit-learn.org/
2. Sastrawi - Indonesian Stemmer - https://github.com/har07/PySastrawi
3. TF-IDF - https://en.wikipedia.org/wiki/Tf%E2%80%93idf
4. SVM - Cortes & Vapnik (1995)
5. Random Forest - Breiman (2001)
6. Naive Bayes - https://en.wikipedia.org/wiki/Naive_Bayes_classifier

---

## Lisensi

Proyek ini dibuat untuk keperluan akademik/pembelajaran.
