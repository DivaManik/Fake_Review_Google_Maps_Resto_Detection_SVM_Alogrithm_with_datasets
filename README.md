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
├── preprocessing.py                    # Pipeline preprocessing teks Indonesia
├── balance_dataset.py                  # Script balancing dataset
├── check_balance.py                    # Verifikasi distribusi dataset
├── debug_gibberish.py                  # Testing deteksi kualitas teks
│
├── dataset_balance/                    # Dataset yang sudah di-balance
│   ├── google_review_balanced_combined.csv         # Metode hybrid (20 MB)
│   ├── google_review_balanced_oversampling.csv     # Metode oversampling (25 MB)
│   └── google_review_balanced_undersampling.csv    # Metode undersampling (14 MB)
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
│                              ALUR KERJA SISTEM                               │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────┐
   │ 1. PENGUMPULAN   │  Scraping data review dari Google Maps
   │    DATA          │  wilayah Yogyakarta menggunakan Apify
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ 2. LABELING      │  Memberikan label FAKE atau REAL
   │                  │  pada setiap review
   └────────┬─────────┘
            │
            ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 3. PREPROCESSING (preprocessing.py)                             │
   │                                                                  │
   │  Case Folding → Cleansing → Normalisasi → Tokenizing →          │
   │  Stopword Removal → Stemming                                     │
   │                                                                  │
   │  Output: kolom text_preprocessed                                 │
   └────────┬────────────────────────────────────────────────────────┘
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
   │                                                                  │
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
   │                                                                  │
   │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
   │  │     SVM       │  │ Random Forest │  │  Naive Bayes  │       │
   │  │  (Linear,     │  │ (30 trees,    │  │   (Hybrid:    │       │
   │  │   manual)     │  │  CART, Gini)  │  │  MNB + GNB)   │       │
   │  └───────────────┘  └───────────────┘  └───────────────┘       │
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
| `TfidfVectorizerManual.fit(documents)` | Mempelajari vocabulary dan IDF |
| `TfidfVectorizerManual.transform(documents)` | Mengubah dokumen menjadi vektor TF-IDF |
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
| `DecisionTreeCART._compute_feature_importances()` | Menghitung feature importance |
| `RandomForestManual` | Implementasi manual Random Forest (ensemble of trees) |
| `RandomForestManual.fit(X, y)` | Training multiple trees dengan bootstrap sampling |
| `RandomForestManual.predict(X)` | Majority voting dari semua trees |
| `RandomForestManual.predict_proba(X)` | Prediksi probabilitas |
| `TreeNode` | Class untuk node dalam decision tree |

### nb_train_akhir.py

| Class/Fungsi | Deskripsi |
|--------------|-----------|
| `MultinomialNB` | Naive Bayes untuk fitur TF-IDF (diskrit) |
| `MultinomialNB.fit(X, y)` | Training dengan menghitung probabilitas kata per kelas |
| `MultinomialNB.predict_log_proba(X)` | Menghitung log probabilitas |
| `GaussianNB` | Naive Bayes untuk fitur numerik (kontinu) |
| `GaussianNB.fit(X, y)` | Training dengan menghitung mean dan variance per kelas |
| `GaussianNB.predict_log_proba(X)` | Menghitung log probabilitas dengan distribusi Gaussian |
| `HybridNaiveBayes` | Kombinasi MultinomialNB (teks) dan GaussianNB (numerik) |
| `HybridNaiveBayes.fit(X_text, X_numeric, y)` | Training kedua model |
| `HybridNaiveBayes.predict(X_text, X_numeric)` | Prediksi dengan weighted combination |

### text_quality_detector.py

| Fungsi | Deskripsi |
|--------|-----------|
| `vowel_consonant_ratio(text)` | Menghitung rasio vokal terhadap konsonan |
| `is_abnormal_vowel_ratio(text)` | Mengecek apakah rasio vokal abnormal |
| `valid_word_ratio(text)` | Menghitung persentase kata valid dalam kamus |
| `is_low_valid_words(text)` | Mengecek apakah teks memiliki sedikit kata valid |
| `average_word_length(text)` | Menghitung rata-rata panjang kata |
| `is_long_words(text)` | Mengecek apakah kata-kata terlalu panjang |
| `text_entropy(text)` | Menghitung Shannon entropy dari teks |
| `is_high_entropy(text)` | Mengecek apakah entropy terlalu tinggi |
| `repeated_char_ratio(text)` | Menghitung rasio karakter berulang |
| `has_repeated_chars(text)` | Mengecek apakah ada karakter berulang berlebihan |
| `consonant_cluster_ratio(text)` | Mendeteksi cluster konsonan |
| `has_long_consonant_cluster(text)` | Mengecek cluster konsonan panjang |
| `detect_gibberish(text)` | Mendeteksi teks gibberish/spam secara keseluruhan |
| `extract_text_quality_features(df, text_column)` | Mengekstrak semua fitur kualitas teks ke dataframe |
| `get_text_quality_feature_names()` | Mengembalikan list nama fitur kualitas teks |

---

## Fitur-Fitur yang Digunakan

### A. TF-IDF Features (1000 fitur)

Mengubah teks menjadi vektor numerik berdasarkan frekuensi dan kepentingan kata.

**Konfigurasi:**
- `max_features`: 1000
- `ngram_range`: (1, 1) - unigram
- `min_df`: 2

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

## Algoritma yang Digunakan

### 1. Support Vector Machine (SVM)

- **Tipe:** Linear SVM dengan implementasi manual
- **Loss Function:** Hinge Loss + L2 Regularization
- **Optimisasi:** Gradient Descent dengan Early Stopping
- **Label:** FAKE (+1), REAL (-1)
- **Keputusan:** score >= 0 → FAKE, score < 0 → REAL

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
- **Feature Selection:** √(total_fitur) per split

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
- **Smoothing:** Laplace smoothing (alpha=1.0)

**Hyperparameter:**
```
alpha = 1.0
text_weight = 0.6
numeric_weight = 0.4
max_features_tfidf = 1000
```

---

## Metrik Evaluasi

| Metrik | Rumus |
|--------|-------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) |

**Confusion Matrix:**
```
                 Predicted
              FAKE    REAL
Actual FAKE    TP      FN
       REAL    FP      TN
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
- Hyperplane Visualization (PCA 2D)

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

**Catatan:** Semua algoritma machine learning (SVM, Random Forest, Naive Bayes, TF-IDF, PCA, Standard Scaler) diimplementasikan secara manual tanpa menggunakan library sklearn untuk keperluan pembelajaran.

---

## Referensi

1. Sastrawi - Indonesian Stemmer - https://github.com/har07/PySastrawi
2. TF-IDF - https://en.wikipedia.org/wiki/Tf%E2%80%93idf
3. SVM - Cortes & Vapnik (1995)
4. Random Forest - Breiman (2001)
5. Naive Bayes - https://en.wikipedia.org/wiki/Naive_Bayes_classifier

---

Proyek ini dibuat untuk keperluan akademik/pembelajaran.
