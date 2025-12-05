"""
Script untuk balancing dataset antara label FAKE dan REAL
- Membaca file google_review_preprocessed.csv
- Memberikan opsi balancing: undersampling, oversampling, atau kombinasi
- Menyimpan hasil balanced dataset untuk training
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample
import os

# ---------------------------
# === KONFIGURASI
# ---------------------------
INPUT_FILE = "label_output/google_review_preprocessed.csv"
OUTPUT_DIR = "label_output"

print("="*80)
print("BALANCING DATASET - FAKE vs REAL REVIEWS")
print("="*80)

# ---------------------------
# === LOAD DATA
# ---------------------------
print(f"\n[1/4] Membaca dataset dari: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Total data: {len(df)} baris")

# Cek kolom label
if 'label' not in df.columns:
    print("\n❌ ERROR: Kolom 'label' tidak ditemukan dalam dataset!")
    print("Pastikan file input sudah memiliki kolom 'label' dengan nilai FAKE/REAL")
    exit(1)

# ---------------------------
# === CEK DISTRIBUSI LABEL
# ---------------------------
print("\n[2/4] Analisis Distribusi Label")
print("-" * 80)

label_counts = df['label'].value_counts()
print("Distribusi label saat ini:")
for label, count in label_counts.items():
    print(f"  - {label}: {count} ({count/len(df)*100:.2f}%)")

# Identifikasi kelas mayoritas dan minoritas
if len(label_counts) < 2:
    print("\n❌ ERROR: Hanya ada 1 label dalam dataset!")
    print("Balancing memerlukan minimal 2 label (FAKE dan REAL)")
    exit(1)

majority_class = label_counts.idxmax()
minority_class = label_counts.idxmin()
majority_count = label_counts.max()
minority_count = label_counts.min()

print(f"\nKelas mayoritas: {majority_class} ({majority_count} data)")
print(f"Kelas minoritas: {minority_class} ({minority_count} data)")
print(f"Rasio imbalance: 1:{majority_count/minority_count:.2f}")

# ---------------------------
# === PILIHAN METODE BALANCING
# ---------------------------
print("\n[3/4] Metode Balancing")
print("-" * 80)

# Pisahkan data berdasarkan label
df_majority = df[df['label'] == majority_class]
df_minority = df[df['label'] == minority_class]

print("\nTersedia 3 metode balancing:\n")

# METODE 1: Undersampling (kurangi kelas mayoritas)
print("1. UNDERSAMPLING (Random Undersampling)")
print(f"   - Mengurangi {majority_class} dari {majority_count} menjadi {minority_count}")
print(f"   - Total data hasil: {minority_count * 2}")
print(f"   - Rasio: 50:50")
print(f"   - Keuntungan: Data balance sempurna, training lebih cepat")
print(f"   - Kerugian: Kehilangan banyak data dari kelas mayoritas")

# METODE 2: Oversampling (tambah kelas minoritas dengan duplikasi)
print("\n2. OVERSAMPLING (Random Oversampling)")
print(f"   - Menambah {minority_class} dari {minority_count} menjadi {majority_count}")
print(f"   - Total data hasil: {majority_count * 2}")
print(f"   - Rasio: 50:50")
print(f"   - Keuntungan: Tidak kehilangan data dari kelas mayoritas")
print(f"   - Kerugian: Duplikasi data (overfitting risk), training lebih lama")

# METODE 3: Kombinasi (middle ground)
target_count = int((majority_count + minority_count) / 2)
print("\n3. KOMBINASI (Hybrid)")
print(f"   - Undersampling {majority_class}: {majority_count} → {target_count}")
print(f"   - Oversampling {minority_class}: {minority_count} → {target_count}")
print(f"   - Total data hasil: {target_count * 2}")
print(f"   - Rasio: 50:50")
print(f"   - Keuntungan: Balance antara data loss dan duplikasi")
print(f"   - Kerugian: Masih ada sedikit data loss dan duplikasi")

# ---------------------------
# === TERAPKAN BALANCING (SEMUA METODE)
# ---------------------------
print("\n[4/4] Menerapkan Balancing")
print("-" * 80)

# METODE 1: Undersampling
print("\nMembuat dataset dengan metode UNDERSAMPLING...")
df_majority_downsampled = resample(df_majority,
                                   replace=False,
                                   n_samples=minority_count,
                                   random_state=42)
df_undersampled = pd.concat([df_majority_downsampled, df_minority])
df_undersampled = df_undersampled.sample(frac=1, random_state=42).reset_index(drop=True)

output_file_1 = os.path.join(OUTPUT_DIR, "google_review_balanced_undersampling.csv")
df_undersampled.to_csv(output_file_1, index=False)
print(f"✓ Metode 1 disimpan: {output_file_1}")
print(f"  Total data: {len(df_undersampled)}")
print(f"  Distribusi: {df_undersampled['label'].value_counts().to_dict()}")

# METODE 2: Oversampling
print("\nMembuat dataset dengan metode OVERSAMPLING...")
df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=majority_count,
                                 random_state=42)
df_oversampled = pd.concat([df_majority, df_minority_upsampled])
df_oversampled = df_oversampled.sample(frac=1, random_state=42).reset_index(drop=True)

output_file_2 = os.path.join(OUTPUT_DIR, "google_review_balanced_oversampling.csv")
df_oversampled.to_csv(output_file_2, index=False)
print(f"✓ Metode 2 disimpan: {output_file_2}")
print(f"  Total data: {len(df_oversampled)}")
print(f"  Distribusi: {df_oversampled['label'].value_counts().to_dict()}")

# METODE 3: Kombinasi
print("\nMembuat dataset dengan metode KOMBINASI...")
df_majority_combined = resample(df_majority,
                                replace=False,
                                n_samples=target_count,
                                random_state=42)
df_minority_combined = resample(df_minority,
                                replace=True,
                                n_samples=target_count,
                                random_state=42)
df_combined = pd.concat([df_majority_combined, df_minority_combined])
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

output_file_3 = os.path.join(OUTPUT_DIR, "google_review_balanced_combined.csv")
df_combined.to_csv(output_file_3, index=False)
print(f"✓ Metode 3 disimpan: {output_file_3}")
print(f"  Total data: {len(df_combined)}")
print(f"  Distribusi: {df_combined['label'].value_counts().to_dict()}")

# ---------------------------
# === STATISTIK PERBANDINGAN
# ---------------------------
print("\n" + "="*80)
print("RINGKASAN BALANCING")
print("="*80)

comparison = pd.DataFrame({
    'Dataset': ['Original', 'Undersampling', 'Oversampling', 'Kombinasi'],
    'Total Data': [
        len(df),
        len(df_undersampled),
        len(df_oversampled),
        len(df_combined)
    ],
    f'{majority_class}': [
        (df['label'] == majority_class).sum(),
        (df_undersampled['label'] == majority_class).sum(),
        (df_oversampled['label'] == majority_class).sum(),
        (df_combined['label'] == majority_class).sum()
    ],
    f'{minority_class}': [
        (df['label'] == minority_class).sum(),
        (df_undersampled['label'] == minority_class).sum(),
        (df_oversampled['label'] == minority_class).sum(),
        (df_combined['label'] == minority_class).sum()
    ]
})

print("\nPerbandingan hasil balancing:")
print(comparison.to_string(index=False))

# Simpan perbandingan
comparison_file = os.path.join(OUTPUT_DIR, "balancing_comparison.csv")
comparison.to_csv(comparison_file, index=False)
print(f"\n✓ Perbandingan disimpan: {comparison_file}")

# ---------------------------
# === REKOMENDASI
# ---------------------------
print("\n" + "="*80)
print("REKOMENDASI")
print("="*80)
print("\n1. UNDERSAMPLING (google_review_balanced_undersampling.csv)")
print("   ✓ Gunakan jika: Anda ingin training cepat dan punya banyak data mayoritas")
print("   ✓ Cocok untuk: Prototyping, testing model cepat")

print("\n2. OVERSAMPLING (google_review_balanced_oversampling.csv)")
print("   ✓ Gunakan jika: Anda ingin mempertahankan semua informasi dari data asli")
print("   ✓ Cocok untuk: Dataset kecil, atau kelas minoritas sangat penting")

print("\n3. KOMBINASI (google_review_balanced_combined.csv) ⭐ RECOMMENDED")
print("   ✓ Gunakan jika: Anda ingin balance terbaik antara data loss dan duplikasi")
print("   ✓ Cocok untuk: Training model production")

print("\n" + "="*80)
print("BALANCING SELESAI!")
print("="*80)
print("\nAnda bisa menggunakan salah satu file hasil balancing untuk training model:")
print(f"  - {output_file_1}")
print(f"  - {output_file_2}")
print(f"  - {output_file_3}")
