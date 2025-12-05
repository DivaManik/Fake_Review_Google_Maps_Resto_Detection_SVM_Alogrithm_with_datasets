"""
Script EDA untuk menganalisis hasil preprocessing
- Membaca file google_review_preprocessed.csv
- Analisis statistik teks sebelum dan sesudah preprocessing
- Distribusi panjang teks
- Kata-kata yang paling sering muncul
- Visualisasi dan simpan plot
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# Set style untuk plot yang lebih bagus
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# ---------------------------
# === KONFIGURASI
# ---------------------------
INPUT_FILE = "label_output/google_review_preprocessed.csv"
OUTPUT_DIR = "output_preprocessed_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("EDA HASIL PREPROCESSING - GOOGLE REVIEWS")
print("="*80)

# ---------------------------
# === LOAD DATA
# ---------------------------
print(f"\n[1/6] Membaca dataset dari: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Total data: {len(df)} baris")
print(f"Kolom tersedia: {df.columns.tolist()}")

# ---------------------------
# === STATISTIK DASAR
# ---------------------------
print("\n[2/6] Statistik Dasar")
print("-" * 80)

# Info kolom
print(f"Jumlah kolom: {len(df.columns)}")
print(f"Kolom utama:")
print(f"  - text_original: {df['text_original'].notna().sum()} (tidak kosong)")
print(f"  - text_preprocessed: {df['text_preprocessed'].notna().sum()} (tidak kosong)")

# Cek label jika ada
if 'label' in df.columns:
    print(f"\nDistribusi Label:")
    print(df['label'].value_counts())
    print(f"  - FAKE: {(df['label'] == 'FAKE').sum()} ({(df['label'] == 'FAKE').mean()*100:.2f}%)")
    print(f"  - REAL: {(df['label'] == 'REAL').sum()} ({(df['label'] == 'REAL').mean()*100:.2f}%)")

# ---------------------------
# === ANALISIS PANJANG TEKS
# ---------------------------
print("\n[3/6] Analisis Panjang Teks")
print("-" * 80)

# Hitung panjang teks (jumlah kata)
df['len_original'] = df['text_original'].fillna('').apply(lambda x: len(str(x).split()))
df['len_preprocessed'] = df['text_preprocessed'].fillna('').apply(lambda x: len(str(x).split()))

# Hitung persentase pengurangan
df['reduction_pct'] = ((df['len_original'] - df['len_preprocessed']) / df['len_original'] * 100).fillna(0)

print(f"Panjang Teks Original:")
print(f"  - Mean: {df['len_original'].mean():.2f} kata")
print(f"  - Median: {df['len_original'].median():.2f} kata")
print(f"  - Min: {df['len_original'].min()} kata")
print(f"  - Max: {df['len_original'].max()} kata")

print(f"\nPanjang Teks Preprocessed:")
print(f"  - Mean: {df['len_preprocessed'].mean():.2f} kata")
print(f"  - Median: {df['len_preprocessed'].median():.2f} kata")
print(f"  - Min: {df['len_preprocessed'].min()} kata")
print(f"  - Max: {df['len_preprocessed'].max()} kata")

print(f"\nPengurangan Panjang Teks:")
print(f"  - Rata-rata pengurangan: {df['reduction_pct'].mean():.2f}%")
print(f"  - Median pengurangan: {df['reduction_pct'].median():.2f}%")

# Cek teks yang hilang setelah preprocessing
empty_after_preprocessing = (df['len_preprocessed'] == 0).sum()
print(f"\nTeks yang kosong setelah preprocessing: {empty_after_preprocessing} ({empty_after_preprocessing/len(df)*100:.2f}%)")

# Plot 1: Perbandingan panjang teks
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['len_original'].clip(0, 100), bins=30, alpha=0.7, label='Original', color='blue')
plt.hist(df['len_preprocessed'].clip(0, 100), bins=30, alpha=0.7, label='Preprocessed', color='red')
plt.title('Distribusi Panjang Teks (0-100 kata)')
plt.xlabel('Jumlah Kata')
plt.ylabel('Frekuensi')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(df['reduction_pct'], bins=30, color='green', alpha=0.7)
plt.title('Distribusi Persentase Pengurangan Panjang Teks')
plt.xlabel('Pengurangan (%)')
plt.ylabel('Frekuensi')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "text_length_comparison.png"), dpi=300)
plt.close()
print(f"\n✓ Plot disimpan: text_length_comparison.png")

# Plot 2: Boxplot perbandingan
plt.figure(figsize=(10, 6))
data_to_plot = [df['len_original'].clip(0, 150), df['len_preprocessed'].clip(0, 150)]
plt.boxplot(data_to_plot, labels=['Original', 'Preprocessed'])
plt.title('Boxplot Perbandingan Panjang Teks')
plt.ylabel('Jumlah Kata')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "text_length_boxplot.png"), dpi=300)
plt.close()
print(f"✓ Plot disimpan: text_length_boxplot.png")

# ---------------------------
# === ANALISIS KATA PALING SERING
# ---------------------------
print("\n[4/6] Analisis Kata Paling Sering Muncul")
print("-" * 80)

# Gabungkan semua teks preprocessed
all_words = []
for text in df['text_preprocessed'].dropna():
    words = str(text).split()
    all_words.extend(words)

# Hitung frekuensi kata
word_freq = Counter(all_words)
most_common = word_freq.most_common(30)

print(f"Total kata unik: {len(word_freq)}")
print(f"Total kata: {len(all_words)}")
print(f"\nTop 30 kata paling sering muncul:")
for idx, (word, freq) in enumerate(most_common, 1):
    print(f"  {idx:2d}. {word:20s} : {freq:6d} ({freq/len(all_words)*100:.2f}%)")

# Plot 3: Bar chart kata paling sering
plt.figure(figsize=(12, 8))
words, counts = zip(*most_common)
plt.barh(range(len(words)), counts, color='steelblue')
plt.yticks(range(len(words)), words)
plt.xlabel('Frekuensi')
plt.title('30 Kata Paling Sering Muncul (Setelah Preprocessing)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_words_frequency.png"), dpi=300)
plt.close()
print(f"\n✓ Plot disimpan: top_words_frequency.png")

# ---------------------------
# === ANALISIS PER LABEL (jika ada)
# ---------------------------
if 'label' in df.columns:
    print("\n[5/6] Analisis Berdasarkan Label")
    print("-" * 80)

    for label in df['label'].unique():
        df_label = df[df['label'] == label]
        print(f"\nLabel: {label}")
        print(f"  - Jumlah data: {len(df_label)}")
        print(f"  - Rata-rata panjang original: {df_label['len_original'].mean():.2f} kata")
        print(f"  - Rata-rata panjang preprocessed: {df_label['len_preprocessed'].mean():.2f} kata")
        print(f"  - Rata-rata pengurangan: {df_label['reduction_pct'].mean():.2f}%")

    # Plot 4: Perbandingan panjang teks per label
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for label in df['label'].unique():
        df_label = df[df['label'] == label]
        plt.hist(df_label['len_original'].clip(0, 100), bins=30, alpha=0.5, label=label)
    plt.title('Distribusi Panjang Teks Original per Label')
    plt.xlabel('Jumlah Kata')
    plt.ylabel('Frekuensi')
    plt.legend()

    plt.subplot(1, 2, 2)
    for label in df['label'].unique():
        df_label = df[df['label'] == label]
        plt.hist(df_label['len_preprocessed'].clip(0, 100), bins=30, alpha=0.5, label=label)
    plt.title('Distribusi Panjang Teks Preprocessed per Label')
    plt.xlabel('Jumlah Kata')
    plt.ylabel('Frekuensi')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "text_length_by_label.png"), dpi=300)
    plt.close()
    print(f"\n✓ Plot disimpan: text_length_by_label.png")

    # Plot 5: Kata paling sering per label
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, label in enumerate(df['label'].unique()):
        df_label = df[df['label'] == label]

        # Gabungkan kata untuk label ini
        words_label = []
        for text in df_label['text_preprocessed'].dropna():
            words_label.extend(str(text).split())

        # Top 15 kata
        word_freq_label = Counter(words_label)
        top_15 = word_freq_label.most_common(15)

        if top_15:
            words, counts = zip(*top_15)
            axes[idx].barh(range(len(words)), counts, color='coral' if label == 'FAKE' else 'skyblue')
            axes[idx].set_yticks(range(len(words)))
            axes[idx].set_yticklabels(words)
            axes[idx].set_xlabel('Frekuensi')
            axes[idx].set_title(f'Top 15 Kata - Label {label}')
            axes[idx].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_words_by_label.png"), dpi=300)
    plt.close()
    print(f"✓ Plot disimpan: top_words_by_label.png")
else:
    print("\n[5/6] Analisis per label dilewati (kolom 'label' tidak ditemukan)")

# ---------------------------
# === SIMPAN STATISTIK KE CSV
# ---------------------------
print("\n[6/6] Menyimpan Statistik")
print("-" * 80)

# Statistik umum
stats_summary = {
    'Metrik': [
        'Total Data',
        'Rata-rata Panjang Original (kata)',
        'Rata-rata Panjang Preprocessed (kata)',
        'Median Panjang Original (kata)',
        'Median Panjang Preprocessed (kata)',
        'Rata-rata Pengurangan (%)',
        'Teks Kosong Setelah Preprocessing',
        'Total Kata Unik',
        'Total Kata'
    ],
    'Nilai': [
        len(df),
        f"{df['len_original'].mean():.2f}",
        f"{df['len_preprocessed'].mean():.2f}",
        f"{df['len_original'].median():.2f}",
        f"{df['len_preprocessed'].median():.2f}",
        f"{df['reduction_pct'].mean():.2f}",
        f"{empty_after_preprocessing} ({empty_after_preprocessing/len(df)*100:.2f}%)",
        len(word_freq),
        len(all_words)
    ]
}

df_stats = pd.DataFrame(stats_summary)
stats_file = os.path.join(OUTPUT_DIR, "preprocessing_statistics.csv")
df_stats.to_csv(stats_file, index=False)
print(f"✓ Statistik disimpan: {stats_file}")

# Top 100 kata paling sering
top_100_words = word_freq.most_common(100)
df_top_words = pd.DataFrame(top_100_words, columns=['Kata', 'Frekuensi'])
df_top_words['Persentase'] = (df_top_words['Frekuensi'] / len(all_words) * 100).round(2)
top_words_file = os.path.join(OUTPUT_DIR, "top_100_words.csv")
df_top_words.to_csv(top_words_file, index=False)
print(f"✓ Top 100 kata disimpan: {top_words_file}")

# Sample data dengan perbandingan
sample_comparison = df[['text_original', 'text_preprocessed', 'len_original', 'len_preprocessed', 'reduction_pct']].head(50)
sample_file = os.path.join(OUTPUT_DIR, "sample_comparison_50.csv")
sample_comparison.to_csv(sample_file, index=False)
print(f"✓ Sample perbandingan disimpan: {sample_file}")

# ---------------------------
# === SUMMARY
# ---------------------------
print("\n" + "="*80)
print("RINGKASAN")
print("="*80)
print(f"Total data dianalisis: {len(df)} baris")
print(f"Pengurangan rata-rata panjang teks: {df['reduction_pct'].mean():.2f}%")
print(f"Kata unik setelah preprocessing: {len(word_freq)}")
print(f"\nFile output tersimpan di folder: {OUTPUT_DIR}")
print("  - text_length_comparison.png")
print("  - text_length_boxplot.png")
print("  - top_words_frequency.png")
if 'label' in df.columns:
    print("  - text_length_by_label.png")
    print("  - top_words_by_label.png")
print("  - preprocessing_statistics.csv")
print("  - top_100_words.csv")
print("  - sample_comparison_50.csv")

print("\n" + "="*80)
print("EDA PREPROCESSING SELESAI!")
print("="*80)
