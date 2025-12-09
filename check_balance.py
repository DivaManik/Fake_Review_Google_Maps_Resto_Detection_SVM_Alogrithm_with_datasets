"""
Script untuk mengecek distribusi label (FAKE vs REAL) di dataset
"""

import pandas as pd
import os

print("="*80)
print("CEK DISTRIBUSI LABEL DATASET")
print("="*80)

# Cek file yang ada
files_to_check = [
    # "label_output/google_review_preprocessed.csv",
    "label_output/google_review_balanced_undersampling.csv",
    "label_output/google_review_balanced_oversampling.csv",
    "label_output/google_review_balanced_combined.csv"
]

print("\n[1] Mengecek file yang tersedia...")
print("-" * 80)

for file_path in files_to_check:
    file_name = os.path.basename(file_path)

    if os.path.exists(file_path):
        print(f"✓ {file_name}")

        # Baca file
        df = pd.read_csv(file_path)

        # Cek kolom label
        if 'label' in df.columns:
            # Hitung distribusi
            total = len(df)
            label_counts = df['label'].value_counts()

            print(f"  Total data: {total:,}")

            for label, count in label_counts.items():
                percentage = (count / total) * 100
                print(f"  - {label}: {count:,} ({percentage:.2f}%)")

            # Cek apakah balanced
            if len(label_counts) == 2:
                counts_list = label_counts.values
                diff_pct = abs(counts_list[0] - counts_list[1]) / total * 100

                if diff_pct < 5:
                    print(f"  Status: ✓ BALANCED (selisih {diff_pct:.2f}%)")
                elif diff_pct < 15:
                    print(f"  Status: ⚠ CUKUP BALANCED (selisih {diff_pct:.2f}%)")
                else:
                    print(f"  Status: ✗ TIDAK BALANCED (selisih {diff_pct:.2f}%)")

            print()
        else:
            print(f"  ⚠ Kolom 'label' tidak ditemukan")
            print()
    else:
        print(f"✗ {file_name} (tidak ditemukan)")
        print()

# Cek file comparison jika ada
print("\n[2] Ringkasan dari balancing sebelumnya...")
print("-" * 80)

comparison_file = "label_output/balancing_comparison.csv"
if os.path.exists(comparison_file):
    df_comp = pd.read_csv(comparison_file)
    print("\nData dari balancing_comparison.csv:")
    print(df_comp.to_string(index=False))
else:
    print("File balancing_comparison.csv tidak ditemukan")

print("\n" + "="*80)
print("KESIMPULAN")
print("="*80)

# Cek apakah ada file balanced
balanced_exists = any(os.path.exists(f) for f in files_to_check[1:])

if balanced_exists:
    print("\n✓ File balanced sudah tersedia!")
    print("  Anda bisa langsung training dengan file balanced.")
else:
    print("\n✗ File balanced belum ada!")
    print("\n📋 Langkah yang perlu dilakukan:")
    print("  1. Jalankan: python balance_dataset.py")
    print("  2. Tunggu hingga selesai membuat 3 file balanced")
    print("  3. Pilih salah satu file balanced untuk training")
    print("\n  Pilihan file balanced:")
    print("  - undersampling: Dataset lebih kecil, training cepat")
    print("  - oversampling: Dataset lebih besar, tidak ada data loss")
    print("  - combined: Middle ground (RECOMMENDED)")

print("\n" + "="*80)
