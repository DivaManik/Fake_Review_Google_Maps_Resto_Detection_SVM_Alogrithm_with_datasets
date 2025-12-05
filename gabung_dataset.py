import pandas as pd
import os

# --- KONFIGURASI ---
# Daftar file CSV yang ingin digabungkan.
# Anda bisa menambahkan lebih banyak file ke dalam daftar ini.
FILES_TO_COMBINE = [
    "dataset.csv",
    "dataset2.csv",
    # "dataset3.csv", # Contoh jika ada file lain
]

# Nama file keluaran setelah digabungkan
OUTPUT_FILE = "dataset_gabungan.csv"
# --------------------

def combine_datasets():
    """
    Fungsi untuk membaca beberapa file CSV dari daftar, menggabungkannya,
    dan menyimpannya sebagai satu file CSV baru.
    """
    print("Memulai proses penggabungan dataset...")

    list_of_dataframes = []
    total_rows = 0

    for file in FILES_TO_COMBINE:
        if os.path.exists(file):
            print(f"Membaca file: {file}...")
            try:
                df_temp = pd.read_csv(file, low_memory=False)
                list_of_dataframes.append(df_temp)
                print(f" -> Ditemukan {len(df_temp)} baris.")
                total_rows += len(df_temp)
            except Exception as e:
                print(f" -> Gagal membaca file {file}. Error: {e}")
        else:
            print(f" -> Peringatan: File '{file}' tidak ditemukan, akan dilewati.")

    if not list_of_dataframes:
        print("\nTidak ada dataset yang berhasil dibaca. Proses dihentikan.")
        return

    print("\nMenggabungkan semua dataset yang ditemukan...")
    # ignore_index=True akan membuat ulang index dari 0 hingga akhir
    combined_df = pd.concat(list_of_dataframes, ignore_index=True)

    print(f"Menyimpan hasil gabungan ke: {OUTPUT_FILE} ({len(combined_df)} baris)")
    combined_df.to_csv(OUTPUT_FILE, index=False)
    print("\nSelesai! Dataset berhasil digabungkan.")

if __name__ == "__main__":
    combine_datasets()