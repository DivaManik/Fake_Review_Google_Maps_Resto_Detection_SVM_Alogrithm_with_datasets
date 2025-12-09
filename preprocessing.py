import pandas as pd
import re
import string
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi Sastrawi
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

factory_stopword = StopWordRemoverFactory()
stopword_remover = factory_stopword.create_stop_word_remover()

# Kamus normalisasi untuk kata-kata tidak baku (slang) bahasa Indonesia
normalization_dict = {
    'gak': 'tidak',
    'ga': 'tidak',
    'ngga': 'tidak',
    'nggak': 'tidak',
    'gk': 'tidak',
    'tdk': 'tidak',
    'gpp': 'tidak apa apa',
    'bgt': 'banget',
    'bgt': 'banget',
    'bgt': 'banget',
    'bgd': 'banget',
    'bngd': 'banget',
    'bngt': 'banget',
    'bener': 'benar',
    'bgt': 'banget',
    'emg': 'memang',
    'emng': 'memang',
    'udh': 'sudah',
    'udah': 'sudah',
    'dah': 'sudah',
    'blm': 'belum',
    'blom': 'belum',
    'jg': 'juga',
    'jgn': 'jangan',
    'dgn': 'dengan',
    'sm': 'sama',
    'gmn': 'bagaimana',
    'gmana': 'bagaimana',
    'knp': 'kenapa',
    'knapa': 'kenapa',
    'krn': 'karena',
    'krna': 'karena',
    'karna': 'karena',
    'tp': 'tetapi',
    'tapi': 'tetapi',
    'yg': 'yang',
    'utk': 'untuk',
    'org': 'orang',
    'sdh': 'sudah',
    'trs': 'terus',
    'bs': 'bisa',
    'bsa': 'bisa',
    'sy': 'saya',
    'kl': 'kalau',
    'klo': 'kalau',
    'kalo': 'kalau',
    'byk': 'banyak',
    'banyak': 'banyak',
    'skrg': 'sekarang',
    'skr': 'sekarang',
    'skg': 'sekarang',
    'hrs': 'harus',
    'trmksh': 'terima kasih',
    'mksh': 'terima kasih',
    'thx': 'terima kasih',
    'thanks': 'terima kasih',
    'thks': 'terima kasih',
    'mantap': 'mantap',
    'mantul': 'mantap',
    'mantep': 'mantap',
    'keren': 'keren',
    'bagus': 'bagus',
    'jelek': 'jelek',
    'buruk': 'buruk',
    'enak': 'enak',
    'mantab': 'mantap',
    'rekomend': 'rekomendasi',
    'rekomen': 'rekomendasi',
    'recommended': 'rekomendasi',
    'poll': 'banget',
    'poolll': 'banget',
    'pol': 'banget',
    'sih': '',
    'si': '',
    'nih': '',
    'nii': '',
    'ya': '',
    'yaa': '',
    'iya': 'ya',
    'yoi': 'ya',
}

def case_folding(text):
    """Mengubah semua huruf menjadi lowercase"""
    if pd.isna(text):
        return ''
    return text.lower()

def cleansing(text):
    """Menghapus karakter yang tidak diperlukan"""
    if pd.isna(text):
        return ''

    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Hapus mention (@username)
    text = re.sub(r'@\w+', '', text)

    # Hapus hashtag
    text = re.sub(r'#\w+', '', text)

    # Hapus emoji dan special characters
    text = re.sub(r'[^\w\s]', ' ', text)

    # Hapus angka
    text = re.sub(r'\d+', '', text)

    # Hapus whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def normalize_text(text):
    """Normalisasi kata-kata tidak baku menjadi baku"""
    if pd.isna(text) or text == '':
        return ''

    words = text.split()
    normalized_words = [normalization_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

def tokenizing(text):
    """Memecah teks menjadi token/kata"""
    if pd.isna(text) or text == '':
        return []
    return text.split()

def stopword_removal(tokens):
    """Menghapus stopword dari list token"""
    if not tokens:
        return []

    # Menggunakan Sastrawi stopword
    text = ' '.join(tokens)
    text_cleaned = stopword_remover.remove(text)
    return text_cleaned.split()

def stemming(tokens):
    """Melakukan stemming pada list token"""
    if not tokens:
        return []

    # Menggunakan Sastrawi stemmer
    text = ' '.join(tokens)
    text_stemmed = stemmer.stem(text)
    return text_stemmed.split()

def preprocess_pipeline(text):
    """Pipeline lengkap preprocessing"""
    # 1. Case folding
    text = case_folding(text)

    # 2. Cleansing
    text = cleansing(text)

    # 3. Normalisasi
    text = normalize_text(text)

    # 4. Tokenizing
    tokens = tokenizing(text)

    # 5. Stopword removal
    tokens = stopword_removal(tokens)

    # 6. Stemming
    tokens = stemming(tokens)

    # Kembalikan sebagai string
    return ' '.join(tokens)

def main():
    print("="*80)
    print("PREPROCESSING TEXT DATA - GOOGLE REVIEWS")
    print("="*80)

    # Load data
    print("\n[1/4] Membaca dataset...")
    df = pd.read_csv('./dataset_organik.csv')
    print(f"Total data awal: {len(df)} baris")

    # Hapus baris dengan kolom ulasan (text) kosong
    print("\n[2/4] Menghapus baris dengan kolom ulasan kosong...")
    df_before = len(df)
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']
    df_after = len(df)
    print(f"Baris yang dihapus: {df_before - df_after}")
    print(f"Total data setelah cleaning: {df_after} baris")

    # Preprocessing
    print("\n[3/4] Melakukan preprocessing pada kolom 'text'...")
    print("  - Case folding")
    print("  - Cleansing")
    print("  - Normalisasi")
    print("  - Tokenizing")
    print("  - Stopword removal")
    print("  - Stemming")

    # Simpan kolom asli
    df['text_original'] = df['text'].copy()

    # Terapkan preprocessing
    df['text_preprocessed'] = df['text'].apply(preprocess_pipeline)

    print("\nContoh hasil preprocessing:")
    print("-" * 80)
    for idx in range(min(3, len(df))):
        print(f"\nContoh {idx+1}:")
        print(f"Original: {df['text_original'].iloc[idx][:100]}...")
        print(f"Preprocessed: {df['text_preprocessed'].iloc[idx][:100]}...")
    print("-" * 80)

    # Simpan hasil
    print("\n[4/4] Menyimpan hasil preprocessing...")
    output_dir = 'data_preprocessed'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'google_review_preprocessed.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"File berhasil disimpan: {output_file}")

    # Statistik
    print("\n" + "="*80)
    print("STATISTIK PREPROCESSING")
    print("="*80)
    print(f"Total data yang diproses: {len(df)}")
    print(f"Kolom yang ditambahkan:")
    print(f"  - text_original: Teks asli sebelum preprocessing")
    print(f"  - text_preprocessed: Teks setelah preprocessing")
    print("\nPreprocessing selesai!")

if __name__ == "__main__":
    main()
