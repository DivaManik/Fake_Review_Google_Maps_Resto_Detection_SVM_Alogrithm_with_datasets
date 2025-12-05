"""
Text Quality Detector untuk deteksi gibberish, random text, dan spam
Fitur untuk mendeteksi review fake dengan kata-kata tidak bermakna
"""

import re
import math
from collections import Counter
import pandas as pd


# Kamus kata Indonesia umum (subset untuk validasi cepat)
COMMON_INDONESIAN_WORDS = {
    'ini', 'itu', 'yang', 'untuk', 'dari', 'dengan', 'tidak', 'ada', 'adalah', 'dan',
    'atau', 'jika', 'maka', 'karena', 'tetapi', 'tapi', 'sudah', 'belum', 'akan',
    'sangat', 'kurang', 'lebih', 'baik', 'buruk', 'bagus', 'jelek', 'enak', 'lezat',
    'makanan', 'minuman', 'tempat', 'lokasi', 'pelayanan', 'layanan', 'harga', 'murah',
    'mahal', 'recommended', 'rekomendasi', 'mantap', 'keren', 'oke', 'suka', 'senang',
    'puas', 'kecewa', 'bersih', 'kotor', 'nyaman', 'ramai', 'sepi', 'cepat', 'lambat',
    'ramah', 'sopan', 'kasar', 'menu', 'porsi', 'rasa', 'besar', 'kecil', 'banyak',
    'sedikit', 'panas', 'dingin', 'hangat', 'segar', 'basi', 'asin', 'manis', 'pahit',
    'pedas', 'gurih', 'hambar', 'saya', 'kamu', 'kami', 'mereka', 'dia', 'anda',
    'disini', 'disana', 'kemari', 'kesana', 'datang', 'pergi', 'tiba', 'sampai',
    'masuk', 'keluar', 'buka', 'tutup', 'beli', 'bayar', 'order', 'pesan', 'makan',
    'minum', 'coba', 'cari', 'lihat', 'tunggu', 'antri', 'duduk', 'berdiri'
}

# Vokal bahasa Indonesia
VOWELS = set('aeiouAEIOU')

# Konsonan bahasa Indonesia
CONSONANTS = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')


def vowel_consonant_ratio(text):
    """
    Hitung rasio vokal vs konsonan
    Teks gibberish biasanya punya rasio yang tidak normal

    Returns:
        float: rasio vokal/konsonan (0 jika tidak ada konsonan)
    """
    if pd.isna(text) or str(text).strip() == '':
        return 0.5  # neutral

    text = str(text)
    vowel_count = sum(1 for c in text if c in VOWELS)
    consonant_count = sum(1 for c in text if c in CONSONANTS)

    if consonant_count == 0:
        return 0.0  # tidak ada konsonan, sangat tidak normal

    ratio = vowel_count / consonant_count
    return ratio


def is_abnormal_vowel_ratio(text, min_ratio=0.25, max_ratio=2.5):
    """
    Cek apakah rasio vokal/konsonan abnormal

    Normal Indonesian: ~0.4-0.6
    Gibberish: < 0.25 atau > 2.5
    """
    ratio = vowel_consonant_ratio(text)
    return 1 if (ratio < min_ratio or ratio > max_ratio) else 0


def valid_word_ratio(text):
    """
    Hitung persentase kata yang valid (ada di kamus umum)
    Teks gibberish biasanya punya valid_word_ratio rendah

    Returns:
        float: 0.0-1.0 (persentase kata valid)
    """
    if pd.isna(text) or str(text).strip() == '':
        return 0.0

    words = str(text).lower().split()
    if len(words) == 0:
        return 0.0

    valid_count = sum(1 for word in words if word in COMMON_INDONESIAN_WORDS)
    return valid_count / len(words)


def is_low_valid_words(text, threshold=0.1):
    """
    Cek apakah teks punya kata valid yang rendah

    Jika < 10% kata valid, kemungkinan gibberish
    """
    ratio = valid_word_ratio(text)
    return 1 if ratio < threshold else 0


def average_word_length(text):
    """
    Hitung rata-rata panjang kata
    Gibberish cenderung punya kata yang terlalu panjang

    Normal Indonesian: ~5-7 karakter per kata
    Gibberish: > 10 karakter
    """
    if pd.isna(text) or str(text).strip() == '':
        return 0.0

    words = str(text).split()
    if len(words) == 0:
        return 0.0

    total_length = sum(len(word) for word in words)
    return total_length / len(words)


def has_long_words(text, threshold=8):
    """
    Cek apakah ada kata yang terlalu panjang
    Kata rata-rata > 8 karakter biasanya gibberish
    """
    if pd.isna(text) or str(text).strip() == '':
        return 0

    words = str(text).split()
    avg_len = average_word_length(text)

    return 1 if avg_len > threshold else 0


def text_entropy(text):
    """
    Hitung entropy dari karakter dalam teks
    Entropy tinggi = lebih random/gibberish

    Returns:
        float: entropy value (0-5 biasanya, higher = more random)
    """
    if pd.isna(text) or str(text).strip() == '':
        return 0.0

    text = str(text)
    if len(text) == 0:
        return 0.0

    # Hitung frekuensi setiap karakter
    char_freq = Counter(text)
    total_chars = len(text)

    # Hitung entropy
    entropy = 0
    for count in char_freq.values():
        probability = count / total_chars
        if probability > 0:
            entropy -= probability * math.log2(probability)

    return entropy


def is_high_entropy(text, threshold=4.5):
    """
    Cek apakah entropy teks terlalu tinggi (terlalu random)
    """
    entropy = text_entropy(text)
    return 1 if entropy > threshold else 0


def repeated_character_ratio(text):
    """
    Hitung rasio karakter yang berulang berturut-turut
    Contoh: "aaaa" "lllll" "kkkkk" adalah pattern spam

    Returns:
        float: 0.0-1.0 (persentase karakter repeated)
    """
    if pd.isna(text) or str(text).strip() == '':
        return 0.0

    text = str(text)
    if len(text) <= 1:
        return 0.0

    repeated_count = 0
    for i in range(1, len(text)):
        if text[i] == text[i-1] and text[i].isalpha():
            repeated_count += 1

    return repeated_count / len(text)


def has_repeated_chars(text, threshold=0.3):
    """
    Cek apakah ada banyak karakter berulang
    > 30% repeated chars = kemungkinan spam
    """
    ratio = repeated_character_ratio(text)
    return 1 if ratio > threshold else 0


def consonant_cluster_score(text):
    """
    Hitung score untuk cluster konsonan yang tidak wajar
    Contoh: "lksdjflksjd" punya banyak konsonan berturut-turut

    Returns:
        float: average max consonant cluster length
    """
    if pd.isna(text) or str(text).strip() == '':
        return 0.0

    text = str(text).lower()
    words = text.split()

    if len(words) == 0:
        return 0.0

    max_clusters = []

    for word in words:
        current_cluster = 0
        max_cluster = 0

        for char in word:
            if char in CONSONANTS:
                current_cluster += 1
                max_cluster = max(max_cluster, current_cluster)
            else:
                current_cluster = 0

        max_clusters.append(max_cluster)

    return sum(max_clusters) / len(max_clusters) if max_clusters else 0.0


def has_long_consonant_clusters(text, threshold=4):
    """
    Cek apakah ada cluster konsonan yang terlalu panjang
    Bahasa Indonesia jarang punya > 3 konsonan berturut-turut
    """
    score = consonant_cluster_score(text)
    return 1 if score > threshold else 0


def detect_gibberish(text):
    """
    Fungsi utama untuk mendeteksi gibberish
    Menggabungkan semua metrik deteksi

    Returns:
        dict: {
            'is_gibberish': bool (True/False),
            'gibberish_score': float (0-1, higher = more gibberish),
            'details': dict of individual scores
        }
    """
    if pd.isna(text) or str(text).strip() == '':
        return {
            'is_gibberish': False,
            'gibberish_score': 0.0,
            'details': {}
        }

    # Hitung semua metrik
    details = {
        'vowel_ratio': vowel_consonant_ratio(text),
        'abnormal_vowel': is_abnormal_vowel_ratio(text),
        'valid_word_ratio': valid_word_ratio(text),
        'low_valid_words': is_low_valid_words(text),
        'avg_word_length': average_word_length(text),
        'has_long_words': has_long_words(text),
        'entropy': text_entropy(text),
        'high_entropy': is_high_entropy(text),
        'repeated_char_ratio': repeated_character_ratio(text),
        'has_repeated_chars': has_repeated_chars(text),
        'consonant_cluster': consonant_cluster_score(text),
        'long_consonant_cluster': has_long_consonant_clusters(text)
    }

    # Hitung gibberish score (0-1)
    # Bobot untuk setiap indikator
    score = 0
    score += details['abnormal_vowel'] * 0.15
    score += details['low_valid_words'] * 0.30  # paling penting - ditingkatkan
    score += details['has_long_words'] * 0.20   # ditingkatkan
    score += details['high_entropy'] * 0.10
    score += details['has_repeated_chars'] * 0.10
    score += details['long_consonant_cluster'] * 0.15

    # Threshold: jika score > 0.4, dianggap gibberish (diturunkan dari 0.5)
    is_gibberish = score > 0.4

    return {
        'is_gibberish': is_gibberish,
        'gibberish_score': score,
        'details': details
    }


# ---------------------------
# === FUNCTIONS UNTUK FEATURE ENGINEERING
# ---------------------------

def extract_text_quality_features(df, text_column='text_preprocessed'):
    """
    Extract text quality features untuk dataframe
    Digunakan di training dan prediction

    Parameters:
        df: pandas DataFrame
        text_column: nama kolom text yang akan dianalisis

    Returns:
        df: DataFrame dengan kolom fitur baru
    """
    print("  [Text Quality] Extracting text quality features...")

    # Feature 1: Vowel-consonant ratio
    df['tq_vowel_ratio'] = df[text_column].apply(vowel_consonant_ratio)
    df['tq_abnormal_vowel'] = df[text_column].apply(is_abnormal_vowel_ratio)

    # Feature 2: Valid word ratio
    df['tq_valid_word_ratio'] = df[text_column].apply(valid_word_ratio)
    df['tq_low_valid_words'] = df[text_column].apply(is_low_valid_words)

    # Feature 3: Word length
    df['tq_avg_word_length'] = df[text_column].apply(average_word_length)
    df['tq_has_long_words'] = df[text_column].apply(has_long_words)

    # Feature 4: Entropy
    df['tq_entropy'] = df[text_column].apply(text_entropy)
    df['tq_high_entropy'] = df[text_column].apply(is_high_entropy)

    # Feature 5: Repeated characters
    df['tq_repeated_ratio'] = df[text_column].apply(repeated_character_ratio)
    df['tq_has_repeated'] = df[text_column].apply(has_repeated_chars)

    # Feature 6: Consonant clusters
    df['tq_consonant_cluster'] = df[text_column].apply(consonant_cluster_score)
    df['tq_long_consonant'] = df[text_column].apply(has_long_consonant_clusters)

    # Feature 7: Overall gibberish score
    gibberish_results = df[text_column].apply(detect_gibberish)
    df['tq_gibberish_score'] = gibberish_results.apply(lambda x: x['gibberish_score'])
    df['tq_is_gibberish'] = gibberish_results.apply(lambda x: 1 if x['is_gibberish'] else 0)

    print(f"    [OK] Detected {df['tq_is_gibberish'].sum()} potential gibberish reviews")
    print(f"    [OK] Mean gibberish score: {df['tq_gibberish_score'].mean():.3f}")

    return df


def get_text_quality_feature_names():
    """
    Return list of text quality feature names
    Untuk digunakan di model training/prediction
    """
    return [
        'tq_vowel_ratio',
        'tq_abnormal_vowel',
        'tq_valid_word_ratio',
        'tq_low_valid_words',
        'tq_avg_word_length',
        'tq_has_long_words',
        'tq_entropy',
        'tq_high_entropy',
        'tq_repeated_ratio',
        'tq_has_repeated',
        'tq_consonant_cluster',
        'tq_long_consonant',
        'tq_gibberish_score',
        'tq_is_gibberish'
    ]


# ---------------------------
# === TESTING
# ---------------------------

if __name__ == "__main__":
    print("="*80)
    print("TEXT QUALITY DETECTOR - TESTING")
    print("="*80)

    # Test cases
    test_texts = [
        "Makanan enak banget, pelayanan ramah, harga terjangkau. Recommended!",
        "lnsalkdjalks djasjd lkasjdlkj alskdjal kjdlka sjdklja sllksajd",
        "aaaaaaaaaaa bbbbbbbbb cccccccc",
        "qwerty asdfgh zxcvbn poiuyt",
        "Tempat nyaman suasana tenang cocok untuk kumpul",
        "xkljdsflkjsdf lskdjflskjdf lksjdflksjdf",
        "mantap poll",
        ""
    ]

    print("\nTest Results:")
    print("-" * 80)

    for i, text in enumerate(test_texts, 1):
        result = detect_gibberish(text)
        print(f"\nTest {i}: \"{text[:50]}...\"")
        print(f"  Is Gibberish: {result['is_gibberish']}")
        print(f"  Gibberish Score: {result['gibberish_score']:.3f}")
        print(f"  Valid Word Ratio: {result['details']['valid_word_ratio']:.3f}")
        print(f"  Vowel Ratio: {result['details']['vowel_ratio']:.3f}")

    print("\n" + "="*80)
    print("Testing completed!")
