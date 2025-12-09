"""
Debug script untuk test text quality detector pada teks gibberish dari tes.csv
"""

import pandas as pd
from file_testing.text_quality_detector import detect_gibberish, extract_text_quality_features
from preprocessing import preprocess_pipeline

print("="*80)
print("DEBUG: GIBBERISH DETECTION TEST")
print("="*80)

# Load tes.csv
print("\n[1] Loading tes.csv...")
df = pd.read_csv('dataset/tes.csv', low_memory=False)
print(f"Total data: {len(df)}")

# Find the gibberish row (row 13, index 11 in 0-based)
print("\n[2] Checking row with gibberish text (row 13, index 11)...")
if len(df) > 11:
    row = df.iloc[11]

    print("\n" + "-"*80)
    print("ORIGINAL TEXT:")
    print(row['text'])
    print("-"*80)

    # Preprocess
    print("\n[3] Preprocessing text...")
    text_preprocessed = preprocess_pipeline(row['text'])
    print(f"PREPROCESSED TEXT: {text_preprocessed}")
    print("-"*80)

    # Test gibberish detection
    print("\n[4] Testing gibberish detection on ORIGINAL text...")
    result_original = detect_gibberish(row['text'])

    print(f"\nIs Gibberish: {result_original['is_gibberish']}")
    print(f"Gibberish Score: {result_original['gibberish_score']:.3f}")
    print("\nDetailed Metrics:")
    details = result_original['details']
    print(f"  - Vowel ratio: {details['vowel_ratio']:.3f} (normal: 0.4-0.6)")
    print(f"  - Abnormal vowel: {details['abnormal_vowel']}")
    print(f"  - Valid word ratio: {details['valid_word_ratio']:.3f} (should be high for real)")
    print(f"  - Low valid words: {details['low_valid_words']}")
    print(f"  - Avg word length: {details['avg_word_length']:.2f} (normal: 5-7)")
    print(f"  - Has long words: {details['has_long_words']}")
    print(f"  - Entropy: {details['entropy']:.3f} (high = random)")
    print(f"  - High entropy: {details['high_entropy']}")
    print(f"  - Repeated char ratio: {details['repeated_char_ratio']:.3f}")
    print(f"  - Has repeated chars: {details['has_repeated_chars']}")
    print(f"  - Consonant cluster: {details['consonant_cluster']:.3f}")
    print(f"  - Long consonant cluster: {details['long_consonant_cluster']}")

    # Test on preprocessed text
    print("\n" + "-"*80)
    print("\n[5] Testing gibberish detection on PREPROCESSED text...")
    result_preprocessed = detect_gibberish(text_preprocessed)

    print(f"\nIs Gibberish: {result_preprocessed['is_gibberish']}")
    print(f"Gibberish Score: {result_preprocessed['gibberish_score']:.3f}")
    print("\nDetailed Metrics:")
    details2 = result_preprocessed['details']
    print(f"  - Valid word ratio: {details2['valid_word_ratio']:.3f}")
    print(f"  - Avg word length: {details2['avg_word_length']:.2f}")
    print(f"  - Entropy: {details2['entropy']:.3f}")

    # Test feature extraction
    print("\n" + "-"*80)
    print("\n[6] Testing full feature extraction...")
    test_df = pd.DataFrame({
        'text_preprocessed': [text_preprocessed]
    })
    test_df = extract_text_quality_features(test_df)

    print("\nExtracted Features:")
    for col in test_df.columns:
        if col.startswith('tq_'):
            print(f"  {col}: {test_df[col].iloc[0]}")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    if result_original['is_gibberish']:
        print("[OK] Text quality detector CORRECTLY identifies this as GIBBERISH")
    else:
        print("[FAIL] Text quality detector FAILS to identify this as gibberish")
        print("\nPossible reasons:")
        print("  1. Threshold might be too strict")
        print("  2. After preprocessing, text becomes too short")
        print("  3. Need to use original text instead of preprocessed")

else:
    print(f"ERROR: tes.csv doesn't have line 39 (only {len(df)} rows)")

# Test on other samples
print("\n" + "="*80)
print("\n[7] Testing on all reviews in tes.csv...")
print("="*80)

# Add preprocessing
df['text_preprocessed'] = df['text'].apply(preprocess_pipeline)

# Extract text quality features
df = extract_text_quality_features(df, text_column='text')  # Use original text!

# Show statistics
print(f"\nGibberish Detection Statistics:")
print(f"  Total reviews: {len(df)}")
print(f"  Detected as gibberish: {df['tq_is_gibberish'].sum()}")
print(f"  Mean gibberish score: {df['tq_gibberish_score'].mean():.3f}")

# Show samples with high gibberish score
print(f"\n[8] Reviews with HIGHEST gibberish scores:")
print("-"*80)
top_gibberish = df.nlargest(5, 'tq_gibberish_score')
for idx, row in top_gibberish.iterrows():
    print(f"\nRow {idx+2} (CSV line {idx+2}):")
    print(f"  Text: {row['text'][:80]}...")
    print(f"  Gibberish Score: {row['tq_gibberish_score']:.3f}")
    print(f"  Is Gibberish: {row['tq_is_gibberish']}")
    print(f"  Stars: {row['stars']}")
    print(f"  Valid Word Ratio: {row['tq_valid_word_ratio']:.3f}")

print("\n" + "="*80)
print("DEBUG COMPLETED!")
print("="*80)
