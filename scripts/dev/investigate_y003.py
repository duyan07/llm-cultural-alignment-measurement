"""
Investigate Y003 (Autonomy Index) Missing Data

Usage: python investigate_y003.py path/to/ivs_2005-2022.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def load_data(path):
    """Load IVS data."""
    print(f"Loading IVS data from {path}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns\n")
    return df

def analyze_y003_structure(df):
    """Understand how Y003 is structured in the dataset."""
    print("="*70)
    print("Y003 STRUCTURE ANALYSIS")
    print("="*70)

    # Find all Y003-related columns
    y003_cols = [col for col in df.columns if 'Y003' in col.upper()]

    print(f"\nFound {len(y003_cols)} Y003-related columns:")
    for col in y003_cols:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null:,} non-null values ({non_null/len(df)*100:.1f}%)")

    if 'Y003' in df.columns:
        print("\n\nY003 value distribution (top 20):")
        print(df['Y003'].value_counts(dropna=False).head(20))

        print(f"\n\nY003 data type: {df['Y003'].dtype}")

        # Check for negative values (missing codes)
        if df['Y003'].dtype in ['int64', 'float64']:
            neg_mask = df['Y003'] < 0
            print(f"\nNegative values (missing codes): {neg_mask.sum():,} ({neg_mask.sum()/len(df)*100:.1f}%)")
            if neg_mask.any():
                print("Negative value codes:")
                print(df.loc[neg_mask, 'Y003'].value_counts().sort_index())

    return y003_cols

def analyze_by_wave(df):
    """Check Y003 availability by survey wave."""
    print("\n" + "="*70)
    print("Y003 AVAILABILITY BY WAVE")
    print("="*70)

    # WVS waves
    if 's002' in df.columns:
        print("\nWVS Waves (s002):")
        for wave in sorted(df['s002'].dropna().unique()):
            wave_df = df[df['s002'] == wave]
            y003_available = wave_df['Y003'].notna().sum()
            y003_valid = ((wave_df['Y003'] >= 0) & wave_df['Y003'].notna()).sum()
            print(f"  Wave {int(wave)}: {len(wave_df):,} total, {y003_valid:,} valid ({y003_valid/len(wave_df)*100:.1f}%)")

    # EVS waves
    if 'S002EVS' in df.columns:
        print("\nEVS Waves (S002EVS):")
        for wave in sorted(df['S002EVS'].dropna().unique()):
            wave_df = df[df['S002EVS'] == wave]
            y003_available = wave_df['Y003'].notna().sum()
            y003_valid = ((wave_df['Y003'] >= 0) & wave_df['Y003'].notna()).sum()
            print(f"  Wave {int(wave)}: {len(wave_df):,} total, {y003_valid:,} valid ({y003_valid/len(wave_df)*100:.1f}%)")

def compare_with_without_y003(df):
    """Compare which countries are lost when requiring Y003."""
    print("\n" + "="*70)
    print("IMPACT OF Y003 REQUIREMENT")
    print("="*70)

    # All 10 questions
    all_questions = ['A008', 'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006', 'Y002', 'Y003']

    # 9 questions without Y003
    nine_questions = ['A008', 'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006', 'Y002']

    # Clean data: remove negative values
    df_clean = df.copy()
    for q in all_questions:
        if q in df_clean.columns:
            df_clean.loc[df_clean[q] < 0, q] = np.nan

    # Complete cases for all 10
    complete_10 = df_clean.dropna(subset=all_questions)
    countries_10 = (complete_10['S024'] // 10).nunique()

    # Complete cases for 9 (without Y003)
    complete_9 = df_clean.dropna(subset=nine_questions)
    countries_9 = (complete_9['S024'] // 10).nunique()

    print(f"\nWith all 10 questions:")
    print(f"  Valid responses: {len(complete_10):,}")
    print(f"  Unique countries: {countries_10}")

    print(f"\nWith 9 questions (excluding Y003):")
    print(f"  Valid responses: {len(complete_9):,}")
    print(f"  Unique countries: {countries_9}")

    print(f"\nImpact of Y003 requirement:")
    print(f"  Lost responses: {len(complete_9) - len(complete_10):,} ({(len(complete_9) - len(complete_10))/len(complete_9)*100:.1f}%)")
    print(f"  Lost countries: {countries_9 - countries_10}")

    # Which countries are lost?
    countries_in_9 = set((complete_9['S024'] // 10).unique())
    countries_in_10 = set((complete_10['S024'] // 10).unique())
    lost_countries = sorted(countries_in_9 - countries_in_10)

    if lost_countries:
        print(f"\nCountries lost due to Y003 requirement ({len(lost_countries)}):")
        for country in lost_countries[:20]:  # Show first 20
            count_9 = len(complete_9[complete_9['S024'] // 10 == country])
            print(f"  Country {int(country)}: {count_9:,} valid responses on 9 questions")
        if len(lost_countries) > 20:
            print(f"  ... and {len(lost_countries) - 20} more")

def main():
    if len(sys.argv) < 2:
        print("Usage: python investigate_y003.py path/to/ivs_2005-2022.csv")
        sys.exit(1)

    ivs_path = sys.argv[1]
    if not Path(ivs_path).exists():
        print(f"Error: File not found: {ivs_path}")
        sys.exit(1)

    df = load_data(ivs_path)

    # Run analyses
    analyze_y003_structure(df)
    analyze_by_wave(df)
    compare_with_without_y003(df)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()