import pandas as pd
import os

def load_csv_file(file_path):
    """Load CSV file and return DataFrame"""
    print(f"Loading CSV file from: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)

    print("Loaded DataFrame successfully")
    print(f"    Rows: {len(df):,}")
    print(f"    Columns: {len(df.columns):,}")
    print(f"    Countries: {df['S003'].nunique() if 'S003' in df.columns else 'N/A'}")

    return df

def load_labels(labels_path):
    """Load variable labels from CSV"""
    if os.path.exists(labels_path):
        labels_df = pd.read_csv(labels_path)
        return dict(zip(labels_df['variable'], labels_df['label']))
    return {}

def explore_labels(labels):
    print("\n Label Information:")
    print(f"    Total labels: {len(labels)}")

    print("\n Sample Labels:")
    for i, (var, label) in enumerate(labels.items()):
        if i >= 10:
            break
        print(f"    {var}: {label}")

def explore_data_structure(df):
    print("\n Data Structure Information:")

    country_cols = [col for col in df.columns if "country" in col.lower() or col in ["S003", "S003A"]]
    print(f"    Country-related columns: {country_cols}")

    wave_cols = [col for col in df.columns if "wave" in col.lower() or col in ["S001", "S002"]]
    print(f"    Wave/Year columns: {wave_cols}")

def main():
    wvs_csv = "data/raw/csv/wvs_trend_1981-2022.csv"
    wvs_labels = "data/raw/csv/wvs_variable_labels.csv"

    evs_csv = "data/raw/csv/evs_trend_1981-2017.csv"
    evs_labels = "data/raw/csv/evs_variable_labels.csv"

    try:
        wvs_df = load_csv_file(wvs_csv)
        wvs_label_dict = load_labels(wvs_labels)
        print("\n" + "="*50)
        print("WVS TREND FILE:")
        explore_labels(wvs_label_dict)
        explore_data_structure(wvs_df)
    except Exception as e:
        print(f"Error loading WVS data: {e}")

    try:
        evs_csv_df = load_csv_file(evs_csv)
        evs_label_dict = load_labels(evs_labels)
        print("\n" + "="*50)
        print("EVS TREND FILE:")
        explore_labels(evs_label_dict)
        explore_data_structure(evs_csv_df)
    except Exception as e:
        print(f"Error loading EVS data: {e}")

if __name__ == "__main__":
    main()