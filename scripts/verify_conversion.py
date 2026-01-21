import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/csv")
DATASETS = {
    "WVS": {
        "data": DATA_DIR / "wvs_trend_1981-2022.csv",
        "labels": DATA_DIR / "wvs_variable_labels.csv"
    },
    "EVS": {
        "data": DATA_DIR / "evs_trend_1981-2017.csv",
        "labels": DATA_DIR / "evs_variable_labels.csv"
    }
}

KEY_VARS = ["s007_01", "S001", "S002", "S002evs", "S024"]
SEARCH_PATTERNS = {
    "wave": ["s002", "wave"],
    "id": ["s007", "respid"]
}

def load_dataset(name):
    """Load both data and labels for a dataset"""
    paths = DATASETS[name]
    return {
        "data": pd.read_csv(paths["data"], low_memory=False),
        "labels": pd.read_csv(paths["labels"])
    }

def print_basic_info(name, dataset):
    """Print basic dataset statistics"""
    df = dataset["data"]
    labels = dataset["labels"]

    print(f"=== {name} ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Labels: {len(labels)}")
    print(f"First few columns: {list(df.columns[:10])}\n")

def find_variables(df, patterns):
    """Find variables matching search patterns (case-insensitive)"""
    matches = []
    for col in df.columns:
        if any(pattern in col.lower() for pattern in patterns):
            matches.append(col)
    return matches

def check_variable_presence(datasets, var_names):
    """Check if variables exist across datasets"""
    print("=== Key Variables Check ===")
    for var in var_names:
        presence = {name: var in ds["data"].columns
                   for name, ds in datasets.items()}
        status = "  ".join(f"{name}: {'Y' if exists else 'N'}"
                          for name, exists in presence.items())
        print(f"{var:12} - {status}")
    print()

def analyze_variable(df, var_name, dataset_name=""):
    """Analyze a specific variable's distribution"""
    if var_name not in df.columns:
        print(f"{var_name} not found in {dataset_name}")
        return

    print(f"=== {var_name} in {dataset_name} ===")

    print(f"Value counts (top 10):")
    print(df[var_name].value_counts(dropna=False).head(10))

    neg_mask = df[var_name] < 0
    if neg_mask.any():
        print(f"\nNegative values (potential missing codes):")
        print(df[var_name][neg_mask].value_counts().sort_index())

    nan_count = df[var_name].isna().sum()
    print(f"\nNaN count: {nan_count:,} ({nan_count/len(df)*100:.1f}%)")
    print()

def main():
    print("Loading datasets...\n")
    datasets = {name: load_dataset(name) for name in DATASETS.keys()}

    for name, ds in datasets.items():
        print_basic_info(name, ds)

    check_variable_presence(datasets, KEY_VARS)

    for pattern_name, patterns in SEARCH_PATTERNS.items():
        print(f"=== Variables matching '{pattern_name}' pattern ===")
        for name, ds in datasets.items():
            matches = find_variables(ds["data"], patterns)
            print(f"{name}: {matches}")
        print()

    analyze_variable(datasets["WVS"]["data"], "A165", "WVS")
    analyze_variable(datasets["WVS"]["data"], "S024", "WVS")

    print("=== Overall Missing Data ===")
    for name, ds in datasets.items():
        df = ds["data"]
        total_cells = df.shape[0] * df.shape[1]
        nan_cells = df.isna().sum().sum()
        print(f"{name}: {nan_cells:,} NaN cells ({nan_cells/total_cells*100:.1f}% of total)")

if __name__ == "__main__":
    main()
