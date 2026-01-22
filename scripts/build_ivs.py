from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import IVSBuilder

WVS_PATH = Path("data/raw/csv/wvs_trend_1981-2022.csv")
EVS_PATH = Path("data/raw/csv/evs_trend_1981-2017.csv")
OUTPUT_PATH = Path("data/processed/ivs_2005-2022.csv")

def main():
    print("Building Integrated Values Survey (IVS) dataset...")
    print("="*60)

    builder = IVSBuilder(WVS_PATH, EVS_PATH)
    ivs = builder.build_ivs(
        wvs_waves=[5, 6, 7],    # 2005-2022
        evs_waves=[4, 5]        # 2008-2017
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ivs.to_csv(OUTPUT_PATH, index=False)

    print(f"\n IVS dataset saved to {OUTPUT_PATH}")
    print(f"  Shape: {ivs.shape[0]:,} rows x {ivs.shape[1]} columns")

    metadata = {
        "n_rows": len(ivs),
        "n_cols": len(ivs.columns),
        "n_countries": builder.get_unique_countries(ivs),
        "wvs_waves": [5, 6, 7],
        "evs_waves": [4, 5]
    }

    import json
    metadata_path = OUTPUT_PATH.with_suffix(".metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f" Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main()
