"""
Data loader for building Integrated Values Survey (IVS) dataset.
Merges World Values Survey (WVS) and European Values Study (EVS) data.
"""

import pandas as pd


class IVSBuilder:
    """
    Builder class for merging WVS and EVS datasets into a single IVS dataset.
    Handles wave filtering, data validation, and standardization.
    """

    def __init__(self, wvs_path, evs_path):
        """
        Initialize builder with paths to WVS and EVS data files.

        Args:
            wvs_path: Path to WVS CSV file
            evs_path: Path to EVS CSV file
        """
        self.wvs_path = wvs_path
        self.evs_path = evs_path

    def load_wvs(self, waves=[5, 6, 7]):
        """
        Load and filter WVS data for specified waves.

        Args:
            waves: List of wave numbers to include (default: [5, 6, 7] for 2005-2022)

        Returns:
            DataFrame filtered to specified waves
        """
        print(f"Loading WVS data from {self.wvs_path} for waves {waves}...")
        df = pd.read_csv(self.wvs_path, low_memory=False)
        print(f"Successfully loaded WVS data!")

        # Verify wave variable exists
        if "s002" not in df.columns:
            raise ValueError("Wave variable 's002' not found in WVS data.")

        # Filter to requested waves
        df_filtered = df[df["s002"].isin(waves)].copy()

        print(f"  Total rows: {len(df):,}")
        print(f"  After filtering to waves {waves}: {len(df_filtered):,}")
        print(f"  Wave distribution:")
        print(df_filtered["s002"].value_counts().sort_index())

        return df_filtered

    def load_evs(self, waves=[4, 5]):
        """
        Load and filter EVS data for specified waves.

        Args:
            waves: List of wave numbers to include (default: [4, 5] for 2008-2017)

        Returns:
            DataFrame filtered to specified waves
        """
        print(f"\nLoading EVS data from {self.evs_path} for waves {waves}...")
        df = pd.read_csv(self.evs_path, low_memory=False)
        print("Successfully loaded EVS data!")

        # Verify wave variable exists (EVS uses different variable name)
        if "S002EVS" not in df.columns:
            raise ValueError("Wave variable 'S002EVS' not found in EVS data.")

        # Filter to requested waves
        df_filtered = df[df["S002EVS"].isin(waves)].copy()

        print(f"  Total rows: {len(df):,}")
        print(f"  After filtering to waves {waves}: {len(df_filtered):,}")
        print(f"  Wave distribution:")
        print(df_filtered["S002EVS"].value_counts().sort_index())

        return df_filtered

    def merge(self, wvs_df, evs_df):
        """
        Merge WVS and EVS datasets into a single IVS dataset.

        Args:
            wvs_df: Filtered WVS DataFrame
            evs_df: Filtered EVS DataFrame

        Returns:
            Merged IVS DataFrame sorted by respondent ID
        """
        print("\nMerging WVS and EVS datasets...")

        # Verify respondent ID exists in both datasets
        if "S007_01" not in wvs_df.columns or "S007_01" not in evs_df.columns:
            raise ValueError("Merge key 'S007_01' not found in one or both of the datasets.")

        # Concatenate datasets (EVS first to match original methodology)
        ivs = pd.concat([evs_df, wvs_df], ignore_index=True)

        # Sort by respondent ID for consistency
        ivs.sort_values("S007_01", inplace=True, ignore_index=True)

        print(f"  Merged dataset: {len(ivs):,} rows")
        print(f"\n  Survey source distribution (S001):")
        print(ivs["S001"].value_counts().sort_index())

        return ivs

    def get_unique_countries(self, ivs_df):
        """Get count of unique country-wave combinations in dataset."""
        return ivs_df["S024"].nunique()

    def build_ivs(self, wvs_waves=[5, 6, 7], evs_waves=[4, 5]):
        """
        Execute full pipeline to build IVS dataset.

        Args:
            wvs_waves: List of WVS wave numbers to include
            evs_waves: List of EVS wave numbers to include

        Returns:
            Merged and validated IVS DataFrame
        """
        wvs = self.load_wvs(wvs_waves)
        evs = self.load_evs(evs_waves)
        ivs = self.merge(wvs, evs)

        self._validate(ivs)
        return ivs

    def _validate(self, ivs):
        """
        Run validation checks on merged IVS dataset.
        Checks for duplicates, expected country counts, and missing data.
        """
        print("\n" + "="*60)
        print("VALIDATION")
        print("="*60)

        # Duplicate check
        n_duplicates = ivs["S007_01"].duplicated().sum()
        status = "PASS" if n_duplicates == 0 else "FAIL"
        print(f"Duplicate respondent IDs (S007_01): {n_duplicates} {status}")

        # Check country-wave combinations
        n_country_waves = ivs["S024"].nunique()
        print(f"Country-wave combinations: {n_country_waves}")

        # Extract unique countries from S024 (S024 = country * 10 + wave)
        country_codes = (ivs["S024"] // 10).astype(int)
        n_countries = country_codes.nunique()
        status = "PASS" if 100 <= n_countries <= 120 else "FAIL"
        print(f"Unique country-wave codes: {n_countries} {status}")
        print(f"  (Expected between ~112 based on PNAS paper)")

        evs_count = (ivs["S001"] == 1).sum()
        wvs_count = (ivs["S001"] == 2).sum()
        total = len(ivs)
        print(f"\nSurvey composition:")
        print(f"  EVS (S001=1): {evs_count:,} ({evs_count/total*100:.1f}%)")
        print(f"  WVS (S001=2): {wvs_count:,} ({wvs_count/total*100:.1f}%)")

        print(f"\nMissing data in key variables")
        for var in ["S007_01", "S024", "S001"]:
            n_missing = ivs[var].isna().sum()
            status = "PASS" if n_missing == 0 else "FAIL"
            print(f"  {var}: {n_missing:,} {status}")

        print("="*60)