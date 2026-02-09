"""
Generate Baseline Cultural Map

This script:
1. Loads the merged IVS dataset
2. Applies PCA to generate cultural map coordinates
3. Saves coordinates and creates visualization

This establishes the "ground truth" baseline that we'll compare LLM outputs against.
"""

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.cultural_map import CulturalMapGenerator

# File paths
IVS_PATH = Path("data/processed/ivs_2005-2022.csv")
OUTPUT_DIR = Path("data/processed/")
COORDS_OUTPUT = OUTPUT_DIR / "cultural_map_coordinates.csv"
VIZ_OUTPUT = OUTPUT_DIR / "cultural_map_baseline.png"


def load_ivs_data():
    """Load the merged IVS dataset."""
    print(f"Loading IVS data from {IVS_PATH}...")

    if not IVS_PATH.exists():
        print(f"\nERROR: IVS dataset not found at {IVS_PATH}")
        print("Please run scripts/build_ivs.py first to create the merged dataset.")
        sys.exit(1)

    df = pd.read_csv(IVS_PATH, low_memory=False)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    return df


def visualize_cultural_map(coords_df, output_path):
    """
    Create a visualization of the cultural map.

    Args:
        coords_df: DataFrame with country coordinates
        output_path: Where to save the plot
    """
    print("\nCreating cultural map visualization...")

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot points
    scatter = ax.scatter(
        coords_df['survival_selfexpression'],
        coords_df['traditional_secular'],
        s=100,
        alpha=0.6,
        c='steelblue',
        edgecolors='black',
        linewidth=0.5
    )

    # Add country labels (just country codes for now)
    for _, row in coords_df.iterrows():
        ax.annotate(
            f"{int(row['country_code'])}",
            (row['survival_selfexpression'], row['traditional_secular']),
            fontsize=6,
            alpha=0.7,
            ha='center'
        )

    # Labels and title
    ax.set_xlabel('Survival ← → Self-Expression Values', fontsize=12, fontweight='bold')
    ax.set_ylabel('Traditional ← → Secular Values', fontsize=12, fontweight='bold')
    ax.set_title('Inglehart-Welzel Cultural Map (Baseline from IVS Data)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add gridlines
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add center lines
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

    # Also show some stats
    print("\nCultural Map Statistics:")
    print(f"  X-axis (Survival-SelfExpression) range: [{coords_df['survival_selfexpression'].min():.2f}, {coords_df['survival_selfexpression'].max():.2f}]")
    print(f"  Y-axis (Traditional-Secular) range: [{coords_df['traditional_secular'].min():.2f}, {coords_df['traditional_secular'].max():.2f}]")
    print(f"  Mean X: {coords_df['survival_selfexpression'].mean():.2f}")
    print(f"  Mean Y: {coords_df['traditional_secular'].mean():.2f}")


def print_summary_stats(coords_df):
    """Print summary statistics about the cultural map."""
    print("\n" + "="*60)
    print("CULTURAL MAP SUMMARY")
    print("="*60)

    print(f"\nTotal countries mapped: {len(coords_df)}")
    print(f"Total respondents: {coords_df['n_respondents'].sum():,}")

    print("\nTop 10 countries by number of respondents:")
    top_10 = coords_df.nlargest(10, 'n_respondents')[['country_code', 'n_respondents']]
    for _, row in top_10.iterrows():
        print(f"  Country {int(row['country_code'])}: {int(row['n_respondents']):,} respondents")

    # Quadrant analysis
    print("\nQuadrant distribution:")
    q1 = len(coords_df[(coords_df['survival_selfexpression'] > 0) & (coords_df['traditional_secular'] > 0)])
    q2 = len(coords_df[(coords_df['survival_selfexpression'] < 0) & (coords_df['traditional_secular'] > 0)])
    q3 = len(coords_df[(coords_df['survival_selfexpression'] < 0) & (coords_df['traditional_secular'] < 0)])
    q4 = len(coords_df[(coords_df['survival_selfexpression'] > 0) & (coords_df['traditional_secular'] < 0)])

    print(f"  Q1 (Self-Expression + Secular): {q1} countries")
    print(f"  Q2 (Survival + Secular): {q2} countries")
    print(f"  Q3 (Survival + Traditional): {q3} countries")
    print(f"  Q4 (Self-Expression + Traditional): {q4} countries")


def main():
    print("="*60)
    print("GENERATING BASELINE CULTURAL MAP FROM IVS DATA")
    print("="*60)

    # Load data
    ivs_df = load_ivs_data()

    # Generate cultural map coordinates
    generator = CulturalMapGenerator(ivs_df)
    coords_df = generator.fit()

    # Save coordinates
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generator.save_coordinates(COORDS_OUTPUT)

    # Print summary
    print_summary_stats(coords_df)

    # Create visualization
    visualize_cultural_map(coords_df, VIZ_OUTPUT)

    print("\n" + "="*60)
    print("BASELINE CULTURAL MAP GENERATION COMPLETE")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  - Coordinates: {COORDS_OUTPUT}")
    print(f"  - Visualization: {VIZ_OUTPUT}")

if __name__ == "__main__":
    main()