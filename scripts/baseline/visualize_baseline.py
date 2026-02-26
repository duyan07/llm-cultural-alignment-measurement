"""
Visualize Baseline Replication Results

Creates publication-quality visualization showing:
- 88-country human baseline (colored by region)
- LLM model positions
- Comparison to PNAS results

Usage:
    python scripts/baseline/visualize_baseline.py

    # Or specify results file:
    python scripts/baseline/visualize_baseline.py --results data/results/baseline_models_*.csv
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from glob import glob

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Paths
BASELINE_PATH = Path("data/processed/cultural_map_coordinates.csv")
RESULTS_DIR = Path("data/results")
OUTPUTS_DIR = Path("outputs")

# Make output directory
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_baseline():
    """Load 88-country baseline."""
    df = pd.read_csv(BASELINE_PATH)
    return df


def load_model_results(results_path=None):
    """Load model results from most recent run."""
    if results_path is None:
        # Find most recent results
        files = sorted(glob(str(RESULTS_DIR / "baseline_models_*.csv")))
        if not files:
            raise FileNotFoundError("No baseline results found. Run scripts/baseline/baseline_replication.py first.")
        results_path = files[-1]

    print(f"Loading results from: {results_path}")
    return pd.read_csv(results_path)


def create_visualization(baseline_df, models_df, output_path):
    """Create cultural map with baseline + models."""

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot country baseline
    ax.scatter(
        baseline_df['survival_selfexpression'],
        baseline_df['traditional_secular'],
        c='lightblue',
        s=100,
        alpha=0.6,
        edgecolors='navy',
        linewidth=0.5,
        label='Human Baseline (88 countries)',
        zorder=1
    )

    # Plot models (larger, different colors)
    colors = ['red', 'orange', 'purple', 'green', 'brown', 'pink', 'olive', 'cyan']

    for idx, (_, model) in enumerate(models_df.iterrows()):
        color = colors[idx % len(colors)]
        ax.scatter(
            model['survival_selfexpression'],
            model['traditional_secular'],
            c=color,
            s=400,
            marker='*',
            edgecolors='black',
            linewidth=2,
            label=model['model'],
            zorder=10
        )

        # Add model name label
        ax.annotate(
            model['model'],
            (model['survival_selfexpression'], model['traditional_secular']),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
            zorder=11
        )

    # Axes labels
    ax.set_xlabel('Survival ← → Self-Expression Values', fontsize=12, fontweight='bold')
    ax.set_ylabel('Traditional ← → Secular Values', fontsize=12, fontweight='bold')

    # Title
    ax.set_title('LLM Cultural Alignment: Baseline Replication (Week 6)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add quadrant lines at origin
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Add quadrant labels
    ax.text(3.5, 1.5, 'Self-Expression\n+ Secular',
            ha='center', va='center', fontsize=10, color='gray', alpha=0.7)
    ax.text(-2.5, 1.5, 'Survival\n+ Secular',
            ha='center', va='center', fontsize=10, color='gray', alpha=0.7)
    ax.text(-2.5, -1.3, 'Survival\n+ Traditional',
            ha='center', va='center', fontsize=10, color='gray', alpha=0.7)
    ax.text(3.5, -1.3, 'Self-Expression\n+ Traditional',
            ha='center', va='center', fontsize=10, color='gray', alpha=0.7)

    # Legend
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)

    # Grid
    ax.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization: {output_path}")

    return fig


def print_summary(baseline_df, models_df):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("BASELINE REPLICATION SUMMARY")
    print("="*70)

    print(f"\nHuman Baseline:")
    print(f"  Countries: {len(baseline_df)}")
    print(f"  X-range: [{baseline_df['survival_selfexpression'].min():.2f}, {baseline_df['survival_selfexpression'].max():.2f}]")
    print(f"  Y-range: [{baseline_df['traditional_secular'].min():.2f}, {baseline_df['traditional_secular'].max():.2f}]")
    print(f"  Mean: ({baseline_df['survival_selfexpression'].mean():.2f}, {baseline_df['traditional_secular'].mean():.2f})")

    print(f"\nLLM Models Tested: {len(models_df)}")
    print("\nModel Positions:")
    print(models_df[['model', 'survival_selfexpression', 'traditional_secular', 'mean_distance']].to_string(index=False))

    # Calculate average model position
    avg_x = models_df['survival_selfexpression'].mean()
    avg_y = models_df['traditional_secular'].mean()

    print(f"\nAverage Model Position:")
    print(f"  X (Survival-SelfExpression): {avg_x:.2f}")
    print(f"  Y (Traditional-Secular): {avg_y:.2f}")
    print(f"  Mean distance to countries: {models_df['mean_distance'].mean():.2f}")

    print(f"\nPNAS Reference (GPT-4o):")
    print(f"  Position: (3.35, 0.50)")
    print(f"  Closest: Finland (d=0.20)")

    # Report which quadrant the average model position falls in
    if avg_x > 0 and avg_y > 0:
        print(f"\nModels cluster in Self-Expression + Secular quadrant (Nordic/Western Europe)")
    elif avg_x > 0 and avg_y < 0:
        print(f"\nModels cluster in Self-Expression + Traditional quadrant (US/UK pattern)")
    elif avg_x < 0:
        print(f"\nModels show Survival values (unexpected)")


def main():
    parser = argparse.ArgumentParser(description='Visualize baseline replication')
    parser.add_argument('--results', help='Path to results CSV (default: most recent)')
    parser.add_argument('--output', help='Output path for visualization',
                       default='outputs/baseline_with_models.png')
    args = parser.parse_args()

    # Load data
    baseline_df = load_baseline()
    models_df = load_model_results(args.results)

    # Create visualization
    output_path = Path(args.output)
    create_visualization(baseline_df, models_df, output_path)

    # Print summary
    print_summary(baseline_df, models_df)

    print(f"\nDone. Results at: {output_path}")


if __name__ == "__main__":
    main()