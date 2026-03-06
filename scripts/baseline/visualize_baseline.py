"""
Visualize Baseline Replication Results

Creates publication-quality visualization showing:
- 88-country human baseline colored by cultural zone
- Country ISO-3 labels on every dot
- LLM model positions with annotations
- Tone-labeled title and tone-specific output filenames

Usage:
    # Generate maps for all tones that have results (default):
    python scripts/baseline/visualize_baseline.py

    # Specific tone only:
    python scripts/baseline/visualize_baseline.py --tone friendly

    # Specific results file:
    python scripts/baseline/visualize_baseline.py --results data/results/baseline_models_combative_20260226_010108.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import re
from glob import glob

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.geo_data import (
    COUNTRY_NAMES, ZONE_COLORS, ISO3_TO_ZONE,
    load_iso3_lookup, quadrant_label, typical_countries,
)

# Paths
BASELINE_PATH = Path("data/processed/cultural_map_coordinates.csv")
IVS_PATH = Path("data/processed/ivs_2005-2022.csv")
RESULTS_DIR = Path("data/results")
OUTPUTS_DIR = Path("outputs")

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_baseline():
    """Load 88-country baseline and annotate with ISO-3, name, and cultural zone."""
    df = pd.read_csv(BASELINE_PATH)
    iso_lookup = load_iso3_lookup(IVS_PATH)
    df['iso3'] = df['country_code'].map(iso_lookup).fillna('???')
    df['name'] = df['iso3'].map(COUNTRY_NAMES).fillna(df['iso3'])
    df['zone'] = df['iso3'].map(ISO3_TO_ZONE).fillna('Other')
    return df


def load_model_results(results_path=None, tone=None):
    """Load model results — from a specific file, by tone, or from the most recent run.

    Returns (models_df, tone, timestamp) where timestamp matches the source filename.
    """
    if results_path:
        print(f"Loading results from: {results_path}")
        return pd.read_csv(results_path), _tone_from_path(results_path), _timestamp_from_path(results_path)

    pattern = f"baseline_models_{tone}_*.csv" if tone else "baseline_models_*.csv"
    files = sorted(glob(str(RESULTS_DIR / pattern)))
    if not files:
        raise FileNotFoundError(
            f"No baseline results found matching '{pattern}'. "
            "Run scripts/baseline/baseline_replication.py first."
        )
    results_path = files[-1]
    print(f"Loading results from: {results_path}")
    return pd.read_csv(results_path), _tone_from_path(results_path), _timestamp_from_path(results_path)


def _tone_from_path(path):
    """Extract tone name from a result filename, e.g. baseline_models_friendly_*.csv -> 'friendly'."""
    m = re.search(r'baseline_models_([a-z]+)_\d{8}', str(path))
    return m.group(1) if m else 'standard'


def _timestamp_from_path(path):
    """Extract timestamp from a result filename, e.g. baseline_models_standard_20260303_121534.csv -> '20260303_121534'."""
    m = re.search(r'(\d{8}_\d{6})', str(path))
    return m.group(1) if m else 'unknown'


# ── Visualization ─────────────────────────────────────────────────────────────

def create_visualization(baseline_df, models_df, output_path, tone='standard'):
    """Create the cultural map: country dots + LLM stars + country labels."""

    _, ax = plt.subplots(figsize=(16, 11))

    # ── Country dots colored by cultural zone ──
    zone_handles = []
    for zone, color in ZONE_COLORS.items():
        zone_data = baseline_df[baseline_df['zone'] == zone]
        if zone_data.empty:
            continue
        ax.scatter(
            zone_data['survival_selfexpression'],
            zone_data['traditional_secular'],
            c=color,
            s=80,
            alpha=0.85,
            edgecolors='white',
            linewidth=0.4,
            zorder=2,
        )
        zone_handles.append(mpatches.Patch(color=color, label=zone))

    # ── ISO-3 labels on every country dot ──
    for _, row in baseline_df.iterrows():
        ax.annotate(
            row['iso3'],
            xy=(row['survival_selfexpression'], row['traditional_secular']),
            xytext=(3, 3),
            textcoords='offset points',
            fontsize=6.5,
            color='#222222',
            zorder=3,
        )

    # ── LLM model stars ──
    model_colors = ['#e41a1c', '#ff7f00', '#984ea3', '#4daf4a',
                    '#377eb8', '#a65628', '#f781bf', '#999999']
    model_handles = []

    for idx, (_, model) in enumerate(models_df.iterrows()):
        color = model_colors[idx % len(model_colors)]
        ax.scatter(
            model['survival_selfexpression'],
            model['traditional_secular'],
            c=color,
            s=450,
            marker='*',
            edgecolors='black',
            linewidth=1.5,
            zorder=10,
        )
        ax.annotate(
            model['model'],
            xy=(model['survival_selfexpression'], model['traditional_secular']),
            xytext=(10, 8),
            textcoords='offset points',
            fontsize=8.5,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=color, alpha=0.75,
                      edgecolor='black', linewidth=0.8),
            zorder=11,
        )
        model_handles.append(
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=color,
                       markeredgecolor='black', markersize=12, label=model['model'])
        )

    # ── Axes ──
    ax.set_xlabel('Survival  ←                    →  Self-Expression Values',
                  fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Traditional  ←               →  Secular Values',
                  fontsize=12, fontweight='bold', labelpad=8)

    ax.set_title(
        f'LLM Cultural Alignment — Baseline Replication  [{tone.capitalize()} tone]',
        fontsize=14, fontweight='bold', pad=16,
    )

    # ── Quadrant reference lines and labels ──
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.6, alpha=0.4)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.6, alpha=0.4)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    pad = 0.15

    ax.text(x_max - pad, y_max - pad, 'Self-Expression\n+ Secular',
            ha='right', va='top', fontsize=9, color='gray', alpha=0.6, style='italic')
    ax.text(x_min + pad, y_max - pad, 'Survival\n+ Secular',
            ha='left', va='top', fontsize=9, color='gray', alpha=0.6, style='italic')
    ax.text(x_min + pad, y_min + pad, 'Survival\n+ Traditional',
            ha='left', va='bottom', fontsize=9, color='gray', alpha=0.6, style='italic')
    ax.text(x_max - pad, y_min + pad, 'Self-Expression\n+ Traditional',
            ha='right', va='bottom', fontsize=9, color='gray', alpha=0.6, style='italic')

    # ── Two-column legend: zones left, models right ──
    legend_zones = ax.legend(
        handles=zone_handles,
        title='Cultural Zone',
        loc='upper left',
        fontsize=7.5,
        title_fontsize=8,
        framealpha=0.9,
    )
    ax.add_artist(legend_zones)
    ax.legend(
        handles=model_handles,
        title='LLM Models',
        loc='lower right',
        fontsize=8,
        title_fontsize=9,
        framealpha=0.9,
    )

    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {output_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(baseline_df, models_df, tone='standard'):
    """Print a detailed summary: model positions, closest countries, quadrant context."""
    print("\n" + "=" * 70)
    print(f"BASELINE REPLICATION SUMMARY  [{tone.upper()} tone]")
    print("=" * 70)

    print(f"\nHuman Baseline ({len(baseline_df)} countries):")
    print(f"  X range : [{baseline_df['survival_selfexpression'].min():.2f}, "
          f"{baseline_df['survival_selfexpression'].max():.2f}]")
    print(f"  Y range : [{baseline_df['traditional_secular'].min():.2f}, "
          f"{baseline_df['traditional_secular'].max():.2f}]")
    print(f"  Centroid: ({baseline_df['survival_selfexpression'].mean():.2f}, "
          f"{baseline_df['traditional_secular'].mean():.2f})")

    print(f"\nLLM Models Tested: {len(models_df)}")
    print("\n" + "-" * 70)

    for _, model in models_df.iterrows():
        x = model['survival_selfexpression']
        y = model['traditional_secular']

        distances = []
        for _, country in baseline_df.iterrows():
            d = np.sqrt((x - country['survival_selfexpression'])**2 +
                        (y - country['traditional_secular'])**2)
            distances.append((d, country['iso3'], country['name'], country['zone']))
        distances.sort()

        print(f"\n  Model : {model['model']}  (tone: {model.get('tone', tone)})")
        print(f"  Position  : ({x:.3f}, {y:.3f})")
        print(f"  Quadrant  : {quadrant_label(x, y)}")
        print(f"  Typical of: {typical_countries(x, y)}")
        print(f"  Closest countries:")
        for d, iso3, name, zone in distances[:5]:
            print(f"    {iso3:3s}  {name:<20s}  d={d:.3f}  [{zone}]")
        print(f"  Mean dist to all countries: {model.get('mean_distance', float('nan')):.3f}")

    print("\n" + "-" * 70)
    avg_x = models_df['survival_selfexpression'].mean()
    avg_y = models_df['traditional_secular'].mean()
    print(f"\nAverage model position: ({avg_x:.2f}, {avg_y:.2f})")
    print(f"  Quadrant  : {quadrant_label(avg_x, avg_y)}")
    print(f"  Typical of: {typical_countries(avg_x, avg_y)}")
    print(f"\nPNAS Reference (GPT-4o): (3.35, 0.50) — closest: Finland (d≈0.20)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize baseline replication')
    parser.add_argument('--results', help='Path to a specific baseline_models_*.csv')
    parser.add_argument('--tone', help='Tone to load (standard/friendly/combative)')
    parser.add_argument('--output', help='Override output image path (single-tone only)')
    args = parser.parse_args()

    baseline_df = load_baseline()

    if args.results or args.tone:
        # Single-file or single-tone mode
        models_df, detected_tone, timestamp = load_model_results(args.results, args.tone)
        tone = args.tone or detected_tone
        output_path = (Path(args.output) if args.output else
                       OUTPUTS_DIR / f"baseline_with_models_{tone}_{timestamp}.png")
        create_visualization(baseline_df, models_df, output_path, tone=tone)
        print_summary(baseline_df, models_df, tone=tone)
        print(f"\nDone. Image saved to: {output_path}")
    else:
        # Default: generate a map for every tone that has results
        generated = []
        for tone in ['standard', 'friendly', 'combative']:
            try:
                models_df, _, timestamp = load_model_results(tone=tone)
                output_path = OUTPUTS_DIR / f"baseline_with_models_{tone}_{timestamp}.png"
                create_visualization(baseline_df, models_df, output_path, tone=tone)
                print_summary(baseline_df, models_df, tone=tone)
                generated.append(str(output_path))
            except FileNotFoundError:
                print(f"Skipping '{tone}' tone — no results file found.")
        print(f"\nDone. {len(generated)} image(s) saved:")
        for p in generated:
            print(f"  {p}")


if __name__ == "__main__":
    main()
