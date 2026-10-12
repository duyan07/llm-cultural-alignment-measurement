"""Plots the 88-country human baseline alongside queried LLM positions on the cultural map, one figure per tone.

The map is cropped vertically; models beyond the crop are pinned to the edge and listed in an inset.

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

from src.geo_data import ZONE_COLORS, quadrant_label, typical_countries
from src.viz_common import API_MODELS, load_baseline, model_label_inline

# Paths
RESULTS_DIR = Path("data/results/baseline")
OUTPUTS_DIR = Path("outputs/baseline")

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model_results(results_path=None, tone=None, model_set=None):
    """Load model results — from a specific file, by tone, or from the most recent run.

    Args:
        results_path: Explicit CSV path; bypasses all filtering.
        tone: Tone to load (standard/friendly/combative).
        model_set: If given, only load files whose name contains _{model_set}_.

    Returns (models_df, tone, timestamp) where timestamp matches the source filename.
    """
    if results_path:
        print(f"Loading results from: {results_path}")
        return pd.read_csv(results_path), _tone_from_path(results_path), _timestamp_from_path(results_path)

    pattern = f"baseline_models_{tone}_*.csv" if tone else "baseline_models_*.csv"
    files = sorted(glob(str(RESULTS_DIR / pattern)))
    files = _filter_by_model_set([Path(f) for f in files], model_set)
    if not files:
        tag = f" with model_set='{model_set}'" if model_set else ""
        raise FileNotFoundError(
            f"No baseline results found matching '{pattern}'{tag}. "
            "Run scripts/baseline/baseline_replication.py first."
        )
    # When no model_set filter is applied, merge all files so open + api
    # models appear together, newer runs superseding older ones. When filtered
    # to a specific set, use latest only.
    if model_set:
        files = [files[-1]]
    files = sorted(files, key=_timestamp_from_path)
    df = pd.DataFrame()
    for f in files:
        print(f"Loading results from: {f}")
        df = _merge_run(df, pd.read_csv(f), _model_set_from_path(f))
    return df, _tone_from_path(files[-1]), _timestamp_from_path(files[-1])


def _merge_run(df, run_df, run_set):
    """Overlay a newer run on earlier rows: the run replaces every earlier row for
    the models it covers ('open' = non-API models, 'api' = API models, 'all' = both),
    so a model absent from the newer run (e.g. unprojectable in a rerun) is dropped."""
    if df.empty:
        return run_df
    is_api = df['model'].isin(API_MODELS)
    covered = {'open': ~is_api, 'api': is_api}.get(run_set, pd.Series(True, index=df.index))
    return pd.concat([df[~covered], run_df], ignore_index=True)


def _model_set_from_path(path):
    """Extract model set ('all'/'open'/'api') from a result filename; old-format files count as 'all'."""
    m = re.search(r'baseline_models_[a-z]+_(all|open|api)_', str(path))
    return m.group(1) if m else 'all'


def _filter_by_model_set(paths, model_set):
    """Keep only paths containing _{model_set}_ if model_set is specified."""
    if not model_set:
        return paths
    return [p for p in paths if f'_{model_set}_' in p.name]


def _tone_from_path(path):
    """Extract tone name from a result filename.

    Handles both old format (baseline_models_standard_20260306.csv)
    and new format (baseline_models_standard_all_20260422.csv).
    """
    m = re.search(r'baseline_models_([a-z]+)_', str(path))
    return m.group(1) if m else 'standard'


def _timestamp_from_path(path):
    """Extract timestamp from a result filename, e.g. baseline_models_standard_20260303_121534.csv -> '20260303_121534'."""
    m = re.search(r'(\d{8}_\d{6})', str(path))
    return m.group(1) if m else 'unknown'


# ── Visualization ─────────────────────────────────────────────────────────────

# Label placement for each model's name box: (dx, dy) in points from the star, text alignment.
# Hand-tuned for the standard-tone map; models not listed get the default.
LABEL_OFFSETS = {
    'gemma2:2b':         (-12, -5, 'right'),
    'claude-sonnet-4-6': (12, -5, 'left'),
    'yi:6b':             (-3, 8, 'left'),
    'gpt-4o':            (34, 9, 'left'),
    'qwen2.5:7b':        (2, 7, 'left'),
    'phi3:mini':         (-10, -6, 'right'),
    'mistral:7b':        (2, -2, 'left'),
    'gpt-4o-mini':       (0, -10, 'center'),
}
DEFAULT_LABEL_OFFSET = (10, 8, 'left')


def create_visualization(baseline_df, models_df, output_path, tone='standard',
                         xlim=(-3.5, 6.5), y_clip=2.6):
    """Create the cultural map: country dots + LLM stars, cropped to |y| <= y_clip.

    Models beyond the crop are drawn as triangles pinned to the edge, numbered,
    and listed with their true coordinates in an inset box.
    """
    models_df = models_df.dropna(subset=['survival_selfexpression', 'traditional_secular'])
    models_df = models_df.sort_values('model').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.4, 5.4))

    # ── Country dots colored by cultural zone ──
    zone_handles = []
    for zone, color in ZONE_COLORS.items():
        zone_data = baseline_df[baseline_df['zone'] == zone]
        if zone_data.empty:
            continue
        ax.scatter(
            zone_data['survival_selfexpression'],
            zone_data['traditional_secular'],
            c=color, s=45, alpha=0.75,
            edgecolors='white', linewidth=0.4, zorder=2,
        )
        zone_handles.append(mpatches.Patch(color=color, label=zone))

    # ── ISO-3 labels on every country dot ──
    for _, row in baseline_df.iterrows():
        ax.annotate(
            row['iso3'],
            xy=(row['survival_selfexpression'], row['traditional_secular']),
            xytext=(3, 3), textcoords='offset points',
            fontsize=6, color='#555555', zorder=3,
        )

    # ── LLM model stars (on-scale) / edge triangles (off-scale) ──
    off_scale = []
    for idx, model in models_df.iterrows():
        color = plt.cm.tab10(idx % 10)
        x, y = model['survival_selfexpression'], model['traditional_secular']
        if abs(y) > y_clip:
            off_scale.append((model, color))
            continue
        ax.scatter(x, y, color=color, s=420, marker='*',
                   edgecolors='black', linewidth=1.3, zorder=10)
        dx, dy, ha = LABEL_OFFSETS.get(model['model'], DEFAULT_LABEL_OFFSET)
        leader = (dict(arrowstyle='-', color='#555555', linewidth=0.8)
                  if np.hypot(dx, dy) > 20 else None)
        ax.annotate(
            model_label_inline(model['model']),
            xy=(x, y), xytext=(dx, dy), textcoords='offset points',
            ha=ha, va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=color, alpha=0.85,
                      edgecolor='black', linewidth=0.8),
            arrowprops=leader, zorder=11,
        )

    off_scale.sort(key=lambda mc: mc[0]['survival_selfexpression'])
    lines = [f'Off-scale (|y| > {y_clip}):' if any(m['traditional_secular'] < 0 for m, _ in off_scale)
             else f'Off-scale (above y = {y_clip}):']
    for n, (model, color) in enumerate(off_scale, start=1):
        x, y = model['survival_selfexpression'], model['traditional_secular']
        top = y > 0
        edge_y = y_clip if top else -y_clip
        ax.scatter(x, edge_y, color=color, s=380, marker='^' if top else 'v',
                   edgecolors='#c0392b', linewidth=1.8, zorder=10, clip_on=False)
        ax.annotate(f'[{n}]', xy=(x, edge_y), xytext=(0, -20 if top else 20),
                    textcoords='offset points', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='#c0392b', zorder=11)
        lines.append(f'[{n}] {model_label_inline(model["model"])}  ({x:.2f}, {y:.2f})')
    if off_scale:
        ax.text(0.985, 0.965, '\n'.join(lines), transform=ax.transAxes,
                ha='right', va='top', multialignment='right', fontsize=8.5,
                family='monospace', zorder=12,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='#999999', alpha=0.95))

    # ── Axes ──
    ax.set_xlim(*xlim)
    ax.set_ylim(-y_clip, y_clip)
    ax.set_xlabel('Survival  ←  →  Self-Expression', fontsize=12, fontweight='bold')
    ax.set_ylabel('Traditional  ←  →  Secular', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Cultural Map — LLM Positions vs {len(baseline_df)} IVS Countries '
        f'({tone.capitalize()} Tone)',
        fontsize=12, fontweight='bold', pad=14,
    )
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.6, alpha=0.4)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.6, alpha=0.4)
    ax.grid(True, alpha=0.2)

    ax.legend(
        handles=zone_handles, title='Cultural Zone',
        loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5,
        fontsize=9, title_fontsize=10, framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {output_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def build_summary(baseline_df, models_df, tone='standard'):
    """Build a detailed summary string: model positions, closest countries, quadrant context."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append(f"BASELINE REPLICATION SUMMARY  [{tone.upper()} tone]")
    lines.append("=" * 70)

    lines.append(f"\nHuman Baseline ({len(baseline_df)} countries):")
    lines.append(f"  X range : [{baseline_df['survival_selfexpression'].min():.2f}, "
                 f"{baseline_df['survival_selfexpression'].max():.2f}]")
    lines.append(f"  Y range : [{baseline_df['traditional_secular'].min():.2f}, "
                 f"{baseline_df['traditional_secular'].max():.2f}]")
    lines.append(f"  Centroid: ({baseline_df['survival_selfexpression'].mean():.2f}, "
                 f"{baseline_df['traditional_secular'].mean():.2f})")

    lines.append(f"\nLLM Models Tested: {len(models_df)}")
    lines.append("\n" + "-" * 70)

    for _, model in models_df.iterrows():
        x = model['survival_selfexpression']
        y = model['traditional_secular']

        distances = []
        for _, country in baseline_df.iterrows():
            d = np.sqrt((x - country['survival_selfexpression'])**2 +
                        (y - country['traditional_secular'])**2)
            distances.append((d, country['iso3'], country['name'], country['zone']))
        distances.sort()

        lines.append(f"\n  Model : {model['model']}  (tone: {model.get('tone', tone)})")
        lines.append(f"  Position  : ({x:.3f}, {y:.3f})")
        lines.append(f"  Quadrant  : {quadrant_label(x, y)}")
        lines.append(f"  Typical of: {typical_countries(x, y)}")
        lines.append(f"  Closest countries:")
        for d, iso3, name, zone in distances[:5]:
            lines.append(f"    {iso3:3s}  {name:<20s}  d={d:.3f}  [{zone}]")
        lines.append(f"  Mean dist to all countries: {model.get('mean_distance', float('nan')):.3f}")

    lines.append("\n" + "-" * 70)
    avg_x = models_df['survival_selfexpression'].mean()
    avg_y = models_df['traditional_secular'].mean()
    lines.append(f"\nAverage model position: ({avg_x:.2f}, {avg_y:.2f})")
    lines.append(f"  Quadrant  : {quadrant_label(avg_x, avg_y)}")
    lines.append(f"  Typical of: {typical_countries(avg_x, avg_y)}")
    lines.append(f"\nPNAS Reference (GPT-4o): (3.35, 0.50) — closest: Finland (d≈0.20)")

    return "\n".join(lines)


def print_summary(baseline_df, models_df, tone='standard'):
    """Print and return the summary string."""
    text = build_summary(baseline_df, models_df, tone=tone)
    print(text)
    return text


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize baseline replication')
    parser.add_argument(
        '--model-set', choices=['all', 'open', 'api'], default=None,
        help='Filter results by model set (default: load all available files)'
    )
    parser.add_argument('--results', help='Path to a specific baseline_models_*.csv')
    parser.add_argument('--tone', help='Tone to load (standard/friendly/combative)')
    parser.add_argument('--output', help='Override output image path (single-tone only)')
    args = parser.parse_args()

    baseline_df = load_baseline()
    ms = args.model_set
    ms_tag = f"_{ms}" if ms else ''

    if args.results or args.tone:
        # Single-file or single-tone mode
        models_df, detected_tone, timestamp = load_model_results(
            args.results, args.tone, model_set=ms)
        tone = args.tone or detected_tone
        output_path = (Path(args.output) if args.output else
                       OUTPUTS_DIR / f"baseline_map_cropped_{tone}{ms_tag}_{timestamp}.png")
        summary_path = output_path.with_suffix('.txt')
        create_visualization(baseline_df, models_df, output_path, tone=tone)
        text = print_summary(baseline_df, models_df, tone=tone)
        summary_path.write_text(text)
        print(f"Saved: {summary_path}")
        print(f"\nDone. Image saved to: {output_path}")
    else:
        # Default: generate a map for every tone that has results
        generated = []
        for tone in ['standard', 'friendly', 'combative']:
            try:
                models_df, _, timestamp = load_model_results(tone=tone, model_set=ms)
                output_path = OUTPUTS_DIR / f"baseline_map_cropped_{tone}{ms_tag}_{timestamp}.png"
                summary_path = output_path.with_suffix('.txt')
                create_visualization(baseline_df, models_df, output_path, tone=tone)
                text = print_summary(baseline_df, models_df, tone=tone)
                summary_path.write_text(text)
                print(f"Saved: {summary_path}")
                generated.append(str(output_path))
            except FileNotFoundError:
                print(f"Skipping '{tone}' tone — no results file found.")
        print(f"\nDone. {len(generated)} image(s) saved:")
        for p in generated:
            print(f"  {p}")


if __name__ == "__main__":
    main()
