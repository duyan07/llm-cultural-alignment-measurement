"""
Stochastic Cultural Map Visualization  (Week 7)

Projects each seed's responses through the IVS-fitted PCA to place each
(model, seed) run as a point on the Inglehart-Welzel cultural map.

Each model appears as a scatter cloud (one dot per seed) + a centroid star,
overlaid on the 88-country human baseline. This shows how much a model's
cultural position drifts due to sampling randomness alone.

Usage:
    python scripts/analysis/visualize_stochastic_map.py
    python scripts/analysis/visualize_stochastic_map.py --flat data/results/stochastic/stochastic_flat_<ts>.csv
    python scripts/analysis/visualize_stochastic_map.py --model-set open
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.response_parser import ResponseParser
from src.prompts import QUESTIONS
from src.cultural_map import CulturalMapGenerator
from src.geo_data import ZONE_COLORS
from src.viz_common import (
    MODEL_COLORS,
    model_label_inline as model_label,
    parse_value,
    load_pca,
    load_baseline,
)

# ── Paths ────────────────────────────────────────────────────────────────────

RESULTS_DIR  = Path("data/results/stochastic")
OUTPUTS_DIR  = Path("outputs/stochastic")


# ── Data loading ─────────────────────────────────────────────────────────────

def _filter_by_model_set(paths, model_set):
    """Keep only paths containing _{model_set}_ if model_set is specified."""
    if not model_set:
        return paths
    return [p for p in paths if f'_{model_set}_' in p.name]


def load_flat(flat_path=None, model_set=None) -> pd.DataFrame:
    """Merge all stochastic flat CSVs (or a specific one if given)."""
    if flat_path:
        paths = [Path(flat_path)]
    else:
        paths = sorted(RESULTS_DIR.glob("stochastic_flat_*.csv"))
        paths = _filter_by_model_set(paths, model_set)
        if not paths:
            tag = f" matching model_set='{model_set}'" if model_set else ""
            raise FileNotFoundError(f"No stochastic_flat_*.csv{tag} in {RESULTS_DIR}")

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    key_cols = ['model', 'tone', 'variant', 'question_id', 'seed']
    df = df.drop_duplicates(subset=key_cols, keep='last')
    print(f"Loaded {len(paths)} file(s) — {len(df)} rows, "
          f"{df['model'].nunique()} models")
    return df


# ── Coordinate computation ───────────────────────────────────────────────────

def seed_to_coordinates(seed_df: pd.DataFrame,
                        pca_gen: CulturalMapGenerator):
    """
    Convert one seed's responses to (x, y) map coordinates by summing only
    the valid question contributions.

    Each invalid response slot contributes 0 to the PC score (mathematically
    equivalent to imputing the IVS standardised mean). Returns (x, y, n_valid)
    where n_valid is the number of slots that contributed real data; if zero
    valid slots, returns (nan, nan, 0).
    """
    standardized = []
    n_valid = 0

    for question_id, question_info in QUESTIONS.items():
        row = seed_df[seed_df['question_id'] == question_id]
        invalid = row.empty or not row.iloc[0]['is_valid']
        numeric = None
        if not invalid:
            parsed = parse_value(row.iloc[0]['parsed_value'])
            numeric = ResponseParser.to_ivs_numeric(parsed, question_info)

        if numeric is None:
            standardized.append(0.0)
            continue

        standardized.append(
            (numeric - pca_gen.question_means_[question_id]) /
            pca_gen.question_stds_[question_id]
        )
        n_valid += 1

    if n_valid == 0:
        return np.nan, np.nan, 0

    pca_scores = pca_gen.pca_model.transform(
        np.array(standardized).reshape(1, -1)
    )[0]

    x = 1.81 * pca_scores[0] + 0.38
    y = 1.61 * pca_scores[1] - 0.01
    return x, y, n_valid


def compute_all_coordinates(flat_df: pd.DataFrame,
                            pca_gen: CulturalMapGenerator) -> pd.DataFrame:
    """
    Compute (x, y) for every (model, seed) combination using pointwise
    imputation: invalid slots contribute 0 to the PC score.

    Models with zero valid responses anywhere (likely API/system errors)
    are excluded entirely.
    """
    excluded_models = set()
    for model in flat_df['model'].unique():
        md = flat_df[flat_df['model'] == model]
        if md['is_valid'].sum() == 0:
            excluded_models.add(model)

    if excluded_models:
        print(f"  Excluding {len(excluded_models)} model(s) with zero valid "
              f"responses (likely API/system error):")
        for m in sorted(excluded_models):
            print(f"    - {model_label(m)}")

    rows = []
    models = [m for m in flat_df['model'].unique() if m not in excluded_models]

    for model in models:
        model_df = flat_df[flat_df['model'] == model]
        seeds = model_df['seed'].unique()
        n_mapped = 0
        n_full = 0

        for seed in seeds:
            seed_df = model_df[model_df['seed'] == seed]
            x, y, n_valid = seed_to_coordinates(seed_df, pca_gen)
            mapped = not (np.isnan(x) or np.isnan(y))
            rows.append({
                'model': model,
                'seed': seed,
                'survival_selfexpression': x,
                'traditional_secular': y,
                'n_valid': n_valid,
                'fully_valid': n_valid == len(QUESTIONS),
                'valid': mapped,
            })
            if mapped:
                n_mapped += 1
            if n_valid == len(QUESTIONS):
                n_full += 1

        print(f"  {model_label(model):<22s}  {n_mapped}/{len(seeds)} seeds mapped"
              f"  ({n_full} fully-valid, {n_mapped - n_full} with imputed slots)")

    return pd.DataFrame(rows)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_stochastic_map(coords_df: pd.DataFrame,
                        baseline_df: pd.DataFrame,
                        out_path: Path,
                        n_seeds: int) -> None:
    """
    Plot country baseline + per-seed clouds + centroids for each model.
    """
    fig, ax = plt.subplots(figsize=(17, 12))

    # ── Country dots coloured by cultural zone ──
    zone_handles = []
    for zone, color in ZONE_COLORS.items():
        zone_data = baseline_df[baseline_df['zone'] == zone]
        if zone_data.empty:
            continue
        ax.scatter(
            zone_data['survival_selfexpression'],
            zone_data['traditional_secular'],
            c=color, s=55, alpha=0.65,
            edgecolors='white', linewidth=0.3, zorder=2,
        )
        zone_handles.append(mpatches.Patch(color=color, label=zone))

    # ── ISO-3 country labels ──
    for _, row in baseline_df.iterrows():
        ax.annotate(
            row['iso3'],
            xy=(row['survival_selfexpression'], row['traditional_secular']),
            xytext=(3, 3), textcoords='offset points',
            fontsize=6, color='#444444', zorder=3,
        )

    # ── Per-model seed clouds + centroids ──
    models = coords_df['model'].unique()
    model_handles = []
    has_imputed_any = False

    for idx, model in enumerate(models):
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        label = model_label(model)

        mdf = coords_df[(coords_df['model'] == model) & (coords_df['valid'])]
        if mdf.empty:
            print(f"  Warning: no valid coordinates for {model}, skipping.")
            continue

        full_mdf = mdf[mdf['fully_valid']]
        partial_mdf = mdf[~mdf['fully_valid']]
        if not partial_mdf.empty:
            has_imputed_any = True

        # Filled markers for fully-valid points
        if not full_mdf.empty:
            ax.scatter(full_mdf['survival_selfexpression'],
                       full_mdf['traditional_secular'],
                       c=color, s=60, alpha=0.45,
                       edgecolors=color, linewidth=0.5, zorder=6)
        # Hollow markers for partially-imputed points
        if not partial_mdf.empty:
            ax.scatter(partial_mdf['survival_selfexpression'],
                       partial_mdf['traditional_secular'],
                       facecolors='none', edgecolors=color,
                       s=60, alpha=0.7, linewidth=1.2, zorder=6)

        xs = mdf['survival_selfexpression'].values
        ys = mdf['traditional_secular'].values
        cx, cy = xs.mean(), ys.mean()

        # Spokes from centroid
        for sx, sy in zip(xs, ys):
            ax.plot([cx, sx], [cy, sy],
                    color=color, alpha=0.2, linewidth=0.8, zorder=5)

        # Centroid star — solid if all points fully valid, hollow otherwise
        all_valid = len(full_mdf) == len(mdf)
        if all_valid:
            ax.scatter(cx, cy, c=color, s=380, marker='*',
                       edgecolors='black', linewidth=1.2, zorder=10)
            label_text = label
        else:
            ax.scatter(cx, cy, facecolors='white', s=380, marker='*',
                       edgecolors=color, linewidth=2.0, zorder=10)
            label_text = f"{label}  ({len(full_mdf)}/{len(mdf)})"

        ax.annotate(
            label_text,
            xy=(cx, cy), xytext=(10, 7),
            textcoords='offset points',
            fontsize=8.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=color,
                      alpha=0.80, edgecolor='black', linewidth=0.7),
            zorder=11,
        )

        # σ from fully-valid points only; omit if too few
        if len(full_mdf) >= 2:
            std_x = full_mdf['survival_selfexpression'].std()
            std_y = full_mdf['traditional_secular'].std()
            ax.annotate(
                f"σ=({std_x:.2f}, {std_y:.2f})",
                xy=(cx, cy), xytext=(10, -8),
                textcoords='offset points',
                fontsize=6.5, color=color, zorder=11,
            )

        model_handles.append(
            plt.Line2D([0], [0], marker='*', color='w',
                       markerfacecolor=color, markeredgecolor='black',
                       markersize=12, label=label)
        )

    # ── Axes & labels ──
    ax.set_xlabel('Survival  ←                    →  Self-Expression Values',
                  fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Traditional  ←               →  Secular Values',
                  fontsize=12, fontweight='bold', labelpad=8)
    impute_subtitle = (
        '\n● filled point = all 10 questions valid    ○ hollow point = ≥1 question imputed'
        ' (slot contributes 0 to the score)    ★ hollow centroid + label "(n_full/N)" '
        'when not all points fully valid    σ from fully-valid subset only'
        if has_imputed_any else ''
    )

    ax.set_title(
        f'Stochastic Sampling Uncertainty on the Cultural Map\n'
        f'({n_seeds} seeds per model, temperature=1.0 — seed cloud shows sampling noise)'
        f'{impute_subtitle}',
        fontsize=13, fontweight='bold', pad=14,
    )

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.6, alpha=0.35)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.6, alpha=0.35)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    pad = 0.15
    ax.text(x_max - pad, y_max - pad, 'Self-Expression\n+ Secular',
            ha='right', va='top', fontsize=8.5, color='gray', alpha=0.55, style='italic')
    ax.text(x_min + pad, y_max - pad, 'Survival\n+ Secular',
            ha='left',  va='top', fontsize=8.5, color='gray', alpha=0.55, style='italic')
    ax.text(x_min + pad, y_min + pad, 'Survival\n+ Traditional',
            ha='left',  va='bottom', fontsize=8.5, color='gray', alpha=0.55, style='italic')
    ax.text(x_max - pad, y_min + pad, 'Self-Expression\n+ Traditional',
            ha='right', va='bottom', fontsize=8.5, color='gray', alpha=0.55, style='italic')

    # ── Legends ──
    legend_zones = ax.legend(
        handles=zone_handles, title='Cultural Zone',
        loc='upper left', fontsize=7, title_fontsize=8, framealpha=0.9,
    )
    ax.add_artist(legend_zones)
    ax.legend(
        handles=model_handles, title='LLM Models  (★ = centroid)',
        loc='lower right', fontsize=8, title_fontsize=8.5, framealpha=0.9,
    )

    ax.grid(True, alpha=0.18)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(coords_df: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("STOCHASTIC CULTURAL MAP — SEED VARIANCE REPORT")
    print("=" * 65)

    for model in coords_df['model'].unique():
        mdf = coords_df[(coords_df['model'] == model) & (coords_df['valid'])]
        if mdf.empty:
            print(f"\n  {model_label(model)}: no valid seeds")
            continue

        xs = mdf['survival_selfexpression'].values
        ys = mdf['traditional_secular'].values
        cx, cy = xs.mean(), ys.mean()

        # Closest country to centroid
        dists = np.sqrt(
            (baseline_df['survival_selfexpression'] - cx) ** 2 +
            (baseline_df['traditional_secular']     - cy) ** 2
        )
        closest_idx = dists.idxmin()
        closest = baseline_df.loc[closest_idx]

        print(f"\n  {model_label(model)}")
        print(f"    Seeds mapped  : {len(mdf)}/{coords_df[coords_df['model']==model]['seed'].nunique()}")
        print(f"    Centroid      : ({cx:.3f}, {cy:.3f})")
        print(f"    σ_x, σ_y      : {xs.std():.3f}, {ys.std():.3f}")
        print(f"    x range       : [{xs.min():.3f}, {xs.max():.3f}]")
        print(f"    y range       : [{ys.min():.3f}, {ys.max():.3f}]")
        print(f"    Closest country: {closest['iso3']}  (d={dists[closest_idx]:.3f})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot stochastic seed clouds on the cultural map.'
    )
    parser.add_argument(
        '--model-set', choices=['all', 'open'], default=None,
        help='Filter to files from a specific model-set run (default: merge all files)'
    )
    parser.add_argument('--flat', type=Path,
                        help='Single flat CSV; overrides --model-set')
    args = parser.parse_args()

    flat_df   = load_flat(args.flat, model_set=args.model_set)
    pca_gen   = load_pca()
    baseline_df = load_baseline()

    print("\nComputing per-seed coordinates...")
    coords_df = compute_all_coordinates(flat_df, pca_gen)

    n_seeds = flat_df['seed'].nunique()
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    ms_tag = f"_{args.model_set}" if args.model_set else ''
    out_path = OUTPUTS_DIR / f"stochastic_cultural_map{ms_tag}_{ts}.png"

    print_report(coords_df, baseline_df)
    plot_stochastic_map(coords_df, baseline_df, out_path, n_seeds)


if __name__ == '__main__':
    main()
