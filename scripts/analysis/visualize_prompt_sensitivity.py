"""
Prompt Sensitivity Visualization  (Stage 4)

Generates four figures from prompt_sensitivity_flat_*.csv:

1. Flip-rate heatmap     — model × question, cell = fraction of variants
                           that gave a different answer than the reference
2. Tone comparison       — for each question, how does the mean/mode shift
                           across standard / friendly / combative tones
3. Cultural map clouds   — each of the 30 (tone×variant) responses projected
                           through the IVS PCA; one dot per variant, centroid star
4. Tone interaction heatmap — std of responses per (model, question, tone);
                           reveals where tone choice drives instability

Usage:
    python scripts/analysis/visualize_prompt_sensitivity.py
    python scripts/analysis/visualize_prompt_sensitivity.py --flat data/results/prompt_sensitivity/prompt_sensitivity_flat_<ts>.csv
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.response_parser import ResponseParser
from src.prompts import QUESTIONS, TONES
from src.cultural_map import CulturalMapGenerator
from src.geo_data import ZONE_COLORS, ISO3_TO_ZONE, load_iso3_lookup

RESULTS_DIR   = Path("data/results/prompt_sensitivity")
OUTPUTS_DIR   = Path("outputs/prompt_sensitivity")
BASELINE_PATH = Path("data/processed/cultural_map_coordinates.csv")
IVS_PATH      = Path("data/processed/ivs_2005-2022.csv")

REFERENCE_TONE    = 'standard'
REFERENCE_VARIANT = 0

QUESTION_ORDER = ['A008', 'A165', 'E018', 'E025', 'F063',
                  'F118', 'F120', 'G006', 'Y002', 'Y003']
QUESTION_LABELS = {
    'A008': 'Happiness\n(1–4)',
    'A165': 'Trust\n(A/B)',
    'E018': 'Authority\n(1–3)',
    'E025': 'Petition\n(A/B/C)',
    'F063': 'God\n(1–10)',
    'F118': 'Homosexuality\n(1–10)',
    'F120': 'Abortion\n(1–10)',
    'G006': 'Nationality\n(1–4)',
    'Y002': 'Post-Mat.\n(rank 2)',
    'Y003': 'Autonomy\n(pick 5)',
}
QUESTION_SHORT = {
    'A008': 'Happiness',   'A165': 'Trust',
    'E018': 'Authority',   'E025': 'Petition',
    'F063': 'God',         'F118': 'Homosexuality',
    'F120': 'Abortion',    'G006': 'Nationality',
    'Y002': 'Post-Mat.',   'Y003': 'Autonomy',
}
NUMERIC_QUESTIONS = {'A008', 'E018', 'F063', 'F118', 'F120', 'G006'}

TONE_COLORS = {
    'standard':  '#4878cf',
    'friendly':  '#6acc65',
    'combative': '#d65f5f',
}

MODEL_PARAMS = {
    'gemma2:2b':            '2B',
    'phi3:mini':            '3.8B',
    'qwen2.5:1.5b':         '1.5B',
    'qwen2.5:3b':           '3B',
    'qwen2.5:7b':           '7B',
    'mistral:7b':           '7B',
    'llama3.1:8b':          '8B',
    'yi:6b':                '6B',
    'salmatrafi/acegpt:7b': '7B',
}
MODEL_COLORS = [
    '#e41a1c', '#ff7f00', '#984ea3', '#4daf4a',
    '#377eb8', '#a65628', '#f781bf', '#999999', '#17becf',
]


def model_label(name):
    short  = name.split('/')[-1].split(':')[0]
    params = MODEL_PARAMS.get(name)
    return f"{short}\n({params})" if params else short


def model_label_inline(name):
    short  = name.split('/')[-1].split(':')[0]
    params = MODEL_PARAMS.get(name)
    return f"{short} ({params})" if params else short


def parse_val(v):
    try:
        return json.loads(v)
    except (json.JSONDecodeError, TypeError):
        return None


# ── Data loading ──────────────────────────────────────────────────────────────

def _filter_by_model_set(paths, model_set):
    """Keep only paths containing _{model_set}_ if model_set is specified."""
    if not model_set:
        return paths
    return [p for p in paths if f'_{model_set}_' in p.name]


def load_data(flat_path=None, model_set=None):
    if flat_path:
        paths = [Path(flat_path)]
    else:
        paths = sorted(RESULTS_DIR.glob("prompt_sensitivity_flat_*.csv"))
        paths = _filter_by_model_set(paths, model_set)
        if not paths:
            tag = f" matching model_set='{model_set}'" if model_set else ""
            raise FileNotFoundError(f"No prompt_sensitivity_flat_*.csv{tag} in {RESULTS_DIR}")

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    key = ['model', 'tone', 'variant', 'question_id']
    df = df.drop_duplicates(subset=key, keep='last')
    df['parsed'] = df['parsed_value'].apply(parse_val)
    print(f"Loaded {len(paths)} file(s) — {len(df)} rows, "
          f"{df['model'].nunique()} models")
    return df


# ── Figure 1: Flip-rate heatmap ───────────────────────────────────────────────

def plot_flip_heatmap(flat_df: pd.DataFrame, out_path: Path) -> None:
    """
    Rows = questions, columns = models.
    Cell colour = sign-flip rate (0=stable green, 1=unstable red).
    """
    models = sorted(flat_df['model'].unique())

    # Compute flip rate per (model, question)
    ref = flat_df[
        (flat_df['tone'] == REFERENCE_TONE) &
        (flat_df['variant'] == REFERENCE_VARIANT)
    ][['model', 'question_id', 'parsed']].rename(columns={'parsed': 'ref_val'})

    merged = flat_df.merge(ref, on=['model', 'question_id'], how='left')
    non_ref = merged[
        ~((merged['tone'] == REFERENCE_TONE) &
          (merged['variant'] == REFERENCE_VARIANT))
    ].copy()

    def is_flip(row):
        if row['parsed'] is None or row['ref_val'] is None:
            return np.nan
        qinfo = QUESTIONS[row['question_id']]
        rtype = qinfo['response_type']
        if rtype == 'numeric':
            return float(row['parsed']) != float(row['ref_val'])
        elif rtype == 'categorical':
            return str(row['parsed']) != str(row['ref_val'])
        else:
            return (frozenset(row['parsed']) if isinstance(row['parsed'], list) else {row['parsed']}) != \
                   (frozenset(row['ref_val']) if isinstance(row['ref_val'], list) else {row['ref_val']})

    non_ref['flipped'] = non_ref.apply(is_flip, axis=1)
    flip_rates = non_ref.groupby(['model', 'question_id'])['flipped'].mean().unstack('model')
    flip_rates = flip_rates.reindex(QUESTION_ORDER)[models]

    cmap = LinearSegmentedColormap.from_list('flip', ['#2ecc71', '#f39c12', '#e74c3c'])

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.3), 6))
    im = ax.imshow(flip_rates.values.astype(float),
                   aspect='auto', cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([model_label(m) for m in models],
                       rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(QUESTION_ORDER)))
    ax.set_yticklabels(
        [QUESTION_LABELS.get(q, q).replace('\n', ' ') for q in QUESTION_ORDER],
        fontsize=9
    )

    for i in range(len(QUESTION_ORDER)):
        for j, model in enumerate(models):
            val = flip_rates.values[i, j]
            if not np.isnan(val):
                txt_color = 'white' if val > 0.6 or val < 0.15 else 'black'
                ax.text(j, i, f'{val:.2f}',
                        ha='center', va='center', fontsize=8, color=txt_color)

    plt.colorbar(im, ax=ax,
                 label='Sign-flip rate  (0 = always same answer, 1 = always different)')
    ax.set_title(
        'Prompt Sensitivity — Sign-Flip Rate per Model × Question\n'
        '(fraction of 29 non-reference variants that gave a different answer)',
        fontsize=11, fontweight='bold'
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Figure 2: Tone comparison ─────────────────────────────────────────────────

def plot_tone_comparison(flat_df: pd.DataFrame, out_path: Path) -> None:
    """
    For each numeric question: grouped box/strip showing response distribution
    per tone, across all models. Reveals whether tone systematically shifts answers.
    """
    numeric_qs = [q for q in QUESTION_ORDER if q in NUMERIC_QUESTIONS]
    n_q = len(numeric_qs)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    tones = ['standard', 'friendly', 'combative']

    for ax, qid in zip(axes, numeric_qs):
        q_df = flat_df[flat_df['question_id'] == qid].copy()
        q_df = q_df[q_df['parsed'].notna()]
        q_df['numeric'] = q_df['parsed'].apply(
            lambda v: float(v) if v is not None else np.nan
        )

        for i, tone in enumerate(tones):
            vals = q_df[q_df['tone'] == tone]['numeric'].dropna().values
            jitter = np.random.uniform(-0.12, 0.12, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter, vals,
                color=TONE_COLORS[tone], alpha=0.35, s=25, zorder=2
            )
            if len(vals):
                ax.plot(
                    [i - 0.25, i + 0.25], [vals.mean(), vals.mean()],
                    color=TONE_COLORS[tone], linewidth=2.5, zorder=3
                )

        ax.set_xticks(range(len(tones)))
        ax.set_xticklabels([t.capitalize() for t in tones], fontsize=9)
        ax.set_title(QUESTION_LABELS[qid].replace('\n', ' '), fontsize=10, fontweight='bold')
        ax.set_ylabel('Response value', fontsize=8)
        ax.grid(True, alpha=0.2)

    # Legend
    handles = [mpatches.Patch(color=TONE_COLORS[t], label=t.capitalize())
               for t in tones]
    fig.legend(handles=handles, loc='lower right', fontsize=9,
               title='Tone', title_fontsize=9)

    # Hide unused axes
    for ax in axes[n_q:]:
        ax.set_visible(False)

    fig.suptitle(
        'Tone Effect on Numeric Responses  (all models × all variants)\n'
        'Dots = individual variant responses; horizontal bar = mean',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Figure 3: Cultural map variant clouds ─────────────────────────────────────

def load_pca():
    print("Fitting PCA on IVS data...")
    ivs_df = pd.read_csv(IVS_PATH, low_memory=False)
    gen = CulturalMapGenerator(ivs_df)
    gen.fit()
    print("PCA ready.\n")
    return gen


def load_baseline():
    df = pd.read_csv(BASELINE_PATH)
    iso_lookup = load_iso3_lookup(IVS_PATH)
    df['iso3'] = df['country_code'].map(iso_lookup).fillna('???')
    df['zone'] = df['iso3'].map(ISO3_TO_ZONE).fillna('Other')
    return df


def variant_to_coords(variant_df: pd.DataFrame,
                      pca_gen: CulturalMapGenerator):
    """Project one (model, tone, variant) response set to (x, y)."""
    question_values = {}
    for question_id, question_info in QUESTIONS.items():
        row = variant_df[variant_df['question_id'] == question_id]
        if row.empty or not row.iloc[0]['is_valid']:
            return np.nan, np.nan
        numeric = ResponseParser.to_ivs_numeric(row.iloc[0]['parsed'], question_info)
        if numeric is None:
            return np.nan, np.nan
        question_values[question_id] = numeric

    standardized = [
        (question_values[qid] - pca_gen.question_means_[qid]) / pca_gen.question_stds_[qid]
        for qid in QUESTIONS
    ]
    pca_scores = pca_gen.pca_model.transform(
        np.array(standardized).reshape(1, -1)
    )[0]
    x = 1.81 * pca_scores[0] + 0.38
    y = 1.61 * pca_scores[1] - 0.01
    return x, y


def compute_variant_coords(flat_df: pd.DataFrame,
                           pca_gen: CulturalMapGenerator) -> pd.DataFrame:
    rows = []
    for model in flat_df['model'].unique():
        mdf = flat_df[flat_df['model'] == model]
        n_valid = 0
        for (tone, variant), grp in mdf.groupby(['tone', 'variant']):
            x, y = variant_to_coords(grp, pca_gen)
            valid = not (np.isnan(x) or np.isnan(y))
            rows.append({
                'model': model, 'tone': tone, 'variant': variant,
                'survival_selfexpression': x,
                'traditional_secular': y,
                'valid': valid,
            })
            if valid:
                n_valid += 1
        total = mdf.groupby(['tone', 'variant']).ngroups
        print(f"  {model_label_inline(model):<22s}  {n_valid}/{total} variants mapped")
    return pd.DataFrame(rows)


def plot_variant_map(coords_df: pd.DataFrame,
                     baseline_df: pd.DataFrame,
                     out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(17, 12))

    # Country dots
    zone_handles = []
    for zone, color in ZONE_COLORS.items():
        zd = baseline_df[baseline_df['zone'] == zone]
        if zd.empty:
            continue
        ax.scatter(zd['survival_selfexpression'], zd['traditional_secular'],
                   c=color, s=50, alpha=0.60,
                   edgecolors='white', linewidth=0.3, zorder=2)
        zone_handles.append(mpatches.Patch(color=color, label=zone))

    for _, row in baseline_df.iterrows():
        ax.annotate(row['iso3'],
                    xy=(row['survival_selfexpression'], row['traditional_secular']),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=6, color='#444444', zorder=3)

    # Per-model variant clouds
    models = coords_df['model'].unique()
    model_handles = []

    for idx, model in enumerate(models):
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        label = model_label_inline(model)
        mdf = coords_df[(coords_df['model'] == model) & (coords_df['valid'])]
        if mdf.empty:
            continue

        # Colour dots by tone
        for tone in ['standard', 'friendly', 'combative']:
            tdf = mdf[mdf['tone'] == tone]
            if tdf.empty:
                continue
            ax.scatter(
                tdf['survival_selfexpression'], tdf['traditional_secular'],
                c=TONE_COLORS[tone], s=55, alpha=0.50,
                edgecolors=color, linewidth=0.8, zorder=6,
                marker={'standard': 'o', 'friendly': 's', 'combative': '^'}[tone]
            )

        cx = mdf['survival_selfexpression'].mean()
        cy = mdf['traditional_secular'].mean()
        std_x = mdf['survival_selfexpression'].std()
        std_y = mdf['traditional_secular'].std()

        # Spokes from centroid
        for _, r in mdf.iterrows():
            ax.plot([cx, r['survival_selfexpression']],
                    [cy, r['traditional_secular']],
                    color=color, alpha=0.15, linewidth=0.7, zorder=5)

        # Centroid star
        ax.scatter(cx, cy, c=color, s=380, marker='*',
                   edgecolors='black', linewidth=1.2, zorder=10)
        ax.annotate(label, xy=(cx, cy), xytext=(10, 7),
                    textcoords='offset points',
                    fontsize=8.5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor=color,
                              alpha=0.80, edgecolor='black', linewidth=0.7),
                    zorder=11)
        ax.annotate(f"σ=({std_x:.2f}, {std_y:.2f})",
                    xy=(cx, cy), xytext=(10, -8),
                    textcoords='offset points',
                    fontsize=6.5, color=color, zorder=11)

        model_handles.append(
            plt.Line2D([0], [0], marker='*', color='w',
                       markerfacecolor=color, markeredgecolor='black',
                       markersize=12, label=label)
        )

    ax.set_xlabel('Survival  ←                    →  Self-Expression Values',
                  fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Traditional  ←               →  Secular Values',
                  fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title(
        'Prompt Sensitivity — Cultural Map Variant Clouds\n'
        '(30 prompt variants per model: ● standard  ■ friendly  ▲ combative  |  ★ = centroid)',
        fontsize=12, fontweight='bold', pad=14,
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

    # Tone shape legend
    tone_handles = [
        plt.Line2D([0], [0], marker='o', color='gray', linestyle='None',
                   markersize=8, label='Standard'),
        plt.Line2D([0], [0], marker='s', color='gray', linestyle='None',
                   markersize=8, label='Friendly'),
        plt.Line2D([0], [0], marker='^', color='gray', linestyle='None',
                   markersize=8, label='Combative'),
    ]

    legend_zones = ax.legend(handles=zone_handles, title='Cultural Zone',
                             loc='upper left', fontsize=7,
                             title_fontsize=8, framealpha=0.9)
    ax.add_artist(legend_zones)
    legend_tones = ax.legend(handles=tone_handles, title='Tone',
                             loc='upper right', fontsize=8,
                             title_fontsize=8, framealpha=0.9)
    ax.add_artist(legend_tones)
    ax.legend(handles=model_handles, title='LLM Models  (★ = centroid)',
              loc='lower right', fontsize=8,
              title_fontsize=8.5, framealpha=0.9)

    ax.grid(True, alpha=0.18)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Figure 4: Tone interaction heatmap ───────────────────────────────────────

def plot_tone_interaction(flat_df: pd.DataFrame, out_path: Path) -> None:
    """
    Three side-by-side heatmaps — one per tone.
    Rows = questions, cols = models, cell = std of responses within that
    (model, question, tone) group (10 variants per cell).

    A large difference in cell colour across the three panels for the same
    (model, question) is a tone interaction — the tone itself is driving
    instability, not just the variant number.
    """
    models = sorted(flat_df['model'].unique())
    tones  = ['standard', 'friendly', 'combative']

    # Compute std per (model, tone, question) in IVS-numeric space
    rows = []
    for (model, tone, qid), grp in flat_df.groupby(['model', 'tone', 'question_id']):
        qinfo = QUESTIONS[qid]
        nums = [
            ResponseParser.to_ivs_numeric(parse_val(v), qinfo)
            for v in grp['parsed_value']
        ]
        nums = [v for v in nums if v is not None]
        rows.append({
            'model':       model,
            'tone':        tone,
            'question_id': qid,
            'std':         np.std(nums, ddof=1) if len(nums) > 1 else 0.0,
        })
    std_df = pd.DataFrame(rows)

    # Global max for shared colour scale
    vmax = std_df['std'].quantile(0.95)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, tone in zip(axes, tones):
        tdf   = std_df[std_df['tone'] == tone]
        pivot = tdf.pivot(index='question_id', columns='model', values='std')
        pivot = pivot.reindex(QUESTION_ORDER)[models]

        cmap = LinearSegmentedColormap.from_list(
            'tone_std', ['#2ecc71', '#f1c40f', '#e74c3c']
        )
        im = ax.imshow(pivot.values.astype(float),
                       aspect='auto', cmap=cmap, vmin=0, vmax=vmax)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([model_label(m) for m in models],
                           rotation=35, ha='right', fontsize=8)
        ax.set_yticks(range(len(QUESTION_ORDER)))
        ax.set_yticklabels(
            [QUESTION_SHORT[q] for q in QUESTION_ORDER], fontsize=9
        )
        ax.set_title(f'{tone.capitalize()} tone', fontsize=11, fontweight='bold',
                     color=TONE_COLORS[tone])

        # Annotate cells
        for i, qid in enumerate(QUESTION_ORDER):
            for j, model in enumerate(models):
                val = pivot.loc[qid, model] if (qid in pivot.index and
                                                model in pivot.columns) else np.nan
                if not np.isnan(val):
                    txt_col = 'white' if val > vmax * 0.6 or val < vmax * 0.1 \
                              else 'black'
                    ax.text(j, i, f'{val:.2f}',
                            ha='center', va='center', fontsize=7, color=txt_col)

    # Shared colourbar on the right
    fig.subplots_adjust(right=0.88, wspace=0.08)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(
        cmap=LinearSegmentedColormap.from_list(
            'tone_std', ['#2ecc71', '#f1c40f', '#e74c3c']
        ),
        norm=plt.Normalize(vmin=0, vmax=vmax)
    )
    fig.colorbar(sm, cax=cbar_ax,
                 label='σ within tone (IVS numeric)  —  green=stable, red=unstable')

    fig.suptitle(
        'Tone Interaction — Response Std per (Model × Question × Tone)\n'
        'Cells that differ strongly across the three panels indicate a tone interaction:\n'
        'that tone is driving instability independently of variant number',
        fontsize=11, fontweight='bold', y=1.03
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualize prompt sensitivity results.')
    parser.add_argument(
        '--model-set', choices=['all', 'open', 'api'], default=None,
        help='Filter to files from a specific model-set run (default: merge all files)'
    )
    parser.add_argument('--flat', type=Path,
                        help='Flat CSV; overrides --model-set')
    args = parser.parse_args()

    flat_df = load_data(args.flat, model_set=args.model_set)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    ms_tag = f"_{args.model_set}" if args.model_set else ''

    # Figure 1: flip-rate heatmap
    plot_flip_heatmap(flat_df, OUTPUTS_DIR / f"flip_rate_heatmap{ms_tag}_{ts}.png")

    # Figure 2: tone comparison (numeric questions)
    plot_tone_comparison(flat_df, OUTPUTS_DIR / f"tone_comparison{ms_tag}_{ts}.png")

    # Figure 3: cultural map variant clouds
    pca_gen     = load_pca()
    baseline_df = load_baseline()
    print("\nComputing per-variant coordinates...")
    coords_df   = compute_variant_coords(flat_df, pca_gen)
    plot_variant_map(coords_df, baseline_df,
                     OUTPUTS_DIR / f"prompt_sensitivity_cultural_map{ms_tag}_{ts}.png")

    # Figure 4: tone interaction heatmap
    plot_tone_interaction(flat_df, OUTPUTS_DIR / f"tone_interaction{ms_tag}_{ts}.png")


if __name__ == '__main__':
    main()
