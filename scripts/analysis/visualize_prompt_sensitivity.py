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
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.response_parser import ResponseParser
from src.prompts import QUESTIONS
from src.cultural_map import CulturalMapGenerator
from src.geo_data import ZONE_COLORS
from src.viz_common import (
    QUESTION_ORDER,
    QUESTION_LABELS,
    QUESTION_SHORT,
    NUMERIC_QUESTIONS,
    TONE_COLORS,
    MODEL_COLORS,
    model_label_inline,
    model_label_multiline as model_label,
    parse_value as parse_val,
    load_pca,
    load_baseline,
)

RESULTS_DIR   = Path("data/results/prompt_sensitivity")
OUTPUTS_DIR   = Path("outputs/prompt_sensitivity")

REFERENCE_TONE    = 'standard'
REFERENCE_VARIANT = 0


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

def variant_to_coords(variant_df: pd.DataFrame,
                      pca_gen: CulturalMapGenerator):
    """Project one (model, tone, variant) response set to (x, y).

    Each invalid response slot contributes 0 to the PC score (mathematically
    equivalent to imputing the IVS standardised mean). Returns (x, y, n_valid).
    """
    standardized = []
    n_valid = 0
    for question_id, question_info in QUESTIONS.items():
        row = variant_df[variant_df['question_id'] == question_id]
        invalid = row.empty or not row.iloc[0]['is_valid']
        numeric = None
        if not invalid:
            numeric = ResponseParser.to_ivs_numeric(row.iloc[0]['parsed'], question_info)

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


def compute_variant_coords(flat_df: pd.DataFrame,
                           pca_gen: CulturalMapGenerator) -> pd.DataFrame:
    """Compute per-(model, tone, variant) coordinates with pointwise imputation.

    Models with zero valid responses anywhere (likely API/system errors)
    are excluded entirely.
    """
    excluded_models = set()
    for model in flat_df['model'].unique():
        if flat_df[flat_df['model'] == model]['is_valid'].sum() == 0:
            excluded_models.add(model)

    if excluded_models:
        print(f"  Excluding {len(excluded_models)} model(s) with zero valid "
              f"responses (likely API/system error):")
        for m in sorted(excluded_models):
            print(f"    - {model_label_inline(m)}")

    rows = []
    models = [m for m in flat_df['model'].unique() if m not in excluded_models]

    for model in models:
        mdf = flat_df[flat_df['model'] == model]
        n_mapped = 0
        n_full = 0
        for (tone, variant), grp in mdf.groupby(['tone', 'variant']):
            x, y, n_valid = variant_to_coords(grp, pca_gen)
            mapped = not (np.isnan(x) or np.isnan(y))
            rows.append({
                'model': model, 'tone': tone, 'variant': variant,
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
        total = mdf.groupby(['tone', 'variant']).ngroups
        print(f"  {model_label_inline(model):<22s}  {n_mapped}/{total} variants mapped"
              f"  ({n_full} fully-valid, {n_mapped - n_full} with imputed slots)")
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
    has_imputed_any = False

    for idx, model in enumerate(models):
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        label = model_label_inline(model)
        mdf = coords_df[(coords_df['model'] == model) & (coords_df['valid'])]
        if mdf.empty:
            continue

        full_mdf = mdf[mdf['fully_valid']]
        partial_mdf = mdf[~mdf['fully_valid']]
        if not partial_mdf.empty:
            has_imputed_any = True

        # Colour dots by tone, split filled (fully-valid) vs hollow (imputed)
        for tone in ['standard', 'friendly', 'combative']:
            marker = {'standard': 'o', 'friendly': 's', 'combative': '^'}[tone]
            tone_full = full_mdf[full_mdf['tone'] == tone]
            tone_part = partial_mdf[partial_mdf['tone'] == tone]
            if not tone_full.empty:
                ax.scatter(
                    tone_full['survival_selfexpression'],
                    tone_full['traditional_secular'],
                    c=TONE_COLORS[tone], s=55, alpha=0.50,
                    edgecolors=color, linewidth=0.8, zorder=6, marker=marker,
                )
            if not tone_part.empty:
                ax.scatter(
                    tone_part['survival_selfexpression'],
                    tone_part['traditional_secular'],
                    facecolors='none', edgecolors=TONE_COLORS[tone],
                    s=55, alpha=0.75, linewidth=1.4, zorder=6, marker=marker,
                )

        cx = mdf['survival_selfexpression'].mean()
        cy = mdf['traditional_secular'].mean()

        # Spokes from centroid
        for _, r in mdf.iterrows():
            ax.plot([cx, r['survival_selfexpression']],
                    [cy, r['traditional_secular']],
                    color=color, alpha=0.15, linewidth=0.7, zorder=5)

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

        ax.annotate(label_text, xy=(cx, cy), xytext=(10, 7),
                    textcoords='offset points',
                    fontsize=8.5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor=color,
                              alpha=0.80, edgecolor='black', linewidth=0.7),
                    zorder=11)

        # σ from fully-valid points only
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

    ax.set_xlabel('Survival  ←                    →  Self-Expression Values',
                  fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Traditional  ←               →  Secular Values',
                  fontsize=12, fontweight='bold', labelpad=8)

    impute_subtitle = (
        '\nFilled markers = all 10 questions valid    Hollow markers = ≥1 question imputed'
        ' (slot contributes 0 to the score)    ★ hollow centroid + label "(n_full/N)" '
        'when not all points fully valid    σ from fully-valid subset only'
        if has_imputed_any else ''
    )

    ax.set_title(
        'Prompt Sensitivity — Cultural Map Variant Clouds\n'
        '(30 prompt variants per model: ● standard  ■ friendly  ▲ combative  |  ★ = centroid)'
        f'{impute_subtitle}',
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


# ── Figure 5: Refusal-rate heatmap ────────────────────────────────────────────

def plot_refusal_heatmap(flat_df: pd.DataFrame, out_path: Path) -> None:
    """
    Per-(model, question) refusal-rate heatmap. Captures the orthogonal
    "what does each model refuse" story so the cultural map is freed from
    having to encode refusal behaviour.
    """
    models = sorted(flat_df['model'].unique())

    rows = []
    for model in models:
        md = flat_df[flat_df['model'] == model]
        for qid in QUESTION_ORDER:
            grp = md[md['question_id'] == qid]
            if len(grp) == 0:
                rate = np.nan
            else:
                rate = (~grp['is_valid']).sum() / len(grp)
            rows.append({'model': model, 'question_id': qid, 'rate': rate})
    rate_df = pd.DataFrame(rows)
    pivot = rate_df.pivot(index='question_id', columns='model', values='rate')
    pivot = pivot.reindex(QUESTION_ORDER)[models]

    cmap = LinearSegmentedColormap.from_list(
        'refusal', ['#ffffff', '#fdbb84', '#e34a33', '#7f0000']
    )

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.3), 6))
    im = ax.imshow(pivot.values.astype(float),
                   aspect='auto', cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([model_label(m) for m in models],
                       rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(QUESTION_ORDER)))
    ax.set_yticklabels(
        [QUESTION_LABELS.get(q, q).replace('\n', ' ') for q in QUESTION_ORDER],
        fontsize=9,
    )

    for i, qid in enumerate(QUESTION_ORDER):
        for j, model in enumerate(models):
            val = pivot.loc[qid, model] if (qid in pivot.index and
                                            model in pivot.columns) else np.nan
            if not np.isnan(val):
                txt_color = 'white' if val > 0.55 else 'black'
                ax.text(j, i, f'{val:.0%}',
                        ha='center', va='center', fontsize=8, color=txt_color)

    plt.colorbar(im, ax=ax,
                 label='Refusal rate  (fraction of responses that were invalid/refused)')
    ax.set_title(
        'Refusal Rate per Model × Question  (prompt-sensitivity dataset)\n'
        'Refusals are an orthogonal signal to cultural-map position: this shows '
        'where each model declines to engage',
        fontsize=11, fontweight='bold',
    )

    plt.tight_layout()
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

    # Figure 5: refusal-rate heatmap
    plot_refusal_heatmap(flat_df, OUTPUTS_DIR / f"refusal_rates{ms_tag}_{ts}.png")


if __name__ == '__main__':
    main()
