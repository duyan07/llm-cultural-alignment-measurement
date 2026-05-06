"""
Variance Decomposition  (Stage 5)

Separates the three sources of disagreement in LLM cultural survey responses:

  σ_seed    — sampling randomness (Stage 3 stochastic: same prompt, 10 different seeds)
  σ_prompt  — wording sensitivity (Stage 4 prompt sensitivity: fixed seed, 30 prompt variants)
  σ_culture — between-model spread per question (the signal we want to detect)

Key question: for each (model, question), does instrument noise (σ_seed + σ_prompt)
dominate the cultural signal (σ_culture)? If so, measurement is unreliable.

All variances are computed in IVS-numeric space so they are comparable across
questions with different response types.

Outputs:
  1. variance_bars_<ts>.png    — σ_seed vs σ_prompt per question, σ_culture as
                                  reference line; one panel per model
  2. noise_signal_heatmap_<ts>.png — noise-to-signal ratio per (model, question)
  3. combined_map_<ts>.png     — seed clouds + prompt clouds overlaid side-by-side

Usage:
    python scripts/analysis/variance_decomposition.py
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
from src.geo_data import ZONE_COLORS
from src.viz_common import (
    QUESTION_ORDER,
    QUESTION_SHORT,
    MODEL_COLORS,
    model_label_inline as model_label,
    parse_value as parse_val,
    load_pca,
    load_baseline as load_baseline_df,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

STOCHASTIC_DIR   = Path("data/results/stochastic")
PROMPT_SENS_DIR  = Path("data/results/prompt_sensitivity")
OUTPUTS_DIR      = Path("outputs/variance_decomposition")


# ── Data loading + variance computation ──────────────────────────────────────

def to_numeric(parsed, qid):
    if parsed is None:
        return None
    return ResponseParser.to_ivs_numeric(parsed, QUESTIONS[qid])


def _filter_by_model_set(paths, model_set):
    """Keep only paths containing _{model_set}_ if model_set is specified."""
    if not model_set:
        return paths
    return [p for p in paths if f'_{model_set}_' in p.name]


def load_stochastic(model_set=None) -> pd.DataFrame:
    """σ_seed per (model, question) — std across 10 seeds."""
    paths = sorted(STOCHASTIC_DIR.glob("stochastic_flat_*.csv"))
    paths = _filter_by_model_set(paths, model_set)
    if not paths:
        tag = f" matching model_set='{model_set}'" if model_set else ""
        raise FileNotFoundError(f"No stochastic flat CSVs{tag} in {STOCHASTIC_DIR}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.drop_duplicates(subset=['model','tone','variant','question_id','seed'], keep='last')

    rows = []
    for (model, qid), grp in df.groupby(['model', 'question_id']):
        nums = [to_numeric(parse_val(v), qid)
                for v in grp['parsed_value']]
        nums = [v for v in nums if v is not None]
        rows.append({
            'model':      model,
            'question_id': qid,
            'sigma_seed': np.std(nums, ddof=1) if len(nums) > 1 else 0.0,
            'mean_seed':  np.mean(nums) if nums else np.nan,
            'n_seed':     len(nums),
        })
    print(f"Stochastic: {len(paths)} file(s), "
          f"{df['model'].nunique()} models, {len(rows)} (model×question) pairs")
    return pd.DataFrame(rows)


def load_prompt_sensitivity(model_set=None) -> pd.DataFrame:
    """σ_prompt per (model, question) — std across 30 prompt variants."""
    paths = sorted(PROMPT_SENS_DIR.glob("prompt_sensitivity_flat_*.csv"))
    paths = _filter_by_model_set(paths, model_set)
    if not paths:
        tag = f" matching model_set='{model_set}'" if model_set else ""
        raise FileNotFoundError(f"No prompt sensitivity flat CSVs{tag} in {PROMPT_SENS_DIR}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.drop_duplicates(subset=['model','tone','variant','question_id'], keep='last')

    rows = []
    for (model, qid), grp in df.groupby(['model', 'question_id']):
        nums = [to_numeric(parse_val(v), qid)
                for v in grp['parsed_value']]
        nums = [v for v in nums if v is not None]
        rows.append({
            'model':        model,
            'question_id':  qid,
            'sigma_prompt': np.std(nums, ddof=1) if len(nums) > 1 else 0.0,
            'mean_prompt':  np.mean(nums) if nums else np.nan,
            'n_prompt':     len(nums),
        })
    print(f"Prompt sensitivity: {len(paths)} file(s), "
          f"{df['model'].nunique()} models, {len(rows)} (model×question) pairs")
    return pd.DataFrame(rows)


def compute_culture_signal(stoch: pd.DataFrame, prompt: pd.DataFrame) -> pd.DataFrame:
    """
    σ_culture per question = std of model mean responses across all models.

    Uses prompt-sensitivity means (30-variant average) as the best estimate of
    each model's stable response for a question.
    """
    rows = []
    for qid in QUESTION_ORDER:
        means = prompt[prompt['question_id'] == qid]['mean_prompt'].dropna().values
        rows.append({
            'question_id':  qid,
            'sigma_culture': np.std(means, ddof=1) if len(means) > 1 else 0.0,
            'n_models':      len(means),
        })
    return pd.DataFrame(rows)


def build_decomposition(stoch, prompt, culture) -> pd.DataFrame:
    """Merge all three variance sources into one flat table.

    σ_seed is left as NaN for models that lack stochastic data (e.g. API
    models without a seed parameter). σ_noise and noise_to_signal then
    propagate NaN for those rows, signalling 'not measured' rather than
    silently treating it as zero.
    """
    merged = stoch.merge(prompt, on=['model', 'question_id'], how='outer')
    merged = merged.merge(culture, on='question_id', how='left')

    merged['sigma_noise'] = merged['sigma_seed'] + merged['sigma_prompt']
    merged['noise_to_signal'] = merged['sigma_noise'] / \
                                merged['sigma_culture'].replace(0, np.nan)
    return merged


# ── Figure 1: Variance bars per model ────────────────────────────────────────

def plot_variance_bars(decomp: pd.DataFrame, out_path: Path) -> None:
    """
    One panel per model.  For each question: stacked bar σ_seed (blue) + σ_prompt (orange).
    Horizontal dashed line = σ_culture (between-model signal).
    """
    models = sorted(decomp['model'].unique())
    n_models = len(models)
    ncols = 3
    nrows = int(np.ceil(n_models / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4 * nrows),
                             sharey=False)
    axes = np.array(axes).flatten()

    x = np.arange(len(QUESTION_ORDER))
    bar_w = 0.5

    for ax, model in zip(axes, models):
        mdf = decomp[decomp['model'] == model].set_index('question_id')

        seed_raw    = [mdf.loc[q, 'sigma_seed']   if q in mdf.index else np.nan
                       for q in QUESTION_ORDER]
        prompt_vals = [mdf.loc[q, 'sigma_prompt'] if q in mdf.index else 0
                       for q in QUESTION_ORDER]
        cult_vals   = [mdf.loc[q, 'sigma_culture'] if q in mdf.index else 0
                       for q in QUESTION_ORDER]

        seed_unmeasured = all(pd.isna(v) for v in seed_raw)
        seed_vals = [0 if pd.isna(v) else v for v in seed_raw]

        # σ_seed bar — drawn only where measured; otherwise stack starts at 0
        ax.bar(x, seed_vals,   bar_w,
               label='σ_seed (stochastic)',   color='#4878cf', alpha=0.85)
        ax.bar(x, prompt_vals, bar_w,
               bottom=seed_vals,
               label='σ_prompt (prompt sensitivity)', color='#e87e3a', alpha=0.85)

        # σ_culture as step line
        for i, cv in enumerate(cult_vals):
            ax.plot([i - bar_w/2, i + bar_w/2], [cv, cv],
                    color='#2ecc71', linewidth=2.2, zorder=5)

        title = model_label(model)
        if seed_unmeasured:
            title += '\n(σ_seed not measured — API has no seed param)'
        ax.set_title(title, fontsize=10, fontweight='bold',
                     color='#777777' if seed_unmeasured else 'black')
        ax.set_xticks(x)
        ax.set_xticklabels([QUESTION_SHORT[q] for q in QUESTION_ORDER],
                           rotation=40, ha='right', fontsize=7.5)
        ax.set_ylabel('σ (IVS numeric)', fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    # Hide unused panels
    for ax in axes[n_models:]:
        ax.set_visible(False)

    # Shared legend
    seed_patch   = mpatches.Patch(color='#4878cf', label='σ_seed — sampling noise (stochastic)')
    prompt_patch = mpatches.Patch(color='#e87e3a', label='σ_prompt — wording sensitivity (prompt)')
    culture_line = plt.Line2D([0], [0], color='#2ecc71', linewidth=2.2,
                              label='σ_culture — between-model signal')
    fig.legend(handles=[seed_patch, prompt_patch, culture_line],
               loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        'Variance Decomposition — Noise Sources vs Cultural Signal\n'
        'Bars = instrument noise (seed + prompt);  Green line = between-model spread',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Figure 2: Noise-to-signal heatmap ────────────────────────────────────────

def plot_noise_signal_heatmap(decomp: pd.DataFrame, out_path: Path) -> None:
    """
    Rows = questions, cols = models.
    Cell = noise-to-signal ratio (σ_noise / σ_culture).
    > 1.0 = noise exceeds signal → red (failure regime).
    """
    models = sorted(decomp['model'].unique())
    pivot  = decomp.pivot(index='question_id', columns='model',
                          values='noise_to_signal')
    pivot  = pivot.reindex(QUESTION_ORDER)[models]

    raw_values = pivot.values.astype(float)
    nan_mask = np.isnan(raw_values)
    display = np.where(nan_mask, 0.0, raw_values).clip(0, 3)

    cmap = LinearSegmentedColormap.from_list(
        'nsr', ['#2ecc71', '#f1c40f', '#e74c3c']
    )

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.4), 6))
    im = ax.imshow(display, aspect='auto', cmap=cmap, vmin=0, vmax=3)

    # Overlay grey hatch on NaN cells (σ_seed unmeasured for API models)
    if nan_mask.any():
        nan_overlay = np.where(nan_mask, 1.0, np.nan)
        ax.imshow(nan_overlay, aspect='auto',
                  cmap=LinearSegmentedColormap.from_list('grey', ['#cccccc', '#cccccc']),
                  vmin=0, vmax=1)
        # Add hatching via patches
        for i in range(nan_mask.shape[0]):
            for j in range(nan_mask.shape[1]):
                if nan_mask[i, j]:
                    ax.add_patch(mpatches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, hatch='///',
                        edgecolor='#777777', linewidth=0,
                    ))

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([model_label(m) for m in models],
                       rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(QUESTION_ORDER)))
    ax.set_yticklabels([QUESTION_SHORT[q] for q in QUESTION_ORDER], fontsize=9)

    for i, qid in enumerate(QUESTION_ORDER):
        for j, model in enumerate(models):
            val = pivot.loc[qid, model] if (qid in pivot.index and
                                            model in pivot.columns) else np.nan
            if np.isnan(val):
                ax.text(j, i, 'n/a',
                        ha='center', va='center', fontsize=7,
                        color='#555555', style='italic')
            else:
                label_txt = f'{val:.2f}'
                txt_color = 'white' if val > 1.5 or val < 0.3 else 'black'
                ax.text(j, i, label_txt,
                        ha='center', va='center', fontsize=8, color=txt_color)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Noise-to-signal ratio  (> 1.0 = noise exceeds cultural signal)',
                   fontsize=9)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['0', '1.0\n(threshold)', '2', '≥3'])

    ax.axhline(y=-0.5, color='black', linewidth=0)   # padding
    nan_subtitle = (
        '\nGrey hatched cells = σ_seed not measured (API models have no seed param)'
        if nan_mask.any() else ''
    )
    ax.set_title(
        'Noise-to-Signal Ratio  —  (σ_seed + σ_prompt) / σ_culture\n'
        'Red = instrument noise dominates cultural signal  |  '
        f'Green = signal detectable{nan_subtitle}',
        fontsize=11, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Figure 3: Combined cultural map (seed + prompt clouds) ───────────────────

def flat_to_coords(flat_df, group_cols, pca_gen):
    """Project each group defined by group_cols to (x, y) map coordinates
    using pointwise imputation: invalid slots contribute 0 to the PC score.

    Models with zero valid responses anywhere (likely API/system errors) are
    excluded entirely. Each output row carries n_valid (slots that contributed
    real data) so downstream plotting can render filled vs hollow markers.
    """
    if 'model' not in group_cols:
        raise ValueError("flat_to_coords requires 'model' in group_cols")

    excluded_models = set()
    for model in flat_df['model'].unique():
        if flat_df[flat_df['model'] == model]['is_valid'].sum() == 0:
            excluded_models.add(model)
    if excluded_models:
        print(f"    Excluding {len(excluded_models)} model(s) with zero valid "
              f"responses: {sorted(excluded_models)}")
    flat_df = flat_df[~flat_df['model'].isin(excluded_models)]

    rows = []
    for keys, grp in flat_df.groupby(group_cols):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)

        standardized = []
        n_valid = 0
        for qid in QUESTIONS:
            qrow = grp[grp['question_id'] == qid]
            invalid = qrow.empty or not qrow.iloc[0]['is_valid']
            num = None
            if not invalid:
                num = to_numeric(parse_val(qrow.iloc[0]['parsed_value']), qid)

            if num is None:
                standardized.append(0.0)
                continue

            standardized.append(
                (num - pca_gen.question_means_[qid]) / pca_gen.question_stds_[qid]
            )
            n_valid += 1

        if n_valid == 0:
            continue

        scores = pca_gen.pca_model.transform(
            np.array(standardized).reshape(1, -1)
        )[0]
        x = 1.81 * scores[0] + 0.38
        y = 1.61 * scores[1] - 0.01

        row = {'survival_selfexpression': x, 'traditional_secular': y,
               'n_valid': n_valid,
               'fully_valid': n_valid == len(QUESTIONS)}
        for col, val in zip(group_cols, keys_tuple):
            row[col] = val
        rows.append(row)
    return pd.DataFrame(rows)


def plot_combined_map(baseline_df, pca_gen, out_path: Path,
                      prompt_flat_path: Path = None) -> None:
    """
    Side-by-side: left = seed clouds (stochastic), right = prompt clouds (prompt sensitivity).
    Same country background, same model colours.

    prompt_flat_path: path to a specific prompt sensitivity flat CSV.
        Pass the temperature=1.0 run here so both panels are on the same
        footing as the stochastic run (also temperature=1.0). If None, merges all files.
    """
    # Load flat data
    stoch_paths = sorted(STOCHASTIC_DIR.glob("stochastic_flat_*.csv"))

    stoch_flat = pd.concat([pd.read_csv(p) for p in stoch_paths], ignore_index=True)
    stoch_flat = stoch_flat.drop_duplicates(
        subset=['model','tone','variant','question_id','seed'], keep='last')
    stoch_flat['parsed'] = stoch_flat['parsed_value'].apply(parse_val)

    if prompt_flat_path:
        prom_flat = pd.read_csv(prompt_flat_path)
        print(f"  Using prompt flat: {prompt_flat_path}")
    else:
        prom_paths = sorted(PROMPT_SENS_DIR.glob("prompt_sensitivity_flat_*.csv"))
        prom_flat = pd.concat([pd.read_csv(p) for p in prom_paths], ignore_index=True)
    prom_flat = prom_flat.drop_duplicates(
        subset=['model','tone','variant','question_id'], keep='last')
    prom_flat['parsed'] = prom_flat['parsed_value'].apply(parse_val)

    # Annotate title with temperature if available
    prompt_temp = prom_flat['temperature'].iloc[0] if 'temperature' in prom_flat.columns \
                  else 'unknown'

    # Compute coordinates
    print("  Computing seed coordinates (stochastic)...")
    stoch_coords = flat_to_coords(stoch_flat, ['model', 'seed'], pca_gen)
    print("  Computing variant coordinates (prompt sensitivity)...")
    prom_coords = flat_to_coords(prom_flat, ['model', 'tone', 'variant'], pca_gen)

    models = sorted(set(stoch_coords['model'].unique()) |
                    set(prom_coords['model'].unique()))
    color_map = {m: MODEL_COLORS[i % len(MODEL_COLORS)]
                 for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    titles = [
        'Stage 3 — Seed Variance\n(10 seeds × fixed prompt)',
        'Stage 4 — Prompt Variance\n(30 prompt variants × fixed seed)',
    ]
    coords_list = [stoch_coords, prom_coords]

    for ax, coords_df, title in zip(axes, coords_list, titles):
        # Country backdrop
        zone_handles = []
        for zone, color in ZONE_COLORS.items():
            zd = baseline_df[baseline_df['zone'] == zone]
            if zd.empty:
                continue
            ax.scatter(zd['survival_selfexpression'], zd['traditional_secular'],
                       c=color, s=40, alpha=0.50,
                       edgecolors='white', linewidth=0.3, zorder=2)
            if ax is axes[0]:
                zone_handles.append(mpatches.Patch(color=color, label=zone))

        for _, row in baseline_df.iterrows():
            ax.annotate(row['iso3'],
                        xy=(row['survival_selfexpression'],
                            row['traditional_secular']),
                        xytext=(2, 2), textcoords='offset points',
                        fontsize=5.5, color='#555555', zorder=3)

        # Model clouds
        model_handles = []
        for model in models:
            color = color_map[model]
            mdf = coords_df[coords_df['model'] == model]
            if mdf.empty:
                continue

            full_mdf = mdf[mdf['fully_valid']]
            partial_mdf = mdf[~mdf['fully_valid']]

            xs = mdf['survival_selfexpression'].values
            ys = mdf['traditional_secular'].values
            cx, cy = xs.mean(), ys.mean()

            # Filled markers for fully-valid points
            if not full_mdf.empty:
                ax.scatter(full_mdf['survival_selfexpression'],
                           full_mdf['traditional_secular'],
                           c=color, s=50, alpha=0.40,
                           edgecolors=color, linewidth=0.5, zorder=6)
            # Hollow markers for partially-imputed points
            if not partial_mdf.empty:
                ax.scatter(partial_mdf['survival_selfexpression'],
                           partial_mdf['traditional_secular'],
                           facecolors='none', edgecolors=color,
                           s=50, alpha=0.7, linewidth=1.2, zorder=6)

            for sx, sy in zip(xs, ys):
                ax.plot([cx, sx], [cy, sy],
                        color=color, alpha=0.15, linewidth=0.7, zorder=5)

            all_valid = len(full_mdf) == len(mdf)
            if all_valid:
                ax.scatter(cx, cy, c=color, s=320, marker='*',
                           edgecolors='black', linewidth=1.1, zorder=10)
                label_text = model_label(model)
            else:
                ax.scatter(cx, cy, facecolors='white', s=320, marker='*',
                           edgecolors=color, linewidth=1.8, zorder=10)
                label_text = f"{model_label(model)}  ({len(full_mdf)}/{len(mdf)})"
            ax.annotate(label_text, xy=(cx, cy),
                        xytext=(8, 5), textcoords='offset points',
                        fontsize=7.5, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=color,
                                  alpha=0.75, edgecolor='black', linewidth=0.6),
                        zorder=11)
            if ax is axes[0]:
                model_handles.append(
                    plt.Line2D([0], [0], marker='*', color='w',
                               markerfacecolor=color, markeredgecolor='black',
                               markersize=10, label=model_label(model))
                )

        ax.set_xlabel('Survival  ←  →  Self-Expression',
                      fontsize=10, fontweight='bold')
        ax.set_ylabel('Traditional  ←  →  Secular',
                      fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.35)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.35)
        ax.grid(True, alpha=0.15)

    # Legends on left panel only
    leg_zones = axes[0].legend(handles=zone_handles, title='Cultural Zone',
                               loc='upper left', fontsize=6.5,
                               title_fontsize=7.5, framealpha=0.9)
    axes[0].add_artist(leg_zones)
    axes[0].legend(handles=model_handles, title='Models  (★ = centroid)',
                   loc='lower right', fontsize=7.5,
                   title_fontsize=8, framealpha=0.9)

    has_imputed = (
        ('fully_valid' in stoch_coords.columns and (~stoch_coords['fully_valid']).any())
        or ('fully_valid' in prom_coords.columns and (~prom_coords['fully_valid']).any())
    )
    impute_subtitle = (
        '\nFilled = all 10 questions valid    Hollow = ≥1 question imputed'
        ' (slot contributes 0 to the score)    ★ hollow centroid + label "(n_full/N)" '
        'when not all points fully valid    (refer to refusal heatmap for which questions were refused)'
        if has_imputed else ''
    )

    fig.suptitle(
        'Variance Decomposition — Cultural Map Clouds\n'
        f'Left: seed uncertainty (stochastic, temp=1.0)  |  '
        f'Right: prompt wording uncertainty (prompt sensitivity, temp={prompt_temp})\n'
        'Larger cloud = greater instability from that source'
        f'{impute_subtitle}',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(decomp: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("VARIANCE DECOMPOSITION REPORT")
    print("=" * 65)

    # Per-question summary
    print("\n  By question (averaged across models):")
    print(f"  {'Question':<14} {'σ_seed':>8} {'σ_prompt':>9} "
          f"{'σ_culture':>10} {'NSR':>6}")
    print("  " + "-" * 52)
    for qid in QUESTION_ORDER:
        qdf = decomp[decomp['question_id'] == qid]
        print(f"  {QUESTION_SHORT[qid]:<14} "
              f"{qdf['sigma_seed'].mean():>8.3f} "
              f"{qdf['sigma_prompt'].mean():>9.3f} "
              f"{qdf['sigma_culture'].mean():>10.3f} "
              f"{qdf['noise_to_signal'].mean():>6.2f}")

    # Failure regimes: NSR > 1.0
    failures = decomp[decomp['noise_to_signal'] > 1.0][
        ['model', 'question_id', 'sigma_seed', 'sigma_prompt',
         'sigma_culture', 'noise_to_signal']
    ].sort_values('noise_to_signal', ascending=False)

    print(f"\n  Failure regimes (noise > signal, NSR > 1.0): "
          f"{len(failures)} cases")
    if not failures.empty:
        for _, r in failures.head(15).iterrows():
            dominant = 'seed' if r['sigma_seed'] > r['sigma_prompt'] else 'prompt'
            print(f"    {model_label(r['model']):<22s}  "
                  f"{QUESTION_SHORT[r['question_id']]:<14s}  "
                  f"NSR={r['noise_to_signal']:.2f}  "
                  f"(dominant: {dominant})")

    # Best behaved models — skip models with NaN NSR (API models without σ_seed)
    print("\n  Average NSR by model (lower = more reliable):")
    avg_nsr = decomp.groupby('model')['noise_to_signal'].mean().sort_values()
    for model, nsr in avg_nsr.items():
        if pd.isna(nsr):
            print(f"    {model_label(model):<22s}  n/a (σ_seed not measured — API model)")
            continue
        bar = '█' * int(nsr * 5)
        print(f"    {model_label(model):<22s}  {nsr:.3f}  {bar}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Decompose variance: seed vs prompt vs culture.')
    parser.add_argument(
        '--model-set', choices=['all', 'open', 'api'], default=None,
        help='Filter both stochastic and prompt sensitivity files by model set (default: use all files)'
    )
    parser.add_argument('--prompt-flat', type=Path, default=None,
                        help='Specific prompt sensitivity flat CSV for the combined map '
                             '(e.g. the temperature=1.0 run). Overrides --model-set for prompt files.')
    args = parser.parse_args()

    print("Loading stochastic results (Stage 3)...")
    stoch = load_stochastic(model_set=args.model_set)

    print("Loading prompt sensitivity results (Stage 4)...")
    prompt = load_prompt_sensitivity(model_set=None if args.prompt_flat else args.model_set)

    print("Computing between-model cultural signal...")
    culture = compute_culture_signal(stoch, prompt)

    decomp = build_decomposition(stoch, prompt, culture)

    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    ms_tag = f"_{args.model_set}" if args.model_set else ''

    print_report(decomp)

    print("\nGenerating figures...")
    plot_variance_bars(
        decomp, OUTPUTS_DIR / f"variance_bars{ms_tag}_{ts}.png")
    plot_noise_signal_heatmap(
        decomp, OUTPUTS_DIR / f"noise_signal_heatmap{ms_tag}_{ts}.png")

    print("Fitting PCA for combined map...")
    pca_gen     = load_pca()
    baseline_df = load_baseline_df()
    plot_combined_map(baseline_df, pca_gen,
                      OUTPUTS_DIR / f"combined_map{ms_tag}_{ts}.png",
                      prompt_flat_path=args.prompt_flat)

    # Save decomposition table
    out_csv = OUTPUTS_DIR / f"variance_decomposition{ms_tag}_{ts}.csv"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    decomp.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == '__main__':
    main()
