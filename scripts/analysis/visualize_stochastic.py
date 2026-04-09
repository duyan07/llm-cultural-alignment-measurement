"""
Stochastic Sensitivity Visualization

Reads the flat + summary CSVs produced by stochastic_sensitivity.py and
generates two figures:

1. Per-question response distributions (strip / bar plots per question)
2. Stability summary — mode consistency across questions and models

Usage:
    python scripts/analysis/visualize_stochastic.py
    python scripts/analysis/visualize_stochastic.py --flat data/results/stochastic/stochastic_flat_<ts>.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

RESULTS_DIR = Path("data/results/stochastic")
OUTPUTS_DIR = Path("outputs/stochastic")

QUESTION_ORDER = ['A008', 'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006', 'Y002', 'Y003']
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

NUMERIC_QUESTIONS = {'A008', 'E018', 'F063', 'F118', 'F120', 'G006'}
CATEGORICAL_QUESTIONS = {'A165', 'E025'}
MULTI_QUESTIONS = {'Y002', 'Y003'}


# ── helpers ─────────────────────────────────────────────────────────────────

def latest_files():
    """Return most recently created flat + summary CSV pair."""
    flats = sorted(RESULTS_DIR.glob("stochastic_flat_*.csv"))
    summaries = sorted(RESULTS_DIR.glob("stochastic_summary_*.csv"))
    if not flats:
        raise FileNotFoundError(f"No stochastic_flat_*.csv in {RESULTS_DIR}")
    return flats[-1], summaries[-1]


def parse_value(v):
    """Deserialise JSON-encoded parsed_value."""
    try:
        return json.loads(v)
    except (json.JSONDecodeError, TypeError):
        return None


# ── Figure 1: per-question distributions ────────────────────────────────────

def plot_distributions(flat_df: pd.DataFrame, model: str, tone: str, variant: int,
                       out_path: Path) -> None:
    """
    One subplot per question.
    - Numeric: scatter strip + mean line + ±1 std band
    - Categorical: horizontal bar chart of choice counts
    - Multi: horizontal bar chart of choice-set counts (Y002) or item frequencies (Y003)
    """
    subset = flat_df[
        (flat_df['model'] == model) &
        (flat_df['tone'] == tone) &
        (flat_df['variant'] == variant)
    ]

    n_q = len(QUESTION_ORDER)
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    fig.suptitle(
        f"Response Distributions — {model}  |  tone={tone}  variant={variant}\n"
        f"(10 seeds, temperature=1.0)",
        fontsize=13, fontweight='bold', y=1.01
    )

    for ax, qid in zip(axes, QUESTION_ORDER):
        q_rows = subset[subset['question_id'] == qid].copy()
        q_rows['parsed'] = q_rows['parsed_value'].apply(parse_value)
        values = list(q_rows['parsed'])
        label = QUESTION_LABELS[qid]

        if qid in NUMERIC_QUESTIONS:
            _plot_numeric_strip(ax, values, qid, label)
        elif qid in CATEGORICAL_QUESTIONS:
            _plot_categorical_bar(ax, values, qid, label)
        elif qid == 'Y002':
            _plot_y002_bar(ax, values, label)
        elif qid == 'Y003':
            _plot_y003_bar(ax, values, label)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def _plot_numeric_strip(ax, values, qid, label):
    valid = [v for v in values if v is not None]
    if not valid:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(label, fontsize=9)
        return

    jitter = np.random.uniform(-0.15, 0.15, size=len(valid))
    ax.scatter(jitter, valid, alpha=0.75, color='steelblue', s=50, zorder=3)

    mean = np.mean(valid)
    std = np.std(valid, ddof=1) if len(valid) > 1 else 0.0
    ax.axhline(mean, color='tomato', linewidth=2, zorder=4, label=f'mean={mean:.2f}')
    ax.axhspan(mean - std, mean + std, alpha=0.15, color='tomato', zorder=2)

    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_title(label, fontsize=9)
    ax.set_ylabel('response', fontsize=8)
    ax.legend(fontsize=7, loc='upper right')

    # Annotate std
    ax.text(0.02, 0.02, f'std={std:.2f}', transform=ax.transAxes,
            fontsize=7, color='gray', va='bottom')


def _plot_categorical_bar(ax, values, qid, label):
    from collections import Counter
    valid = [v for v in values if v is not None]
    counts = Counter(str(v) for v in valid)

    if qid == 'A165':
        all_opts = ['A', 'B']
    else:
        all_opts = ['A', 'B', 'C']

    freqs = [counts.get(o, 0) for o in all_opts]
    colors = ['steelblue', 'salmon', 'mediumseagreen'][:len(all_opts)]
    bars = ax.bar(all_opts, freqs, color=colors[:len(all_opts)], edgecolor='white', linewidth=0.5)

    for bar, f in zip(bars, freqs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(f), ha='center', va='bottom', fontsize=8)

    ax.set_ylim(0, 11)
    ax.set_title(label, fontsize=9)
    ax.set_ylabel('count (of 10 seeds)', fontsize=8)


def _plot_y002_bar(ax, values, label):
    from collections import Counter
    valid = [tuple(v) if isinstance(v, list) else v for v in values if v is not None]
    counts = Counter(str(v) for v in valid)

    labels = list(counts.keys())
    freqs = list(counts.values())
    y_pos = range(len(labels))

    ax.barh(y_pos, freqs, color='steelblue', edgecolor='white')
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('count', fontsize=8)
    ax.set_title(label, fontsize=9)
    ax.set_xlim(0, 11)


def _plot_y003_bar(ax, values, label):
    """Show per-item selection frequency across seeds."""
    from collections import Counter
    items = [
        'Good manners', 'Independence', 'Hard work', 'Responsibility',
        'Imagination', 'Tolerance', 'Thrift', 'Determination',
        'Religious faith', 'Unselfishness', 'Obedience'
    ]
    full_names = [
        'Good manners', 'Independence', 'Hard work', 'Feeling of responsibility',
        'Imagination', 'Tolerance and respect for other people',
        'Thrift, saving money and things', 'Determination, perseverance',
        'Religious faith', 'Not being selfish (unselfishness)', 'Obedience'
    ]

    counts = Counter()
    for v in values:
        if isinstance(v, list):
            for item in v:
                counts[item] += 1

    freqs = [counts.get(fn, 0) for fn in full_names]
    y_pos = range(len(items))

    colors = ['steelblue' if f > 0 else 'lightgray' for f in freqs]
    ax.barh(list(y_pos), freqs, color=colors, edgecolor='white')
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(items, fontsize=6)
    ax.set_xlabel('times selected (of 10 seeds)', fontsize=8)
    ax.set_title(label, fontsize=9)
    ax.set_xlim(0, 11)


# ── Figure 2: stability summary ──────────────────────────────────────────────

def plot_stability_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    """
    Heatmap-style grid: rows = questions, columns = models.
    Cell colour = mode consistency (1.0 = fully stable, 0.0 = max variance).
    """
    # Pivot: one row per question, one col per model (averaged over tone/variant)
    pivot = summary_df.groupby(['question_id', 'model'])['mode_consistency'].mean().unstack('model')
    pivot = pivot.reindex(QUESTION_ORDER)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.4), 5))

    im = ax.imshow(pivot.values.astype(float), aspect='auto',
                   cmap='RdYlGn', vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.split(':')[0] for c in pivot.columns],
                       rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([QUESTION_LABELS.get(q, q).replace('\n', ' ')
                        for q in pivot.index], fontsize=9)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color='black' if 0.3 < val < 0.85 else 'white')

    plt.colorbar(im, ax=ax, label='Mode consistency (1.0 = always same answer)')
    ax.set_title('Stochastic Stability — Mode Consistency per Question × Model\n'
                 '(10 seeds, temperature=1.0)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize stochastic sensitivity results.')
    parser.add_argument('--flat', type=Path, help='Path to stochastic_flat_*.csv')
    parser.add_argument('--summary', type=Path, help='Path to stochastic_summary_*.csv')
    parser.add_argument('--tone', default='standard', help='Tone to plot (default: standard)')
    parser.add_argument('--variant', type=int, default=0, help='Variant index to plot (default: 0)')
    args = parser.parse_args()

    flat_path, summary_path = (args.flat, args.summary) if args.flat else latest_files()

    print(f"Reading: {flat_path}")
    flat_df = pd.read_csv(flat_path)
    summary_df = pd.read_csv(summary_path)

    models = flat_df['model'].unique()
    print(f"Models found: {list(models)}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = flat_path.stem.replace('stochastic_flat_', '')

    # Figure 1: per-question distributions for each model
    for model in models:
        safe_model = model.replace('/', '_').replace(':', '_')
        out = OUTPUTS_DIR / f"distributions_{safe_model}_{args.tone}_v{args.variant}_{ts}.png"
        plot_distributions(flat_df, model, args.tone, args.variant, out)

    # Figure 2: stability summary heatmap (all models)
    out2 = OUTPUTS_DIR / f"stability_summary_{ts}.png"
    plot_stability_summary(summary_df, out2)


if __name__ == '__main__':
    main()
