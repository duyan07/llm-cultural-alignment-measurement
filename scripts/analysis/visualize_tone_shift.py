"""
Directional Tone-Shift Heatmap

Visualizes how each model's baseline (x, y) coordinate on the cultural map
moves when the system-prompt tone changes from standard to friendly or
combative.

For each model, we compute four signed shifts:
    Δx_fri = x_friendly  − x_standard
    Δy_fri = y_friendly  − y_standard
    Δx_com = x_combative − x_standard
    Δy_com = y_combative − y_standard

Positive Δx = shifted toward Self-Expression;   negative = toward Survival.
Positive Δy = shifted toward Secular-Rational;  negative = toward Traditional.

Reported as a heatmap with a diverging colormap (RdBu_r) so both direction and
magnitude are readable at a glance. A final "Panel mean" row summarizes the
average directional effect across all projectable models — this reveals
systematic tone effects (e.g. combative tone shifting the whole panel toward
Secular-Rational) that per-model rows alone would obscure.

Data sources (kept in sync with the rest of the pipeline):
  open-weight baseline: baseline_models_{tone}_open_20260707_050850.csv
  API-model baseline:   baseline_models_{tone}_all_20260506_192420.csv

Llama3.1-8B and AceGPT-7B are unprojectable in all tones (categorical refusal
on multiple items) and are excluded from the panel automatically.

Usage:
    python scripts/analysis/visualize_tone_shift.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

BASELINE_DIR = Path('data/results/baseline')
OUTPUTS_DIR  = Path('outputs/prompt_sensitivity')

OPEN_TS = '20260707_050850'
API_TS  = '20260506_192420'
API_MODELS = {'gpt-4o', 'gpt-4o-mini', 'claude-sonnet-4-6'}

MODEL_LABEL = {
    'gpt-4o':            'GPT-4o',
    'gpt-4o-mini':       'GPT-4o-mini',
    'claude-sonnet-4-6': 'Claude Sonnet 4.6',
    'gemma2:2b':         'Gemma2-2B',
    'phi3:mini':         'Phi3-Mini',
    'mistral:7b':        'Mistral-7B',
    'qwen2.5:1.5b':      'Qwen2.5-1.5B',
    'qwen2.5:3b':        'Qwen2.5-3B',
    'qwen2.5:7b':        'Qwen2.5-7B',
    'yi:6b':             'Yi-6B',
}

DEFAULT_ORDER = [
    'Mistral-7B', 'Yi-6B', 'Claude Sonnet 4.6', 'Qwen2.5-1.5B',
    'GPT-4o', 'Qwen2.5-7B', 'GPT-4o-mini', 'Gemma2-2B',
    'Phi3-Mini', 'Qwen2.5-3B',
]


def load_tone_coords(tone: str) -> pd.DataFrame:
    """Load per-model (x, y) for a single tone, merging open + API sources."""
    op = pd.read_csv(BASELINE_DIR / f'baseline_models_{tone}_open_{OPEN_TS}.csv')
    ap = pd.read_csv(BASELINE_DIR / f'baseline_models_{tone}_all_{API_TS}.csv')
    ap = ap[ap.model.isin(API_MODELS)]
    df = pd.concat([op, ap], ignore_index=True)[
        ['model', 'survival_selfexpression', 'traditional_secular']
    ]
    df.columns = ['model', f'x_{tone}', f'y_{tone}']
    return df


def build_shift_table() -> pd.DataFrame:
    m = (
        load_tone_coords('standard')
        .merge(load_tone_coords('friendly'), on='model')
        .merge(load_tone_coords('combative'), on='model')
    )
    m['label'] = m.model.map(MODEL_LABEL)
    m = m.set_index('label').reindex(DEFAULT_ORDER)

    m['dx_fri'] = m.x_friendly  - m.x_standard
    m['dy_fri'] = m.y_friendly  - m.y_standard
    m['dx_com'] = m.x_combative - m.x_standard
    m['dy_com'] = m.y_combative - m.y_standard
    return m


def plot(m: pd.DataFrame, out_path: Path) -> None:
    per_model = m[['dx_fri', 'dy_fri', 'dx_com', 'dy_com']].values
    panel_mean = per_model.mean(axis=0)          # column-wise mean = panel-level shift
    data = np.vstack([per_model, panel_mean])

    row_labels = list(m.index) + ['Panel mean']
    col_labels = [
        r'$\Delta x$ friendly', r'$\Delta y$ friendly',
        r'$\Delta x$ combative', r'$\Delta y$ combative',
    ]

    vmax = float(np.abs(data).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, aspect='auto', cmap='RdBu_r', norm=norm)

    ax.set_xticks(range(4))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9.5)

    for i, row in enumerate(data):
        for j, val in enumerate(row):
            color = 'white' if abs(val) > vmax * 0.55 else 'black'
            ax.text(j, i, f'{val:+.2f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    ax.axhline(len(m) - 0.5, color='black', linewidth=1.2)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label('Shift on cultural map (Δ from standard tone)', fontsize=9)

    ax.set_title(
        'Directional Tone Shift — how each model moves on the cultural map\n'
        'when tone changes from standard to friendly or combative',
        fontsize=11, fontweight='bold', pad=12,
    )
    ax.set_xlabel(
        'Blue = shifted toward Survival / Traditional     '
        'Red = shifted toward Self-Expression / Secular',
        fontsize=8.5, labelpad=8,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def main() -> None:
    m = build_shift_table()

    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    plot(m, OUTPUTS_DIR / f'tone_shift_heatmap_{ts}.png')

    print('\nSigned shifts (Δ from standard):')
    print(m[['dx_fri', 'dy_fri', 'dx_com', 'dy_com']].round(2).to_string())

    panel_mean = m[['dx_fri', 'dy_fri', 'dx_com', 'dy_com']].mean()
    print('\nPanel mean shift (across projectable models):')
    print(panel_mean.round(3).to_string())


if __name__ == '__main__':
    main()
