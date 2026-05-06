"""
Shared constants and helpers for visualization scripts.

Anything that was previously duplicated across the analysis/baseline viz
scripts (model metadata, question metadata, tone colours, the
parsed-value JSON decoder, the PCA / baseline loaders) lives here so each
script imports a single source of truth.
"""

import json
from pathlib import Path

import pandas as pd

from src.cultural_map import CulturalMapGenerator
from src.geo_data import COUNTRY_NAMES, ISO3_TO_ZONE, load_iso3_lookup

# ── Paths ─────────────────────────────────────────────────────────────────────

BASELINE_PATH = Path("data/processed/cultural_map_coordinates.csv")
IVS_PATH      = Path("data/processed/ivs_2005-2022.csv")

# ── Question metadata ─────────────────────────────────────────────────────────

QUESTION_ORDER = ['A008', 'A165', 'E018', 'E025', 'F063',
                  'F118', 'F120', 'G006', 'Y002', 'Y003']

# Multi-line labels for axis ticks
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

# Single-line labels for inline use
QUESTION_SHORT = {
    'A008': 'Happiness',   'A165': 'Trust',
    'E018': 'Authority',   'E025': 'Petition',
    'F063': 'God',         'F118': 'Homosexuality',
    'F120': 'Abortion',    'G006': 'Nationality',
    'Y002': 'Post-Mat.',   'Y003': 'Autonomy',
}

NUMERIC_QUESTIONS = {'A008', 'E018', 'F063', 'F118', 'F120', 'G006'}

# ── Tone colours ──────────────────────────────────────────────────────────────

TONE_COLORS = {
    'standard':  '#4878cf',
    'friendly':  '#6acc65',
    'combative': '#d65f5f',
}

# ── Model metadata ────────────────────────────────────────────────────────────

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

# ── Helpers ───────────────────────────────────────────────────────────────────

def model_label_inline(name: str) -> str:
    """Single-line display label, e.g. 'qwen2.5 (7B)'. Used in legends, plot annotations."""
    short = name.split('/')[-1].split(':')[0]
    params = MODEL_PARAMS.get(name)
    return f"{short} ({params})" if params else short


def model_label_multiline(name: str) -> str:
    """Two-line label for axis ticks, e.g. 'qwen2.5\\n(7B)'."""
    short = name.split('/')[-1].split(':')[0]
    params = MODEL_PARAMS.get(name)
    return f"{short}\n({params})" if params else short


def parse_value(v):
    """JSON-decode a parsed_value field from a flat CSV. Returns None on failure."""
    try:
        return json.loads(str(v))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_pca(verbose: bool = True) -> CulturalMapGenerator:
    """Re-fit PCA on the IVS dataset to recover the baseline projection."""
    if verbose:
        print("Fitting PCA on IVS data...")
    ivs_df = pd.read_csv(IVS_PATH, low_memory=False)
    gen = CulturalMapGenerator(ivs_df)
    gen.fit()
    if verbose:
        print("PCA ready.\n")
    return gen


def load_baseline() -> pd.DataFrame:
    """Load 88-country baseline with ISO-3, country name, and cultural-zone annotations."""
    df = pd.read_csv(BASELINE_PATH)
    iso_lookup = load_iso3_lookup(IVS_PATH)
    df['iso3'] = df['country_code'].map(iso_lookup).fillna('???')
    df['name'] = df['iso3'].map(COUNTRY_NAMES).fillna(df['iso3'])
    df['zone'] = df['iso3'].map(ISO3_TO_ZONE).fillna('Other')
    return df
