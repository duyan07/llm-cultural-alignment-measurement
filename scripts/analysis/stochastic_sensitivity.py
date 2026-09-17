"""
Stochastic Sensitivity Analysis

For each (model, question) pair, runs the same prompt 10 times to measure
sampling-randomness variance. Tone and variant are held fixed (standard,
variant 0) so this isolates seed/sampling noise from prompt-wording noise
(the latter is prompt_sensitivity.py's job).

Provider handling:
  - Ollama and OpenAI both accept a `seed` parameter — we run seeds 0–9 to
    get 10 deterministic-given-seed samples from the temperature distribution.
  - Anthropic (Claude) doesn't accept a seed — we run 10 independent calls at
    temperature=1.0 instead. Each call is an i.i.d. sample from the same
    response distribution, so the variance estimate is the same population
    quantity. The only difference is reproducibility of individual outputs.
    The 'seed' column for Anthropic rows is a sample index (0–9), not a
    reproducibility key.

Usage:
    # All available models (Ollama + API) auto-detected
    python scripts/analysis/stochastic_sensitivity.py

    # Open Ollama models only
    python scripts/analysis/stochastic_sensitivity.py --model-set open

    # Proprietary API models only
    python scripts/analysis/stochastic_sensitivity.py --model-set api

    # Specific models (provider auto-detected from name)
    python scripts/analysis/stochastic_sensitivity.py --models gemma2:2b gpt-4o claude-sonnet-4-6
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.llm_interface import LLMQueryWrapper
from src.response_parser import ResponseParser
from src.prompts import QUESTIONS, TONES, format_full_prompt

PROPRIETARY_MODELS = {
    'gpt-4o':            'openai',
    'gpt-4o-mini':       'openai',
    'claude-sonnet-4-6': 'anthropic',
}

RESULTS_DIR = Path("data/results/stochastic")
LOGS_DIR = Path("logs/stochastic")

DEFAULT_SEEDS = list(range(10))

DEFAULT_TEMPERATURE = 1.0

def _numeric_stats(values: list) -> dict:
    """Descriptive statistics for a list of numeric values."""
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv = std / abs(mean) if mean != 0 else float('nan')
    return {
        'mean': round(mean, 4),
        'std': round(std, 4),
        'cv': round(cv, 4),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
    }


def _categorical_stats(values: list) -> dict:
    """Mode consistency and entropy for a list of categorical values."""
    if not values:
        return {}
    counts = Counter(str(v) for v in values)
    total = len(values)
    mode_val, mode_count = counts.most_common(1)[0]
    consistency = mode_count / total

    # Shannon entropy (bits)
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    return {
        'mode': mode_val,
        'mode_consistency': round(consistency, 4),
        'entropy_bits': round(entropy, 4),
        'n_unique': len(counts),
        'value_counts': dict(counts),
    }


def compute_distribution_stats(parsed_values: list, question_info: dict) -> dict:
    """
    Compute distribution statistics over a list of parsed values.

    For numeric questions: mean, std, CV, range.
    For categorical/multi questions: mode consistency, entropy.
    """
    valid = [v for v in parsed_values if v is not None]
    n_valid = len(valid)
    n_total = len(parsed_values)

    base = {
        'n_total': n_total,
        'n_valid': n_valid,
        'parse_rate': round(n_valid / n_total, 4) if n_total > 0 else 0.0,
    }

    if n_valid == 0:
        return base

    rtype = question_info['response_type']

    if rtype == 'numeric':
        base.update(_numeric_stats(valid))
        base.update(_categorical_stats(valid))   # adds mode consistency too

    elif rtype == 'categorical':
        base.update(_categorical_stats(valid))

    elif rtype == 'multi_numeric':
        # Flatten lists to tuples for consistency counting
        as_tuples = [tuple(v) if isinstance(v, list) else v for v in valid]
        base.update(_categorical_stats(as_tuples))

    elif rtype == 'multi_categorical':
        # Treat each selection set as a frozenset for mode counting
        as_frozen = [frozenset(v) if isinstance(v, list) else v for v in valid]
        base.update(_categorical_stats(as_frozen))

    return base

class StochasticSensitivityRunner:
    """
    Runs each prompt N times to measure response sampling variance.

    For Ollama and OpenAI models, the seed parameter is passed (seeds 0–9)
    so each call is deterministic-given-seed. For Anthropic models, the API
    does not expose a seed parameter, so we run 10 independent calls at
    temperature=1.0 — the variance estimate is the same population quantity,
    only per-call reproducibility differs (see module docstring).
    """

    def __init__(self,
                 models: Optional[List] = None,
                 seeds: List[int] = DEFAULT_SEEDS,
                 temperature: float = DEFAULT_TEMPERATURE,
                 tones: Optional[List[str]] = None,
                 variants: Optional[List[int]] = None,
                 model_set: str = 'all'):
        self.model_set = model_set
        # Internally, models is a list of (name, provider) tuples.
        if models is None:
            self.models = self._build_model_list()
        else:
            self.models = self._normalize_models(models)
        self.seeds = seeds
        self.temperature = temperature
        # Fixed to one tone and variant — prompt wording is held constant so that
        # only seed variation is measured. Cross-prompt sensitivity is prompt_sensitivity.py's job.
        self.tones = tones or ['standard']
        self.variants = variants if variants is not None else [0]

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = LOGS_DIR / f"stochastic_{self.timestamp}.jsonl"
        print(f"Logging raw responses to: {self.log_file}")

        total = (len(self.models) * len(self.tones) * len(self.variants)
                 * len(QUESTIONS) * len(self.seeds))
        print(f"\nPlan: {len(self.models)} models × {len(self.tones)} tones "
              f"× {len(self.variants)} variants × {len(QUESTIONS)} questions "
              f"× {len(self.seeds)} seeds = {total} queries\n")
        for name, provider in self.models:
            seed_note = ('seedless — 10 i.i.d. samples'
                         if provider == 'anthropic' else 'seeds 0–9')
            print(f"  - {name}  [{provider}]  ({seed_note})")
        print()

    # ── Model discovery ──────────────────────────────────────────────────────

    def _normalize_models(self, models: List) -> List[Tuple[str, str]]:
        """Accept list of names or (name, provider) tuples; infer provider when missing."""
        out = []
        for entry in models:
            if isinstance(entry, tuple):
                out.append(entry)
            else:
                provider = PROPRIETARY_MODELS.get(entry, 'ollama')
                out.append((entry, provider))
        return out

    def _build_model_list(self) -> List[Tuple[str, str]]:
        result: List[Tuple[str, str]] = []
        if self.model_set in ('all', 'open'):
            result.extend((m, 'ollama') for m in self._discover_ollama_models())
        if self.model_set in ('all', 'api'):
            for model, provider in PROPRIETARY_MODELS.items():
                if provider == 'openai' and os.getenv('OPENAI_API_KEY'):
                    result.append((model, provider))
                elif provider == 'anthropic' and os.getenv('ANTHROPIC_API_KEY'):
                    result.append((model, provider))
        if not result:
            print(f"Warning: no models found for --model-set '{self.model_set}'")
        return result

    def _discover_ollama_models(self) -> List[str]:
        """Auto-detect local Ollama models."""
        try:
            import ollama
            models = ollama.list()
            names = [m['model'] for m in models.get('models', [])]
            print(f"Discovered {len(names)} Ollama models: {names}")
            return names
        except Exception as e:
            print(f"Could not auto-detect Ollama models: {e}")
            return []

    def _make_wrapper(self, model_name: str, provider: str,
                      seed: int) -> LLMQueryWrapper:
        """Build a query wrapper for a given (model, provider, seed)."""
        kwargs = dict(
            provider=provider,
            model_name=model_name,
            temperature=self.temperature,
            max_tokens=256,
        )
        # Ollama and OpenAI accept a seed parameter; Anthropic does not.
        # For Anthropic the 'seed' is just a sample index recorded in our
        # output for bookkeeping — the call itself is genuinely stochastic.
        if provider in ('ollama', 'openai'):
            kwargs['seed'] = seed
        if provider == 'openai':
            kwargs['api_key'] = os.getenv('OPENAI_API_KEY')
        elif provider == 'anthropic':
            kwargs['api_key'] = os.getenv('ANTHROPIC_API_KEY')
        return LLMQueryWrapper(**kwargs)

    # ── Core query loop ──────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Run all seeds for all (model, tone, variant, question) combinations.

        Returns a flat DataFrame with one row per individual query.
        """
        all_rows = []

        for model_name, provider in self.models:
            print(f"\n{'='*70}")
            print(f"MODEL: {model_name}  ({provider})")
            print(f"{'='*70}")

            for tone in self.tones:
                for variant_idx in self.variants:
                    rows = self._run_model_variant(
                        model_name, provider, tone, variant_idx
                    )
                    all_rows.extend(rows)

        df = pd.DataFrame(all_rows)

        flat_path = RESULTS_DIR / f"stochastic_flat_{self.model_set}_{self.timestamp}.csv"
        df.to_csv(flat_path, index=False)
        print(f"\nSaved flat results: {flat_path}")

        summary_df = self._compute_summary(df)
        summary_path = RESULTS_DIR / f"stochastic_summary_{self.model_set}_{self.timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved distribution summary: {summary_path}")

        self._print_report(summary_df)

        return df

    def _run_model_variant(self, model_name: str, provider: str,
                           tone: str, variant_idx: int) -> list:
        """Run all seeds × questions for one (model, tone, variant) combo."""
        rows = []

        for question_id, question_info in QUESTIONS.items():
            prompt = format_full_prompt(
                question_id=question_id,
                country_name=None,
                variant=variant_idx,
                tone=tone
            )

            seed_responses = []

            for seed in self.seeds:
                wrapper = self._make_wrapper(model_name, provider, seed)

                result = wrapper.query(
                    system_prompt=prompt['system'],
                    user_prompt=prompt['user'],
                    metadata={
                        'question_id': question_id,
                        'variant': variant_idx,
                        'tone': tone,
                        'seed': seed,
                        'run_type': 'stochastic',
                    }
                )

                raw = result.get('response') or ''
                parsed = ResponseParser.parse_by_type(raw, question_info)
                is_valid = parsed is not None

                row = {
                    'model': model_name,
                    'provider': provider,
                    'tone': tone,
                    'variant': variant_idx,
                    'question_id': question_id,
                    'seed': seed,
                    'raw_response': raw,
                    'parsed_value': json.dumps(parsed),
                    'is_valid': is_valid,
                    'error': result.get('error'),
                }
                rows.append(row)
                seed_responses.append(parsed)

                self._log_entry({
                    'timestamp': datetime.now().isoformat(),
                    'model': model_name,
                    'provider': provider,
                    'tone': tone,
                    'variant': variant_idx,
                    'question_id': question_id,
                    'question_name': question_info['name'],
                    'seed': seed,
                    'temperature': self.temperature,
                    'system_prompt': prompt['system'],
                    'user_prompt': prompt['user'],
                    'raw_response': raw,
                    'parsed_value': parsed,
                    'is_valid': is_valid,
                    'error': result.get('error'),
                })

            n_valid = sum(1 for v in seed_responses if v is not None)
            print(f"  {model_name} | {tone} | v{variant_idx} | {question_id}"
                  f" — {n_valid}/{len(self.seeds)} valid seeds")

        return rows

    def _log_entry(self, entry: dict) -> None:
        """Append one entry to the JSONL log."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    # ── Distribution summary ─────────────────────────────────────────────────

    def _compute_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate flat results into per-(model, tone, variant, question) stats.
        """
        summary_rows = []
        group_cols = ['model', 'tone', 'variant', 'question_id']

        for keys, group in df.groupby(group_cols):
            model, tone, variant, question_id = keys
            question_info = QUESTIONS[question_id]

            parsed_values = []
            for pv in group['parsed_value']:
                try:
                    parsed_values.append(json.loads(pv))
                except (json.JSONDecodeError, TypeError):
                    parsed_values.append(None)

            stats = compute_distribution_stats(parsed_values, question_info)

            row = {
                'model': model,
                'tone': tone,
                'variant': variant,
                'question_id': question_id,
                'question_name': question_info['name'],
                'response_type': question_info['response_type'],
                'n_seeds': len(self.seeds),
            }
            row.update(stats)
            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    # ── Report ───────────────────────────────────────────────────────────────

    def _print_report(self, summary_df: pd.DataFrame) -> None:
        """Print a human-readable summary of response distributions."""
        print("\n" + "="*70)
        print("STOCHASTIC SENSITIVITY REPORT")
        print(f"Temperature: {self.temperature}  |  Seeds: {self.seeds}")
        print("="*70)

        for model, model_df in summary_df.groupby('model'):
            print(f"\n  Model: {model}")

            for tone, tone_df in model_df.groupby('tone'):
                print(f"    Tone: {tone}")

                for _, row in tone_df.iterrows():
                    rtype = row['response_type']
                    qid = row['question_id']
                    parse_rate = row.get('parse_rate', float('nan'))

                    if rtype == 'numeric':
                        detail = (f"mean={row.get('mean', 'N/A'):.3f}  "
                                  f"std={row.get('std', 'N/A'):.3f}  "
                                  f"cv={row.get('cv', 'N/A'):.3f}")
                    else:
                        cons = row.get('mode_consistency', float('nan'))
                        ent = row.get('entropy_bits', float('nan'))
                        detail = (f"mode_consistency={cons:.3f}  "
                                  f"entropy={ent:.3f} bits")

                    print(f"      {qid:<6s}  parse={parse_rate:.0%}  {detail}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Stochastic sensitivity: run each prompt with N seeds.'
    )
    parser.add_argument(
        '--model-set', choices=['all', 'open', 'api'], default='all',
        help='all (default) runs Ollama + API models; open runs Ollama only; '
             'api runs proprietary models only. Anthropic models are queried '
             'seedlessly (10 i.i.d. samples at temperature=1.0).'
    )
    parser.add_argument('--models', nargs='+',
                        help='Explicit model names; provider auto-detected '
                             '(known proprietary models routed to OpenAI/Anthropic, '
                             'rest to Ollama). Overrides --model-set.')
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=DEFAULT_SEEDS,
                        help=f'Seeds to use (default: {DEFAULT_SEEDS})')
    parser.add_argument('--temperature', type=float,
                        default=DEFAULT_TEMPERATURE,
                        help=f'Sampling temperature (default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--tones', nargs='+',
                        choices=list(TONES.keys()),
                        help='Override tone (default: standard only)')
    parser.add_argument('--variants', nargs='+', type=int,
                        help='Override variant indices (default: 0 only)')
    args = parser.parse_args()

    runner = StochasticSensitivityRunner(
        models=args.models,
        seeds=args.seeds,
        temperature=args.temperature,
        tones=args.tones,
        variants=args.variants,
        model_set=args.model_set,
    )
    runner.run()


if __name__ == '__main__':
    main()
