"""
Prompt Sensitivity Analysis  (Stage 4)

Measures instrument instability by varying prompt wording while holding
everything else fixed. For each (model, question), runs all 30 prompt
combinations (3 tones × 10 variants) at a fixed seed / temperature=0,
so any response change is purely attributable to wording, not randomness.

Contrast with stochastic_sensitivity.py (Stage 3), which fixes the prompt
and varies the seed.

Open models (Ollama): seed=0, temperature=0  → fully deterministic
Proprietary models  : temperature=0          → as deterministic as the API allows

Usage:
    # All models — Ollama + any API keys set (default)
    python scripts/analysis/prompt_sensitivity.py

    # Open/local models only
    python scripts/analysis/prompt_sensitivity.py --model-set open

    # Proprietary API models only (requires OPENAI_API_KEY / ANTHROPIC_API_KEY)
    python scripts/analysis/prompt_sensitivity.py --model-set api

    # Specific models (overrides --model-set)
    python scripts/analysis/prompt_sensitivity.py --models gemma2:2b qwen2.5:7b

    # Run at temp=1.0 for map-level comparison with Stage 3 (stochastic)
    python scripts/analysis/prompt_sensitivity.py --temperature 1.0
"""

import sys
import os
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

RESULTS_DIR = Path("data/results/prompt_sensitivity")
LOGS_DIR    = Path("logs/prompt_sensitivity")

# Reference prompt — all sign flips measured relative to this
REFERENCE_TONE    = 'standard'
REFERENCE_VARIANT = 0

PROPRIETARY_MODELS = {
    'gpt-4o':         'openai',
    'gpt-4-turbo':    'openai',
    'claude-sonnet-4-5': 'anthropic',
}


# ── Statistics helpers ────────────────────────────────────────────────────────

def _numeric_stats(values: list) -> dict:
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv   = std / abs(mean) if mean != 0 else float('nan')
    return {
        'mean':   round(mean, 4),
        'std':    round(std,  4),
        'cv':     round(cv,   4),
        'min':    float(np.min(arr)),
        'max':    float(np.max(arr)),
        'range':  float(np.max(arr) - np.min(arr)),
    }


def _categorical_stats(values: list) -> dict:
    if not values:
        return {}
    counts = Counter(str(v) for v in values)
    total  = len(values)
    mode_val, mode_count = counts.most_common(1)[0]
    consistency = mode_count / total
    probs   = [c / total for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return {
        'mode':             mode_val,
        'mode_consistency': round(consistency, 4),
        'entropy_bits':     round(entropy,     4),
        'n_unique':         len(counts),
        'value_counts':     dict(counts),
    }


def compute_sign_flips(values: list, reference_value,
                       question_info: dict) -> Tuple[int, float]:
    """
    Count how many variant responses differ from the reference value.

    For numeric questions a flip is any change in value.
    For categorical / multi questions a flip is any different selection.

    Returns (n_flips, flip_rate).
    """
    if reference_value is None:
        return 0, float('nan')

    n_flips = 0
    n_comparable = 0

    rtype = question_info['response_type']

    for v in values:
        if v is None:
            continue
        n_comparable += 1

        if rtype == 'numeric':
            if float(v) != float(reference_value):
                n_flips += 1

        elif rtype == 'categorical':
            if str(v) != str(reference_value):
                n_flips += 1

        elif rtype in ('multi_numeric', 'multi_categorical'):
            ref_set = frozenset(reference_value) if isinstance(reference_value, list) else {reference_value}
            val_set = frozenset(v)               if isinstance(v, list)             else {v}
            if ref_set != val_set:
                n_flips += 1

    rate = n_flips / n_comparable if n_comparable > 0 else float('nan')
    return n_flips, round(rate, 4)


# ── Runner ────────────────────────────────────────────────────────────────────

class PromptSensitivityRunner:
    """
    For each (model, tone, variant, question) runs one query at fixed seed/temp,
    then computes sign flips relative to the reference prompt (standard, variant 0).
    """

    def __init__(self,
                 models: Optional[List[Tuple[str, str]]] = None,
                 tones: Optional[List[str]] = None,
                 variants: Optional[List[int]] = None,
                 model_set: str = 'all',
                 temperature: float = 0.0):
        """
        Args:
            models: Explicit list of (model_name, provider) tuples; overrides
                    model_set when provided.
            tones: Tone names to sweep (default: all 3).
            variants: Variant indices to sweep (default: all 10).
            model_set: 'all' (default) runs open + API models, 'open' runs
                       Ollama-only, 'api' runs proprietary models only.
            temperature: Sampling temperature. Use 0.0 (default) for clean
                         question-level isolation of prompt variance. Use 1.0
                         to match Stage 3 (stochastic) conditions for map-level comparison.
        """
        self.model_set   = model_set
        self.tones       = tones    or list(TONES.keys())
        self.variants    = variants if variants is not None else list(range(10))
        self.temperature = temperature

        self.models = models if models is not None else self._build_model_list()

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file  = LOGS_DIR / f"prompt_sensitivity_{self.timestamp}.jsonl"
        print(f"Logging to: {self.log_file}")

        n_prompts = len(self.tones) * len(self.variants)
        total = len(self.models) * n_prompts * len(QUESTIONS)
        print(f"\nPlan: {len(self.models)} models × {n_prompts} prompt variants "
              f"× {len(QUESTIONS)} questions = {total} queries")
        print(f"Temperature: {self.temperature}")
        print(f"Reference prompt: tone='{REFERENCE_TONE}', variant={REFERENCE_VARIANT}")
        print(f"Models: {[m for m, _ in self.models]}\n")

    # ── Model discovery ───────────────────────────────────────────────────────

    def _build_model_list(self) -> List[Tuple[str, str]]:
        models = []
        if self.model_set in ('all', 'open'):
            models += self._discover_ollama_models()
        if self.model_set in ('all', 'api'):
            models += self._discover_proprietary_models()
        if not models:
            print(f"Warning: no models found for --model-set '{self.model_set}'")
        return models

    def _discover_ollama_models(self) -> List[Tuple[str, str]]:
        try:
            import ollama
            names = [m['model'] for m in ollama.list().get('models', [])]
            print(f"Discovered {len(names)} Ollama models: {names}")
            return [(n, 'ollama') for n in names]
        except Exception as e:
            print(f"Could not auto-detect Ollama models: {e}")
            return []

    def _discover_proprietary_models(self) -> List[Tuple[str, str]]:
        available = []
        for model, provider in PROPRIETARY_MODELS.items():
            if provider == 'openai' and os.getenv('OPENAI_API_KEY'):
                available.append((model, provider))
            elif provider == 'anthropic' and os.getenv('ANTHROPIC_API_KEY'):
                available.append((model, provider))
        if available:
            print(f"API models available: {[m for m, _ in available]}")
        elif self.model_set == 'api':
            print("Warning: --model-set api requested but no API keys found "
                  "(set OPENAI_API_KEY / ANTHROPIC_API_KEY)")
        return available

    # ── Core query loop ───────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        all_rows = []

        for model_name, provider in self.models:
            print(f"\n{'='*70}")
            print(f"MODEL: {model_name}  ({provider})")
            print(f"{'='*70}")

            rows = self._run_model(model_name, provider)
            all_rows.extend(rows)

        flat_df = pd.DataFrame(all_rows)

        flat_path = RESULTS_DIR / f"prompt_sensitivity_flat_{self.model_set}_{self.timestamp}.csv"
        flat_df.to_csv(flat_path, index=False)
        print(f"\nSaved flat results: {flat_path}")

        summary_df = self._compute_summary(flat_df)
        summary_path = RESULTS_DIR / f"prompt_sensitivity_summary_{self.model_set}_{self.timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")

        self._print_report(summary_df)
        return flat_df

    def _make_wrapper(self, model_name: str, provider: str) -> LLMQueryWrapper:
        """Build an LLMQueryWrapper with fixed seed for the given provider."""
        kwargs = dict(
            provider=model_name if provider == 'ollama' else provider,
            model_name=model_name,
            temperature=self.temperature,
            max_tokens=256,
        )
        # Ollama and OpenAI support seed; Anthropic does not
        if provider in ('ollama', 'openai'):
            kwargs['seed'] = 0

        if provider == 'openai':
            kwargs['provider'] = 'openai'
            kwargs['api_key'] = os.getenv('OPENAI_API_KEY')
        elif provider == 'anthropic':
            kwargs['provider'] = 'anthropic'
            kwargs['api_key'] = os.getenv('ANTHROPIC_API_KEY')
        else:
            kwargs['provider'] = 'ollama'

        return LLMQueryWrapper(**kwargs)

    def _run_model(self, model_name: str, provider: str) -> list:
        rows = []

        for tone in self.tones:
            for variant_idx in self.variants:
                wrapper = self._make_wrapper(model_name, provider)

                for question_id, question_info in QUESTIONS.items():
                    prompt = format_full_prompt(
                        question_id=question_id,
                        country_name=None,
                        variant=variant_idx,
                        tone=tone,
                    )

                    result = wrapper.query(
                        system_prompt=prompt['system'],
                        user_prompt=prompt['user'],
                        metadata={
                            'question_id': question_id,
                            'tone': tone,
                            'variant': variant_idx,
                            'run_type': 'prompt_sensitivity',
                        }
                    )

                    raw    = result.get('response') or ''
                    parsed = ResponseParser.parse_by_type(raw, question_info)

                    row = {
                        'model':        model_name,
                        'provider':     provider,
                        'temperature':  self.temperature,
                        'tone':         tone,
                        'variant':      variant_idx,
                        'question_id':  question_id,
                        'raw_response': raw,
                        'parsed_value': json.dumps(parsed),
                        'is_valid':     parsed is not None,
                        'error':        result.get('error'),
                    }
                    rows.append(row)

                    self._log(row | {
                        'timestamp':     datetime.now().isoformat(),
                        'question_name': question_info['name'],
                        'system_prompt': prompt['system'],
                        'user_prompt':   prompt['user'],
                        'parsed_value':  parsed,   # non-JSON for log
                    })

                n_valid = sum(1 for r in rows[-len(QUESTIONS):]
                              if r['is_valid'])
                print(f"  {model_name} | {tone} | v{variant_idx}"
                      f" — {n_valid}/{len(QUESTIONS)} valid")

        return rows

    def _log(self, entry: dict) -> None:
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    # ── Summary + sign flips ──────────────────────────────────────────────────

    def _compute_summary(self, flat_df: pd.DataFrame) -> pd.DataFrame:
        summary_rows = []

        for (model, question_id), group in flat_df.groupby(['model', 'question_id']):
            question_info = QUESTIONS[question_id]

            # Deserialise
            all_parsed = []
            for pv in group['parsed_value']:
                try:
                    all_parsed.append(json.loads(pv))
                except (json.JSONDecodeError, TypeError):
                    all_parsed.append(None)

            # Reference value: (standard, variant 0)
            ref_row = group[
                (group['tone'] == REFERENCE_TONE) &
                (group['variant'] == REFERENCE_VARIANT)
            ]
            ref_parsed = None
            if not ref_row.empty:
                try:
                    ref_parsed = json.loads(ref_row.iloc[0]['parsed_value'])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Non-reference values for sign-flip counting
            non_ref = group[
                ~((group['tone'] == REFERENCE_TONE) &
                  (group['variant'] == REFERENCE_VARIANT))
            ]
            non_ref_parsed = []
            for pv in non_ref['parsed_value']:
                try:
                    non_ref_parsed.append(json.loads(pv))
                except (json.JSONDecodeError, TypeError):
                    non_ref_parsed.append(None)

            n_flips, flip_rate = compute_sign_flips(
                non_ref_parsed, ref_parsed, question_info
            )

            rtype  = question_info['response_type']
            valid  = [v for v in all_parsed if v is not None]

            row = {
                'model':            model,
                'question_id':      question_id,
                'question_name':    question_info['name'],
                'response_type':    rtype,
                'n_variants':       len(group),
                'n_valid':          len(valid),
                'parse_rate':       round(len(valid) / len(group), 4) if group.shape[0] > 0 else 0,
                'reference_value':  json.dumps(ref_parsed),
                'n_sign_flips':     n_flips,
                'sign_flip_rate':   flip_rate,
            }

            if rtype == 'numeric' and valid:
                row.update(_numeric_stats([float(v) for v in valid]))
            if valid:
                row.update(_categorical_stats(valid))

            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    # ── Report ────────────────────────────────────────────────────────────────

    def _print_report(self, summary_df: pd.DataFrame) -> None:
        print("\n" + "="*70)
        print("PROMPT SENSITIVITY REPORT")
        print(f"Reference: tone='{REFERENCE_TONE}', variant={REFERENCE_VARIANT}")
        print("="*70)

        for model, mdf in summary_df.groupby('model'):
            print(f"\n  Model: {model}")

            for _, row in mdf.iterrows():
                rtype      = row['response_type']
                qid        = row['question_id']
                flip_rate  = row.get('sign_flip_rate', float('nan'))
                n_flips    = int(row.get('n_sign_flips', 0))
                parse_rate = row.get('parse_rate', float('nan'))

                if rtype == 'numeric':
                    detail = (f"std={row.get('std', float('nan')):.3f}  "
                              f"range={row.get('range', float('nan')):.3f}  "
                              f"mean={row.get('mean', float('nan')):.3f}")
                else:
                    cons = row.get('mode_consistency', float('nan'))
                    ent  = row.get('entropy_bits',     float('nan'))
                    detail = f"mode_consistency={cons:.3f}  entropy={ent:.3f} bits"

                flip_flag = " !!!" if flip_rate > 0.3 else ""
                print(f"    {qid:<6s}  parse={parse_rate:.0%}  "
                      f"flips={n_flips}/{int(row['n_variants'])-1} ({flip_rate:.0%})"
                      f"{flip_flag}  {detail}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Prompt sensitivity: run all tone×variant combinations at fixed seed.'
    )
    parser.add_argument(
        '--model-set', choices=['all', 'open', 'api'], default='all',
        help='all=open+API (default), open=Ollama only, api=proprietary only'
    )
    parser.add_argument('--models', nargs='+',
                        help='Explicit model names; overrides --model-set')
    parser.add_argument('--tones', nargs='+', choices=list(TONES.keys()),
                        help='Tones to sweep (default: all)')
    parser.add_argument('--variants', nargs='+', type=int,
                        help='Variant indices to sweep (default: all 0-9)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (default: 0.0 for clean isolation; '
                             'use 1.0 to match Week 7 for map-level comparison)')
    args = parser.parse_args()

    if args.models:
        models = [(m, PROPRIETARY_MODELS.get(m, 'ollama')) for m in args.models]
    else:
        models = None

    runner = PromptSensitivityRunner(
        models=models,
        tones=args.tones,
        variants=[int(v) for v in args.variants] if args.variants else None,
        model_set=args.model_set,
        temperature=args.temperature,
    )
    runner.run()


if __name__ == '__main__':
    main()
