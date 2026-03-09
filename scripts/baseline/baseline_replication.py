"""
Baseline Replication

Replicates PNAS Figure 1 by querying LLMs with IVS questions
and plotting them on the cultural map.

Usage:
    python scripts/baseline/baseline_replication.py

    # Or with specific models:
    python scripts/baseline/baseline_replication.py --models gemma2:2b qwen2.5:1.5b
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.llm_interface import LLMQueryWrapper
from src.response_parser import ResponseParser
from src.query_logger import QueryLogger
from src.prompts import QUESTIONS, TONES, format_full_prompt
from src.cultural_map import CulturalMapGenerator
from src.geo_data import COUNTRY_NAMES, ISO3_TO_ZONE, load_iso3_lookup, quadrant_label, typical_countries

# Paths
BASELINE_PATH = Path("data/processed/cultural_map_coordinates.csv")
IVS_PATH = Path("data/processed/ivs_2005-2022.csv")
RESULTS_DIR = Path("data/results")
LOGS_DIR = Path("logs")


class BaselineReplicator:
    """Replicates PNAS baseline by querying models without cultural prompting."""

    def __init__(self, models_to_test=None):
        """Initialize replicator."""
        self.models_to_test = models_to_test or self._get_default_models()

        # Create output directories
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # Load baseline and annotate with ISO-3, name, and cultural zone
        print("Loading baseline coordinates...")
        self.baseline_df = pd.read_csv(BASELINE_PATH)
        iso_lookup = load_iso3_lookup(IVS_PATH)
        self.baseline_df['iso3'] = self.baseline_df['country_code'].map(iso_lookup).fillna('???')
        self.baseline_df['name'] = self.baseline_df['iso3'].map(COUNTRY_NAMES).fillna(self.baseline_df['iso3'])
        self.baseline_df['zone'] = self.baseline_df['iso3'].map(ISO3_TO_ZONE).fillna('Other')
        print(f"  Loaded {len(self.baseline_df)} countries\n")

        self.logger = QueryLogger(log_dir=LOGS_DIR)

        # Re-fit PCA on IVS data so we can project LLM responses into the same space
        print("Loading PCA transformation from baseline...")
        self.pca_generator = self._load_pca_transformation()
        print("  PCA loaded successfully\n")

    def _get_default_models(self):
        """Get list of local Ollama models to test."""
        try:
            import ollama
            models = ollama.list()
            model_names = [m['model'] for m in models.get('models', [])]
            print(f"Found {len(model_names)} local Ollama models")
            return model_names
        except:
            print("No Ollama models found, using default list")
            return [
                'gemma2:2b', 'phi3:mini',
                'qwen2.5:1.5b', 'qwen2.5:3b', 'qwen2.5:7b',
                'mistral:7b', 'llama3.1:8b', 'yi:6b',
                'salmatrafi/acegpt:7b',
            ]

    def _load_pca_transformation(self):
        """Load PCA transformation from baseline generation."""
        ivs_df = pd.read_csv(IVS_PATH, low_memory=False)
        generator = CulturalMapGenerator(ivs_df)
        generator.fit()
        return generator

    def query_model_baseline(self, model_name, provider='ollama', tone='standard'):
        """Query a model with all IVS questions (no cultural prompting) for one tone."""
        print(f"\n{'='*70}")
        print(f"Querying {model_name} (provider: {provider}, tone: {tone})")
        print(f"{'='*70}")

        # Create wrapper
        wrapper = LLMQueryWrapper(
            provider=provider,
            model_name=model_name,
            temperature=0.0,
            seed=42
        )

        tone_prompts = TONES[tone]
        results = []
        total_queries = len(QUESTIONS) * len(tone_prompts)

        # Query each question with each variant
        for question_id, question_info in QUESTIONS.items():
            print(f"\n  Question {question_id}: {question_info['name']}")

            for variant_idx in range(len(tone_prompts)):
                print(f"    Variant {variant_idx + 1}/{len(tone_prompts)}... ", end='', flush=True)

                # Format prompt with the current tone
                prompt = format_full_prompt(
                    question_id=question_id,
                    country_name=None,
                    variant=variant_idx,
                    tone=tone
                )

                # Query model
                result = wrapper.query(
                    system_prompt=prompt['system'],
                    user_prompt=prompt['user'],
                    metadata={
                        'question_id': question_id,
                        'variant': variant_idx,
                        'tone': tone
                    }
                )

                # Parse response
                parsed = ResponseParser.parse_by_type(
                    result.get('response', ''),
                    question_info
                )

                is_valid = parsed is not None
                print(f"{'OK' if is_valid else 'FAIL'}")

                # Log
                self.logger.log_query(
                    model=model_name,
                    provider=provider,
                    system_prompt=prompt['system'],
                    user_prompt=prompt['user'],
                    raw_response=result.get('response', ''),
                    parsed_response=parsed,
                    is_valid=is_valid,
                    metadata={
                        'question_id': question_id,
                        'variant': variant_idx,
                        'tone': tone
                    },
                    error=result.get('error')
                )

                # Store result
                results.append({
                    'model': model_name,
                    'provider': provider,
                    'tone': tone,
                    'question_id': question_id,
                    'variant': variant_idx,
                    'raw_response': result.get('response', ''),
                    'parsed_value': parsed,
                    'is_valid': is_valid
                })

        results_df = pd.DataFrame(results)
        valid_count = results_df['is_valid'].sum()
        print(f"\n  Summary: {valid_count}/{total_queries} valid ({valid_count/total_queries*100:.1f}%)")

        return results_df

    def calculate_model_coordinates(self, responses_df, model_name):
        """Calculate model's coordinates on cultural map."""
        print(f"\nCalculating coordinates for {model_name}...")

        # Average across valid responses for each question.
        # Categorical and multi-choice responses must be converted to their IVS
        # numeric equivalents before averaging so all inputs to PCA are numeric.
        question_means = {}

        for question_id, question_info in QUESTIONS.items():
            valid_responses = responses_df[
                (responses_df['question_id'] == question_id) &
                (responses_df['is_valid'] == True)
            ]

            if len(valid_responses) > 0:
                # Convert each parsed value to its IVS numeric code
                numeric_values = [
                    ResponseParser.to_ivs_numeric(v, question_info)
                    for v in valid_responses['parsed_value']
                ]
                numeric_values = [v for v in numeric_values if v is not None]

                if numeric_values:
                    mean_value = np.mean(numeric_values)
                    question_means[question_id] = mean_value
                    print(f"  {question_id}: {mean_value:.2f} (n={len(numeric_values)})")
                else:
                    print(f"  {question_id}: NO VALID RESPONSES")
                    question_means[question_id] = np.nan
            else:
                print(f"  {question_id}: NO VALID RESPONSES")
                question_means[question_id] = np.nan

        # Abort if any question produced no valid responses — PCA cannot handle NaN
        missing_qs = [qid for qid, v in question_means.items() if np.isnan(v)]
        if missing_qs:
            print(f"  Skipping coordinates: no valid responses for {missing_qs}")
            return np.nan, np.nan

        # Project averaged responses through the baseline PCA transformation
        means = self.pca_generator.question_means_
        stds = self.pca_generator.question_stds_

        standardized = [
            (question_means[qid] - means[qid]) / stds[qid]
            for qid in QUESTIONS.keys()
        ]

        # Apply PCA transformation and rescale using PNAS formula
        pca_scores = self.pca_generator.pca_model.transform(
            np.array(standardized).reshape(1, -1)
        )[0]

        x = 1.81 * pca_scores[0] + 0.38
        y = 1.61 * pca_scores[1] - 0.01

        print(f"  Coordinates: ({x:.3f}, {y:.3f})")
        return x, y

    def calculate_distances(self, model_x, model_y):
        """Calculate Euclidean distance from model to each country."""
        distances = []

        for _, country in self.baseline_df.iterrows():
            dist = np.sqrt(
                (model_x - country['survival_selfexpression'])**2 +
                (model_y - country['traditional_secular'])**2
            )
            distances.append({
                'country_code': country['country_code'],
                'iso3': country['iso3'],
                'name': country['name'],
                'zone': country['zone'],
                'distance': dist,
            })

        return pd.DataFrame(distances).sort_values('distance').reset_index(drop=True)

    def run(self, tones=None):
        """Run complete baseline replication across all tones."""
        tones = tones or list(TONES.keys())

        print("\n" + "="*70)
        print("BASELINE REPLICATION")
        print("="*70)
        print(f"\nModels to test: {self.models_to_test}")
        print(f"Tones to test:  {tones}\n")

        # Results keyed by tone
        all_results = {tone: {'models': [], 'responses': [], 'distances': [], 'summary': []} for tone in tones}

        for tone in tones:
            print(f"\n{'#'*70}")
            print(f"# TONE: {tone.upper()}")
            print(f"{'#'*70}")
            all_results[tone]['summary'].append(f"{'#'*70}\n# TONE: {tone.upper()}\n{'#'*70}\n")

            for model_name in self.models_to_test:
                try:
                    responses_df = self.query_model_baseline(model_name, tone=tone)
                    x, y = self.calculate_model_coordinates(responses_df, model_name)

                    # Per-question refusal rates (always computed, regardless of mappability)
                    refusal_rates = {}
                    for qid in QUESTIONS:
                        q_rows = responses_df[responses_df['question_id'] == qid]
                        n_valid = int(q_rows['is_valid'].sum())
                        n_total = len(q_rows)
                        refusal_rates[qid] = round(1 - n_valid / n_total, 3) if n_total > 0 else 1.0

                    if not (np.isnan(x) or np.isnan(y)):
                        distances_df = self.calculate_distances(x, y)
                        top = distances_df.head(5)
                        closest = distances_df.iloc[0]
                        mean_dist = distances_df['distance'].mean()

                        model_record = {
                            'model': model_name,
                            'tone': tone,
                            'survival_selfexpression': x,
                            'traditional_secular': y,
                            'closest_country': closest['iso3'],
                            'closest_country_name': closest['name'],
                            'closest_zone': closest['zone'],
                            'closest_distance': closest['distance'],
                            'mean_distance': mean_dist,
                        }
                        model_record.update({f'{qid}_refusal_rate': r for qid, r in refusal_rates.items()})
                        all_results[tone]['models'].append(model_record)

                        all_results[tone]['responses'].append(responses_df)
                        distances_df['model'] = model_name
                        distances_df['tone'] = tone
                        all_results[tone]['distances'].append(distances_df)

                        # ── Per-model summary (printed + accumulated for file) ──
                        refused = [(qid, r) for qid, r in refusal_rates.items() if r > 0]
                        lines = [
                            f"\n  ┌─ {model_name}  [{tone}]",
                            f"  │  Position : ({x:.3f}, {y:.3f})",
                            f"  │  Quadrant : {quadrant_label(x, y)}",
                            f"  │  Typical  : {typical_countries(x, y)}",
                            f"  │  Mean dist to all countries: {mean_dist:.3f}",
                            f"  │  Top-5 closest countries:",
                        ]
                        for _, row in top.iterrows():
                            lines.append(f"  │    {row['iso3']:3s}  {row['name']:<20s}"
                                         f"  d={row['distance']:.3f}  [{row['zone']}]")
                        if refused:
                            lines.append("  │  Refusals: " +
                                         "  ".join(f"{qid}={r*100:.0f}%" for qid, r in refused))
                        lines.append("  └─────────────────────────────────────────")
                        for line in lines:
                            print(line)
                        all_results[tone]['summary'].extend(lines)

                    else:
                        skipped = f"\n  ✗ {model_name} [{tone}] — skipped (missing question responses)"
                        print(skipped)
                        all_results[tone]['summary'].append(skipped)

                except Exception as e:
                    err = f"\n  Error ({model_name}, tone={tone}): {e}"
                    print(err)
                    all_results[tone]['summary'].append(err)

        return all_results

    def save_results(self, all_results):
        """Save results to CSV files and a human-readable summary, one set per tone."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)

        for tone, results in all_results.items():
            if results['models']:
                models_df = pd.DataFrame(results['models'])
                path = RESULTS_DIR / f"baseline_models_{tone}_{timestamp}.csv"
                models_df.to_csv(path, index=False)
                print(f"\nSaved: {path}")

            if results['distances']:
                distances_df = pd.concat(results['distances'], ignore_index=True)
                path = RESULTS_DIR / f"baseline_distances_{tone}_{timestamp}.csv"
                distances_df.to_csv(path, index=False)
                print(f"Saved: {path}")

        # Write combined plain-text summary — filename includes the tones that were run
        tones_run = "_".join(all_results.keys())
        summary_path = outputs_dir / f"baseline_summary_{tones_run}_{timestamp}.txt"
        header = (
            f"Baseline Replication Summary\n"
            f"Run   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Tones : {list(all_results.keys())}\n"
            f"Models: {self.models_to_test}\n"
            f"{'='*70}\n"
        )
        all_lines = [header]
        for results in all_results.values():
            all_lines.extend(results['summary'])
        summary_path.write_text("\n".join(all_lines))
        print(f"Saved: {summary_path}")

        self.logger.print_stats()


def main():
    parser = argparse.ArgumentParser(description='Baseline Replication')
    parser.add_argument('--models', nargs='+', help='Models to test')
    parser.add_argument('--tones', nargs='+', choices=list(TONES.keys()),
                        help='Tones to test (default: all)')
    args = parser.parse_args()

    replicator = BaselineReplicator(models_to_test=args.models)
    all_results = replicator.run(tones=args.tones)
    replicator.save_results(all_results)

    print("\nDone. Generate visualization with scripts/baseline/visualize_baseline.py")


if __name__ == "__main__":
    main()