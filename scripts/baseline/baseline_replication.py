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
import json
from datetime import datetime
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.llm_interface import LLMQueryWrapper
from src.response_parser import ResponseParser
from src.query_logger import QueryLogger
from src.prompts import QUESTIONS, SYSTEM_PROMPTS, format_full_prompt
from src.cultural_map import CulturalMapGenerator

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

        # Load the pre-computed country coordinates from the IVS baseline
        print("Loading baseline coordinates...")
        self.baseline_df = pd.read_csv(BASELINE_PATH)
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
            return ['gemma2:2b', 'qwen2.5:1.5b', 'phi3:mini']

    def _load_pca_transformation(self):
        """Load PCA transformation from baseline generation."""
        ivs_df = pd.read_csv(IVS_PATH, low_memory=False)
        generator = CulturalMapGenerator(ivs_df)
        generator.fit()
        return generator

    def query_model_baseline(self, model_name, provider='ollama'):
        """Query a model with all IVS questions (no cultural prompting)."""
        print(f"\n{'='*70}")
        print(f"Querying {model_name} (provider: {provider})")
        print(f"{'='*70}")

        # Create wrapper
        wrapper = LLMQueryWrapper(
            provider=provider,
            model_name=model_name,
            temperature=0.0,
            seed=42
        )

        results = []
        total_queries = len(QUESTIONS) * len(SYSTEM_PROMPTS)
        query_count = 0

        # Query each question with each variant
        for question_id, question_info in QUESTIONS.items():
            print(f"\n  Question {question_id}: {question_info['name']}")

            for variant_idx in range(len(SYSTEM_PROMPTS)):
                query_count += 1
                print(f"    Variant {variant_idx + 1}/{len(SYSTEM_PROMPTS)}... ", end='', flush=True)

                # Format prompt
                prompt = format_full_prompt(
                    question_id=question_id,
                    country_name=None,
                    variant=variant_idx
                )

                # Query model
                result = wrapper.query(
                    system_prompt=prompt['system'],
                    user_prompt=prompt['user'],
                    metadata={
                        'question_id': question_id,
                        'variant': variant_idx
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
                        'variant': variant_idx
                    },
                    error=result.get('error')
                )

                # Store result
                results.append({
                    'model': model_name,
                    'provider': provider,
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

        # Project averaged responses through the baseline PCA transformation
        try:
            # Standardize using baseline statistics stored in the fitted generator
            means = self.pca_generator.question_means_
            stds = self.pca_generator.question_stds_

            standardized = [
                (question_means.get(qid, np.nan) - means[qid]) / stds[qid]
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
        except Exception as e:
            print(f"  Error: {e}")
            return np.nan, np.nan

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
                'distance': dist
            })

        return pd.DataFrame(distances).sort_values('distance')

    def run(self):
        """Run complete baseline replication."""
        print("\n" + "="*70)
        print("BASELINE REPLICATION")
        print("="*70)
        print(f"\nModels to test: {self.models_to_test}\n")

        all_results = {'models': [], 'responses': [], 'distances': []}

        for model_name in self.models_to_test:
            try:
                # Query model
                responses_df = self.query_model_baseline(model_name)

                # Calculate coordinates
                x, y = self.calculate_model_coordinates(responses_df, model_name)

                if not (np.isnan(x) or np.isnan(y)):
                    # Calculate distances
                    distances_df = self.calculate_distances(x, y)

                    all_results['models'].append({
                        'model': model_name,
                        'survival_selfexpression': x,
                        'traditional_secular': y,
                        'closest_country': distances_df.iloc[0]['country_code'],
                        'mean_distance': distances_df['distance'].mean()
                    })

                    all_results['responses'].append(responses_df)
                    distances_df['model'] = model_name
                    all_results['distances'].append(distances_df)

                    print(f"\n  Done. Closest to country {distances_df.iloc[0]['country_code']}")

            except Exception as e:
                print(f"\n  Error: {e}")

        return all_results

    def save_results(self, results):
        """Save results to CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if results['models']:
            models_df = pd.DataFrame(results['models'])
            path = RESULTS_DIR / f"baseline_models_{timestamp}.csv"
            models_df.to_csv(path, index=False)
            print(f"\nSaved: {path}")

        if results['distances']:
            distances_df = pd.concat(results['distances'], ignore_index=True)
            path = RESULTS_DIR / f"baseline_distances_{timestamp}.csv"
            distances_df.to_csv(path, index=False)
            print(f"Saved: {path}")

        self.logger.print_stats()


def main():
    parser = argparse.ArgumentParser(description='Week 6: Baseline Replication')
    parser.add_argument('--models', nargs='+', help='Models to test')
    args = parser.parse_args()

    replicator = BaselineReplicator(models_to_test=args.models)
    results = replicator.run()
    replicator.save_results(results)

    print("\nDone. Generate visualization with scripts/baseline/visualize_baseline.py")


if __name__ == "__main__":
    main()