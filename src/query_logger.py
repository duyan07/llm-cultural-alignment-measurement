"""
Query Logger

Logs all LLM queries and responses for reproducibility and analysis.
Supports both JSON Lines format and CSV export.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import hashlib


class QueryLogger:
    """Appends LLM queries and responses to a timestamped JSONL log file."""

    def __init__(self, log_dir: Path = Path("logs")):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"queries_{timestamp}.jsonl"

        print(f"Logging to: {self.log_file}")

    def log_query(self,
                  model: str,
                  provider: str,
                  system_prompt: str,
                  user_prompt: str,
                  raw_response: str,
                  parsed_response: Any,
                  is_valid: bool,
                  metadata: Optional[Dict] = None,
                  error: Optional[str] = None) -> None:
        """Log a single query-response pair as one line in the JSONL file."""
        prompt_hash = self._hash_prompt(system_prompt, user_prompt, model)

        entry = {
            'timestamp': datetime.now().isoformat(),
            'prompt_hash': prompt_hash,
            'model': {
                'provider': provider,
                'name': model,
            },
            'prompts': {
                'system': system_prompt,
                'user': user_prompt,
            },
            'response': {
                'raw': raw_response,
                'parsed': parsed_response,
                'is_valid': is_valid,
            },
            'metadata': metadata or {},
            'error': error
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def log_batch(self, entries: List[Dict]) -> None:
        """Append multiple pre-built log entries at once."""
        with open(self.log_file, 'a') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')

    @staticmethod
    def _hash_prompt(system: str, user: str, model: str) -> str:
        """Create unique hash for prompt + model combination."""
        content = f"{model}||{system}||{user}"
        return hashlib.md5(content.encode()).hexdigest()

    def export_to_csv(self, output_path: Path) -> None:
        """Flatten the JSONL log to a CSV for analysis."""
        entries = []
        with open(self.log_file, 'r') as f:
            for line in f:
                entries.append(json.loads(line))

        if not entries:
            print("No entries to export")
            return

        with open(output_path, 'w', newline='') as f:
            fieldnames = [
                'timestamp', 'provider', 'model', 'question_id', 'country',
                'variant', 'raw_response', 'parsed_response', 'is_valid',
                'temperature', 'seed', 'error'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in entries:
                row = {
                    'timestamp': entry['timestamp'],
                    'provider': entry['model']['provider'],
                    'model': entry['model']['name'],
                    'question_id': entry['metadata'].get('question_id', ''),
                    'country': entry['metadata'].get('country', ''),
                    'variant': entry['metadata'].get('variant', ''),
                    'raw_response': entry['response']['raw'],
                    'parsed_response': json.dumps(entry['response']['parsed']),
                    'is_valid': entry['response']['is_valid'],
                    'temperature': entry['metadata'].get('temperature', ''),
                    'seed': entry['metadata'].get('seed', ''),
                    'error': entry.get('error', '')
                }
                writer.writerow(row)

        print(f"Exported {len(entries)} entries to {output_path}")

    def get_stats(self) -> Dict:
        """Get summary statistics from log file."""
        entries = []
        with open(self.log_file, 'r') as f:
            for line in f:
                entries.append(json.loads(line))

        if not entries:
            return {'total_queries': 0}

        total = len(entries)
        valid = sum(1 for e in entries if e['response']['is_valid'])
        errors = sum(1 for e in entries if e.get('error'))

        models = {}
        for entry in entries:
            model = entry['model']['name']
            models[model] = models.get(model, 0) + 1

        return {
            'total_queries': total,
            'valid_responses': valid,
            'invalid_responses': total - valid,
            'errors': errors,
            'valid_rate': valid / total if total > 0 else 0,
            'models_used': models,
            'log_file': str(self.log_file)
        }

    def print_stats(self) -> None:
        """Print summary statistics."""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("QUERY LOG STATISTICS")
        print("="*60)
        print(f"Log file: {stats['log_file']}")
        print(f"\nTotal queries: {stats['total_queries']}")
        print(f"Valid responses: {stats['valid_responses']} ({stats['valid_rate']:.1%})")
        print(f"Invalid responses: {stats['invalid_responses']}")
        print(f"Errors: {stats['errors']}")

        if stats['models_used']:
            print("\nQueries by model:")
            for model, count in stats['models_used'].items():
                print(f"  {model}: {count}")


class ProgressTracker:
    """Tracks completed query hashes in a checkpoint file so interrupted runs can resume."""

    def __init__(self, total_queries: int, checkpoint_file: Path = Path("logs/checkpoint.json")):
        self.total = total_queries
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self.completed = self._load_progress()

    def _load_progress(self) -> set:
        """Load completed query hashes from checkpoint file."""
        if not self.checkpoint_file.exists():
            return set()

        with open(self.checkpoint_file, 'r') as f:
            data = json.load(f)
            return set(data.get('completed', []))

    def mark_complete(self, query_hash: str) -> None:
        """Mark a query as completed."""
        self.completed.add(query_hash)
        self._save_progress()

    def is_complete(self, query_hash: str) -> bool:
        """Check if query was already completed."""
        return query_hash in self.completed

    def _save_progress(self) -> None:
        """Save progress to checkpoint file."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'completed': list(self.completed),
                'total': self.total,
                'timestamp': datetime.now().isoformat()
            }, f)

    def get_progress(self) -> float:
        """Get completion percentage."""
        return len(self.completed) / self.total if self.total > 0 else 0

    def print_progress(self) -> None:
        """Print current progress."""
        pct = self.get_progress() * 100
        print(f"Progress: {len(self.completed)}/{self.total} ({pct:.1f}%)")