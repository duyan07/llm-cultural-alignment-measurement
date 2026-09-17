"""
Unified query interface for Ollama, OpenAI, and Anthropic chat models.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

# Each provider's SDK is optional; only the ones installed get enabled.
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class LLMQueryWrapper:
    """Dispatches queries to the configured provider ('ollama', 'openai', or 'anthropic')."""

    def __init__(self, provider: str, model_name: str,
                 api_key: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 256,
                 seed: Optional[int] = None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

        if self.provider == 'ollama':
            if not OLLAMA_AVAILABLE:
                raise ImportError("ollama package not installed. Run: pip install ollama")
            self.client = None  # Ollama uses module-level functions

        elif self.provider == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed. Run: pip install openai")
            api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key required")
            self.client = OpenAI(api_key=api_key)

        elif self.provider == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
            api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key required")
            self.client = Anthropic(api_key=api_key)

        elif self.provider == 'google':
            raise NotImplementedError("Google Gemini support coming soon")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def query(self, system_prompt: str, user_prompt: str,
              metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Query the LLM, returning a dict with 'response', 'metadata', and 'raw'
        (the raw provider response object, for debugging) — or 'error' set on failure.
        """
        start_time = time.time()

        try:
            if self.provider == 'ollama':
                result = self._query_ollama(system_prompt, user_prompt)
            elif self.provider == 'openai':
                result = self._query_openai(system_prompt, user_prompt)
            elif self.provider == 'anthropic':
                result = self._query_anthropic(system_prompt, user_prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            elapsed_time = time.time() - start_time

            result['metadata'] = {
                'provider': self.provider,
                'model': self.model_name,
                'temperature': self.temperature,
                'seed': self.seed,
                'timestamp': datetime.now().isoformat(),
                'elapsed_seconds': elapsed_time,
                **(metadata or {})
            }

            return result

        except Exception as e:
            return {
                'response': None,
                'error': str(e),
                'metadata': {
                    'provider': self.provider,
                    'model': self.model_name,
                    'timestamp': datetime.now().isoformat(),
                    'elapsed_seconds': time.time() - start_time,
                    **(metadata or {})
                }
            }

    def _query_ollama(self, system_prompt: str, user_prompt: str) -> Dict:
        """Query Ollama model using chat endpoint for proper instruct model support."""
        options = {
            'temperature': self.temperature,
            'num_predict': self.max_tokens,
        }

        if self.seed is not None:
            options['seed'] = self.seed

        # Use chat endpoint to preserve system/user role structure.
        # ollama.generate() concatenates prompts into a single string, which
        # bypasses chat templates and causes instruct models (e.g. llama3.2)
        # to misinterpret context and trigger safety refusals on survey questions.
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            options=options
        )

        return {
            'response': response['message']['content'].strip(),
            'raw': response
        }

    def _query_openai(self, system_prompt: str, user_prompt: str) -> Dict:
        """Query OpenAI model."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        params = {
            'model': self.model_name,
            'messages': messages,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }

        if self.seed is not None:
            params['seed'] = self.seed

        response = self.client.chat.completions.create(**params)

        return {
            'response': response.choices[0].message.content.strip(),
            'raw': response.model_dump()
        }

    def _query_anthropic(self, system_prompt: str, user_prompt: str) -> Dict:
        """Query Anthropic Claude model."""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        return {
            'response': response.content[0].text.strip(),
            'raw': response.model_dump()
        }

    def batch_query(self, prompts: List[Dict[str, str]],
                    metadata: Optional[List[Dict]] = None) -> List[Dict]:
        """Query a list of {'system', 'user'} prompt dicts in sequence."""
        results = []

        for i, prompt in enumerate(prompts):
            meta = metadata[i] if metadata else None
            result = self.query(
                system_prompt=prompt['system'],
                user_prompt=prompt['user'],
                metadata=meta
            )
            results.append(result)

            # Small delay to avoid rate limits
            if self.provider != 'ollama':
                time.sleep(0.5)

        return results


# Convenience functions
def query_ollama(model: str, system_prompt: str, user_prompt: str,
                 temperature: float = 0.0, seed: Optional[int] = None) -> str:
    """Quick function to query Ollama model."""
    wrapper = LLMQueryWrapper('ollama', model, temperature=temperature, seed=seed)
    result = wrapper.query(system_prompt, user_prompt)
    return result['response']


def query_openai(model: str, system_prompt: str, user_prompt: str,
                 temperature: float = 0.0, seed: Optional[int] = None,
                 api_key: Optional[str] = None) -> str:
    """Quick function to query OpenAI model."""
    wrapper = LLMQueryWrapper('openai', model, api_key=api_key,
                             temperature=temperature, seed=seed)
    result = wrapper.query(system_prompt, user_prompt)
    return result['response']


def query_anthropic(model: str, system_prompt: str, user_prompt: str,
                   temperature: float = 0.0, api_key: Optional[str] = None) -> str:
    """Quick function to query Anthropic model."""
    wrapper = LLMQueryWrapper('anthropic', model, api_key=api_key,
                             temperature=temperature)
    result = wrapper.query(system_prompt, user_prompt)
    return result['response']