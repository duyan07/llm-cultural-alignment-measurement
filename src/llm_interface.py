"""
LLM Query Wrapper

Provides a unified interface for querying different LLM providers:
- Ollama (local models)
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude)
- Google (Gemini)

Handles:
- Prompt formatting
- Temperature/seed control
- Error handling and retries
- Response extraction
"""

import os
import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

# Import libraries (will be optional based on what's available)
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
    """
    Unified interface for querying different LLM providers.
    """

    def __init__(self, provider: str, model_name: str,
                 api_key: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 256,
                 seed: Optional[int] = None):
        """
        Initialize the query wrapper.

        Args:
            provider: 'ollama', 'openai', 'anthropic', or 'google'
            model_name: Specific model identifier
            api_key: API key (not needed for Ollama)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            seed: Random seed for reproducibility
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

        # Initialize provider-specific client
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
        Query the LLM with system and user prompts.

        Args:
            system_prompt: System message (role/context)
            user_prompt: User message (actual question)
            metadata: Additional metadata to log

        Returns:
            Dictionary with:
                - response: Model's text response
                - metadata: Query metadata (model, timestamp, etc.)
                - raw: Raw response object (for debugging)
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

            # Add metadata
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
        """Query Ollama model."""
        # Combine system and user prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Build options
        options = {
            'temperature': self.temperature,
            'num_predict': self.max_tokens,
        }

        if self.seed is not None:
            options['seed'] = self.seed

        # Query model
        response = ollama.generate(
            model=self.model_name,
            prompt=full_prompt,
            options=options
        )

        return {
            'response': response['response'].strip(),
            'raw': response
        }

    def _query_openai(self, system_prompt: str, user_prompt: str) -> Dict:
        """Query OpenAI model."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Build parameters
        params = {
            'model': self.model_name,
            'messages': messages,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }

        if self.seed is not None:
            params['seed'] = self.seed

        # Query model
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
        """
        Query multiple prompts in sequence.

        Args:
            prompts: List of dicts with 'system' and 'user' keys
            metadata: Optional list of metadata dicts (same length as prompts)

        Returns:
            List of response dictionaries
        """
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