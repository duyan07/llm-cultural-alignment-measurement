"""
Test script to verify Ollama installation and model availability.
Checks that at least one model is installed and can generate responses.
"""

import ollama
import sys


def test_ollama():
    """
    Verify Ollama setup by listing available models and testing generation.

    Returns:
        bool: True if Ollama is properly configured with working models, False otherwise
    """
    print("Testing Ollama setup...")

    # List all available models
    models = ollama.list()
    if not models["models"]:
        print("No models found. Please run 'ollama pull <model_name>' to download a model.")
        return False

    print(f"Found {len(models["models"])} model(s):")
    for model in models["models"]:
        print(f"   - {model['model']}")

    # Test generation with the first available model
    test_model = models["models"][0]["model"]
    print(f"Testing model: {test_model}")

    response = ollama.generate(
        model=test_model,
        prompt="Hello, Ollama!",
    )

    print(f"Response from model: {response["response"]}")
    return True


if __name__ == '__main__':
    success = test_ollama()
    sys.exit(0 if success else 1)