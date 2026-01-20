import ollama
import sys

def test_ollama():
    print("Testing Ollama setup...")

    models = ollama.list()
    if not models["models"]:
        print("No models found. Please run 'ollama pull <model_name>' to download a model.")
        return False

    print(f"Found {len(models["models"])} model(s):")
    for model in models["models"]:
        print(f"   - {model['model']}")

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