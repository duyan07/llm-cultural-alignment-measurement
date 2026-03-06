"""
Week 2 Validation Script

Checks that all Week 2 deliverables are complete:
1. IVS data is merged and ready
2. 10 questions are extractable
3. Prompts are formatted correctly
4. Cultural map can be generated
5. Model configuration is defined
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prompts import QUESTIONS, TONES, format_full_prompt
from config.models import MODELS, get_ollama_models, get_api_models
import pandas as pd


def check_ivs_data():
    """Check that merged IVS data exists and has required columns."""
    print("="*60)
    print("1. CHECKING IVS DATA")
    print("="*60)

    ivs_path = Path("data/processed/ivs_2005-2022.csv")

    if not ivs_path.exists():
        print("FAIL: IVS data not found")
        print(f"   Expected: {ivs_path}")
        print("   Run: python scripts/build_ivs.py")
        return False

    print(f"PASS: IVS data file exists: {ivs_path}")

    # Load and check columns
    df = pd.read_csv(ivs_path, low_memory=False, nrows=100)

    required_cols = ['S007_01', 'S024', 'S001', 'S017']  # ID, country, survey source, weight
    question_cols = ['A008', 'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006', 'Y002', 'Y003']

    missing_required = [col for col in required_cols if col not in df.columns]
    missing_questions = [col for col in question_cols if col not in df.columns]

    if missing_required:
        print(f"FAIL: Missing required columns: {missing_required}")
        return False

    print(f"PASS: Required metadata columns present: {required_cols}")

    if missing_questions:
        print(f"WARNING: Missing question columns: {missing_questions}")
        print("   (This is OK if questions use different variable names)")
    else:
        print(f"PASS: All 10 question columns present")

    return True


def check_prompts():
    """Check that all 10 prompts are defined correctly."""
    print("\n" + "="*60)
    print("2. CHECKING PROMPT CONFIGURATION")
    print("="*60)

    if len(QUESTIONS) != 10:
        print(f"FAIL: Expected 10 questions, found {len(QUESTIONS)}")
        return False

    print(f"PASS: All 10 questions defined")

    # Check each question has required fields
    for qid, qdata in QUESTIONS.items():
        required_fields = ['name', 'ivs_variable', 'prompt', 'response_type']
        missing = [f for f in required_fields if f not in qdata]
        if missing:
            print(f"FAIL: Question {qid} missing fields: {missing}")
            return False

    print("PASS: All questions have required fields")

    # Check tone system prompts
    expected_tones = ['standard', 'friendly', 'combative']
    missing_tones = [t for t in expected_tones if t not in TONES]
    if missing_tones:
        print(f"FAIL: Missing tones: {missing_tones}")
        return False

    for tone_name, prompts in TONES.items():
        if len(prompts) != 10:
            print(f"FAIL: Tone '{tone_name}' expected 10 variants, found {len(prompts)}")
            return False

    print(f"PASS: All 3 tones defined, each with 10 prompt variants")

    # Test prompt formatting
    try:
        test_prompt = format_full_prompt('A008', country_name=None, variant=0)
        if 'system' not in test_prompt or 'user' not in test_prompt:
            print("FAIL: Prompt formatting doesn't return system/user keys")
            return False
        print("PASS: Prompt formatting works correctly")
    except Exception as e:
        print(f"FAIL: Error formatting prompt: {e}")
        return False

    # Test cultural prompting
    try:
        cultural_prompt = format_full_prompt('A008', country_name='Thailand', variant=0)
        if 'Thailand' not in cultural_prompt['system']:
            print("FAIL: Cultural prompting not inserting country name")
            return False
        print("PASS: Cultural prompting works correctly")
    except Exception as e:
        print(f"FAIL: Error with cultural prompting: {e}")
        return False

    return True


def check_model_config():
    """Check that model configuration is complete."""
    print("\n" + "="*60)
    print("3. CHECKING MODEL CONFIGURATION")
    print("="*60)

    total_models = sum(len(models) for models in MODELS.values())
    print(f"PASS: Total models defined: {total_models}")

    ollama_models = get_ollama_models()
    print(f"PASS: Ollama models: {len(ollama_models)}")
    for model in ollama_models[:5]:  # Show first 5
        print(f"    - {model}")
    if len(ollama_models) > 5:
        print(f"    ... and {len(ollama_models) - 5} more")

    api_models = get_api_models()
    print(f"PASS: API models: {len(api_models)}")
    for model, provider in api_models[:5]:
        print(f"    - {model} ({provider})")
    if len(api_models) > 5:
        print(f"    ... and {len(api_models) - 5} more")

    return True


def check_cultural_map_generator():
    """Check that cultural map generation code exists."""
    print("\n" + "="*60)
    print("4. CHECKING CULTURAL MAP GENERATOR")
    print("="*60)

    try:
        from src.cultural_map import CulturalMapGenerator
        print("PASS: CulturalMapGenerator class imported successfully")

        # Check key methods exist
        required_methods = ['fit', 'get_country_coordinates', 'save_coordinates']
        for method in required_methods:
            if not hasattr(CulturalMapGenerator, method):
                print(f"FAIL: Missing method: {method}")
                return False

        print(f"PASS: All required methods present: {required_methods}")
        return True

    except ImportError as e:
        print(f"FAIL: Cannot import CulturalMapGenerator: {e}")
        return False


def print_next_steps():
    """Print what to do next."""
    print("\n" + "="*60)
    print("NEXT STEPS - COMPLETING WEEK 2")
    print("="*60)

    print("\nTo finish Week 2, run:")
    print("  1. python scripts/baseline/generate_cultural_map.py")
    print("     → Generates cultural map coordinates from IVS data")
    print("     → Creates baseline visualization")
    print("     → This is your 'ground truth' for comparison")

    print("\n" + "="*60)
    print("WEEK 3 PREPARATION")
    print("="*60)

    print("\nBefore starting Week 3, you'll need to:")
    print("  1. Install Ollama: https://ollama.ai")
    print("  2. Download models:")
    print("     ollama pull gemma2:2b")
    print("     ollama pull phi3:mini")
    print("     ollama pull qwen2.5:1.5b")
    print("     ollama pull mistral:7b")
    print("     ollama pull llama3.1:8b")
    print("  3. Set up API keys for OpenAI/Anthropic/Google")


def main():
    print("\n" + "="*70)
    print("WEEK 2 VALIDATION: INSTRUMENT & DATA FREEZING")
    print("="*70)

    checks = [
        ("IVS Data", check_ivs_data),
        ("Prompts", check_prompts),
        ("Models", check_model_config),
        ("Cultural Map Generator", check_cultural_map_generator),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {name}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\nALL CHECKS PASSED - Week 2 infrastructure is ready!")
        print_next_steps()
    else:
        print("\nSOME CHECKS FAILED - Please address the issues above")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)