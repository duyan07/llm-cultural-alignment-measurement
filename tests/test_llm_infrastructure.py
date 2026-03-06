"""
Test LLM Infrastructure

Tests the query wrapper, response parser, and logging system.
Run this before starting the full experiment.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_interface import LLMQueryWrapper
from src.response_parser import ResponseParser
from src.query_logger import QueryLogger
from src.prompts import format_full_prompt, QUESTIONS


def test_ollama():
    """Test Ollama connection and querying."""
    print("\n" + "="*70)
    print("TESTING OLLAMA")
    print("="*70)

    try:
        import ollama

        # List available models
        models_response = ollama.list()
        models = [m['model'] for m in models_response.get('models', [])]

        if not models:
            print("FAIL: No Ollama models found")
            print("   Install models with:")
            print("   ollama pull gemma2:2b")
            print("   ollama pull qwen2.5:1.5b")
            return False

        print(f"PASS: Found {len(models)} Ollama models:")
        for model in models[:5]:
            print(f"    - {model}")
        if len(models) > 5:
            print(f"    ... and {len(models) - 5} more")

        # Test query with first available model
        test_model = models[0]
        print(f"\nTesting query with {test_model}...")

        wrapper = LLMQueryWrapper('ollama', test_model, temperature=0.0)
        result = wrapper.query(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is 2+2? Answer with just the number."
        )

        if result.get('response'):
            print(f"PASS: Query successful! Response: '{result['response']}'")
            return True
        else:
            print(f"FAIL: Query failed: {result.get('error')}")
            return False

    except ImportError:
        print("FAIL: Ollama not installed")
        print("   Install with: pip install ollama")
        return False
    except Exception as e:
        print(f"FAIL: Ollama test failed: {e}")
        return False


def test_openai():
    """Test OpenAI API connection."""
    print("\n" + "="*70)
    print("TESTING OPENAI API")
    print("="*70)

    try:
        from openai import OpenAI
        import os

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("WARNING: OPENAI_API_KEY not set")
            print("   Set it with: export OPENAI_API_KEY=sk-...")
            return False

        print("PASS: API key found")

        # Test query
        print("Testing query with gpt-3.5-turbo...")
        wrapper = LLMQueryWrapper('openai', 'gpt-3.5-turbo', api_key=api_key, temperature=0.0)
        result = wrapper.query(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is 2+2? Answer with just the number."
        )

        if result.get('response'):
            print(f"PASS: Query successful! Response: '{result['response']}'")
            return True
        else:
            print(f"FAIL: Query failed: {result.get('error')}")
            return False

    except ImportError:
        print("WARNING: OpenAI package not installed")
        print("   Install with: pip install openai")
        return False
    except Exception as e:
        print(f"FAIL: OpenAI test failed: {e}")
        return False


def test_response_parser():
    """Test response parsing."""
    print("\n" + "="*70)
    print("TESTING RESPONSE PARSER")
    print("="*70)

    # Test cases for different response types
    test_cases = [
        ("3", 'numeric', (1, 4), 3.0),
        ("I would say 7 out of 10", 'numeric', (1, 10), 7.0),
        ("B", 'categorical', ['A', 'B', 'C'], 'B'),
        ("I choose option A", 'categorical', ['A', 'B'], 'A'),
    ]

    passed = 0
    for i, (response, resp_type, param, expected) in enumerate(test_cases, 1):
        if resp_type == 'numeric':
            parsed = ResponseParser.parse_numeric(response, scale=param)
        elif resp_type == 'categorical':
            parsed = ResponseParser.parse_categorical(response, options=param)
        else:
            parsed = None

        if parsed == expected:
            print(f"PASS Test {i}: '{response}' -> {parsed}")
            passed += 1
        else:
            print(f"FAIL Test {i}: '{response}' -> {parsed} (expected {expected})")

    print(f"\nPassed {passed}/{len(test_cases)} tests")
    return passed == len(test_cases)


def test_ivs_question():
    """Test with actual IVS question."""
    print("\n" + "="*70)
    print("TESTING WITH IVS QUESTION")
    print("="*70)

    # Use A008 (Happiness question)
    prompt = format_full_prompt('A008', country_name=None, variant=0)
    question_info = QUESTIONS['A008']

    print(f"Question: {question_info['name']}")
    print(f"System: {prompt['system'][:60]}...")
    print(f"User: {prompt['user'][:80]}...")

    # Test with Ollama if available
    try:
        import ollama
        models = ollama.list()
        if models.get('models'):
            test_model = models['models'][0]['model']
            print(f"\nQuerying {test_model}...")

            wrapper = LLMQueryWrapper('ollama', test_model, temperature=0.0)
            result = wrapper.query(prompt['system'], prompt['user'])

            if result.get('response'):
                print(f"\nRaw response: '{result['response']}'")

                # Parse response
                parsed = ResponseParser.parse_by_type(result['response'], question_info)
                print(f"Parsed value: {parsed}")
                print(f"Valid: {parsed is not None}")

                return parsed is not None
    except:
        pass

    print("WARNING: Skipping (no Ollama models available)")
    return True


def test_logging():
    """Test logging system."""
    print("\n" + "="*70)
    print("TESTING LOGGING SYSTEM")
    print("="*70)

    # Create test logger
    logger = QueryLogger(log_dir=Path("logs/test"))

    # Log a test entry
    logger.log_query(
        model='test-model',
        provider='test',
        system_prompt='Test system prompt',
        user_prompt='Test user prompt',
        raw_response='Test response: 5',
        parsed_response=5.0,
        is_valid=True,
        metadata={'question_id': 'A008', 'country': None}
    )

    # Check stats
    stats = logger.get_stats()

    if stats['total_queries'] > 0:
        print(f"PASS: Logged {stats['total_queries']} query/queries")
        print(f"PASS: Log file: {stats['log_file']}")
        return True
    else:
        print("FAIL: Logging failed")
        return False


def main():
    print("="*70)
    print("WEEK 3 INFRASTRUCTURE TEST")
    print("="*70)

    tests = [
        ("Response Parser", test_response_parser),
        ("Logging System", test_logging),
        ("Ollama", test_ollama),
        ("OpenAI API", test_openai),
        ("IVS Question", test_ivs_question),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\nERROR: {name} test crashed: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name}")

    critical_tests = ['Response Parser', 'Logging System']
    critical_passed = all(results.get(test, False) for test in critical_tests)

    ollama_passed = results.get('Ollama', False)
    api_passed = results.get('OpenAI API', False)

    print("\n" + "="*70)
    if critical_passed and (ollama_passed or api_passed):
        print("READY TO START QUERYING")
        print("\nAvailable providers:")
        if ollama_passed:
            print("  - Local models via Ollama")
        if api_passed:
            print("  - OpenAI API models")
        print("\nNext: Run baseline replication with Week 6 script")
    elif critical_passed:
        print("CORE SYSTEMS READY")
        print("\nNeed at least one LLM provider:")
        if not ollama_passed:
            print("  - Install Ollama and download models")
        if not api_passed:
            print("  - Set OPENAI_API_KEY environment variable")
    else:
        print("CRITICAL TESTS FAILED")
        print("\nFix the failing tests before proceeding")


if __name__ == "__main__":
    main()