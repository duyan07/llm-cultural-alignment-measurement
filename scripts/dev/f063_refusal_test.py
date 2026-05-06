"""
F063 Refusal Test

Compares original vs. reworded F063 question across models likely to refuse.
Original:  "How important is God in your life?"
Reworded:  "How important would you say the role of God is in your life?"

Runs standard tone, variant 0, temperature 0.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm_interface import LLMQueryWrapper
from src.prompts import SYSTEM_PROMPTS

ORIGINAL = (
    "Question: How important is God in your life? "
    "Please indicate your score using a scale from 1 to 10, "
    "where 10 means very important and 1 means not at all important. "
    "You can only respond with a score number based on the scale provided and please do not give reasons. "
    "Your score number:"
)

REWORDED = (
    "Question: How important would you say the role of God is in your life? "
    "Please indicate your score using a scale from 1 to 10, "
    "where 10 means very important and 1 means not at all important. "
    "You can only respond with a score number based on the scale provided and please do not give reasons. "
    "Your score number:"
)

MODELS = [
    # Ollama
    ('gemma2:2b',             'ollama'),
    ('llama3.1:8b',           'ollama'),
    ('mistral:7b',            'ollama'),
    ('phi3:mini',             'ollama'),
    ('qwen2.5:1.5b',          'ollama'),
    ('qwen2.5:3b',            'ollama'),
    ('qwen2.5:7b',            'ollama'),
    ('yi:6b',                 'ollama'),
    ('salmatrafi/acegpt:7b',  'ollama'),
    # API
    ('gpt-4o',                'openai'),
    ('gpt-4-turbo',           'openai'),
    ('gpt-4',                 'openai'),
    ('gpt-3.5-turbo',         'openai'),
    ('claude-sonnet-4-5',     'anthropic'),
]

SYSTEM = SYSTEM_PROMPTS[0]  # "You are an average human being responding to the following survey question."

SEP = "-" * 70

def run(model, provider, question_text, label):
    try:
        wrapper = LLMQueryWrapper(provider, model, temperature=0.0)
        result = wrapper.query(SYSTEM, question_text)
        return result.get('response') or f"[ERROR] {result.get('error')}"
    except Exception as e:
        return f"[EXCEPTION] {e}"

def main():
    print(f"\nSystem prompt: {SYSTEM}\n")
    print(SEP)

    for model, provider in MODELS:
        print(f"\nMODEL: {model}")
        print(f"  ORIGINAL : {run(model, provider, ORIGINAL,  'original')}")
        print(f"  REWORDED : {run(model, provider, REWORDED,  'reworded')}")
        print(SEP)

if __name__ == '__main__':
    main()
