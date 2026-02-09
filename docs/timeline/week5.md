# Week 5: LLM Infrastructure and API Model Setup

## Overview

Week 5 implements the complete LLM querying infrastructure for the cultural alignment replication study. This document describes the LLM interface, response parsing, and logging systems that enable querying both local and API-based language models with the 10 IVS questions.

## Components

### 1. LLM Query Wrapper: [src/llm_interface.py](../src/llm_interface.py)

Unified interface for querying multiple LLM providers.

Supported providers:
- Ollama (local models: Gemma-2, Qwen-2.5, Phi-3, LLaMA-3, Mistral)
- OpenAI (GPT-4o, GPT-4, GPT-3.5-turbo)
- Anthropic (Claude Opus, Sonnet, Haiku)
- Google (Gemini Pro, Ultra)

Main class: `LLMQueryWrapper(provider, model_name, temperature, seed)`
- `query(system_prompt, user_prompt)` - send prompt and get response
- `batch_query(prompts)` - query multiple prompts efficiently
- Automatic error handling with retries
- Token counting and cost tracking

### 2. Response Parser: [src/response_parser.py](../src/response_parser.py)

Extracts structured answers from natural language LLM responses.

Parsing capabilities:
- Numeric scales (1-4 for happiness, 1-10 for importance)
- Categorical choices (A/B/C/D options)
- Multi-choice questions (Y002, Y003 with multiple selections)
- Robust extraction from conversational responses

Main class: `ResponseParser`
- `parse_by_type(response, question_info)` - parse based on question type
- `validate_response(parsed, question_info)` - check validity
- `extract_numeric(response, min_val, max_val)` - extract numbers
- Error detection for ambiguous or invalid responses

### 3. Query Logger: [src/query_logger.py](../src/query_logger.py)

Tracks all queries and responses for analysis and reproducibility.

Features:
- JSONL format for efficient storage and analysis
- Checkpoint system for resuming interrupted runs
- Automatic CSV export for statistical analysis
- Summary statistics and progress tracking

Main class: `QueryLogger(log_dir)`
- `log_query(model, provider, prompts, response, metadata)` - record query
- `export_to_csv(output_path)` - convert JSONL to CSV
- `load_checkpoint()` - resume from last successful query
- `print_stats()` - display query counts and success rates

### 4. Testing Script: [tests/test_llm_infrastructure.py](../tests/test_llm_infrastructure.py)

Validates the complete LLM infrastructure.

Checks:
- Ollama service running and models available
- API keys configured correctly
- Query wrapper connects to each provider
- Response parser handles all question types
- Logger creates and updates files correctly
- Round-trip query → parse → log pipeline works

---

## Usage

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install Ollama for local models
curl -fsSL https://ollama.ai/install.sh | sh

# Download required models
ollama pull gemma2:2b-instruct
ollama pull qwen2.5:1.5b-instruct
ollama pull phi3:mini

# Configure API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Validation

```bash
python tests/test_llm_infrastructure.py
```

Expected output:
```
PASS: Ollama connection
PASS: OpenAI API
PASS: Response parser
PASS: Query logger

ALL CHECKS PASSED - Week 5 infrastructure is ready
```

### Query a Model

Basic query example:
```python
from src.llm_interface import LLMQueryWrapper
from src.prompts import format_full_prompt

# Initialize wrapper
wrapper = LLMQueryWrapper(
    provider='ollama',
    model_name='gemma2:2b-instruct',
    temperature=0.0,
    seed=42
)

# Format prompt with cultural context
prompt = format_full_prompt('A008', country_name='Thailand', variant=0)

# Query model
result = wrapper.query(
    system_prompt=prompt['system'],
    user_prompt=prompt['user']
)
```

### Parse and Log Responses

Complete pipeline example:
```python
from src.response_parser import ResponseParser
from src.query_logger import QueryLogger
from src.prompts import QUESTIONS

# Parse response
question_info = QUESTIONS['A008']
parsed = ResponseParser.parse_by_type(result['response'], question_info)

# Log query
logger = QueryLogger(log_dir='logs')
logger.log_query(
    model='gemma2:2b-instruct',
    provider='ollama',
    system_prompt=prompt['system'],
    user_prompt=prompt['user'],
    raw_response=result['response'],
    parsed_response=parsed,
    is_valid=True,
    metadata={'question_id': 'A008', 'country': 'Thailand'}
)
```

### Output Files

Expected outputs:
- `logs/queries_YYYYMMDD_HHMMSS.jsonl` - Raw query logs
- `logs/queries_YYYYMMDD_HHMMSS.csv` - Parsed responses
- `logs/checkpoints/model_progress.json` - Resume checkpoints

JSONL format example:
```json
{
  "timestamp": "2026-02-08T10:30:00",
  "model": "gemma2:2b-instruct",
  "provider": "ollama",
  "question_id": "A008",
  "country": "Thailand",
  "variant": 0,
  "raw_response": "I would say about 2",
  "parsed_response": 2.0,
  "is_valid": true
}
```

---

## Model Configuration

### Temperature Settings
- **0.0**: Deterministic responses (baseline testing)
- **1.0**: Stochastic responses (variance testing)

Default: 0.0 for replication consistency

### Prompt Variants
Following PNAS methodology, 10 system prompt variations:
- "You are an average human being"
- "You are a typical person"
- "You are a world citizen"
- (7 additional variants in [src/prompts.py](../src/prompts.py))

Tests sensitivity to prompt wording.

### Cultural Prompting
Insert country context into system prompt:
- Base: "You are a typical person"
- Cultural: "You are a typical person from Thailand"

PNAS findings: Reduces bias for 71-81% of countries.

---

## Completion Checklist

- Ollama installed and service running
- Required models downloaded (gemma2:2b, qwen2.5:1.5b, phi3:mini)
- API keys configured for OpenAI/Anthropic
- Validation script passes all checks
- Can query at least one local model
- Can query at least one API model
- Response parser handles all 10 question types
- Logger creates JSONL and CSV files
- Checkpoint system tested for resume capability

Ready to proceed to Week 6: Baseline Replication

---

## File Structure

```
src/llm_interface.py                LLM query wrapper
src/response_parser.py              Extract structured answers
src/query_logger.py                 Query/response logging
src/prompts.py                      IVS questions and variants
config/models.py                    Model configurations
tests/test_llm_infrastructure.py    Infrastructure validation
logs/queries_*.jsonl                Raw query logs
logs/queries_*.csv                  Parsed responses
logs/checkpoints/*.json             Resume checkpoints
```
