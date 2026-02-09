# Week 2: Instrument and Data Freezing

## Overview

Week 2 infrastructure for the cultural alignment replication study. This document describes the implemented components and their usage.

## Components

### 1. Core Pipeline: [src/cultural_map.py](../src/cultural_map.py)

Replicates the Inglehart-Welzel Cultural Map using PCA.

Pipeline steps:
- Extract 10 IVS questions (A008, A165, E018, E025, F063, F118, F120, G006, Y002, Y003)
- Standardize responses (mean=0, std=1)
- Apply PCA to create 2D coordinates
- Rescale using PNAS formulas: PC1' = 1.81 * PC1 + 0.38
- Aggregate to country level using sample weights

Main class: `CulturalMapGenerator(ivs_df)`
- `fit()` - execute full pipeline
- `get_country_coordinates()` - return results
- `save_coordinates(path)` - save to CSV

### 2. Prompt Configuration: [src/prompts.py](../src/prompts.py)

Stores exact questions from PNAS Table 1.

Contents:
- 10 question prompts with exact wording
- Response format instructions
- 10 system prompt variants ("average human being", "typical person", etc.)
- Cultural prompting function

Functions:
- `format_full_prompt(question_id, country_name, variant)` - build complete prompt
- `get_cultural_prompt(country_name, variant)` - generate system message
- `get_all_question_ids()` - return question IDs

### 3. Model Configuration: [config/models.py](../config/models.py)

Defines all models to test.

Model categories:
- Western open: Gemma-2, LLaMA-3, Mistral
- East Asian: Qwen-2.5 (multiple sizes)
- Arabic: Jais
- Proprietary: GPT-4o/4/3.5-turbo, Claude, Gemini

Metadata includes size, origin, languages, priority.
Default parameters: temperature=0.0 for baseline.

Functions:
- `get_ollama_models()` - list local models
- `get_api_models()` - list API models
- `get_model_info(name)` - get model config

### 4. Generation Script: [scripts/generate_baseline_cultural_map.py](../scripts/generate_baseline_cultural_map.py)

Executes the full pipeline.

Steps:
1. Load IVS data from `data/processed/ivs_2005-2022.csv`
2. Run `CulturalMapGenerator.fit()`
3. Save coordinates to `data/processed/cultural_map_coordinates.csv`
4. Create visualization: `data/processed/cultural_map_baseline.png`
5. Print summary statistics

Usage: `python scripts/generate_baseline_cultural_map.py`

### 5. Validation Script: [scripts/w2_validation.py](../scripts/w2_validation.py)

Validates Week 2 infrastructure.

Checks:
- IVS data file exists
- Required columns present (S007_01, S024, S001, S017, 10 questions)
- 10 prompts configured correctly
- Cultural prompting works
- Model configuration complete
- CulturalMapGenerator imports successfully

Usage: `python scripts/w2_validation.py`

### 6. Dependencies: [requirements.txt](../requirements.txt)

Required Python packages:
- pandas, numpy, scikit-learn, scipy
- matplotlib, seaborn
- ollama, openai, anthropic

---

## Usage

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify your IVS data exists
ls data/processed/ivs_2005-2022.csv

# If not, run:
# python scripts/build_ivs.py
```

### Validation

```bash
python scripts/w2_validation.py
```

Expected output:
```
PASS: IVS Data
PASS: Prompts
PASS: Models
PASS: Cultural Map Generator

ALL CHECKS PASSED - Week 2 infrastructure is ready
```

### Generate Baseline Map

```bash
python scripts/generate_baseline_cultural_map.py
```

Creates the ground truth baseline from IVS survey data.

Expected output:
```
CULTURAL MAP GENERATION COMPLETE
Total countries mapped: 107
Variance explained: 39%
Cultural Map Statistics:
  X-axis range: [-2.5, 3.0]
  Y-axis range: [-1.8, 2.0]
```

### Output Files

Expected outputs:
- `data/processed/cultural_map_coordinates.csv` - Country coordinates
- `data/processed/cultural_map_baseline.png` - Visualization

The CSV should look like:
```csv
country_code,traditional_secular,survival_selfexpression,n_respondents
276,1.23,0.87,1500
840,-0.45,1.95,2000
...
```

---

## Question Dimensions

### Self-Expression (X-axis)
- F118 (Homosexuality): Tolerance of diversity
- F120 (Abortion): Personal autonomy
- A165 (Trust): Social capital
- E025 (Petition): Political engagement

High values indicate tolerant, autonomous, engaged societies.

### Traditional-Secular (Y-axis)
- F063 (Importance of God): Religiosity
- E018 (Respect for Authority): Traditional hierarchy
- G006 (National Pride): In-group loyalty
- Y003 (Autonomy): Child-rearing values

High values indicate secular, rational-authority societies.

---

## Completion Checklist

- Dependencies installed
- IVS data at `data/processed/ivs_2005-2022.csv`
- Validation script passes all checks
- Baseline map generated
- ~107 countries mapped (±5)
- Variance explained ~39% (±5%)
- Visualization shows expected clustering
- Coordinates saved to CSV

Ready to proceed to Week 3/4: Local Model Infrastructure

---

## File Structure

```
config/models.py                                        Model configuration
scripts/data/build_ivs.py                               Merge WVS+EVS
scripts/baseline/generate_baseline_cultural_map.py      Generate baseline
tests/validate_week2.py                                 Validation
src/cultural_map.py                                     PCA pipeline
src/data_loader.py                                      IVS loader
src/prompts.py                                          Question prompts
data/processed/ivs_2005-2022.csv                        Merged IVS data
data/processed/cultural_map_coordinates.csv             Baseline coordinates
data/processed/cultural_map_baseline.png                Visualization
requirements.txt                                        Dependencies
```