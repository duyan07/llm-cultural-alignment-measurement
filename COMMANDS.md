# Commands Reference

All commands are run from the **project root** so that relative `Path("data/...")` paths resolve correctly.

---

## Main Pipeline

The full pipeline runs in three ordered stages: data preparation → baseline generation → querying & visualization.

### Stage 1 — Data Preparation

#### Build IVS Dataset
**File:** [scripts/data/build_ivs.py](scripts/data/build_ivs.py)

Merges WVS waves 5–7 and EVS waves 4–5 (2005–2022) into a single processed dataset.
Reads from `data/raw/csv/` and writes `data/processed/ivs_2005-2022.csv`.

```bash
python scripts/data/build_ivs.py
```

**Outputs:**
- `data/processed/ivs_2005-2022.csv` — merged IVS dataset (~500k+ rows)
- `data/processed/ivs_2005-2022.metadata.json` — row/column/country counts and wave info

---

### Stage 2 — Baseline Generation

#### Generate Cultural Map Coordinates
**File:** [scripts/baseline/generate_cultural_map.py](scripts/baseline/generate_cultural_map.py)

Runs the PCA pipeline on IVS data to produce the ground-truth 88-country Inglehart-Welzel
cultural map coordinates. Must be run before baseline replication.

```bash
python scripts/baseline/generate_cultural_map.py
```

**Outputs:**
- `data/processed/cultural_map_coordinates.csv` — country-level (x, y) coordinates
- `data/processed/cultural_map_baseline.png` — plain scatter plot (no zone colors or LLM positions)

**Requires:** `data/processed/ivs_2005-2022.csv` (run Stage 1 first)

---

### Stage 3 — LLM Querying & Visualization

#### Run Baseline Replication
**File:** [scripts/baseline/baseline_replication.py](scripts/baseline/baseline_replication.py)

Queries each LLM with all 10 IVS survey questions across 10 prompt variants × 3 tones
(standard, friendly, combative). Projects responses into the cultural map space and saves
results CSVs and a JSONL query log.

```bash
# Run all default models across all tones
python scripts/baseline/baseline_replication.py

# Run specific models only
python scripts/baseline/baseline_replication.py --models gemma2:2b qwen2.5:1.5b

# Run specific tones only
python scripts/baseline/baseline_replication.py --tones standard friendly

# Combine: specific models and tones
python scripts/baseline/baseline_replication.py --models llama3.1:8b --tones combative
```

**Options:**

| Flag | Description | Example |
|------|-------------|---------|
| `--models` | Space-separated list of model names (Ollama format) | `--models gemma2:2b llama3.1:8b` |
| `--tones` | One or more of `standard`, `friendly`, `combative` | `--tones standard combative` |

**Outputs (per tone, timestamped):**
- `data/results/baseline_models_{tone}_{timestamp}.csv` — model positions and closest country
- `data/results/baseline_distances_{tone}_{timestamp}.csv` — distance from each model to all 88 countries
- `logs/queries_{timestamp}.jsonl` — full query/response log
- `outputs/baseline_summary_{timestamp}.txt` — human-readable findings summary across all tones

**Requires:** `data/processed/cultural_map_coordinates.csv` and `data/processed/ivs_2005-2022.csv`

---

#### Visualize Baseline Results
**File:** [scripts/baseline/visualize_baseline.py](scripts/baseline/visualize_baseline.py)

Reads the most recent results CSVs and generates a publication-quality cultural map image
with country dots colored by cultural zone, ISO-3 country labels, and LLM star markers.
Also prints a detailed per-model summary to stdout.

```bash
# Visualize most recent results for all 3 tones
python scripts/baseline/visualize_baseline.py

# Visualize most recent results for a specific tone
python scripts/baseline/visualize_baseline.py --tone friendly

# Visualize a specific results file
python scripts/baseline/visualize_baseline.py --results data/results/baseline_models_combative_20260226_010108.csv

# Save to a custom output path
python scripts/baseline/visualize_baseline.py --tone standard --output outputs/my_map.png
```

**Options:**

| Flag | Description | Example |
|------|-------------|---------|
| `--tone` | Filter to results from a specific tone | `--tone combative` |
| `--results` | Path to a specific `baseline_models_*.csv` file | `--results data/results/baseline_models_standard_20260301_120000.csv` |
| `--output` | Override default output image path | `--output outputs/custom.png` |

**Outputs (per tone):**
- `outputs/baseline_with_models_{tone}_{timestamp}.png` — cultural map with LLM positions
- `outputs/baseline_with_models_{tone}_{timestamp}.txt` — summary text: positions, quadrant, top-5 closest countries, average model position

**Requires:** `data/results/baseline_models_*.csv` (run Stage 3 querying first)

---

## Analysis Scripts

#### Investigate Y003 Missing Data
**File:** [scripts/analysis/investigate_y003.py](scripts/analysis/investigate_y003.py)

Analyzes missingness of Y003 (Autonomy Index) across survey waves and quantifies how many
countries/responses are lost when requiring all 10 questions vs. 9 questions.

```bash
python scripts/analysis/investigate_y003.py data/processed/ivs_2005-2022.csv
```

**Stdout includes:** Y003 value distribution, availability by WVS/EVS wave, and comparison
of country coverage with vs. without the Y003 requirement.

---

#### Explore Raw WVS/EVS Data
**File:** [scripts/data/explore_wvs_evs.py](scripts/data/explore_wvs_evs.py)

Prints basic structural statistics (row count, column count, country count, variable labels)
for the raw WVS and EVS CSV files. Useful for sanity-checking raw data before building IVS.

```bash
python scripts/data/explore_wvs_evs.py
```

**Reads:** `data/raw/csv/wvs_trend_1981-2022.csv`, `data/raw/csv/evs_trend_1981-2017.csv`
and their respective `*_variable_labels.csv` files.

---

## Test / Validation Scripts

#### Test Ollama Setup
**File:** [tests/test_ollama_setup.py](tests/test_ollama_setup.py)

Verifies that Ollama is running and at least one model is installed and can generate responses.
Run this first when setting up a new environment.

```bash
python tests/test_ollama_setup.py
```

---

#### Test LLM Infrastructure
**File:** [tests/test_llm_infrastructure.py](tests/test_llm_infrastructure.py)

End-to-end test of the query wrapper, response parser, and logging system. Tests Ollama
connectivity, prompt formatting, response parsing, and log writing. Run before starting
a full experiment to catch configuration issues early.

```bash
python tests/test_llm_infrastructure.py
```

---

#### Validate Week 2 Completion
**File:** [tests/validate_week2.py](tests/validate_week2.py)

Validates that the Week 2 deliverables (IVS merge, PCA pipeline) are working correctly.

```bash
python tests/validate_week2.py
```

---

#### Verify Data Conversion
**File:** [tests/verify_data_conversion.py](tests/verify_data_conversion.py)

Checks that the WVS/EVS merge output is well-formed (correct columns, no corrupt rows, etc.).

```bash
python tests/verify_data_conversion.py
```

---

## Ollama Model Management

These are standard Ollama CLI commands for managing local models used in the pipeline.

```bash
# List all locally available models
ollama list

# Pull a model (downloads it)
ollama pull gemma2:2b
ollama pull phi3:mini
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull llama3.1:8b
ollama pull yi:6b
ollama pull salmatrafi/acegpt:7b

# Remove a model
ollama rm gemma2:2b

# Check if Ollama server is running
ollama ps
```

---

## Quick Reference

| Goal | Command |
|------|---------|
| Set up IVS data from scratch | `python scripts/data/build_ivs.py` |
| Generate country baselines | `python scripts/baseline/generate_cultural_map.py` |
| Query all models, all tones | `python scripts/baseline/baseline_replication.py` |
| Query specific models | `python scripts/baseline/baseline_replication.py --models llama3.1:8b` |
| Visualize latest standard results | `python scripts/baseline/visualize_baseline.py --tone standard` |
| Visualize a specific file | `python scripts/baseline/visualize_baseline.py --results data/results/baseline_models_friendly_20260301_120000.csv` |
| Investigate Y003 missingness | `python scripts/analysis/investigate_y003.py data/processed/ivs_2005-2022.csv` |
| Verify Ollama works | `python tests/test_ollama_setup.py` |
| Full infrastructure check | `python tests/test_llm_infrastructure.py` |
