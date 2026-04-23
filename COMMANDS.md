# Commands Reference

All commands are run from the **project root** so that relative `Path("data/...")` paths resolve correctly.

---

## Stage 0 — Data Preparation (one-time)

```bash
# Convert raw .sav files to CSV (requires R)
Rscript scripts/data/convert_sav.R

# Merge WVS + EVS into a single IVS dataset
python scripts/data/build_ivs.py
```

**Outputs:** `data/processed/ivs_2005-2022.csv`, `data/processed/ivs_2005-2022.metadata.json`

---

## Stage 1 — Generate Cultural Map Baseline (one-time)

```bash
python scripts/baseline/generate_cultural_map.py
```

**Outputs:** `data/processed/cultural_map_coordinates.csv`, `outputs/baseline/cultural_map_baseline.png`

---

## Stage 2 — Baseline Replication

```bash
# All models (Ollama auto-detected + any API keys set)
python scripts/baseline/baseline_replication.py

# Open/local models only
python scripts/baseline/baseline_replication.py --model-set open

# Proprietary API models only (requires OPENAI_API_KEY / ANTHROPIC_API_KEY)
python scripts/baseline/baseline_replication.py --model-set api

# Specific tones only
python scripts/baseline/baseline_replication.py --tones standard friendly

# Specific models (overrides --model-set)
python scripts/baseline/baseline_replication.py --models gemma2:2b llama3.1:8b
```

**Outputs (per tone, in `data/results/baseline/` and `outputs/baseline/`):**
- `baseline_models_{tone}_{model_set}_{ts}.csv`
- `baseline_distances_{tone}_{model_set}_{ts}.csv`
- `baseline_summary_{tones}_{model_set}_{ts}.txt`
- `logs/baseline/` — JSONL query log

```bash
# Visualize — one map per tone found in results
python scripts/baseline/visualize_baseline.py

# Filter to a specific model-set run
python scripts/baseline/visualize_baseline.py --model-set open

# Single tone
python scripts/baseline/visualize_baseline.py --tone standard

# Explicit file
python scripts/baseline/visualize_baseline.py --results data/results/baseline/baseline_models_standard_open_<ts>.csv
```

**Outputs:** `outputs/baseline/baseline_with_models_{tone}{_model_set}_{ts}.png`

---

## Stage 3 — Stochastic Sensitivity

Fix prompt (standard tone, variant 0), vary seed 0–9 → isolates sampling noise.
Only Ollama models are supported (seed control requires a local runtime).

```bash
# All open models (default)
python scripts/analysis/stochastic_sensitivity.py

# Explicit models
python scripts/analysis/stochastic_sensitivity.py --models gemma2:2b qwen2.5:7b
```

**Outputs:** `data/results/stochastic/stochastic_flat_{model_set}_{ts}.csv`

```bash
# Stability heatmap + per-model distributions
python scripts/analysis/visualize_stochastic.py
python scripts/analysis/visualize_stochastic.py --model-set open

# Seed clouds on the cultural map
python scripts/analysis/visualize_stochastic_map.py
python scripts/analysis/visualize_stochastic_map.py --model-set open

# Explicit flat file
python scripts/analysis/visualize_stochastic_map.py --flat data/results/stochastic/stochastic_flat_open_<ts>.csv
```

**Outputs:** `outputs/stochastic/`

---

## Stage 4 — Prompt Sensitivity

Fix seed (seed=0), vary all 30 prompt variants (3 tones × 10 variants) → isolates wording noise.

```bash
# All models, temp=0 (for NSR analysis — clean isolation)
python scripts/analysis/prompt_sensitivity.py

# Open models only
python scripts/analysis/prompt_sensitivity.py --model-set open

# API models only
python scripts/analysis/prompt_sensitivity.py --model-set api

# temp=1.0 — run separately to match Stage 3 for the combined map visual
python scripts/analysis/prompt_sensitivity.py --temperature 1.0
```

**Outputs:** `data/results/prompt_sensitivity/prompt_sensitivity_flat_{model_set}_{ts}.csv`

```bash
# Flip-rate heatmap, tone comparison, variant map, tone interaction
python scripts/analysis/visualize_prompt_sensitivity.py
python scripts/analysis/visualize_prompt_sensitivity.py --model-set open

# Explicit flat file
python scripts/analysis/visualize_prompt_sensitivity.py --flat data/results/prompt_sensitivity/prompt_sensitivity_flat_open_<ts>.csv
```

**Outputs:** `outputs/prompt_sensitivity/`

---

## Stage 5 — Variance Decomposition

Combines Stages 3 + 4 to compute σ_seed, σ_prompt, σ_culture and NSR per (model, question).

```bash
# NSR analysis (use temp=0 prompt file — correct for comparing noise sources)
python scripts/analysis/variance_decomposition.py

# Filter to a specific model-set
python scripts/analysis/variance_decomposition.py --model-set open

# Combined cultural map with matched temperatures (pass temp=1.0 prompt run explicitly)
python scripts/analysis/variance_decomposition.py \
  --prompt-flat data/results/prompt_sensitivity/prompt_sensitivity_flat_open_<ts>.csv
```

**Outputs** (in `outputs/variance_decomposition/`)**:** `variance_bars_{model_set}_{ts}.png`, `noise_signal_heatmap_{model_set}_{ts}.png`, `combined_map_{model_set}_{ts}.png`, `variance_decomposition_{model_set}_{ts}.csv`

---

## Tests

```bash
python tests/test_ollama_setup.py        # Verify Ollama is running
python tests/test_llm_infrastructure.py  # End-to-end query + parse + log test
python tests/validate_week2.py           # Validate IVS merge + PCA pipeline
python tests/verify_data_conversion.py   # Check WVS/EVS merge output
```

---

## Ollama Model Management

```bash
ollama list                         # List installed models
ollama pull gemma2:2b               # Download a model
ollama pull phi3:mini
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull llama3.1:8b
ollama pull yi:6b
ollama pull salmatrafi/acegpt:7b
ollama rm gemma2:2b                 # Remove a model
ollama ps                           # Check server status
```

---

## Quick Reference

| Goal | Command |
|------|---------|
| Build IVS dataset | `python scripts/data/build_ivs.py` |
| Generate country map | `python scripts/baseline/generate_cultural_map.py` |
| Run baseline (all models) | `python scripts/baseline/baseline_replication.py` |
| Run baseline (open only) | `python scripts/baseline/baseline_replication.py --model-set open` |
| Visualize baseline | `python scripts/baseline/visualize_baseline.py` |
| Stage 3 — stochastic | `python scripts/analysis/stochastic_sensitivity.py` |
| Stage 3 — visualize | `python scripts/analysis/visualize_stochastic.py` |
| Stage 3 — cultural map | `python scripts/analysis/visualize_stochastic_map.py` |
| Stage 4 — prompt sensitivity | `python scripts/analysis/prompt_sensitivity.py` |
| Stage 4 — visualize | `python scripts/analysis/visualize_prompt_sensitivity.py` |
| Stage 5 — variance decomposition | `python scripts/analysis/variance_decomposition.py` |
| Test Ollama setup | `python tests/test_ollama_setup.py` |
