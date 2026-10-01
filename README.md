# Measurement Validity in LLM Cultural Alignment

Does an LLM's position on the Inglehart-Welzel cultural map reflect a real cultural signal, or is it mostly measurement noise? This project treats LLM survey responses as a psychometric instrument, replicates the Tao et al. (PNAS Nexus) approach of projecting model answers onto the map, and then quantifies how much of that projection is explained by sampling randomness and prompt wording rather than genuine between-model differences.

See [docs/experiment_overview.md](docs/experiment_overview.md) for the full methodology, math, and current findings. See [docs/STRUCTURE.md](docs/STRUCTURE.md) for a map of the repository, and [docs/COMMANDS.md](docs/COMMANDS.md) for the full command reference.

## Known issue — cultural map coefficients

The affine transform that places PCA scores on the Inglehart-Welzel map (`x = 1.81·PC1 + 0.38, y = 1.61·PC2 - 0.01`, in `src/cultural_map.py` and the scripts that duplicate it) has a transposed-digit error inherited from Tao et al. (PNAS Nexus, 2024). Per the [official WVS SPSS syntax](https://worldvaluessurvey.org/WVSContents.jsp?CMSID=tradrat) (`COMPUTE SurvSAgg = 1.81 * SurvSelf + .038`, `COMPUTE TradAgg = 1.61 * TradRat5 - .1`), the correct constants are `+0.038` and `-0.1`. Matt Hall (Equinor) flagged this in September 2026 — thank you, Matt.

This repo intentionally keeps the erroneous constants so the code stays consistent with all already-generated data, figures, and reported numbers (regenerating everything would mean rerunning the full LLM-querying pipeline). The error is a uniform translation of every coordinate and does not change any comparison computed entirely within this pipeline (model-to-model, model-to-closest-country, clustering, NSR) — it only affects comparisons against externally-sourced absolute coordinates, such as the literal PNAS GPT-4o reference point quoted in `scripts/baseline/visualize_baseline.py`. A version of this pipeline with the corrected constants is maintained separately for follow-on work.

## Pipeline

The pipeline is one continuous measurement process, not a set of independent steps — each part feeds the next:

1. **Build the cultural map** — merge WVS/EVS survey data into a single IVS dataset and run PCA to reproduce the 88-country Inglehart-Welzel coordinate system.
2. **Query the models** — prompt each LLM with the 10 IVS questions (3 tones × 10 phrasings) and project its answers onto the map (baseline replication).
3. **Vary the seed** — hold the prompt fixed and resample with different seeds to measure stochastic noise (σ_seed).
4. **Vary the prompt** — hold the seed fixed and resample across all 30 semantically-equivalent prompt variants to measure wording sensitivity (σ_prompt).
5. **Decompose the variance** — combine σ_seed and σ_prompt against the between-model spread (σ_culture) into a noise-to-signal ratio (NSR) per model/question, which is the paper's core validity metric.

## Quick start

```bash
# 1. One-time data setup
Rscript scripts/data/convert_sav.R
python scripts/data/build_ivs.py
python scripts/baseline/generate_cultural_map.py

# 2. Query models and project onto the map
python scripts/baseline/baseline_replication.py
python scripts/baseline/visualize_baseline.py

# 3. Measure stochastic and prompt-wording noise
python scripts/analysis/stochastic_sensitivity.py
python scripts/analysis/prompt_sensitivity.py

# 4. Decompose variance into signal vs. noise
python scripts/analysis/variance_decomposition.py
```

Full flags, outputs, and file-by-file walkthroughs are in [docs/COMMANDS.md](docs/COMMANDS.md).

## Requirements

- Python 3, packages in `requirements.txt`
- R (only for the one-time `.sav` → CSV conversion)
- [Ollama](https://ollama.com) running locally for open-weight models
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` for proprietary models (optional — `--model-set open` skips these)
