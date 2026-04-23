# Project Structure

```
.
├── config/
│   └── models.py                   LLM model definitions, priorities, and parameters
│
├── data/
│   ├── raw/                        Original WVS/EVS data (gitignored)
│   │   ├── csv/                    CSV exports of raw survey files
│   │   └── sav/                    Original SPSS .sav files
│   ├── processed/                  Intermediate pipeline outputs
│   │   ├── ivs_2005-2022.csv       Merged IVS dataset (~500k rows)
│   │   ├── ivs_2005-2022.metadata.json
│   │   └── cultural_map_coordinates.csv   88-country (x, y) ground-truth positions
│   └── results/                    Per-run query results
│       ├── baseline/               baseline_replication.py outputs
│       ├── stochastic/             stochastic_sensitivity.py outputs
│       └── prompt_sensitivity/     prompt_sensitivity.py outputs
│
├── docs/
│   ├── data/
│   │   ├── variable_mapping.md
│   │   └── wvs_evs_merge_syntax.md
│   └── timeline/                   Stage-by-stage progress notes
│
├── logs/                           JSONL query/response logs (one file per run)
│   ├── baseline/                   Logs from baseline_replication.py
│   ├── stochastic/                 Logs from stochastic_sensitivity.py
│   └── prompt_sensitivity/         Logs from prompt_sensitivity.py
│
├── outputs/                        Generated figures and summaries
│   ├── baseline/                   Baseline cultural maps and summaries
│   ├── stochastic/                 Stochastic stability heatmaps and distributions
│   ├── prompt_sensitivity/         Flip-rate heatmaps, tone comparisons, variant maps
│   └── variance_decomposition/     NSR heatmaps, variance bars, combined maps
│
├── scripts/                        Executable pipeline scripts (run from project root)
│   ├── data/                       Data preparation (run once)
│   │   ├── build_ivs.py            Merge WVS + EVS into ivs_2005-2022.csv
│   │   ├── convert_sav.R           Convert raw .sav files to CSV
│   │   └── explore_wvs_evs.py      Sanity-check raw data structure
│   ├── baseline/                   Baseline generation (Steps 1–2)
│   │   ├── generate_cultural_map.py   PCA → 88-country coordinates + baseline plot
│   │   ├── baseline_replication.py    Query LLMs, project to cultural map
│   │   └── visualize_baseline.py      Plot cultural map with LLM positions
│   ├── analysis/                   Sensitivity analysis (Steps 3–5)
│   │   ├── stochastic_sensitivity.py      Stage 3: vary seed, fix prompt
│   │   ├── visualize_stochastic.py        Stage 3: stability heatmap + distributions
│   │   ├── visualize_stochastic_map.py    Stage 3: seed clouds on cultural map
│   │   ├── prompt_sensitivity.py          Stage 4: vary prompt, fix seed
│   │   ├── visualize_prompt_sensitivity.py  Stage 4: flip-rate, tone comparison, variant map
│   │   └── variance_decomposition.py      Stage 5: σ_seed vs σ_prompt vs σ_culture
│   └── dev/                        One-off exploration and debugging scripts
│       └── investigate_y003.py     Y003 missingness analysis
│
├── src/                            Core library (imported by scripts)
│   ├── cultural_map.py             PCA pipeline replicating the Inglehart-Welzel map
│   ├── data_loader.py              IVS data loader and merger
│   ├── geo_data.py                 Country names, cultural zones, zone colors, helpers
│   ├── llm_interface.py            Unified LLM query wrapper (Ollama, OpenAI, Anthropic)
│   ├── prompts.py                  IVS question prompts and system prompt tones
│   ├── query_logger.py             JSONL query/response logging
│   └── response_parser.py          Parse and validate LLM responses by question type
│
├── tests/                          Tests and validation scripts
│   ├── test_ollama_setup.py        Verify Ollama is running and a model responds
│   ├── test_llm_infrastructure.py  End-to-end test of query wrapper + parser + logger
│   ├── validate_week2.py           Validate IVS merge + PCA pipeline
│   └── verify_data_conversion.py   Check WVS/EVS merge output integrity
│
├── COMMANDS.md                     Full command reference for every pipeline step
├── README.md                       Project overview and quick-start
└── requirements.txt                Python dependencies
```

## Key conventions

- **Run all scripts from the project root** so that relative `Path("data/...")` paths resolve correctly.
- **Model sets**: `--model-set all` (default, open + API), `--model-set open` (Ollama only), `--model-set api` (proprietary only). The tag is embedded in output filenames for traceability.
- **Output filenames**: `{script_output}_{model_set}_{timestamp}.{ext}` — the model_set tag is omitted when not specified (backward compatible).
- **Tones**: `standard` (neutral), `friendly` (warm), `combative` (blunt).
- **Results subdirs mirror script subdirs**: baseline → `data/results/baseline/`, stochastic → `data/results/stochastic/`, prompt_sensitivity → `data/results/prompt_sensitivity/`.
- **Shared country data**: All country names, cultural zones, zone colors, and helpers live in `src/geo_data.py`.
