# Project Structure

```
.
├── config/                         Configuration files
│   └── models.py                   LLM model definitions and priority tiers
│
├── docs/                           Documentation
│   ├── data/                       Data-related documentation
│   │   ├── variable_mapping.md
│   │   └── wvs_evs_merge_syntax.md
│   └── timeline/                   Week-by-week progress
│       ├── week2.md
│       └── week5.md
│
├── scripts/                        Executable scripts (run from project root)
│   ├── data/                       Data preparation
│   │   ├── build_ivs.py           Merge WVS and EVS datasets into ivs_2005-2022.csv
│   │   └── explore_wvs_evs.py     Explore raw data
│   ├── baseline/                   Baseline generation and visualization
│   │   ├── generate_cultural_map.py  Generate ground-truth country coordinates
│   │   ├── baseline_replication.py   Query LLMs across all tones, save results CSVs
│   │   └── visualize_baseline.py     Plot cultural map with LLM positions
│   └── analysis/                   Analysis scripts
│       └── investigate_y003.py    Analyze Y003 (Autonomy Index) missing data
│
├── src/                            Core library (imported by scripts)
│   ├── cultural_map.py            PCA pipeline replicating the Inglehart-Welzel map
│   ├── data_loader.py             IVS data loader and merger
│   ├── geo_data.py                Country names, cultural zones, zone colors, helpers
│   ├── llm_interface.py           Unified LLM query wrapper (Ollama, OpenAI, Anthropic)
│   ├── prompts.py                 IVS question prompts and system prompt tones
│   ├── query_logger.py            JSONL query/response logging
│   └── response_parser.py         Parse and validate LLM responses by question type
│
├── tests/                          Tests and validation
│   ├── test_ollama_setup.py       Test Ollama installation
│   ├── test_llm_infrastructure.py Test LLM query system
│   ├── validate_week2.py          Validate Week 2 completion
│   └── verify_data_conversion.py  Verify WVS/EVS conversion
│
├── data/                           Data files (gitignored)
│   ├── raw/                        Original WVS/EVS data
│   ├── processed/                  Processed datasets (ivs_2005-2022.csv, coordinates)
│   └── results/                    Per-run CSVs (baseline_models_{tone}_{ts}.csv, etc.)
│
├── logs/                           JSONL query logs (one file per run)
├── outputs/                        Generated visualizations (baseline_with_models_{tone}.png)
├── README.md                       Project overview
└── requirements.txt                Python dependencies
```

## Key conventions

- **Tones**: `standard` (neutral), `friendly` (warm), `combative` (blunt). Each generates
  its own baseline CSV and visualization PNG.
- **Result filenames**: `baseline_models_{tone}_{timestamp}.csv` and
  `baseline_distances_{tone}_{timestamp}.csv`.
- **Shared country data**: All country names, cultural zones, zone colors, and helper
  functions live in `src/geo_data.py` — do not duplicate in scripts.
- **Running scripts**: Always run from the project root so relative `Path("data/...")` paths
  resolve correctly, e.g. `python scripts/baseline/baseline_replication.py`.
```
