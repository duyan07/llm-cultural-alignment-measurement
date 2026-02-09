# Project Structure

```
.
├── config/                         Configuration files
│   └── models.py                   LLM model definitions
│
├── docs/                           Documentation
│   ├── data/                       Data-related documentation
│   │   ├── variable_mapping.md
│   │   └── wvs_evs_merge_syntax.md
│   └── timeline/                   Week-by-week progress
│       ├── week2.md
│       └── week4.md
│
├── scripts/                        Executable scripts
│   ├── data/                       Data preparation
│   │   ├── build_ivs.py           Merge WVS and EVS datasets
│   │   └── explore_wvs_evs.py     Explore raw data
│   ├── baseline/                   Baseline generation
│   │   └── generate_cultural_map.py  Generate ground truth map
│   └── analysis/                   Analysis scripts
│       └── investigate_y003.py    Analyze Y003 missing data
│
├── src/                            Core library code
│   ├── cultural_map.py            PCA pipeline for cultural map
│   ├── data_loader.py             IVS data loader
│   ├── llm_interface.py           Unified LLM query wrapper
│   ├── prompts.py                 IVS question prompts
│   ├── query_logger.py            Query/response logging
│   └── response_parser.py         Parse LLM responses
│
├── tests/                          Tests and validation
│   ├── test_ollama_setup.py       Test Ollama installation
│   ├── test_llm_infrastructure.py Test LLM query system
│   ├── validate_week2.py          Validate Week 2 completion
│   └── verify_data_conversion.py  Verify WVS/EVS conversion
│
├── data/                           Data files (gitignored)
│   ├── raw/                        Original WVS/EVS data
│   └── processed/                  Processed datasets
│
├── README.md                       Project overview
└── requirements.txt                Python dependencies
```
