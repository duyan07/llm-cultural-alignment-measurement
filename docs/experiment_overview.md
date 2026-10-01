# Experiment Overview

Concise reference for what the experiment measures, how the math is set up, and what each output represents. Companion to `README.md` (which is about how to run the code).

---

## 1. The question

*When an LLM answers a value-survey question and we plot its "position" on the Inglehart-Welzel cultural map, is that position a meaningful measurement, or is it noise?*

We treat model outputs as a psychometric instrument and ask whether the instrument is valid — i.e., whether the between-model differences we're reading off the map exceed the instrument's own noise floor.

## 2. Sources of disagreement

For any (model $m$, question $q$) pair, three separate forces produce disagreement:

| Symbol | What it captures | How we vary it |
|---|---|---|
| $\sigma_\text{seed}$ | **Stochastic noise** — same prompt, different sampling seed | 10 random seeds at temperature 1.0, same prompt |
| $\sigma_\text{prompt}$ | **Prompt-wording sensitivity** — semantically equivalent phrasings | 30 prompt variants (3 tones × 10 personas) at temperature 0 |
| $\sigma_\text{culture}$ | **Between-model spread** (the signal) — how much models actually differ | Std of mean responses across all 12 models |

The first two are *instrument noise*. The third is what we want to interpret culturally. If noise ≥ signal, the map coordinate is uninterpretable.

## 3. The math

### 3.1 σ_seed — stochastic variance

$$\sigma_\text{seed}(m, q) = \operatorname{std}\bigl(\{r_{m,q,s}\}_{s=1}^{10}\bigr)$$

10 responses, each with a different sampling seed, on the standard prompt.

- Undefined ($\sigma = \text{NaN}$) when fewer than 2 valid responses returned.
- API models (GPT-4o, GPT-4o-mini, Claude Sonnet 4.6): sampled at their default temperature. Anthropic doesn't expose a seed parameter, so Claude's $\sigma_\text{seed} = 0$ in practice — its NSR is a lower bound.

### 3.2 σ_prompt — prompt-wording variance

$$\sigma_\text{prompt}(m, q) = \operatorname{std}\bigl(\{r_{m,q,v}\}_{v=1}^{30}\bigr)$$

30 semantically-equivalent prompts, temperature 0.

- Same NaN rule.
- The 30 = 3 tones (standard / friendly / combative) × 10 persona wordings.

### 3.3 σ_culture — between-model signal

$$\sigma_\text{culture}(q) = \operatorname{std}\bigl(\{\bar{r}_{m,q}\}_{m=1}^{M}\bigr)$$

Std of the *mean* prompt-sensitivity response across models. $\bar{r}_{m,q}$ is the average of the 30 prompt-sensitivity responses for model $m$ on question $q$. This is the between-model spread the instrument would need to resolve above its noise.

### 3.4 Noise-to-signal ratio (NSR)

$$\text{NSR}(m, q) = \frac{\sigma_\text{seed}(m, q) + \sigma_\text{prompt}(m, q)}{\sigma_\text{culture}(q)}$$

- **NSR = 1**: instrument noise equals cultural signal. Cannot reliably distinguish this model from the panel mean.
- **NSR > 1**: noise exceeds signal — treat coordinate as uninterpretable.
- **NSR < 1**: signal exceeds noise — comparison is meaningful.

### 3.5 Why standard deviation instead of variance?

Std dev is in the same units as the underlying response. F063 is on a 1–10 Likert scale, so a std of 1.9 means "responses vary by ~2 scale points" — directly interpretable. Variance would be squared, which loses the intuitive scale. NSR is a ratio anyway, so unit-preservation matters only for interpreting each σ standalone; the ratio is unitless either way.

### 3.6 Why additive $\sigma_\text{seed} + \sigma_\text{prompt}$ instead of root-sum-of-squares?

$\sqrt{\sigma_\text{seed}^2 + \sigma_\text{prompt}^2}$ (RSS) is the correct combined std under independence and produces strictly smaller values. We use additive as a **conservative upper bound**: the threshold $\text{NSR} > 1$ then has a strict reading — even if we generously combine the two noise sources, instrument noise alone exceeds the cultural signal. Passing the additive threshold is a stronger validity claim than passing the RSS one.

### 3.7 Why the $\bar{r}_{m,q}$ in σ_culture comes from prompt sensitivity, not stochastic

$\bar{r}_{m,q}$ needs to be the most stable estimate of "this model's typical response to this question." The 30-variant prompt-sensitivity average uses more samples (30 vs 10) and covers a broader operating regime (multiple tones × personas), so it's a better anchor than the seed average. Baseline coords are also from prompt sensitivity-like sampling, but at only 10 variants per tone.

## 4. Data flow

```
IVS (survey data, 88 countries) ──┐
                                  ├─→ PCA → cultural map (2D projection)
Cultural map coord CSV ──────────┘

Baseline queries (10 variants × 3 tones)     ──→ baseline_models_{tone}_*.csv
    │                                                 │
    │                                                 └──→ per-model (x,y) on map (Table 2)
    │                                                 └──→ tone displacement (Table 4)
    │
Stochastic queries (10 seeds × standard × v0)──→ stochastic_flat_*.csv → σ_seed ─┐
                                                                                 │
Prompt-sensitivity queries (30 variants)      ──→ prompt_flat_*.csv    → σ_prompt├─→ variance_decomposition.csv
                                                                       → σ_culture┘         │
                                                                                            │
                                                                                            └─→ NSR, per-model/question summaries (Tables 5–7)
```

## 5. Outputs — what each file shows

### `outputs/baseline/`

- **`baseline_with_models_{tone}_20260506_192420.png`** — the Inglehart-Welzel country map (88 dots) with LLM positions overlaid as stars. One PNG per tone. Reads directly: *where does the model land relative to human cultures?*
- **`.txt`** — same data as text: coordinates, closest countries, closest cultural zone.

### `outputs/prompt_sensitivity/`

All from prompt-sensitivity flat CSVs (temperature 0, 30 variants per model).

- **`refusal_rates_20260704_023005.png`** — heatmap of parse-failure rates per (model, question). This is where refusal patterns are visible.
- **`flip_rate_heatmap_20260704_023005.png`** — % of variants disagreeing with the model's modal response. High values = the model gives different answers to semantically equivalent prompts.
- **`tone_comparison_20260704_023005.png`** — how mean coordinates shift by tone (standard vs friendly vs combative).
- **`tone_interaction_20260704_023005.png`** — model × tone heatmap of position magnitude.
- **`prompt_sensitivity_cultural_map_20260704_023005.png`** — all 30 variants projected as point clouds per model. Large cloud = position not robust to rephrasing.

### `outputs/stochastic/`

All from stochastic flat CSVs (temperature 1.0, 10 seeds, standard tone, variant 0).

- **`distributions_{model}_standard_v0_20260704_023051.png`** — one per model. Per-question histograms across the 10 seeds. Shows sampling noise directly.
- **`stability_summary_20260704_023051.png`** — std / coefficient-of-variation / entropy per (model, question). Compact reliability overview.
- **`stochastic_cultural_map_20260704_023133.png`** — 10-seed point clouds per model on the map. Analogous to the prompt-sensitivity map, but noise source is sampling not wording.

### `outputs/variance_decomposition/`

The core NSR outputs.

- **`variance_decomposition_20260704_022449.csv`** — master table. 120 rows (12 models × 10 questions), columns for each σ, n counts, and NSR. **Every downstream summary derives from this.**
- **`variance_bars_20260704_022449.png`** — per-model panel, one bar per question showing σ_seed (blue) + σ_prompt (orange) stacked, with σ_culture as a horizontal reference line. When the bar towers over the line, that pair is NSR > 1.
- **`noise_signal_heatmap_20260704_022449.png`** — 12 × 10 grid of NSR values. Green = signal exceeds noise, red = noise exceeds signal. Grey-hatched cells = undefined (n < 2 valid samples, e.g., Llama F118/F120).
- **`combined_map_20260704_022449.png`** — side-by-side cultural maps: left panel shows seed clouds, right panel shows prompt clouds. Reveals which noise source is dominant per model.

### `outputs/tables/` (paper-ready CSVs)

- **`nsr_overall_20260704.csv`** — one row: totals, median, %>1, %>2
- **`nsr_per_model_20260704.csv`** — 12 rows, sortable summary per model
- **`nsr_per_question_20260704.csv`** — 10 rows, per question
- **`nsr_top10_pairs_20260704.csv`** — worst 10 pairs (paper Table 7)
- **`nsr_full_120pairs_20260704.csv`** — the full model × question grid, for appendix
- **`llama_detail_20260704.csv`** — Llama's row with F118/F120 marked undefined
- **`baseline_coords_all_tones_20260704.csv`** — 11 models × 3 tones
- **`tone_max_displacement_20260704.csv`** — per-model max Euclidean shift across tones
- **`parse_rates_prompt_sensitivity_20260704.csv`** — model × question refusal-rate matrix

## 6. Data included / excluded

### Included

**All 12 models** were queried across every part of the pipeline (baseline, prompt sensitivity, stochastic):

| Region | Models |
|---|---|
| Western (US) | GPT-4o, GPT-4o-mini, Claude Sonnet 4.6, Gemma2-2B, Phi3-Mini, Llama3.1-8B |
| Western (EU) | Mistral-7B |
| East Asian | Qwen2.5-{1.5B, 3B, 7B}, Yi-6B |
| MENA | AceGPT-7B |

**10 IVS questions** — the Inglehart-Welzel canonical set (A008, A165, E018, E025, F063, F118, F120, G006, Y002, Y003).

**Ground truth: 88 IVS countries** — stricter than Tao et al.'s 107, driven by our all-10-items completeness criterion.

### Excluded / handled specially

**Llama3.1-8B refuses categorically on F118 (justif. homosexuality) and F120 (justif. abortion).**

- In the raw stochastic data: F118 has 3/10 valid responses, F120 has 0/10.
- In the raw prompt-sensitivity data: both have 0/30 valid responses.
- **Consequences:**
  - Llama does not appear in `baseline_models_{tone}_*.csv` — the baseline projection requires all 10 items, so Llama has no (x, y) coordinate. Absent from Table 2, Table 4, and the baseline PNGs.
  - Llama appears in the prompt-sensitivity and stochastic cultural maps with **hollow markers**: refused items contribute 0 to the standardized PCA score (equivalent to imputing the IVS mean). The `(0/30)` label on Llama's centroid signals every variant used at least one imputed slot.
  - Llama's Table 5 NSR summary is computed over **8 valid items**, not 10 — F118 and F120 show as NaN in the variance decomposition CSV and as grey-hatched cells in the noise-signal heatmap.

**AceGPT-7B has partial refusal on G006** (parse rate 0% in prompt sensitivity for standard tone). Its variance_decomposition row for G006 also carries NaN, so its Table 5 summary aggregates over 9 items.

**API models: no true σ_seed.** OpenAI exposes a seed parameter, so GPT-4o and GPT-4o-mini get real 10-seed variance. Anthropic does not, so Claude Sonnet 4.6's $\sigma_\text{seed} = 0$ on every item. Its NSR is therefore a **lower bound**; the paper reports this in Methods and Limitations.

**σ_seed for API models is only measured under the standard tone with variant 0**, not the full 3 × 10 grid — otherwise the API cost would triple. Open-weight σ_seed is under the same condition for consistency.

### Not measured (deliberate scope limits)

- **Only English prompts.** Whether models behave differently in their target language is a separate question (see §6.4 Limitations in the paper).
- **Only 10 IVS items.** The Inglehart-Welzel projection is defined on this exact set; broader value coverage would need a different instrument.
- **No live-country calibration.** The map is built from IVS country-year data 2005–2022; there's no attempt to project models onto contemporary human samples.

## 7. Which prompt wording is in force

`src/prompts.py` is the source of truth. F063 uses the IVS-native wording *"How important is God in your life?"*. An earlier version used a paraphrase (*"role of God"*) to work around AceGPT refusals; that was reverted May 5. **All CSVs under the 20260704 timestamp use the IVS-native wording.** Any results in outputs older than that predate the fix and should not be mixed with current data.

## 8. Provenance summary

Single source of truth per pipeline part, all consistent as of July 4:

| Part | File | Timestamp |
|---|---|---|
| Baseline | `data/results/baseline/baseline_models_{tone}_all_20260506_192420.csv` | 20260506 |
| Stochastic (open) | `data/results/stochastic/stochastic_flat_open_20260704_003513.csv` | 20260704 |
| Stochastic (API) | `data/results/stochastic/stochastic_flat_all_20260506_175429.csv` | 20260506 |
| Prompt sensitivity (open) | `data/results/prompt_sensitivity/prompt_sensitivity_flat_open_20260704_012744.csv` | 20260704 |
| Prompt sensitivity (API) | `data/results/prompt_sensitivity/prompt_sensitivity_flat_api_20260704_003513.csv` | 20260704 |
| Variance decomposition | `outputs/variance_decomposition/variance_decomposition_20260704_022449.csv` | 20260704 |

Older files with matching names have been moved to `data/results/_archive/` and `outputs/_archive/` — kept for auditability, out of the way of the current glob-based pipeline.

**Known coefficient discrepancy.** The map-rescaling constants in `src/cultural_map.py` (`+0.38`, `-0.01`) are the values inherited from Tao et al., not the correct WVS constants (`+0.038`, `-0.1` — see the [official SPSS syntax](https://worldvaluessurvey.org/WVSContents.jsp?CMSID=tradrat)). Flagged by Matt Hall (Equinor), September 2026; see the README's "Known issue" section for why it's left uncorrected here. Every coordinate below is on the uncorrected scale, consistently — relative comparisons (distances, clustering, NSR) are unaffected, but the "0.38 map units from the PNAS GPT-4o reference" figure in 9.1 below carries the same uncertainty as that external reference point.

## 9. Takeaways — what the numbers say

### 9.1 Western bias replicates, but its explanation shifts

Every model that produces a projectable coordinate lands in the self-expressive half of the cultural map ($x > 0$). The panel mean sits at $(2.98, 0.58)$ — 0.38 map units from the PNAS GPT-4o reference $(3.35, 0.50)$. This holds across institutional origin (US, EU, China, Saudi–Chinese), training-language target (English, Chinese, Arabic), and parameter scale (1.5B → API scale). AceGPT-7B, explicitly post-trained on Arabic corpora, lands at $(0.52, 1.52)$ with closest neighbors in East Asia and Eastern Europe — nowhere near any MENA country. The Qwen2.5 family (Chinese-developed, Chinese-trained) all sit firmly in self-expression territory.

**What this means:** Tao et al.'s "training-data composition drives Western bias" hypothesis doesn't survive the counterfactual. Targeted training does *not* predictably relocate models toward their target culture. The bias more likely reflects the survey instrument as administered to LLMs — a 10-question English-language PCA projection of human survey responses — than the training corpus per se.

### 9.2 Instrument noise exceeds cultural signal on ~40% of pairs

Overall NSR distribution: median **0.87**, and **49 of 117 valid pairs (42%)** exceed the validity threshold of 1.0. Worst pair: AceGPT-7B × F120 (Justif. Abortion) at **NSR = 5.56**.

**What this means:** the paper's headline. For a substantial fraction of model-question pairs, the coordinate you'd draw on the cultural map has more noise than the signal it's supposed to encode. Any downstream claim ("model X is closer to culture Y than culture Z") on those pairs is uninterpretable without demonstrated validity.

### 9.3 Instrument wording is load-bearing

Under an earlier F063 phrasing (*"role of God"*), F063 was the highest-NSR question in the panel (median 1.45, max 7.31). Reverting to the IVS-native wording (*"How important is God"*) collapsed F063's median NSR to 0.95 (rank 6 of 10) — and F120 (Justif. Abortion) took its place as the least reliable item.

**What this means:** the top-line "NSR = 7.31" number in the previous draft was partly an artifact of prompt paraphrasing. The genuine instrument (IVS-native wording) is more reliable on F063 than a "workaround" phrase, but exposes AceGPT's F120 refusal-driven collapse. This is an argument *for* fidelity to the target instrument — paraphrases that reduce refusals can hide the very failure modes worth measuring.

### 9.4 Refusal is a distinct failure mode from noise

Llama3.1-8B refuses categorically on F118 and F120 (0/30 valid prompt-sensitivity responses). AceGPT refuses fully on G006 (0/30). Under the old-buggy pipeline this was silently coded as $\sigma = 0$, i.e., "perfectly reliable." The fix marks these as NaN — undefined, no measurement possible.

**What this means:** a model that refuses is not producing a "quiet, stable answer." It's producing no answer at all. Two failure modes we distinguish:
- **Noisy answers** (Yi-6B on Y003, Qwen2.5-1.5B on F063): the model answers, but its answer swings around under seed / prompt variation. NSR captures this.
- **Refusal** (Llama on F118/F120, AceGPT on G006): the model won't answer. NSR is undefined; the coordinate cannot exist.

The refusal pattern is **ideologically targeted** — Llama refuses on the two most value-laden items in the panel (justifiability of homosexuality and abortion), not on happiness or authority. That is safety training screening off exactly the questions on which the cultural map's discriminating power depends.

### 9.5 Panel-level reliability can mask within-model instability

F118 has the second-lowest panel-level NSR (median 0.57) — most models return deterministic ceiling responses (10 = fully justifiable). But Claude Sonnet 4.6 alone produces sign-flips on 25 of 29 prompt variants on this item (86%). The low panel-level NSR reflects that the modal model defaults to the same value, not that the response is stable to rephrasings within any given model.

**What this means:** low cross-model NSR does not imply within-model robustness. Two distinct properties. A paper reporting only panel-level NSR would miss Claude's F118 instability entirely. Any future replication should report both.

### 9.6 The API / open-weight dichotomy is not a clean predictor

Median NSR: API models 0.46 (GPT-4o-mini 0.00, Claude 0.46, GPT-4o 1.14), open-weight 1.05. But the API category contains both the most reliable model in the panel (GPT-4o-mini) and one of the least reliable (GPT-4o at 60% pairs > 1). Meanwhile Qwen2.5-3B and Qwen2.5-7B (both open-weight) score better than GPT-4o.

**What this means:** "API vs open-weight" is not the relevant axis for reliability. Frontier scale doesn't guarantee measurement validity — GPT-4o's cultural-map coordinate is more uncertain than several sub-10B open-weight models. Model choice for cultural probing should be driven by per-item NSR, not access tier.

### 9.7 Scaling within a family is non-monotonic

The three Qwen2.5 models span 1.5B → 3B → 7B parameters. Their standard-tone coordinates: $(3.20, 4.43)$, $(1.81, 3.96)$, $(3.38, -0.60)$. Their median NSRs: 1.54, 0.00, 0.25. Neither position nor reliability moves monotonically with scale.

**What this means:** "bigger models are more Western" or "bigger models are more reliable" — both narratives fail within a controlled family. Sample-size claims about scaling behavior on cultural probes need many more model families before they can be trusted.

### 9.8 Bottom line

The four claims that survive the analysis:
1. **Western bias replicates** at the panel level, but is instrument-dependent, not training-data-dependent.
2. **~40% of model-question pairs have NSR > 1** — instrument noise exceeds cultural signal.
3. **Refusal and noise are distinct failure modes**, and refusal is ideologically targeted at the highest-signal items.
4. **Panel-level reliability and within-model reliability are separate properties** and can dissociate sharply (Claude × F118).

The claim that does *not* survive is any cultural attribution that treats an $(x, y)$ coordinate as face-valid without first demonstrating that its underlying items pass the NSR threshold.
