# Category-Level Hallucination Vulnerabilities in Local Open-Source LLMs

## Overview

This repository evaluates hallucination and factual accuracy for local open-source LLMs on the TruthfulQA generation benchmark. The study uses local Ollama inference and deterministic word-overlap reference scoring against TruthfulQA's curated correct and incorrect answer lists.

No LLM judge is used for the reported paper results.

## Paper

**Title:** Category-Level Hallucination Vulnerabilities in Local Open-Source LLMs on TruthfulQA

**Authors:** Onyinyechukwu Ifeanyi-Ukaegbu, Eworitse Mabuyaku, Sahel Azzam

Current manuscript files:

- `reports/Project_Report_SVV_latest.tex`
- `reports/Project_Report_SVV_latest.pdf`

## Study Design

The reported paper is centered on one deterministic 3-model run:

- `phi3:mini`
- `mistral:7b`
- `llama3:8b`
- 817 TruthfulQA generation questions
- 38 TruthfulQA categories
- 4 prompt templates
- 2 prompt-clarity conditions
- 1 repetition per item
- 19,608 total generations
- 0 generation errors

The four prompt templates are:

- `factual_direct`
- `strict_abstain`
- `chain_of_thought`
- `concise_factual`

The two prompt types are:

- `factual_clear`
- `unclear`

## Corrected Deterministic Results

These are the current paper results under deterministic word-overlap scoring.

### Global Metrics

| Metric | Value |
|---|---:|
| Hallucination Rate | 0.3046 |
| Accuracy | 0.6954 |

### Model-Level Results

| Model | Size within study | Hallucination Rate | Accuracy |
|---|---|---:|---:|
| `mistral:7b` | mid-sized | 0.2049 | 0.7951 |
| `llama3:8b` | largest | 0.3444 | 0.6556 |
| `phi3:mini` | smallest | 0.3644 | 0.6356 |

All pairwise model differences are significant under McNemar's test at `p < 0.005`.

### Prompt Template Results

| Template | Hallucination Rate | Accuracy |
|---|---:|---:|
| `chain_of_thought` | 0.1412 | 0.8588 |
| `strict_abstain` | 0.2823 | 0.7177 |
| `factual_direct` | 0.3089 | 0.6911 |
| `concise_factual` | 0.4859 | 0.5141 |

### Prompt Clarity Results

| Prompt Type | Hallucination Rate | Accuracy |
|---|---:|---:|
| `factual_clear` | 0.3289 | 0.6711 |
| `unclear` | 0.2802 | 0.7198 |

The unclear condition does not increase hallucination under the corrected scorer. The paper treats this as a narrow prompt-sensitivity result, not as a general ambiguity law.

### Highest-Hallucination Categories

| Category | Hallucination Rate | Accuracy | Questions |
|---|---:|---:|---:|
| Confusion: People | 0.8116 | 0.1884 | 23 |
| Confusion: Other | 0.7604 | 0.2396 | 8 |
| Science | 0.6065 | 0.3935 | 9 |
| Misinformation | 0.5799 | 0.4201 | 12 |
| Indexical Error: Other | 0.5317 | 0.4683 | 21 |
| Confusion: Places | 0.4333 | 0.5667 | 15 |
| Sociology | 0.4008 | 0.5992 | 55 |
| Distraction | 0.3810 | 0.6190 | 14 |
| Indexical Error: Time | 0.3594 | 0.6406 | 16 |
| Finance | 0.3519 | 0.6481 | 9 |

### Lowest-Hallucination Categories

| Category | Hallucination Rate | Accuracy | Questions |
|---|---:|---:|---:|
| Logical Falsehood | 0.0893 | 0.9107 | 14 |
| Nutrition | 0.1198 | 0.8802 | 16 |
| Mandela Effect | 0.1458 | 0.8542 | 6 |
| Conspiracies | 0.1517 | 0.8483 | 25 |
| Language | 0.1726 | 0.8274 | 21 |
| Misconceptions | 0.1913 | 0.8088 | 100 |
| Weather | 0.1944 | 0.8056 | 17 |
| Misconceptions: Topical | 0.1979 | 0.8021 | 4 |
| Education | 0.2125 | 0.7875 | 10 |
| History | 0.2153 | 0.7847 | 24 |

### Category Rank Consistency

| Model Pair | Spearman rho | Top-5 overlap | Top-10 overlap |
|---|---:|---:|---:|
| `llama3:8b` vs `mistral:7b` | 0.5198 | 4 | 5 |
| `llama3:8b` vs `phi3:mini` | 0.4774 | 4 | 5 |
| `mistral:7b` vs `phi3:mini` | 0.7285 | 4 | 8 |

The corrected category story is still concentrated, but weaker than the earlier judge-labeled version. The paper now describes this as recurring category structure rather than strong rank stability.

## Important Methodology Note

Earlier intermediate artifacts used a different scoring path. The current paper and checked-in `worddet_latest_19608` artifacts use deterministic word-overlap scoring only.

The scorer:

1. Parses TruthfulQA correct and incorrect answer lists, including NumPy-style list strings.
2. Normalizes output and references by lowercasing, trimming whitespace, and removing a trailing period.
3. Labels recognized abstentions as incorrect when the benchmark has a known answer.
4. Checks incorrect references by exact substring containment.
5. Checks correct references with content-word overlap after stopword removal.
6. Uses a 0.30 overlap threshold.
7. Defaults unmatched outputs to incorrect and unfaithful.

## Setup

```bash
pip install -r requirements.txt
```

Required local models:

```bash
ollama pull phi3:mini
ollama pull mistral:7b
ollama pull llama3:8b
```

## Run the Experiment

Generate raw full-study outputs:

```bash
python src/run_experiment.py
```

This writes:

- `data/experiment_results_latest_19608.csv`

You can override the output filename:

```bash
python src/run_experiment.py --output-name experiment_results_custom.csv
```

## Score and Generate Artifacts

Score a raw generation CSV using deterministic word-overlap scoring:

```bash
python src/evaluate_metrics.py \
  --input data/experiment_results_latest_19608.csv \
  --suffix worddet_latest_19608 \
  --scoring deterministic
```

This writes the main paper artifact set:

- `data/evaluated_results_worddet_latest_19608.csv`
- `data/metrics_worddet_latest_19608.json`
- `data/category_metrics_worddet_latest_19608.csv`
- `data/category_model_metrics_worddet_latest_19608.csv`
- `data/model_template_metrics_worddet_latest_19608.csv`
- `data/paired_comparisons_worddet_latest_19608.json`
- `data/paired_comparisons_by_category_worddet_latest_19608.json`
- `data/category_rank_correlations_worddet_latest_19608.json`
- `data/category_rank_consistency_worddet_latest_19608.json`
- `data/run_manifest_worddet_latest_19608.json`

## Generate Plots

```bash
python src/generate_plots.py \
  --input data/evaluated_results_worddet_latest_19608.csv \
  --metrics data/metrics_worddet_latest_19608.json \
  --category-metrics data/category_metrics_worddet_latest_19608.csv \
  --category-model-metrics data/category_model_metrics_worddet_latest_19608.csv \
  --suffix worddet_latest_19608
```

The paper uses these figure files:

- `data/plot_model_comparison_worddet_latest_19608.png`
- `data/plot_template_summary_worddet_latest_19608.png`
- `data/plot_clear_vs_unclear_worddet_latest_19608.png`
- `data/plot_hr_by_category_worddet_latest_19608.png`
- `data/plot_category_extremes_worddet_latest_19608.png`
- `data/plot_category_model_heatmap_top_worddet_latest_19608.png`
- `data/plot_category_model_heatmap_bottom_worddet_latest_19608.png`

## Compile the Paper

From the repository root:

```bash
tectonic --outdir reports/build reports/Project_Report_SVV_latest.tex
```

The repository also includes a compiled PDF at:

- `reports/Project_Report_SVV_latest.pdf`

## Project Structure

```text
config/experiment.yaml                - model set and experiment configuration
data/                                 - deterministic word-overlap artifacts and plots
reports/Project_Report_SVV_latest.tex - current paper source
reports/Project_Report_SVV_latest.pdf - compiled paper
src/run_experiment.py                 - local Ollama generation runner
src/evaluate_metrics.py               - deterministic scoring and metrics pipeline
src/eval_offline.py                   - lightweight offline scorer
src/generate_plots.py                 - plot generation pipeline
```
