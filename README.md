# Evaluating Hallucination Detection and Reliability in LLMs Using Phoenix

## Project Personnel
Onyinyechukwu Ifeanyi-Ukaegbu, Eworitse Mabuyaku, Sahel Azzam

## Overview
This project evaluates hallucination rates, factual accuracy, and output consistency of large language models using the Arize Phoenix observability platform and TruthfulQA benchmark. All inference runs locally via Ollama, while evaluation uses deterministic reference-answer scoring over saved CSV outputs. Phoenix traces the generation pipeline; judging happens offline from the saved results.

## Prerequisites
- **Ollama** installed and running (`ollama serve`)
- For the **full 817-question study** (3 models), pull all targets:
  - `ollama pull llama3.2`
  - `ollama pull mistral`
  - `ollama pull llama3.3:70b`
- For **Round 1 / Round 2** pilots only, `llama3.2` (and `mistral` for Round 2) are enough — see `config/experiment.yaml`.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Round 1 — Baseline
```bash
python src/run_round1_baseline.py
```

### Round 2 — Expanded Matrix (2 models)
```bash
python src/run_round2_matrix.py
```
Uses `llama3.2` and `mistral` per `config/experiment.yaml` (`round2_models`).

### Full 817-Question Study (Publishable Mode)
```bash
python src/run_round2_matrix.py --mode full
```
This mode uses:
- 817 questions (TruthfulQA generation split)
- 3 models: `llama3.2`, `mistral`, `llama3.3:70b`
- 4 templates (`factual_direct`, `strict_abstain`, `chain_of_thought`, `concise_factual`) x 2 prompt types
- resumable checkpointed execution (safe to restart)

### Google Colab (optional)
Upload `colab_full_study_runner.ipynb` to Colab, enable a **GPU** runtime, then run cells top to bottom. The notebook sets `RUN_STRONG_MACHINE` to choose **3 models** (`llama3.2`, `mistral`, `llama3.3:70b`) or **2 models** for lighter hardware.

### Evaluation Metrics
```bash
python src/evaluate_metrics.py
```
This step evaluates existing results offline with deterministic non-LLM scoring.
Generated full-study files include:
- `data/evaluated_results_full.csv`
- `data/metrics_full.json`
- `data/category_metrics_full.csv`
- `data/category_why_signals_full.csv`
- `data/paired_comparisons_full.json`
- `data/paired_comparisons_by_category_full.json`

### Failure Analysis Notebook
Open `notebooks/failure_analysis.ipynb` in Jupyter.

### Plots and Publication Artifacts
```bash
python src/generate_plots.py
python src/generate_full_study_artifacts.py --suffix full
```
This generates:
- category vulnerability plots
- model/category heatmap
- deterministic "why" plots
- publication tables in `data/`
- reproducibility appendix in `reports/full_study_reproducibility_appendix.md`

## Final Full-Scale Results (latest completed run)

The metrics below come from the latest completed **3-model x 2-template** full run.
After adding two new templates (`chain_of_thought`, `concise_factual`), rerun full mode to produce the expanded **3-model x 4-template** matrix.

### Experiment Coverage
| Item | Value |
|---|---|
| Dataset | TruthfulQA generation split |
| Questions | 817 |
| Categories | 38 |
| Models | `llama3.2`, `mistral`, `llama3.3:70b` |
| Templates | `factual_direct`, `strict_abstain` (latest completed run) |
| Prompt types | `factual_clear`, `unclear` |
| Repetitions | 1 |
| Total generations | 9804 |
| Generation errors | 0 |

### Global Metrics (from `data/metrics_full.json`)
| Metric | Value | 95% CI |
|---|---:|---|
| Hallucination Rate | 0.9469 | [0.9423, 0.9512] |
| Accuracy | 0.0531 | [0.0488, 0.0577] |
| Consistency | 1.0000 | [1.0000, 1.0000] |

### Model-Level Summary
| Model | Hallucination Rate | Accuracy |
|---|---:|---:|
| `llama3.2` | 0.9614 | 0.0386 |
| `mistral` | 0.9498 | 0.0502 |
| `llama3.3:70b` | 0.9293 | 0.0707 |

### Prompt Sensitivity
| Template | Hallucination Rate |
|---|---:|
| `factual_direct` | 0.9347 |
| `strict_abstain` | 0.9590 |

- Max template delta in hallucination rate: **0.0243**

### Rerun Target After Template Expansion
| Item | Value |
|---|---|
| Templates (expanded) | `factual_direct`, `strict_abstain`, `chain_of_thought`, `concise_factual` |
| Expected generations | 19608 (817 x 3 models x 4 templates x 2 prompt types x 1 rep) |
| Command | `python src/run_round2_matrix.py --mode full --disable-phoenix` |

### Top 10 Most Vulnerable Categories (from `data/table_top10_vulnerable_categories_full.csv`)
| Category | Hallucination Rate | Misconception Rate |
|---|---:|---:|
| Politics | 1.0000 | 0.0250 |
| Nutrition | 1.0000 | 0.0104 |
| Subjective | 1.0000 | 0.0093 |
| Indexical Error: Location | 1.0000 | 0.0076 |
| Superstitions | 1.0000 | 0.0000 |
| Misinformation | 1.0000 | 0.0000 |
| Education | 1.0000 | 0.0000 |
| Finance | 1.0000 | 0.0000 |
| Science | 1.0000 | 0.0000 |
| Misconceptions: Topical | 1.0000 | 0.0000 |

### Plots (Final Full Study)
#### Hallucination by Category
![Hallucination by Category](data/plot_hr_by_category.png)

#### Top Vulnerable Categories
![Top Vulnerable Categories](data/plot_top_vulnerable_categories.png)

#### Category Failure Reasons ("Why")
![Category Why Signals](data/plot_category_why_signals.png)

#### Model x Category Heatmap
![Model Category Heatmap](data/plot_model_category_heatmap.png)

#### Accuracy by Model and Template
![Accuracy by Model Template](data/plot_accuracy_model_template.png)

#### Clear vs Unclear Prompt Comparison
![Clear vs Unclear](data/plot_clear_vs_unclear.png)

#### Consistency Heatmap
![Consistency Heatmap](data/plot_consistency_heatmap.png)

#### Hallucination Rate with Bootstrap CIs
![Hallucination CI](data/plot_hr_bootstrap_ci.png)

#### Model Comparison (Hallucination vs Accuracy)
![Model Comparison](data/plot_model_comparison.png)

### Final Artifact Locations
- Full raw generations: `data/full_study_results.csv`
- Scored full outputs: `data/evaluated_results_full.csv`
- Global metrics: `data/metrics_full.json`
- Category metrics: `data/category_metrics_full.csv`
- Category why-signals: `data/category_why_signals_full.csv`
- Global paired significance: `data/paired_comparisons_full.json`
- Category paired significance: `data/paired_comparisons_by_category_full.json`
- Publication tables:
  - `data/table_top10_vulnerable_categories_full.csv`
  - `data/table_model_category_hallucination_full.csv`
  - `data/table_top_failure_reason_by_model_category_full.csv`
- Reproducibility appendix: `reports/full_reproducibility_appendix.md`

## Project Structure
```
colab_full_study_runner.ipynb — Colab runner (Ollama + full study + export)
config/experiment.yaml     — Experiment parameters (models, templates, Ollama config)
data/                      — Dataset files and results (auto-generated)
src/prompt_templates.py    — Prompt template definitions
src/run_round1_baseline.py — Round 1 baseline runner (llama3.2)
src/run_round2_matrix.py   — Round 2 and full-study runner with resume/checkpoint support
src/reference_scoring.py   — Deterministic non-LLM scoring and misconception labels
src/evaluate_metrics.py    — Global/category metrics + significance tests
src/generate_full_study_artifacts.py — Publication tables + reproducibility appendix
notebooks/                 — Analysis notebooks
reports/                   — Generated reports
```
