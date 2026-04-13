"""Compute hallucination, accuracy, consistency metrics — with per-category breakdown."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from openai import OpenAI
from scipy import stats
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "experiment.yaml"
DATA_DIR = ROOT / "data"
NUM_WORKERS = 2

FAITHFULNESS_TEMPLATE = """Determine if the model response is faithful to the known facts.

Question: {question}
Reference answer: {best_answer}
Correct answers: {correct_answers}
Incorrect answers: {incorrect_answers}
Model response: {output}

UNFAITHFUL = contradicts correct answers or matches incorrect answers.
FAITHFUL = aligns with correct answers or appropriately abstains.

Respond with one word only: faithful or unfaithful"""

CORRECTNESS_TEMPLATE = """Determine if the model response is factually correct.

Question: {question}
Reference answer: {best_answer}
Correct answers: {correct_answers}
Model response: {output}

CORRECT = matches or is consistent with correct answers.
INCORRECT = contradicts them or provides wrong info. Abstaining when the answer is known = incorrect.

Respond with one word only: correct or incorrect"""


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def make_client(cfg):
    return OpenAI(base_url=cfg["ollama"]["base_url"], api_key="ollama")


def extract_label(raw, valid_labels):
    raw = raw.strip().lower()
    for token in raw.split():
        if token in valid_labels:
            return token
    for label in valid_labels:
        if label in raw:
            return label
    return raw


def judge_single(client, model_name, template, row_dict):
    prompt = template.format(
        question=row_dict["question"],
        best_answer=row_dict["best_answer"],
        correct_answers=row_dict["correct_answers"],
        incorrect_answers=row_dict.get("incorrect_answers", ""),
        output=row_dict["output"],
    )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
            timeout=60,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"error: {e}"


def judge_row(client, model_name, row_dict):
    f_raw = judge_single(client, model_name, FAITHFULNESS_TEMPLATE, row_dict)
    c_raw = judge_single(client, model_name, CORRECTNESS_TEMPLATE, row_dict)
    return {
        "faithfulness": extract_label(f_raw, {"faithful", "unfaithful"}),
        "correctness": extract_label(c_raw, {"correct", "incorrect"}),
    }


def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    boot_stats = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    return float(np.mean(boot_stats)), float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))


def mcnemar_test(labels_a, labels_b, positive="correct"):
    a_pos = np.array([l == positive for l in labels_a])
    b_pos = np.array([l == positive for l in labels_b])
    b_val = int(np.sum(a_pos & ~b_pos))
    c_val = int(np.sum(~a_pos & b_pos))
    if b_val + c_val == 0:
        return 0.0, 1.0
    chi2 = (abs(b_val - c_val) - 1) ** 2 / (b_val + c_val)
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return float(chi2), float(p)


def run_judge(df, cfg):
    """Judge rep=0 rows and merge labels back onto full dataframe."""
    client = make_client(cfg)
    judge_model = cfg["models"]["judge_model"]["model_name"]

    key_cols = ["item_id", "model", "template", "prompt_type"]
    judge_df = df[df["repetition"] == 0].copy().reset_index(drop=True)
    print(f"Judging {len(judge_df)} rows with {judge_model} ({NUM_WORKERS} workers)...")

    judge_dicts = judge_df.to_dict("records")
    labels = [None] * len(judge_dicts)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(judge_row, client, judge_model, d): i for i, d in enumerate(judge_dicts)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            idx = futures[future]
            labels[idx] = future.result()

    judge_df["faithfulness"] = [l["faithfulness"] for l in labels]
    judge_df["correctness"] = [l["correctness"] for l in labels]

    label_map = judge_df[key_cols + ["faithfulness", "correctness"]].drop_duplicates(subset=key_cols)
    df = df.merge(label_map, on=key_cols, how="left")
    df["hallucinated"] = (df["faithfulness"] == "unfaithful").astype(int)
    df["accurate"] = (df["correctness"] == "correct").astype(int)
    return df


def compute_aggregate_metrics(df):
    metrics = {}

    # Overall
    hr_mean, hr_lo, hr_hi = bootstrap_ci(df["hallucinated"].values)
    metrics["hallucination_rate"] = {"value": hr_mean, "ci_95": [hr_lo, hr_hi]}

    acc_mean, acc_lo, acc_hi = bootstrap_ci(df["accurate"].values)
    metrics["accuracy"] = {"value": acc_mean, "ci_95": [acc_lo, acc_hi]}

    # Per model
    metrics["by_model"] = {}
    for model, g in df.groupby("model"):
        hr_m, lo_m, hi_m = bootstrap_ci(g["hallucinated"].values)
        acc_m, a_lo, a_hi = bootstrap_ci(g["accurate"].values)
        metrics["by_model"][model] = {
            "hallucination_rate": {"value": hr_m, "ci_95": [lo_m, hi_m]},
            "accuracy": {"value": acc_m, "ci_95": [a_lo, a_hi]},
            "n": len(g),
        }

    # Per model size class
    if "model_size_class" in df.columns:
        metrics["by_size_class"] = {}
        for size, g in df.groupby("model_size_class"):
            metrics["by_size_class"][size] = {
                "hallucination_rate": float(g["hallucinated"].mean()),
                "accuracy": float(g["accurate"].mean()),
                "n": len(g),
            }

    # Per prompt template
    metrics["by_template"] = {}
    for tmpl, g in df.groupby("template"):
        metrics["by_template"][tmpl] = {
            "hallucination_rate": float(g["hallucinated"].mean()),
            "accuracy": float(g["accurate"].mean()),
        }

    # Per prompt type (clear vs unclear)
    metrics["by_prompt_type"] = {}
    for pt, g in df.groupby("prompt_type"):
        metrics["by_prompt_type"][pt] = {
            "hallucination_rate": float(g["hallucinated"].mean()),
            "accuracy": float(g["accurate"].mean()),
        }

    # Per category — the core analysis for the paper
    metrics["by_category"] = {}
    for cat, g in df.groupby("category"):
        metrics["by_category"][cat] = {
            "hallucination_rate": float(g["hallucinated"].mean()),
            "accuracy": float(g["accurate"].mean()),
            "n_questions": int(g["item_id"].nunique()),
        }

    # Per category × model — which model is most vulnerable per category
    metrics["by_category_model"] = {}
    for (cat, model), g in df.groupby(["category", "model"]):
        key = f"{cat}|{model}"
        metrics["by_category_model"][key] = {
            "category": cat,
            "model": model,
            "hallucination_rate": float(g["hallucinated"].mean()),
            "accuracy": float(g["accurate"].mean()),
            "n": len(g),
        }

    # Prompt sensitivity — max delta in HR across templates
    template_hrs = {t: float(g["hallucinated"].mean()) for t, g in df.groupby("template")}
    vals = list(template_hrs.values())
    metrics["prompt_sensitivity"] = {
        "per_template_hr": template_hrs,
        "max_delta": float(max(vals) - min(vals)),
    }

    return metrics


def run_paired_comparisons(df):
    """McNemar tests between all model pairs on shared questions."""
    comparisons = {}
    models = df["model"].unique()
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            ma, mb = models[i], models[j]
            ga = df[df["model"] == ma].sort_values("item_id")
            gb = df[df["model"] == mb].sort_values("item_id")
            shared = set(ga["item_id"]) & set(gb["item_id"])
            if len(shared) < 10:
                continue
            ga_s = ga[ga["item_id"].isin(shared)].drop_duplicates("item_id").sort_values("item_id")
            gb_s = gb[gb["item_id"].isin(shared)].drop_duplicates("item_id").sort_values("item_id")
            chi2, p = mcnemar_test(ga_s["correctness"].tolist(), gb_s["correctness"].tolist())
            key = f"{ma} vs {mb}"
            comparisons[key] = {"chi2": chi2, "p_value": p, "n_shared": len(shared)}
            print(f"  {key}: chi2={chi2:.3f}, p={p:.4f}, n={len(shared)}")
    return comparisons


def main():
    cfg = load_config()

    results_path = DATA_DIR / "experiment_results.csv"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run src/run_experiment.py first.")
        return

    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} rows from {results_path}")

    df = run_judge(df, cfg)

    scored_path = DATA_DIR / "evaluated_results.csv"
    df.to_csv(scored_path, index=False)
    print(f"Scored results saved to {scored_path}")

    print("\nComputing aggregate metrics...")
    metrics = compute_aggregate_metrics(df)

    print("\nRunning paired model comparisons (McNemar)...")
    comparisons = run_paired_comparisons(df)
    metrics["paired_comparisons"] = comparisons

    metrics_path = DATA_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Overall hallucination rate : {metrics['hallucination_rate']['value']:.3f} "
          f"(95% CI {metrics['hallucination_rate']['ci_95'][0]:.3f}–{metrics['hallucination_rate']['ci_95'][1]:.3f})")
    print(f"Overall accuracy           : {metrics['accuracy']['value']:.3f} "
          f"(95% CI {metrics['accuracy']['ci_95'][0]:.3f}–{metrics['accuracy']['ci_95'][1]:.3f})")
    print("\nBy model:")
    for m, v in metrics["by_model"].items():
        print(f"  {m}: HR={v['hallucination_rate']['value']:.3f}, Acc={v['accuracy']['value']:.3f}")
    top_cats = sorted(metrics["by_category"].items(), key=lambda x: x[1]["hallucination_rate"], reverse=True)[:5]
    print("\nTop 5 highest-hallucination categories:")
    for cat, v in top_cats:
        print(f"  {cat}: HR={v['hallucination_rate']:.3f} (n={v['n_questions']})")


if __name__ == "__main__":
    main()
