"""Evaluate results offline using reference-answer matching (no LLM judge needed)."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def parse_answer_list(raw):
    if pd.isna(raw) or not raw:
        return []
    try:
        return ast.literal_eval(raw)
    except Exception:
        return [s.strip() for s in str(raw).split(";") if s.strip()]


def normalize(text):
    return str(text).strip().lower().rstrip(".")


def judge_row(row):
    output = normalize(row["output"])
    best = normalize(row["best_answer"])
    correct_list = parse_answer_list(row.get("correct_answers", ""))
    incorrect_list = parse_answer_list(row.get("incorrect_answers", ""))

    if not output or output in ("i'm not sure", "i'm not sure", "i don't know"):
        has_correct = len(correct_list) > 0 or bool(best)
        return ("unfaithful" if has_correct else "faithful",
                "incorrect" if has_correct else "correct")

    for inc in incorrect_list:
        if normalize(inc) in output or output in normalize(inc):
            return "unfaithful", "incorrect"

    for cor in correct_list:
        if normalize(cor) in output or output in normalize(cor):
            return "faithful", "correct"

    if best in output or output in best:
        return "faithful", "correct"

    return "unfaithful", "incorrect"


def bootstrap_ci(values, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    stats_list = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    return float(np.mean(stats_list)), float(np.percentile(stats_list, 2.5)), float(np.percentile(stats_list, 97.5))


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


def output_consistency(group):
    texts = group.tolist()
    n = len(texts)
    if n < 2:
        return 1.0
    agree = sum(1 for i in range(n) for j in range(i + 1, n) if texts[i] == texts[j])
    return agree / (n * (n - 1) / 2)


def main():
    r2_path = DATA_DIR / "round2_results.csv"
    df = pd.read_csv(r2_path)
    print(f"Loaded {len(df)} rows from {r2_path}")

    results = df.apply(judge_row, axis=1, result_type="expand")
    df["faithfulness"] = results[0]
    df["correctness"] = results[1]
    df["hallucinated"] = (df["faithfulness"] == "unfaithful").astype(int)
    df["accurate"] = (df["correctness"] == "correct").astype(int)

    scored_path = DATA_DIR / "evaluated_results_round2.csv"
    df.to_csv(scored_path, index=False)
    print(f"Scored results saved to {scored_path}")

    metrics = {}

    hr = df["hallucinated"].mean()
    hr_mean, hr_lo, hr_hi = bootstrap_ci(df["hallucinated"].values)
    metrics["hallucination_rate"] = {"value": float(hr), "ci_95": [hr_lo, hr_hi]}

    acc = df["accurate"].mean()
    acc_mean, acc_lo, acc_hi = bootstrap_ci(df["accurate"].values)
    metrics["accuracy"] = {"value": float(acc), "ci_95": [acc_lo, acc_hi]}

    group_cols = ["item_id", "prompt_type", "template", "model"]
    cons_series = df.groupby(group_cols)["output"].apply(output_consistency)
    cons_mean = float(cons_series.mean())
    metrics["consistency"] = {"value": cons_mean, "ci_95": [cons_mean, cons_mean]}

    for pt in df["prompt_type"].unique():
        sub = df[df["prompt_type"] == pt]
        metrics[f"hallucination_rate_{pt}"] = float(sub["hallucinated"].mean())
        metrics[f"accuracy_{pt}"] = float(sub["accurate"].mean())

    for m in df["model"].unique():
        sub = df[df["model"] == m]
        metrics[f"hallucination_rate_{m}"] = float(sub["hallucinated"].mean())
        metrics[f"accuracy_{m}"] = float(sub["accurate"].mean())

    for t in df["template"].unique():
        sub = df[df["template"] == t]
        metrics[f"hallucination_rate_{t}"] = float(sub["hallucinated"].mean())
        metrics[f"accuracy_{t}"] = float(sub["accurate"].mean())

    templates = df["template"].unique()
    if len(templates) > 1:
        ps = {t: float(df[df["template"] == t]["hallucinated"].mean()) for t in templates}
        vals = list(ps.values())
        metrics["prompt_sensitivity_hr"] = {"per_template": ps, "max_delta": float(max(vals) - min(vals))}

    metrics_path = DATA_DIR / "metrics_round2.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print("\n=== Paired Comparisons (McNemar) ===")
    comparisons = {}
    groups = df.groupby(["template", "model"])
    group_keys = list(groups.groups.keys())
    for i in range(len(group_keys)):
        for j in range(i + 1, len(group_keys)):
            ga = groups.get_group(group_keys[i])
            gb = groups.get_group(group_keys[j])
            shared = set(ga["item_id"]) & set(gb["item_id"])
            if len(shared) < 5:
                continue
            ga_s = ga[ga["item_id"].isin(shared)].sort_values("item_id")
            gb_s = gb[gb["item_id"].isin(shared)].sort_values("item_id")
            chi2, p = mcnemar_test(ga_s["correctness"].tolist(), gb_s["correctness"].tolist())
            key = f"{group_keys[i]} vs {group_keys[j]}"
            comparisons[key] = {"chi2": chi2, "p_value": p, "n_shared": len(shared)}
            print(f"  {key}: chi2={chi2:.3f}, p={p:.4f}")

    comp_path = DATA_DIR / "paired_comparisons.json"
    with open(comp_path, "w") as f:
        json.dump(comparisons, f, indent=2)
    print(f"Paired comparisons saved to {comp_path}")


if __name__ == "__main__":
    main()
