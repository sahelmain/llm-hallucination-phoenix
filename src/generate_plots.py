"""Generate all plots from evaluated results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data():
    path = DATA_DIR / "evaluated_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run evaluate_metrics.py first.")
    return pd.read_csv(path)


def load_metrics():
    path = DATA_DIR / "metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Plot 1: Hallucination rate by category (sorted bar chart) ─────────────────
def plot_hr_by_category(df):
    cat_hr = df.groupby("category")["hallucinated"].mean().sort_values(ascending=False)
    overall = df["hallucinated"].mean()

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(cat_hr.index, cat_hr.values, color="coral", edgecolor="white")
    ax.axvline(overall, color="black", linestyle="--", linewidth=1.5,
               label=f"Overall mean: {overall:.2f}")
    ax.set_xlabel("Hallucination Rate", fontsize=12)
    ax.set_title("Hallucination Rate by TruthfulQA Category", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "plot_hr_by_category.png", dpi=150)
    plt.close()
    print("Saved plot_hr_by_category.png")


# ── Plot 2: Per-model per-category hallucination heatmap ──────────────────────
def plot_category_model_heatmap(df):
    pivot = df.groupby(["category", "model"])["hallucinated"].mean().unstack()
    # Sort categories by mean HR descending
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2.5), max(10, len(pivot) * 0.4)))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="YlOrRd",
        ax=ax, vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Hallucination Rate"},
    )
    ax.set_title("Hallucination Rate by Category × Model", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Category", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "plot_category_model_heatmap.png", dpi=150)
    plt.close()
    print("Saved plot_category_model_heatmap.png")


# ── Plot 3: Model comparison — HR vs Accuracy ─────────────────────────────────
def plot_model_comparison(df):
    model_stats = df.groupby("model").agg(
        hallucination_rate=("hallucinated", "mean"),
        accuracy=("accurate", "mean"),
    ).reset_index()

    x = np.arange(len(model_stats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, model_stats["hallucination_rate"], w, label="Hallucination Rate", color="coral")
    ax.bar(x + w / 2, model_stats["accuracy"], w, label="Accuracy", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(model_stats["model"], fontsize=10)
    ax.set_ylabel("Rate")
    ax.set_title("Model Comparison: Hallucination Rate vs Accuracy", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(DATA_DIR / "plot_model_comparison.png", dpi=150)
    plt.close()
    print("Saved plot_model_comparison.png")


# ── Plot 4: Accuracy by model × template ──────────────────────────────────────
def plot_accuracy_model_template(df):
    pivot = df.groupby(["model", "template"])["accurate"].mean().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot.bar(ax=ax, edgecolor="white")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Model and Prompt Template", fontsize=13, fontweight="bold")
    ax.legend(title="Template", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "plot_accuracy_model_template.png", dpi=150)
    plt.close()
    print("Saved plot_accuracy_model_template.png")


# ── Plot 5: Clear vs Unclear prompts ──────────────────────────────────────────
def plot_clear_vs_unclear(df):
    comp = df.groupby("prompt_type").agg(
        hallucination_rate=("hallucinated", "mean"),
        accuracy=("accurate", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["steelblue", "coral"]
    comp["hallucination_rate"].plot.bar(ax=axes[0], color=colors)
    axes[0].set_title("Hallucination Rate")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=15)
    comp["accuracy"].plot.bar(ax=axes[1], color=colors)
    axes[1].set_title("Accuracy")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=15)
    plt.suptitle("Factual-Clear vs Unclear Prompts", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "plot_clear_vs_unclear.png", dpi=150)
    plt.close()
    print("Saved plot_clear_vs_unclear.png")


# ── Plot 6: Hallucination rate with 95% bootstrap CI ─────────────────────────
def plot_hr_bootstrap_ci(df, metrics):
    by_model = metrics.get("by_model", {})
    if not by_model:
        return
    records = []
    for model, v in by_model.items():
        records.append({
            "model": model,
            "HR_mean": v["hallucination_rate"]["value"],
            "HR_lo": v["hallucination_rate"]["ci_95"][0],
            "HR_hi": v["hallucination_rate"]["ci_95"][1],
        })
    ci_df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(ci_df))
    ax.bar(x, ci_df["HR_mean"],
           yerr=[ci_df["HR_mean"] - ci_df["HR_lo"], ci_df["HR_hi"] - ci_df["HR_mean"]],
           capsize=6, color="steelblue", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ci_df["model"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Hallucination Rate")
    ax.set_title("Hallucination Rate with 95% Bootstrap CI by Model", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "plot_hr_bootstrap_ci.png", dpi=150)
    plt.close()
    print("Saved plot_hr_bootstrap_ci.png")


# ── Plot 7: Model size class comparison ───────────────────────────────────────
def plot_size_class_comparison(df):
    if "model_size_class" not in df.columns:
        return
    size_stats = df.groupby("model_size_class").agg(
        hallucination_rate=("hallucinated", "mean"),
        accuracy=("accurate", "mean"),
    ).reindex(["small", "medium", "large"]).dropna()

    x = np.arange(len(size_stats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, size_stats["hallucination_rate"], w, label="Hallucination Rate", color="coral")
    ax.bar(x + w / 2, size_stats["accuracy"], w, label="Accuracy", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(size_stats.index, fontsize=11)
    ax.set_ylabel("Rate")
    ax.set_title("Hallucination Rate vs Accuracy by Model Size Class", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(DATA_DIR / "plot_size_class_comparison.png", dpi=150)
    plt.close()
    print("Saved plot_size_class_comparison.png")


# ── Plot 8: Consistency heatmap ───────────────────────────────────────────────
def plot_consistency_heatmap(df):
    def pairwise_agreement(labels):
        labels = list(labels)
        n = len(labels)
        if n < 2:
            return 1.0
        agree = sum(1 for i in range(n) for j in range(i + 1, n) if labels[i] == labels[j])
        return agree / (n * (n - 1) / 2)

    cons = df.groupby(["model", "template", "item_id"])["hallucinated"].apply(pairwise_agreement)
    cons_avg = cons.groupby(["model", "template"]).mean().unstack(fill_value=1.0)
    fig, ax = plt.subplots(figsize=(10, max(4, len(cons_avg) * 0.8)))
    sns.heatmap(cons_avg, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, vmin=0, vmax=1)
    ax.set_title("Mean Output Consistency by Model & Template", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "plot_consistency_heatmap.png", dpi=150)
    plt.close()
    print("Saved plot_consistency_heatmap.png")


def main():
    df = load_data()
    metrics = load_metrics()
    print(f"Loaded {len(df)} rows, {df['model'].nunique()} models, {df['category'].nunique()} categories.")

    plot_hr_by_category(df)
    plot_category_model_heatmap(df)
    plot_model_comparison(df)
    plot_accuracy_model_template(df)
    plot_clear_vs_unclear(df)
    plot_hr_bootstrap_ci(df, metrics)
    plot_size_class_comparison(df)
    plot_consistency_heatmap(df)
    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
