"""Generate publication plots from evaluated rows or summary metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="",
        help="Optional evaluated results CSV. Leave empty for summary-backed plotting.",
    )
    parser.add_argument(
        "--metrics",
        default=str(DATA_DIR / "metrics.json"),
        help="Metrics JSON path.",
    )
    parser.add_argument(
        "--category-metrics",
        default=str(DATA_DIR / "category_metrics.csv"),
        help="Category metrics CSV path.",
    )
    parser.add_argument(
        "--category-model-metrics",
        default=str(DATA_DIR / "category_model_metrics.csv"),
        help="Category x model metrics CSV path.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix appended to output filenames, e.g. latest_19608.",
    )
    return parser.parse_args()


def normalize_suffix(raw_suffix: str) -> str:
    if not raw_suffix:
        return ""
    return raw_suffix if raw_suffix.startswith("_") else f"_{raw_suffix}"


def plot_path(name: str, suffix: str) -> Path:
    return DATA_DIR / f"{name}{suffix}.png"


def load_optional_csv(path_str):
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_optional_json(path_str):
    path = Path(path_str)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_current(fig, name: str, suffix: str):
    out_path = plot_path(name, suffix)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path.name}")


def category_frame_from_df(df):
    return (
        df.groupby("category")
        .agg(
            hallucination_rate=("hallucinated", "mean"),
            accuracy=("accurate", "mean"),
            questions=("item_id", "nunique"),
        )
        .reset_index()
        .sort_values(["hallucination_rate", "category"], ascending=[False, True])
    )


def category_model_frame_from_df(df):
    return (
        df.groupby(["category", "model"])
        .agg(
            hallucination_rate=("hallucinated", "mean"),
            accuracy=("accurate", "mean"),
        )
        .reset_index()
    )


def category_frame_from_metrics(metrics):
    by_category = metrics.get("by_category", {})
    if not by_category:
        return None
    rows = [{"category": category, **values} for category, values in by_category.items()]
    return pd.DataFrame(rows).sort_values(
        ["hallucination_rate", "category"], ascending=[False, True]
    )


def plot_hr_by_category(category_df, suffix: str):
    if category_df is None or category_df.empty:
        return
    title = (
        "Hallucination Rate by TruthfulQA Category"
        if len(category_df) >= 20
        else "Reported Hallucination Rate by Category (Available Extremes)"
    )
    fig, ax = plt.subplots(figsize=(14, max(5, len(category_df) * 0.35)))
    ordered = category_df.sort_values("hallucination_rate", ascending=True)
    ax.barh(ordered["category"], ordered["hallucination_rate"], color="coral", edgecolor="white")
    ax.set_xlabel("Hallucination Rate")
    ax.set_ylabel("Category")
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=14, fontweight="bold")
    save_current(fig, "plot_hr_by_category", suffix)


def plot_category_extremes(category_df, suffix: str):
    if category_df is None or category_df.empty:
        return

    ordered = category_df.sort_values("hallucination_rate", ascending=False).reset_index(drop=True)
    top_n = min(10, len(ordered))
    bottom_n = min(10, len(ordered) - top_n) if len(ordered) > top_n else min(5, len(ordered))

    top_df = ordered.head(top_n).sort_values("hallucination_rate", ascending=True)
    bottom_df = ordered.tail(bottom_n).sort_values("hallucination_rate", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4.5, max(top_n, bottom_n) * 0.45)))
    axes[0].barh(top_df["category"], top_df["hallucination_rate"], color="coral", edgecolor="white")
    axes[0].set_title(f"Top {top_n} vulnerable categories")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Hallucination Rate")

    axes[1].barh(bottom_df["category"], bottom_df["hallucination_rate"], color="seagreen", edgecolor="white")
    axes[1].set_title(f"Lowest {bottom_n} available categories")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Hallucination Rate")
    axes[1].set_ylabel("")

    fig.suptitle("Category Vulnerability Extremes", fontsize=13, fontweight="bold")
    save_current(fig, "plot_category_extremes", suffix)


def plot_category_model_heatmap(category_model_df, suffix: str):
    if category_model_df is None or category_model_df.empty:
        return
    pivot = category_model_df.pivot(index="category", columns="model", values="hallucination_rate")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2.5), max(8, len(pivot) * 0.35)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Hallucination Rate"},
    )
    ax.set_title("Hallucination Rate by Category x Model", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Category")
    ax.tick_params(axis="x", rotation=30)
    save_current(fig, "plot_category_model_heatmap", suffix)

    split_index = max(1, int(np.ceil(len(pivot) / 2)))
    split_frames = {
        "plot_category_model_heatmap_top": pivot.iloc[:split_index],
        "plot_category_model_heatmap_bottom": pivot.iloc[split_index:],
    }
    for name, split_pivot in split_frames.items():
        if split_pivot.empty:
            continue
        fig, ax = plt.subplots(
            figsize=(max(8, len(split_pivot.columns) * 2.5), max(6, len(split_pivot) * 0.5))
        )
        sns.heatmap(
            split_pivot,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            ax=ax,
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Hallucination Rate"},
        )
        panel_name = "Higher-risk half" if name.endswith("_top") else "Lower-risk half"
        ax.set_title(
            f"Hallucination Rate by Category x Model ({panel_name})",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("Category")
        ax.tick_params(axis="x", rotation=30)
        save_current(fig, name, suffix)


def plot_model_comparison(metrics, suffix: str):
    by_model = metrics.get("by_model", {})
    if not by_model:
        return
    records = []
    for model, values in by_model.items():
        hr = values["hallucination_rate"]
        acc = values["accuracy"]
        records.append(
            {
                "model": model,
                "hallucination_rate": hr["value"] if isinstance(hr, dict) else hr,
                "accuracy": acc["value"] if isinstance(acc, dict) else acc,
                "hr_lo": hr["ci_95"][0] if isinstance(hr, dict) and "ci_95" in hr else None,
                "hr_hi": hr["ci_95"][1] if isinstance(hr, dict) and "ci_95" in hr else None,
                "acc_lo": acc["ci_95"][0] if isinstance(acc, dict) and "ci_95" in acc else None,
                "acc_hi": acc["ci_95"][1] if isinstance(acc, dict) and "ci_95" in acc else None,
            }
        )
    model_df = pd.DataFrame(records)
    x = np.arange(len(model_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.5, 5))

    hr_yerr = None
    acc_yerr = None
    if model_df["hr_lo"].notna().all() and model_df["hr_hi"].notna().all():
        hr_yerr = [
            model_df["hallucination_rate"] - model_df["hr_lo"],
            model_df["hr_hi"] - model_df["hallucination_rate"],
        ]
    if model_df["acc_lo"].notna().all() and model_df["acc_hi"].notna().all():
        acc_yerr = [
            model_df["accuracy"] - model_df["acc_lo"],
            model_df["acc_hi"] - model_df["accuracy"],
        ]

    ax.bar(
        x - width / 2,
        model_df["hallucination_rate"],
        width,
        label="Hallucination Rate",
        color="coral",
        yerr=hr_yerr,
        capsize=5,
        edgecolor="white",
    )
    ax.bar(
        x + width / 2,
        model_df["accuracy"],
        width,
        label="Accuracy",
        color="steelblue",
        yerr=acc_yerr,
        capsize=5,
        edgecolor="white",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_df["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Model Comparison with 95% Wilson Confidence Intervals", fontsize=13, fontweight="bold")
    ax.legend()
    save_current(fig, "plot_model_comparison", suffix)


def plot_accuracy_model_template(df, metrics, suffix: str):
    if df is not None and {"model", "template", "accurate"}.issubset(df.columns):
        pivot = df.groupby(["model", "template"])["accurate"].mean().unstack(fill_value=0)
    else:
        by_model_template = metrics.get("by_model_template", {})
        if not by_model_template:
            return
        pivot = (
            pd.DataFrame(by_model_template.values())
            .pivot(index="model", columns="template", values="accuracy")
            .fillna(0)
        )

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot.bar(ax=ax, edgecolor="white")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Model and Prompt Template", fontsize=13, fontweight="bold")
    ax.legend(title="Template", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=25)
    save_current(fig, "plot_accuracy_model_template", suffix)


def plot_template_summary(metrics, suffix: str):
    by_template = metrics.get("by_template", {})
    if not by_template:
        return
    template_df = pd.DataFrame(
        [
            {
                "template": template,
                "hallucination_rate": values["hallucination_rate"],
                "accuracy": values["accuracy"],
            }
            for template, values in by_template.items()
        ]
    ).sort_values("accuracy", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(template_df["template"], template_df["hallucination_rate"], color="coral", edgecolor="white")
    axes[0].set_title("Hallucination Rate by Template")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(template_df["template"], template_df["accuracy"], color="steelblue", edgecolor="white")
    axes[1].set_title("Accuracy by Template")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("Prompt Template Summary", fontsize=13, fontweight="bold")
    save_current(fig, "plot_template_summary", suffix)


def plot_clear_vs_unclear(df, metrics, suffix: str):
    if df is not None and {"prompt_type", "hallucinated", "accurate"}.issubset(df.columns):
        comp = df.groupby("prompt_type").agg(
            hallucination_rate=("hallucinated", "mean"),
            accuracy=("accurate", "mean"),
        )
    else:
        by_prompt_type = metrics.get("by_prompt_type", {})
        if not by_prompt_type:
            return
        comp = pd.DataFrame.from_dict(by_prompt_type, orient="index")[
            ["hallucination_rate", "accuracy"]
        ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["steelblue", "coral"]
    comp["hallucination_rate"].plot.bar(ax=axes[0], color=colors)
    comp["accuracy"].plot.bar(ax=axes[1], color=colors)
    axes[0].set_title("Hallucination Rate")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.set_ylabel("Rate")
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Factual-Clear vs Unclear Prompts", fontsize=13, fontweight="bold")
    save_current(fig, "plot_clear_vs_unclear", suffix)


def plot_hr_bootstrap_ci(metrics, suffix: str):
    by_model = metrics.get("by_model", {})
    if not by_model:
        return
    rows = []
    for model, values in by_model.items():
        hr = values["hallucination_rate"]
        if not isinstance(hr, dict) or "ci_95" not in hr:
            continue
        rows.append(
            {
                "model": model,
                "mean": hr["value"],
                "lo": hr["ci_95"][0],
                "hi": hr["ci_95"][1],
            }
        )
    if not rows:
        return

    ci_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(ci_df))
    ax.bar(
        x,
        ci_df["mean"],
        yerr=[ci_df["mean"] - ci_df["lo"], ci_df["hi"] - ci_df["mean"]],
        capsize=6,
        color="steelblue",
        edgecolor="white",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(ci_df["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Hallucination Rate")
    ax.set_title("Hallucination Rate with 95% CI by Model", fontsize=13, fontweight="bold")
    save_current(fig, "plot_hr_bootstrap_ci", suffix)


def plot_size_class_comparison(df, metrics, suffix: str):
    if df is not None and "model_size_class" in df.columns and df["model_size_class"].notna().any():
        size_df = df.groupby("model_size_class").agg(
            hallucination_rate=("hallucinated", "mean"),
            accuracy=("accurate", "mean"),
        )
    else:
        by_size_class = metrics.get("by_size_class", {})
        if not by_size_class:
            return
        size_df = pd.DataFrame.from_dict(by_size_class, orient="index")[
            ["hallucination_rate", "accuracy"]
        ]
    if size_df.empty:
        return

    x = np.arange(len(size_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, size_df["hallucination_rate"], width, label="Hallucination Rate", color="coral")
    ax.bar(x + width / 2, size_df["accuracy"], width, label="Accuracy", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(size_df.index)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Hallucination Rate vs Accuracy by Model Size Class", fontsize=13, fontweight="bold")
    ax.legend()
    save_current(fig, "plot_size_class_comparison", suffix)


def plot_consistency_heatmap(df, suffix: str):
    if df is None or "hallucinated" not in df.columns:
        return

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
    ax.set_title("Mean Output Consistency by Model and Template", fontsize=13, fontweight="bold")
    save_current(fig, "plot_consistency_heatmap", suffix)


def main():
    args = parse_args()
    suffix = normalize_suffix(args.suffix)

    df = load_optional_csv(args.input)
    metrics = load_optional_json(args.metrics)
    category_df = load_optional_csv(args.category_metrics)
    category_model_df = load_optional_csv(args.category_model_metrics)

    if df is not None:
        if category_df is None:
            category_df = category_frame_from_df(df)
        if category_model_df is None:
            category_model_df = category_model_frame_from_df(df)
        print(f"Loaded {len(df)} evaluated rows from {args.input}")
    else:
        print("No evaluated row-level CSV found; generating summary-backed plots only.")
        if category_df is None:
            category_df = category_frame_from_metrics(metrics)

    plot_hr_by_category(category_df, suffix)
    plot_category_extremes(category_df, suffix)
    plot_category_model_heatmap(category_model_df, suffix)
    plot_model_comparison(metrics, suffix)
    plot_accuracy_model_template(df, metrics, suffix)
    plot_template_summary(metrics, suffix)
    plot_clear_vs_unclear(df, metrics, suffix)
    plot_hr_bootstrap_ci(metrics, suffix)
    plot_size_class_comparison(df, metrics, suffix)
    plot_consistency_heatmap(df, suffix)
    print("\nPlot generation complete.")


if __name__ == "__main__":
    main()
