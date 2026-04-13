"""Full experiment runner — all 817 TruthfulQA questions, all models, all templates.

Checkpoint/resume: results are flushed to disk every CHECKPOINT_EVERY completions.
If the session disconnects, re-running will skip already-completed rows automatically.
"""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_templates import render, make_unclear_variant

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "experiment.yaml"
DATA_DIR = ROOT / "data"
NUM_WORKERS = 4
CHECKPOINT_EVERY = 100  # flush to disk every N completed rows


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def make_client(cfg):
    return OpenAI(base_url=cfg["ollama"]["base_url"], api_key="ollama")


def get_truthfulqa(cfg):
    print("Loading TruthfulQA (all 817 questions)...")
    ds = load_dataset(cfg["dataset"]["hf_path"], cfg["dataset"]["hf_subset"], split="validation")
    df = ds.to_pandas()
    sample_size = cfg["dataset"]["sample_size"]
    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=cfg["dataset"]["seed"]).reset_index(drop=True)
    df["item_id"] = df.index.astype(str)
    print(f"  Loaded {len(df)} questions across {df['category'].nunique()} categories.")
    return df


def run_generation(client, model_name, temperature, max_tokens, prompt):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=180,
    )
    return response.choices[0].message.content.strip()


def build_tasks(df, models, template_names, n_reps):
    tasks = []
    for model_cfg in models:
        for template_name in template_names:
            for _, row in df.iterrows():
                question = row["question"]
                for prompt_type in ["factual_clear", "unclear"]:
                    q = question if prompt_type == "factual_clear" else make_unclear_variant(question)
                    prompt_text = render(template_name, q)
                    for rep in range(n_reps):
                        tasks.append({
                            "model_cfg": model_cfg,
                            "template_name": template_name,
                            "prompt_text": prompt_text,
                            "row": row.to_dict(),
                            "prompt_type": prompt_type,
                            "rep": rep,
                        })
    return tasks


def execute_task(client, task):
    mcfg = task["model_cfg"]
    row = task["row"]
    try:
        output = run_generation(
            client,
            mcfg["model_name"],
            mcfg["temperature"],
            mcfg["max_tokens"],
            task["prompt_text"],
        )
    except Exception as e:
        output = f"[ERROR] {e}"

    correct_answers = row.get("correct_answers", [])
    incorrect_answers = row.get("incorrect_answers", [])
    return {
        "item_id": row["item_id"],
        "question": row["question"],
        "category": row.get("category", ""),
        "prompt_type": task["prompt_type"],
        "prompt_text": task["prompt_text"],
        "template": task["template_name"],
        "model": mcfg["model_name"],
        "model_size_class": mcfg.get("size_class", ""),
        "repetition": task["rep"],
        "output": output,
        "best_answer": row.get("best_answer", ""),
        "correct_answers": json.dumps(correct_answers) if isinstance(correct_answers, list) else str(correct_answers),
        "incorrect_answers": json.dumps(incorrect_answers) if isinstance(incorrect_answers, list) else str(incorrect_answers),
    }


def load_checkpoint(out_path):
    """Return set of already-completed (model, template, item_id, prompt_type, rep) tuples."""
    if not out_path.exists():
        return set()
    try:
        existing = pd.read_csv(out_path)
        if existing.empty:
            return set()
        return set(zip(
            existing["model"],
            existing["template"],
            existing["item_id"].astype(str),
            existing["prompt_type"],
            existing["repetition"].astype(int),
        ))
    except Exception:
        return set()


def flush(buffer, out_path, write_header):
    """Append buffer contents to CSV output file. Must be called while lock is held."""
    if not buffer:
        return
    pd.DataFrame(buffer).to_csv(out_path, mode="a", header=write_header[0], index=False)
    write_header[0] = False
    buffer.clear()


def main():
    cfg = load_config()
    client = make_client(cfg)

    models = cfg["models"]["experiment_models"]
    template_names = cfg["prompt_templates"]["full_study"]
    n_reps = cfg["evaluation"]["repetitions_per_item"]

    df = get_truthfulqa(cfg)
    tasks = build_tasks(df, models, template_names, n_reps)

    out_path = DATA_DIR / "experiment_results.csv"
    done = load_checkpoint(out_path)

    pending = [
        t for t in tasks
        if (
            t["model_cfg"]["model_name"],
            t["template_name"],
            str(t["row"]["item_id"]),
            t["prompt_type"],
            t["rep"],
        ) not in done
    ]

    total_calls = len(tasks)
    skipped = len(done)
    remaining = len(pending)

    print(f"\nExperiment matrix:")
    print(f"  Questions   : {len(df)}")
    print(f"  Models      : {len(models)} ({', '.join(m['model_name'] for m in models)})")
    print(f"  Templates   : {len(template_names)}")
    print(f"  Prompt types: 2 (factual_clear, unclear)")
    print(f"  Repetitions : {n_reps}")
    print(f"  Total calls : {total_calls}")
    if skipped:
        print(f"  Resuming    : {skipped} already done, {remaining} remaining\n")
    else:
        print()

    if not pending:
        print("All tasks already complete.")
        return

    lock = threading.Lock()
    buffer = []
    write_header = [not out_path.exists() or skipped == 0]
    completed_count = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(execute_task, client, t): t for t in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            result = future.result()
            with lock:
                buffer.append(result)
                completed_count += 1
                if completed_count % CHECKPOINT_EVERY == 0:
                    flush(buffer, out_path, write_header)

    # Final flush for any remaining rows
    with lock:
        flush(buffer, out_path, write_header)

    total_written = skipped + remaining
    print(f"\nDone. {total_written} total rows in {out_path}")


if __name__ == "__main__":
    main()
