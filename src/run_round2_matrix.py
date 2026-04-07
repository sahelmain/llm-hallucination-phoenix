"""Round 2: expanded model/template matrix with parallel Ollama calls."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

import phoenix as px
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_templates import render, make_unclear_variant

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "experiment.yaml"
DATA_DIR = ROOT / "data"
NUM_WORKERS = 4


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def make_client(cfg):
    return OpenAI(base_url=cfg["ollama"]["base_url"], api_key="ollama")


def get_truthfulqa(cfg):
    ds = load_dataset(cfg["dataset"]["hf_path"], cfg["dataset"]["hf_subset"], split="validation")
    df = ds.to_pandas()
    sample_size = cfg["dataset"]["round2_sample_size"]
    seed = cfg["dataset"]["seed"]
    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    df["item_id"] = df.index.astype(str)
    return df


def run_generation(client, model_name, temperature, max_tokens, prompt):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=120,
    )
    return response.choices[0].message.content.strip()


def build_tasks(df, models, template_names, n_reps):
    """Build a flat list of all (metadata_dict, prompt_text) tasks."""
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
            client, mcfg["model_name"], mcfg["temperature"], mcfg["max_tokens"], task["prompt_text"]
        )
    except Exception as e:
        output = f"[ERROR] {e}"

    correct_answers = row.get("correct_answers", [])
    incorrect_answers = row.get("incorrect_answers", [])
    return {
        "item_id": row["item_id"],
        "question": row["question"],
        "prompt_type": task["prompt_type"],
        "prompt_text": task["prompt_text"],
        "template": task["template_name"],
        "model": mcfg["model_name"],
        "repetition": task["rep"],
        "output": output,
        "best_answer": row.get("best_answer", ""),
        "correct_answers": json.dumps(correct_answers) if isinstance(correct_answers, list) else str(correct_answers),
        "incorrect_answers": json.dumps(incorrect_answers) if isinstance(incorrect_answers, list) else str(incorrect_answers),
        "category": row.get("category", ""),
    }


def main():
    cfg = load_config()
    phoenix_cfg = cfg["phoenix"]

    px.launch_app(port=phoenix_cfg["port"])
    tracer_provider = register(project_name=phoenix_cfg["project_name"])
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    client = make_client(cfg)
    models = cfg["models"]["round2_models"]
    template_names = cfg["prompt_templates"]["round2"]
    n_reps = cfg["evaluation"]["repetitions_per_item"]

    print(f"Loading TruthfulQA (round 2: {cfg['dataset']['round2_sample_size']})...")
    df = get_truthfulqa(cfg)

    tasks = build_tasks(df, models, template_names, n_reps)
    print(f"Round 2: {len(tasks)} total calls, {NUM_WORKERS} parallel workers")

    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(execute_task, client, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Round 2"):
            results.append(future.result())

    results_df = pd.DataFrame(results)
    out_path = DATA_DIR / "round2_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nDone. {len(results_df)} rows saved to {out_path}")
    print(f"Phoenix UI: http://localhost:{phoenix_cfg['port']}")


if __name__ == "__main__":
    main()
