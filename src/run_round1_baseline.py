"""Round 1 baseline: single model, single prompt template, N repetitions via Ollama."""

import json
import sys
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


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def make_client(cfg):
    base_url = cfg["ollama"]["base_url"]
    return OpenAI(base_url=base_url, api_key="ollama")


def get_truthfulqa_subset(cfg):
    ds = load_dataset(cfg["dataset"]["hf_path"], cfg["dataset"]["hf_subset"], split="validation")
    df = ds.to_pandas()
    sample_size = cfg["dataset"]["round1_sample_size"]
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
    )
    return response.choices[0].message.content.strip()


def main():
    cfg = load_config()
    phoenix_cfg = cfg["phoenix"]

    px.launch_app(port=phoenix_cfg["port"])
    tracer_provider = register(project_name=phoenix_cfg["project_name"])
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    client = make_client(cfg)
    model_cfg = cfg["models"]["baseline"]
    model_name = model_cfg["model_name"]
    temp = model_cfg["temperature"]
    max_tok = model_cfg["max_tokens"]
    template_name = cfg["prompt_templates"]["baseline"]
    n_reps = cfg["evaluation"]["repetitions_per_item"]

    print(f"Loading TruthfulQA (round 1 subset: {cfg['dataset']['round1_sample_size']})...")
    df = get_truthfulqa_subset(cfg)

    csv_path = DATA_DIR / "truthfulqa_subset.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset subset to {csv_path}")

    results = []
    total = len(df) * 2 * n_reps
    print(f"Running baseline: model={model_name}, template={template_name}, "
          f"items={len(df)}, reps={n_reps}, total_calls={total}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Items"):
        question = row["question"]
        best_answer = row.get("best_answer", "")
        correct_answers = row.get("correct_answers", [])
        incorrect_answers = row.get("incorrect_answers", [])
        category = row.get("category", "")

        for prompt_type in ["factual_clear", "unclear"]:
            q = question if prompt_type == "factual_clear" else make_unclear_variant(question)
            prompt_text = render(template_name, q)

            for rep in range(n_reps):
                try:
                    output = run_generation(client, model_name, temp, max_tok, prompt_text)
                except Exception as e:
                    output = f"[ERROR] {e}"
                    print(f"  Error on item {row['item_id']} rep {rep}: {e}")

                results.append({
                    "item_id": row["item_id"],
                    "question": question,
                    "prompt_type": prompt_type,
                    "prompt_text": prompt_text,
                    "template": template_name,
                    "model": model_name,
                    "repetition": rep,
                    "output": output,
                    "best_answer": best_answer,
                    "correct_answers": json.dumps(correct_answers) if isinstance(correct_answers, list) else str(correct_answers),
                    "incorrect_answers": json.dumps(incorrect_answers) if isinstance(incorrect_answers, list) else str(incorrect_answers),
                    "category": category,
                })

    results_df = pd.DataFrame(results)
    out_path = DATA_DIR / "round1_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nDone. {len(results_df)} rows saved to {out_path}")
    print(f"Phoenix UI: http://localhost:{phoenix_cfg['port']}")


if __name__ == "__main__":
    main()
