#!/bin/bash
#------------------------------------------------------------
# TTU HPCC — RedRaider Matador GPU partition
# Runs full hallucination experiment: generate + evaluate + plots
#
# Submit with:  sbatch hpcc_job.sh
# Monitor with: squeue -u $USER
# Output log:   slurm-<jobid>.out
#------------------------------------------------------------

#SBATCH --job-name=hallucination-study
#SBATCH --partition=matador
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emabuyak@ttu.edu

echo "======================================================"
echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "======================================================"

# ── Paths ──────────────────────────────────────────────────
PROJECT_DIR="$HOME/llm-hallucination-phoenix-main"   # adjust if needed
VENV_DIR="$PROJECT_DIR/.venv"
OLLAMA_BIN="$HOME/.ollama/bin/ollama"                # default Ollama install path
OLLAMA_MODELS_DIR="$HOME/.ollama/models"
OLLAMA_HOST="127.0.0.1"
OLLAMA_PORT="11434"

# ── Activate virtual environment ───────────────────────────
echo "[1/6] Activating Python virtual environment..."
source "$VENV_DIR/bin/activate"
python --version

# ── Start Ollama server ────────────────────────────────────
echo "[2/6] Starting Ollama server on port $OLLAMA_PORT..."
export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"
export OLLAMA_HOST="${OLLAMA_HOST}:${OLLAMA_PORT}"
"$OLLAMA_BIN" serve &
OLLAMA_PID=$!
echo "  Ollama PID: $OLLAMA_PID"

# Wait for Ollama to be ready
OLLAMA_BASE_URL="http://${OLLAMA_HOST}"
echo "  Waiting for Ollama to be ready at $OLLAMA_BASE_URL..."
for i in $(seq 1 30); do
    if curl -s "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; then
        echo "  Ollama is ready."
        break
    fi
    sleep 2
done

# ── Pull models ────────────────────────────────────────────
echo "[3/6] Pulling models (this may take a while on first run)..."
"$OLLAMA_BIN" pull phi3:mini
"$OLLAMA_BIN" pull mistral:7b
"$OLLAMA_BIN" pull llama3:70b

echo "  Models ready."
"$OLLAMA_BIN" list

# ── Install/update Python dependencies ────────────────────
echo "[4/6] Installing Python dependencies..."
pip install -q -r "$PROJECT_DIR/requirements.txt"

cd "$PROJECT_DIR"

# ── Run generation ─────────────────────────────────────────
echo "[5/6] Running experiment generation (all 817 questions)..."
echo "  Started: $(date)"
python src/run_experiment.py
echo "  Generation done: $(date)"

# ── Run evaluation ─────────────────────────────────────────
echo "[6/6] Running judge evaluation and metrics..."
echo "  Started: $(date)"
python src/evaluate_metrics.py
echo "  Evaluation done: $(date)"

# ── Generate plots ─────────────────────────────────────────
echo "[7/7] Generating plots..."
python src/generate_plots.py

# ── Cleanup ────────────────────────────────────────────────
echo "Stopping Ollama server (PID $OLLAMA_PID)..."
kill "$OLLAMA_PID" 2>/dev/null

echo "======================================================"
echo "Job finished: $(date)"
echo "Results in: $PROJECT_DIR/data/"
echo "Plots:   $(ls $PROJECT_DIR/data/*.png 2>/dev/null | wc -l) PNG files"
echo "Metrics: $PROJECT_DIR/data/metrics.json"
echo "======================================================"
