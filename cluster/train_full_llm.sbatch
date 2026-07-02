#!/usr/bin/env bash
#SBATCH --job-name=charrua-llm
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=condejuanpablo10@gmail.com

set -euo pipefail

cd /clusteruy/home/jconde/charruaLLM

source ~/miniconda3/etc/profile.d/conda.sh
conda activate charruallm

TOKENIZER="${TOKENIZER:-tokenizer.json}"
PROCESSED_DIR="${PROCESSED_DIR:-processed}"
OUT_DIR="${OUT_DIR:-runs/run3}"

MAX_STEPS="${MAX_STEPS:-32000}"

echo "TOKENIZER=$TOKENIZER"
echo "PROCESSED_DIR=$PROCESSED_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "MAX_STEPS=$MAX_STEPS"

if [[ ! -f "$TOKENIZER" ]]; then
  echo "ERROR: no existe TOKENIZER=$TOKENIZER" >&2
  exit 1
fi

if [[ ! -d "$PROCESSED_DIR" ]]; then
  echo "ERROR: no existe PROCESSED_DIR=$PROCESSED_DIR" >&2
  exit 1
fi

echo "Entrenando LLM..."
python -u train.py \
  --device cuda \
  --data-dir "$PROCESSED_DIR" \
  --out-dir "$OUT_DIR" \
  --max-steps "$MAX_STEPS" 

echo "OK: checkpoints en $OUT_DIR"
