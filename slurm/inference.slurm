#!/bin/bash
# One model per array task, four data-parallel shards per model, one GPU each.
#
# Resume is the point of the layout: each shard owns a disjoint, stably hashed
# set of prompts and appends to its own file, so a task killed by the wall clock
# is restarted by resubmitting the same array -- it skips what it already wrote.
#
# Set the account, partition and node for your site before submitting, and
# submit from a login node.
#
#SBATCH --job-name=defer-infer
#SBATCH --array=0-7
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/infer_%A_%a.out
#SBATCH --error=logs/slurm/infer_%A_%a.err

set -euo pipefail

REPO="${REPO:-$PWD}"
ENV_NAME="${ENV_NAME:-deferral}"
NUM_SHARDS="${NUM_SHARDS:-4}"

cd "$REPO"
mkdir -p logs/slurm raw_responses

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONPATH="$REPO"

MODEL="$(python -m src.inference.run_inference --list-models | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")"
echo "task ${SLURM_ARRAY_TASK_ID}: ${MODEL}"

for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES="$SHARD" python -m src.inference.run_inference \
    --model "$MODEL" \
    --manifest data/exam_set.csv \
    --output-dir raw_responses \
    --phrasing neutral \
    --format MC OE \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD" &
done
wait

echo "done: ${MODEL}"
