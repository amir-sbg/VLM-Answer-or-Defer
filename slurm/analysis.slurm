#!/bin/bash
# Transcripts -> decision table -> tables, figures and the verification report.
# CPU only; it needs no GPU and runs in a few minutes.
#
#SBATCH --job-name=defer-analysis
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/analysis_%j.out
#SBATCH --error=logs/slurm/analysis_%j.err

set -euo pipefail

REPO="${REPO:-$PWD}"
ENV_NAME="${ENV_NAME:-deferral}"

cd "$REPO"
mkdir -p logs/slurm

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONPATH="$REPO"

python -m src.analysis.build_study_table
make tables figures verify
