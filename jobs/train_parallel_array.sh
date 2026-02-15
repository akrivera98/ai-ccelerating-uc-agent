#!/bin/bash
#SBATCH --job-name=uc_train_lp_bsweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=15
#SBATCH --mem=128G
#SBATCH --array=0-3

set -euo pipefail
mkdir -p logs

source .venv/bin/activate

export LP_WORKERS=${SLURM_CPUS_PER_TASK:-1}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG=${1:-configs/exclude_gens.yaml}

# Sweep values
BATCH_SIZES=(64 128 256 512)
BS=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

# ---- Encode batch size directly in base output directory ----
BASE_OUT=${2:-results/less_gens_parallel_bsweep}
OUT_DIR="${BASE_OUT}_bs${BS}"

echo "JobID=${SLURM_JOB_ID} ArrayTaskID=${SLURM_ARRAY_TASK_ID}"
echo "Config: ${CONFIG}"
echo "Batch size: ${BS}"
echo "OUT_DIR: ${OUT_DIR}"

srun python -u main.py \
  --config "$CONFIG" \
  --batch-size "$BS" \
  --out-dir "$OUT_DIR"
