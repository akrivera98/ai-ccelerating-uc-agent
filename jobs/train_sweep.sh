#!/bin/bash
#SBATCH --job-name=uc_train_sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
mkdir -p logs

cd "$SLURM_SUBMIT_DIR"

source .venv-lp/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

CSV=${1:-sweeps.csv}
CONFIG=${2:-configs/ablations_default_config.yaml}

bash jobs/run_train_one.sh "$CSV" "$CONFIG" "$SLURM_ARRAY_TASK_ID"
