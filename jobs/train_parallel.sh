#!/bin/bash
#SBATCH --job-name=uc_train_lp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G

set -euo pipefail
mkdir -p logs

# Activate environment
source .venv/bin/activate

# Use SLURM allocation for joblib workers
export LP_WORKERS=${SLURM_CPUS_PER_TASK}

# Avoid nested parallelism (BLAS / OpenMP)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG=${1:-configs/exclude_gens.yaml}

echo "Using LP_WORKERS=${LP_WORKERS}"
echo "Config: ${CONFIG}"

python -u main.py --config "$CONFIG"
