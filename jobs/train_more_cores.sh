#!/bin/bash
#SBATCH --job-name=uc_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # you can override this at submission time
#SBATCH --mem=128G
#SBATCH --partition=cpu

set -euo pipefail
mkdir -p logs

source .venv/bin/activate

CONFIG=${1:-configs/idea_1_config.yaml}

# Use all cores Slurm gives you
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# (optional but often helpful for PyTorch CPU ops)
export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Run under srun so Slurm enforces CPU binding
srun --cpu-bind=cores python -u main.py --config "$CONFIG"
