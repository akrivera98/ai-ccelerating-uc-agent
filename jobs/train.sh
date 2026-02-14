#!/bin/bash
#SBATCH --job-name=uc_train_cvxpylayers
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
mkdir -p logs

source .venv/bin/activate

CONFIG=${1:-configs/exclude_gens.yaml}

python -u main.py --config "$CONFIG"