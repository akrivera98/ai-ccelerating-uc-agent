#!/bin/bash
#SBATCH --job-name=uc_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
mkdir -p logs

source .venv-lp/bin/activate

CONFIG=${1:-configs/idea_1_config.yaml}

python -u main.py --config "$CONFIG"