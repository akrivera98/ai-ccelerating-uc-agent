#!/bin/bash
#SBATCH --job-name=uc_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu   # or remove if cpu is the default

set -euo pipefail
mkdir -p logs

source .venv/bin/activate

CONFIG=${1:-configs/idea_1_config.yaml}

python -u main.py --config "$CONFIG"
