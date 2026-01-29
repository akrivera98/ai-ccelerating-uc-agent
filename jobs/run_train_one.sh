#!/bin/bash
set -euo pipefail

CSV="$1"
CONFIG="$2"
TASK_ID="$3"

# Read the (TASK_ID+2)-th line: +1 for header, +1 because awk is 1-indexed
LINE=$(awk -F',' -v i=$((TASK_ID+2)) 'NR==i{print}' "$CSV")
LINE=$(echo "$LINE" | tr -d ' ')

if [[ -z "${LINE}" ]]; then
  echo "ERROR: No line found for task_id=${TASK_ID} in ${CSV}"
  exit 1
fi

IFS=',' read -r run_name seed use_ste solve_lp_in_loss \
  w_supervised w_violation w_ed_objective w_startup <<< "$LINE"

echo "=== Running: ${run_name} (seed=${seed}) ==="
echo "use_ste=${use_ste} solve_lp_in_loss=${solve_lp_in_loss}"
echo "weights: sup=${w_supervised} viol=${w_violation} obj=${w_ed_objective} startup=${w_startup}"

python -u main_ablations.py \
  --config "$CONFIG" \
  --run-name "$run_name" \
  --seed "$seed" \
  --use-ste "$use_ste" \
  --use-ed-in-training "$solve_lp_in_loss" \
  --w-supervised "$w_supervised" \
  --w-violation "$w_violation" \
  --w-ed-objective "$w_ed_objective" \
  --w-startup "$w_startup"
