#!/usr/bin/env bash
#SBATCH --job-name=lp_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=45
#SBATCH --mem=128G

set -euo pipefail
mkdir -p logs

echo "Running worker sweep on node: $SLURMD_NODENAME"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"

# ---- Prevent BLAS thread oversubscription ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Fixed during worker sweep ----
REPEATS=3
BATCH_SZ=512

# ---- CHUNK sweep ----
CHUNKS_LIST=(1 2 4 8 10 12)
W=8
source .venv/bin/activate

for C in "${CHUNKS_LIST[@]}"; do
    echo ""
    echo "======================================="
    echo "Testing chunks = $C"
    echo "======================================="

    for r in $(seq 1 $REPEATS); do
        echo "---- Repeat $r / $REPEATS ----"

        python -u tests/test_parallel_lp.py $W --batch_size $BATCH_SZ --chunks_per_worker $C
    done
done

echo "Sweep complete."
