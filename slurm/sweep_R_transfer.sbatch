#!/bin/bash
#SBATCH --job-name=R-transfer
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --array=0-5
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%A_%a.out
set -euo pipefail

# Does the R=4x bandwidth optimum transfer off adult?
#
# R is the one hyperparameter whose failure mode is silent, and it is validated
# on adult only. What it has to bracket is the ratio of gen-gen to gen->data
# distance, which is governed by distance concentration and therefore by the
# effective dimension after one-hot encoding. That varies enormously across the
# suite, so this tests the two extremes rather than assuming:
#
#   magic     10 numerical, 0 categorical  -- lowest dimension in the suite
#   diabetes   9 numerical, 27 categorical -- highest, with high-cardinality codes
#
# If 4x wins at both ends, it is justified for the middle four and no further
# sweeping is needed. If the optimum moves, bandwidth needs per-dataset
# calibration -- itself a finding, and one that changes the recommended recipe.
#
# 30k steps is enough to rank arms (the adult sweep separated 0.84 from 0.97 at
# this budget); the winner gets a full 200k run afterwards.
#
# Usage: sbatch slurm/sweep_R_transfer.sbatch

DATASETS=(magic diabetes)
R_LISTS=("0.04 0.1 0.4" "0.08 0.2 0.8" "0.16 0.4 1.6")
MULTS=(2 4 8)

D="${DATASETS[$((SLURM_ARRAY_TASK_ID / 3))]}"
I=$((SLURM_ARRAY_TASK_ID % 3))
R="${R_LISTS[$I]}"; MULT="${MULTS[$I]}"

REPO=/playpen-shared/dinara/TabDiff
source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff
cd "$REPO"

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID on $(hostname): $D R=${MULT}x [$R]"
nvidia-smi -L

if [ ! -f "$REPO/synthetic/$D/real.csv" ]; then
    echo "[$(date)] ERROR: synthetic/$D/real.csv missing -- run process_dataset.py --dataname $D" >&2
    exit 1
fi

python -u -m drifting.train \
    --dataname "$D" --steps 30000 \
    --phi whitened --R $R \
    --c_gen 1024 --c_pos 2048 --eval_every 10000 \
    --out "drifting/result/$D/sweep_R/${MULT}x"

echo "[$(date)] Done: $D ${MULT}x"
