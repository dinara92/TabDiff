#!/bin/bash
#SBATCH --job-name=R-sweep
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --array=0-13
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%A_%a.out
set -euo pipefail

# Per-dataset bandwidth calibration.
#
# R=4x does NOT transfer. Measured at 30k steps, whitened phi:
#   magic    (dim   12)  2x 0.6523 (collapse) | 4x 0.9809 | 8x 0.9850
#   adult    (dim  110)  2x 0.9718 | 4x 0.9726 | 8x 0.9680
#   diabetes (dim 2218)  2x 0.9140 | 4x 0.9131 | 8x 0.9157  (flat)
#
# The same multiplier is near-optimal on adult and catastrophic on magic, so a
# fixed default cannot be used. Sensitivity tracks dimension: low-dimensional
# data needs a large R and collapses under a small one, high-dimensional data is
# nearly indifferent (distance concentration compresses the range R must span).
# The loss gives no warning in either case -- magic's collapsed arm sat at
# 7.2-8.7, inside the normal range.
#
# Arms 0-1  magic 16x/32x -- 4x->8x was still climbing, so its peak is not yet
#           bracketed and may lie well above 8x.
# Arms 2-13 the four unswept datasets at 2x/4x/8x.
#
# Usage: sbatch slurm/sweep_R_all.sbatch

R_LISTS=("0.04 0.1 0.4" "0.08 0.2 0.8" "0.16 0.4 1.6" "0.32 0.8 3.2" "0.64 1.6 6.4")
MULTS=(2 4 8 16 32)

if [ "$SLURM_ARRAY_TASK_ID" -lt 2 ]; then
    D=magic; I=$((SLURM_ARRAY_TASK_ID + 3))          # 16x, 32x
else
    DS=(default shoppers beijing news)
    K=$((SLURM_ARRAY_TASK_ID - 2))
    D="${DS[$((K / 3))]}"; I=$((K % 3))              # 2x, 4x, 8x
fi
R="${R_LISTS[$I]}"; MULT="${MULTS[$I]}"

REPO=/playpen-shared/dinara/TabDiff
source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff
cd "$REPO"

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID on $(hostname): $D R=${MULT}x [$R]"
nvidia-smi -L
[ -f "$REPO/synthetic/$D/real.csv" ] || { echo "ERROR: synthetic/$D/real.csv missing" >&2; exit 1; }

python -u -m drifting.train \
    --dataname "$D" --steps 30000 \
    --phi whitened --R $R \
    --c_gen 1024 --c_pos 2048 --eval_every 10000 \
    --out "drifting/result/$D/sweep_R/${MULT}x"

echo "[$(date)] Done: $D ${MULT}x"
