#!/bin/bash
#SBATCH --job-name=drift-eval
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%A_%a.out
set -euo pipefail

# Re-evaluate a saved long-run checkpoint under TabDiff's reporting protocol:
# 20 independently sampled tables, averaged, with the per-run spread kept.
#
# The original long run reported a single draw, which is not comparable to the
# baseline in eval_tabdiff.sbatch (that one averages 20). Training is skipped
# entirely (--steps 0 --init_from), so this only costs sampling + metrics.
#
# Usage: sbatch --export=DATANAME=adult eval_drift.sbatch

DATANAME="${DATANAME:-adult}"
RUNS="${RUNS:-20}"

R_LISTS=("0.04 0.1 0.4" "0.08 0.2 0.8")
MULTS=(2 4)
R="${R_LISTS[$SLURM_ARRAY_TASK_ID]}"
MULT="${MULTS[$SLURM_ARRAY_TASK_ID]}"
OUT="drifting/result/$DATANAME/long_whitened_${MULT}x"

REPO=/playpen-shared/dinara/TabDiff
source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff
cd "$REPO"

if [ ! -f "$OUT/generator.pt" ]; then
    echo "[$(date)] ERROR: $OUT/generator.pt missing. Run train_drift_long.sbatch first." >&2
    exit 1
fi

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID on $(hostname): $DATANAME R=${MULT}x over $RUNS runs"
nvidia-smi -L

python -u -m drifting.train \
    --dataname "$DATANAME" --steps 0 \
    --phi whitened --R $R \
    --init_from "$OUT/generator.pt" \
    --final_metrics density mle c2st --final_runs "$RUNS" \
    --out "$OUT"

echo "[$(date)] Done: R=${MULT}x"
