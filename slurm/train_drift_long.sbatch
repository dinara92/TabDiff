#!/bin/bash
#SBATCH --job-name=drift-long
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%A_%a.out
set -euo pipefail

# Long run at the sweep's winning configuration, reported on the same metrics
# as the TabDiff baseline.
#
# Sweep result (adult, 30k steps, EMA density/Overall): whitened phi peaked at
# 2x/4x (0.9718 / 0.9726) and was still improving at the final eval. The
# baseline (20-run average, eval/report_runs/learnable_schedule/adult/avg_std.csv)
# is 0.9884 density / 0.9127 MLE / 0.9887 C2ST. Two arms because the 2x-vs-4x
# gap (0.0008) is well inside single-seed noise.
#
# The final suite here averages 20 sampled tables (--final_runs), matching the
# baseline's protocol; re-evaluate a finished checkpoint with eval_drift.sbatch.
#
# Usage: sbatch --export=DATANAME=adult train_drift_long.sbatch

DATANAME="${DATANAME:-adult}"
STEPS="${STEPS:-200000}"
C_GEN="${C_GEN:-2048}"
C_POS="${C_POS:-4096}"

R_LISTS=("0.04 0.1 0.4" "0.08 0.2 0.8")
MULTS=(2 4)
R="${R_LISTS[$SLURM_ARRAY_TASK_ID]}"
MULT="${MULTS[$SLURM_ARRAY_TASK_ID]}"

REPO=/playpen-shared/dinara/TabDiff
source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff
cd "$REPO"

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID on $(hostname): $DATANAME whitened R=${MULT}x [$R]"
nvidia-smi -L

python -u -m drifting.train \
    --dataname "$DATANAME" --steps "$STEPS" \
    --phi whitened --R $R \
    --c_gen "$C_GEN" --c_pos "$C_POS" --eval_every 10000 \
    --final_metrics density mle c2st \
    --out "drifting/result/$DATANAME/long_whitened_${MULT}x"

echo "[$(date)] Done: R=${MULT}x"
