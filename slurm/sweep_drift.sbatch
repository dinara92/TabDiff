#!/bin/bash
#SBATCH --job-name=drift-sweep
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --array=0-17%6
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%A_%a.out
set -euo pipefail

# Joint sweep over the feature space (phi) and the kernel bandwidth.
#
# Swept together on purpose: settling bandwidth on one phi and then changing
# phi would mean tuning against a geometry we are about to replace. ReMasker's
# embedding space and whitened raw space have different distance distributions,
# so their optimal R need not agree.
#
# R is parameterised as a MULTIPLIER on the reference default [0.02, 0.05, 0.2]
# rather than as free triples. That is the search space the TabSyn port used,
# where it found an inverted-U with the peak at 2x (0.25x and 0.5x collapsed
# outright, 8x reversed the gains). The grid runs to 32x because an early
# TabDiff run at roughly 34x scored well in whitened space -- so the two ports
# disagree, and the grid brackets both claims.
#
# phi variants:
#   whitened        raw space, per-column distance equalisation. The control.
#   remasker_flat   frozen tabular MAE, all column tokens in one vector.
#                   What the TabSyn port measured -- the baseline to beat.
#   remasker_percol one kernel per column token. Mirrors how the reference
#                   folds patch tokens into the kernel group axis. Untested.
#
# Usage: sbatch --export=DATANAME=adult sweep_drift.sbatch

DATANAME="${DATANAME:-adult}"
STEPS="${STEPS:-30000}"
# Held equal across arms so the comparison is fair, and sized for the most
# expensive arm (remasker_percol: 111 ms/step, 2.7 GB at C_GEN=1024).
C_GEN="${C_GEN:-1024}"
C_POS="${C_POS:-2048}"

PHIS=(whitened remasker_flat remasker_percol)
MULTS=(1 2 4 8 16 32)
R_LISTS=(
  "0.02 0.05 0.2"     # 1x  -- reference default
  "0.04 0.1 0.4"      # 2x  -- TabSyn's adopted value
  "0.08 0.2 0.8"      # 4x
  "0.16 0.4 1.6"      # 8x
  "0.32 0.8 3.2"      # 16x
  "0.64 1.6 6.4"      # 32x -- near the early TabDiff whitened result
)

PHI="${PHIS[$((SLURM_ARRAY_TASK_ID / 6))]}"
MI=$((SLURM_ARRAY_TASK_ID % 6))
R="${R_LISTS[$MI]}"
MULT="${MULTS[$MI]}"

REPO=/playpen-shared/dinara/TabDiff
source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff
cd "$REPO"

echo "[$(date)] Job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "[$(date)] dataset=$DATANAME phi=$PHI R=${MULT}x [$R] steps=$STEPS"
nvidia-smi -L

if [ "$PHI" != "whitened" ] && [ ! -f "$REPO/remasker/ckpt/$DATANAME/model.pt" ]; then
    echo "[$(date)] ERROR: remasker/ckpt/$DATANAME/model.pt missing. Pretrain it first." >&2
    exit 1
fi

python -u -m drifting.train \
    --dataname "$DATANAME" --steps "$STEPS" \
    --phi "$PHI" --R $R \
    --c_gen "$C_GEN" --c_pos "$C_POS" --eval_every 5000 \
    --out "drifting/result/$DATANAME/sweep/${PHI}_${MULT}x"

echo "[$(date)] Done: phi=$PHI R=${MULT}x"
