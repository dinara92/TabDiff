#!/bin/bash
#SBATCH --job-name=drift
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%j.out
set -euo pipefail

# Trains the one-step drift generator (drifting/train.py). Does NOT train
# TabDiff -- that is train_tabdiff.sbatch, and it stays the multi-step baseline.
#
# Usage:
#   sbatch --export=DATANAME=adult train_drift.sbatch
#   sbatch --export=DATANAME=adult,STEPS=50000,GROUP_BY=target train_drift.sbatch
#   sbatch --export=DATANAME=adult,R="0.1 0.3 1.0" train_drift.sbatch
#
# Defaults follow the reference recipe (drifting/configs/gen/latent_sota_B.yaml):
# 200k steps, lr 4e-4 with 10k warmup then constant, EMA 0.999. Particle counts
# are NOT copied from it: the reference splits 8192 samples across 128 class
# groups, so its kernel solves many small problems, while unconditional
# generation gives one large group. C_GEN=2048 already exceeds the reference's
# per-step kernel work. ~3.7h on one Ada GPU; raise --time before raising these.
#
# Leaving R unset calibrates the bandwidths from the data at step 0. That is
# the setting to touch first if the run does not move: the loss sits at
# O(len(R_list)) whether or not anything is being learned, so a flat loss curve
# is not evidence either way. See drifting/README.md.

DATANAME="${DATANAME:-adult}"
STEPS="${STEPS:-200000}"
C_GEN="${C_GEN:-2048}"
C_POS="${C_POS:-4096}"
LR="${LR:-4e-4}"
GROUP_BY="${GROUP_BY:-none}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
SEED="${SEED:-0}"
R="${R:-}"
OUT="${OUT:-}"

REPO=/playpen-shared/dinara/TabDiff
CONDA_SH=/playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
ENV_NAME=tabdiff

source "$CONDA_SH"
conda activate "$ENV_NAME"
cd "$REPO"

echo "[$(date)] Job $SLURM_JOB_ID on $(hostname), dataset=$DATANAME"
nvidia-smi -L

if [ ! -f "$REPO/data/$DATANAME/info.json" ]; then
    echo "[$(date)] Preparing data for $DATANAME"
    python download_dataset.py
    python process_dataset.py --dataname "$DATANAME"
fi

# Evaluation compares against synthetic/<data>/real.csv, written by TabDiff's
# own data prep. Without it there is nothing to score against.
if [ ! -f "$REPO/synthetic/$DATANAME/real.csv" ]; then
    echo "[$(date)] ERROR: synthetic/$DATANAME/real.csv missing. Run process_dataset.py first." >&2
    exit 1
fi

ARGS=(--dataname "$DATANAME" --steps "$STEPS" --c_gen "$C_GEN" --c_pos "$C_POS"
      --lr "$LR" --group_by "$GROUP_BY" --eval_every "$EVAL_EVERY" --seed "$SEED")
[ -n "$R" ]   && ARGS+=(--R $R)
[ -n "$OUT" ] && ARGS+=(--out "$OUT")

echo "[$(date)] python -m drifting.train ${ARGS[*]}"
python -u -m drifting.train "${ARGS[@]}"

echo "[$(date)] Done: $DATANAME"
