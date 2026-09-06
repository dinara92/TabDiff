#!/bin/bash
#SBATCH --job-name=dcr-ckpt
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --array=0-4
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%A_%a.out
set -euo pipefail

# Does TabDiff's DCR on the 50/50 split degrade with training time?
#
# Our repro scored 58.53 +-0.28 against the paper's 50.10 +-0.32 -- a 26-sigma
# gap, so one of the two is measuring something different. Training itself was
# healthy (all 8000 epochs, final loss 1.5893 vs 1.5875 on the main split), so
# the leading hypothesis is overfitting from the split change: adult_dcr has
# 24421 training rows against the main split's 32561, on an unchanged 8000-epoch
# budget. The best EMA checkpoint arrives at epoch 7722 here versus 4546 on the
# main split -- it kept improving on training loss far longer, which is what
# memorisation looks like.
#
# If DCR rises monotonically with checkpoint epoch, the hypothesis holds and our
# 58.53 is an artefact of overtraining rather than a property of TabDiff. If it
# is flat and high, something else differs from the paper's protocol.
#
# Costs no retraining -- every checkpoint is already on disk. The periodic EMA
# dumps store a flat state_dict rather than the {denoise_fn, num_schedule,
# cat_schedule} form the trainer loads, so they are repacked first; the learnable
# schedules are not saved per-epoch and are taken from the final checkpoint,
# which is the one inexactness in this sweep (weights vary, schedules do not).
#
# Usage: sbatch --export=DATANAME=adult_dcr dcr_ckpt_sweep.sbatch

DATANAME="${DATANAME:-adult_dcr}"
REPO=/playpen-shared/dinara/TabDiff
CKPT_DIR="$REPO/tabdiff/ckpt/$DATANAME/learnable_schedule"

CKPTS=(repacked/best_ema_model_0.0_2000.pt repacked/best_ema_model_0.0_4000.pt repacked/best_ema_model_0.0_6000.pt repacked/best_ema_model_0.0_8000.pt BEST)
C="${CKPTS[$SLURM_ARRAY_TASK_ID]}"
if [ "$C" = "BEST" ]; then
    CKPT=$(ls "$CKPT_DIR"/best_ema_model_*.pt | head -1)
    TAG="best"
else
    CKPT="$CKPT_DIR/$C"
    TAG=$(basename "${C%.pt}" | sed "s/best_ema_model_0.0_/ep/")
fi

source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff
cd "$REPO"

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID on $(hostname): $TAG -> $CKPT"
[ -f "$CKPT" ] || { echo "ERROR: $CKPT missing" >&2; exit 1; }
nvidia-smi -L

# --report writes to a fixed eval/report_runs/<exp>/<dataname> path, so each
# arm gets its own exp_name to avoid clobbering the others.
EXP="dcr_ckpt_$TAG"
mkdir -p "eval/report_runs/$EXP"
python -u main.py --dataname "$DATANAME" --mode test --report --no_wandb \
    --ckpt_path "$CKPT" --exp_name "$EXP"

echo "[$(date)] Done: $TAG"
