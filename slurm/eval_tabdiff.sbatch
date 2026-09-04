#!/bin/bash
#SBATCH --job-name=tabdiff-eval
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%j.out
set -euo pipefail

# Usage: sbatch --export=DATANAME=adult eval_tabdiff.sbatch
# Samples 20 synthetic tables from an already-trained model and reports the
# avg/std of density, MLE, and C2ST scores (README "Sampling and Evaluating"
# section). Does NOT train -- run train_tabdiff.sbatch first.
#
# To also compute alpha-precision/beta-recall (README "Additional Fidelity
# Metrics"), pass EVAL_QUALITY=1; this needs samples already on disk from the
# report step above and runs in the separate `synthcity` env.
DATANAME="${DATANAME:-adult}"
EVAL_QUALITY="${EVAL_QUALITY:-0}"

REPO=/playpen-shared/dinara/TabDiff
CONDA_SH=/playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
TABDIFF_ENV=tabdiff
SYNTHCITY_ENV=synthcity

source "$CONDA_SH"
cd "$REPO"

echo "[$(date)] Job $SLURM_JOB_ID on $(hostname), dataname=$DATANAME"
nvidia-smi -L

CKPT_DIR="$REPO/tabdiff/ckpt/$DATANAME/learnable_schedule"
if ! compgen -G "$CKPT_DIR/best_ema_model*" > /dev/null; then
    echo "[$(date)] ERROR: no checkpoint found in $CKPT_DIR. Train first (e.g. sbatch train_tabdiff.sbatch)." >&2
    exit 1
fi

# --- Sample + report density/MLE/C2ST (tabdiff env) ---
conda activate "$TABDIFF_ENV"

echo "[$(date)] Sampling + evaluating (density, MLE, C2ST) over 20 runs"
python main.py --dataname "$DATANAME" --mode test --report --no_wandb

conda deactivate

# --- Optional: alpha-precision / beta-recall (synthcity env) ---
if [ "$EVAL_QUALITY" = "1" ]; then
    conda activate "$SYNTHCITY_ENV"

    echo "[$(date)] Alpha Precision / Beta Recall"
    python eval/eval_quality.py --dataname "$DATANAME"

    conda deactivate
fi

echo "[$(date)] Done evaluating: $DATANAME"
