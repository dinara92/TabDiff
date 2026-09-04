#!/bin/bash
#SBATCH --job-name=tabdiff
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%j.out
set -euo pipefail

# Usage: sbatch --export=DATANAME=adult train_tabdiff.sbatch
# DATANAME options: adult, default, shoppers, magic, beijing, news, diabetes
DATANAME="${DATANAME:-adult}"

REPO=/playpen-shared/dinara/TabDiff
CONDA_SH=/playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
ENV_NAME=tabdiff

source "$CONDA_SH"
conda activate "$ENV_NAME"
cd "$REPO"

echo "[$(date)] Job $SLURM_JOB_ID on $(hostname), dataset=$DATANAME"
nvidia-smi -L

# --- Data: download + process once, skipped if already present ---
if [ ! -f "$REPO/data/$DATANAME/info.json" ]; then
    echo "[$(date)] Preparing data for $DATANAME"
    python download_dataset.py
    python process_dataset.py --dataname "$DATANAME"
fi

# --- Train TabDiff (unconditional, learnable noise schedule) ---
echo "[$(date)] Training TabDiff on $DATANAME"
python main.py --dataname "$DATANAME" --mode train --no_wandb

echo "[$(date)] Done training: $DATANAME. Run eval_tabdiff.sbatch to sample + evaluate."
