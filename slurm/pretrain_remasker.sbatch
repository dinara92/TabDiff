#!/bin/bash
#SBATCH --job-name=remasker-pre
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --array=0-5
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%A_%a.out
set -euo pipefail

# Pretrain ReMasker architecture variants.
#
# The inherited checkpoints all use embed_dim=64, encoder_layers=8 -- the
# argparse defaults in remasker/pretrain.py, applied uniformly across every
# dataset and every pretrain_data ablation. The architecture was never varied,
# so "ReMasker phi loses to whitened raw space" is currently confounded with
# "this one architecture loses".
#
# Two hypotheses for why the flat readout underperformed at 64x15=960 dims:
#   embed_dim   distance concentration -- a 960-d kernel has less dynamic
#               range than whitened raw space's 110-d.
#   depth       8 layers is deep for a 15-column table; MAE features are
#               reconstruction-invariant and deep layers may discard the
#               detail the kernel needs.
#
# Usage: sbatch --export=DATANAME=adult pretrain_remasker.sbatch

DATANAME="${DATANAME:-adult}"
EPOCHS="${EPOCHS:-500}"

EMBED=(16 32 64 64 64 32)
LAYERS=(8  8  8  4  2  4)
E="${EMBED[$SLURM_ARRAY_TASK_ID]}"
L="${LAYERS[$SLURM_ARRAY_TASK_ID]}"
RUN="e${E}_l${L}"

REPO=/playpen-shared/dinara/TabDiff
source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff
cd "$REPO"

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID on $(hostname): $DATANAME embed=$E layers=$L"
nvidia-smi -L

python -u -m remasker.pretrain \
    --dataname "$DATANAME" --pretrain-data real \
    --embed-dim "$E" --encoder-layers "$L" \
    --epochs "$EPOCHS" --run-name "$RUN"

echo "[$(date)] Done: $RUN"
