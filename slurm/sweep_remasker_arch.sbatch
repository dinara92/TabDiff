#!/bin/bash
#SBATCH --job-name=drift-arch
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --array=0-23%6
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%A_%a.out
set -euo pipefail

# Does a differently-shaped ReMasker beat whitened raw space?
#
# The first sweep found whitened phi (0.9726) beating remasker_flat (0.9487) at
# every bandwidth. But every ReMasker checkpoint in existence uses embed_dim=64
# / 8 layers -- pretrain.py's defaults, never varied -- so that result confounds
# "ReMasker doesn't help here" with "this one architecture doesn't help".
#
# Runs each pretrained variant at the bandwidth its readout family preferred in
# the first sweep, bracketed one step either side:
#   flat       peaked at 4x  -> test 4x, 16x
#   percol     peaked at 32x -> test 4x, 16x (its optimum ran the other way,
#              consistent with per-column kernels living in a much lower-
#              dimensional space with a different distance scale)
#
# Requires slurm/pretrain_remasker.sbatch to have finished.
# Usage: sbatch --dependency=afterok:<pretrain_jobid> --export=DATANAME=adult sweep_remasker_arch.sbatch

DATANAME="${DATANAME:-adult}"
STEPS="${STEPS:-30000}"
C_GEN="${C_GEN:-1024}"
C_POS="${C_POS:-2048}"

ARCHS=(e16_l8 e32_l8 e64_l8 e64_l4 e64_l2 e32_l4)
READOUTS=(remasker_flat remasker_percol)
R_LISTS=("0.08 0.2 0.8" "0.32 0.8 3.2")
MULTS=(4 16)

A=$(( SLURM_ARRAY_TASK_ID % 6 ))
RD=$(( (SLURM_ARRAY_TASK_ID / 6) % 2 ))
RI=$(( SLURM_ARRAY_TASK_ID / 12 ))

ARCH="${ARCHS[$A]}"; PHI="${READOUTS[$RD]}"; R="${R_LISTS[$RI]}"; MULT="${MULTS[$RI]}"
CKPT="remasker/ckpt/${DATANAME}-${ARCH}/model.pt"

REPO=/playpen-shared/dinara/TabDiff
source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff
cd "$REPO"

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID on $(hostname): $ARCH $PHI R=${MULT}x"
nvidia-smi -L

if [ ! -f "$CKPT" ]; then
    echo "[$(date)] ERROR: $CKPT missing -- run pretrain_remasker.sbatch first." >&2
    exit 1
fi

python -u -m drifting.train \
    --dataname "$DATANAME" --steps "$STEPS" \
    --phi "$PHI" --remasker_ckpt "$CKPT" --R $R \
    --c_gen "$C_GEN" --c_pos "$C_POS" --eval_every 10000 \
    --out "drifting/result/$DATANAME/arch/${ARCH}_${PHI}_${MULT}x"

echo "[$(date)] Done: $ARCH $PHI ${MULT}x"
