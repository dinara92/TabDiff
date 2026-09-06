#!/bin/bash
#SBATCH --job-name=drift-dcr
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%A_%a.out
set -euo pipefail

# Privacy (DCR) arms, trained on the equal 50/50 split.
#
# DCR cannot be computed from the existing runs: it needs a train/holdout split
# of the same size (the default is 90/10), so every model has to be retrained on
# <dataset>_dcr. See README "Evaluating Data Privacy (DCR score)".
#
# This is the failure mode the architecture is most exposed to, and nothing in
# the density/MLE/C2ST table would reveal it: the attraction term is a soft
# nearest-neighbour match onto stored real rows, so a particle can converge onto
# a training row while every distributional metric stays healthy. 50% is ideal
# (synthetic rows equidistant from train and holdout); higher means the
# synthetic data sits closer to the training rows than a real holdout row does.
#
# Three arms, chosen so a bad ReMasker DCR number is attributable rather than
# confounded:
#   0 whitened          raw space, the sweep winner. phi is fit on the training
#                       half too, but has ~110 parameters (means and scales) and
#                       cannot encode individual rows.
#   1 remasker_flat     best ReMasker readout, phi pretrained on the training
#                       half -- so it *can* encode individual rows.
#   2 remasker_flat     same architecture and output dimensionality, phi frozen
#     (random init)     at random init: zero training-set information.
#
# Arm 2 is the control that separates the two mechanisms. If arms 1 and 2 have
# similar DCR, phi's training is not what drives memorisation and any gap is the
# generator. If arm 2 is markedly better, phi memorising the training half is
# carrying it. Direction is genuinely unpredictable a priori: a phi optimised to
# reconstruct training rows has fine structure around them (sharper attraction
# onto individual rows), but MAE representations are also smooth and detail-
# invariant, which would blur attraction and target the distribution instead.
#
# IMPORTANT: the ReMasker checkpoint must be pretrained on the *_dcr* split.
# Reusing remasker/ckpt/adult/model.pt would leak: it was pretrained on the 90%
# split, which contains rows that are holdout under the 50/50 split, and DCR is
# precisely a train-vs-holdout comparison. train.py resolves the checkpoint per
# dataname, so this fails loudly rather than silently using the wrong one.
#
# Usage: sbatch --dependency=afterok:<pretrain_id> --export=DATANAME=adult_dcr train_drift_dcr.sbatch

DATANAME="${DATANAME:-adult_dcr}"
STEPS="${STEPS:-200000}"
C_GEN="${C_GEN:-2048}"
C_POS="${C_POS:-4096}"
R="${R:-0.08 0.2 0.8}"      # 4x -- the sweep winner on adult
MULT="${MULT:-4}"
# e64_l4 was the best ReMasker architecture in the arch sweep (0.9546 vs 0.9269
# for the inherited e64_l8), so the privacy arm uses the best available ReMasker,
# not the default one. pretrain_remasker.sbatch writes to <dataname>-<arch>.
ARCH="${ARCH:-e64_l4}"

PHIS=(whitened remasker_flat remasker_flat)
ARCHS=("" "$ARCH" "${ARCH}_randinit")
PHI="${PHIS[$SLURM_ARRAY_TASK_ID]}"
ARCH="${ARCHS[$SLURM_ARRAY_TASK_ID]}"
TAG="$PHI"; [ "$SLURM_ARRAY_TASK_ID" = "2" ] && TAG="${PHI}_randinit"

REPO=/playpen-shared/dinara/TabDiff
source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff
cd "$REPO"

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID on $(hostname): $DATANAME phi=$TAG arch=${ARCH:-n/a} R=${MULT}x"
nvidia-smi -L

CKPT="$REPO/remasker/ckpt/${DATANAME}-${ARCH}/model.pt"
if [ "$PHI" != "whitened" ] && [ ! -f "$CKPT" ]; then
    echo "[$(date)] ERROR: $CKPT missing." >&2
    echo "  Pretrain on the _dcr split first -- do NOT substitute the adult checkpoint." >&2
    exit 1
fi

python -u -m drifting.train \
    --dataname "$DATANAME" --steps "$STEPS" \
    --phi "$PHI" --R $R \
    $( [ "$PHI" != "whitened" ] && echo "--remasker_ckpt $CKPT" ) \
    --c_gen "$C_GEN" --c_pos "$C_POS" --eval_every 20000 \
    --final_metrics dcr density \
    --out "drifting/result/$DATANAME/${TAG}_${MULT}x"

echo "[$(date)] Done: $DATANAME $PHI"
