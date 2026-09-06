#!/bin/bash
#SBATCH --job-name=drift-cond
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%A_%a.out
set -euo pipefail

# Does conditioning fix the coverage deficit?
#
# beta-recall is the method's one real weakness: 34.36 against TabDiff's 53.67,
# a 19-point gap, while every distributional metric is within ~1.3 points. It is
# a coverage metric, and a one-step deterministic map onto a disconnected support
# is structurally bad at coverage (the 2D gate showed the same: 87% mode capture
# with samples bridging between modes).
#
# Every run so far has been unconditional -- one kernel group holding every
# generated sample, so repulsion is global. --group_by target makes the group
# axis the target column, so repulsion is *within class*: >50K particles repel
# only each other and are attracted only to >50K rows. On adult that class is 24%
# of the data, so under global repulsion its internal structure is easily swamped
# by the majority class. Within-class repulsion is the direct fix, and it is the
# structure Drifting was designed around (its ImageNet results group by class).
#
# CFG stays off deliberately. It sharpens p(x|y), and sharpening is the opposite
# of coverage -- it should make beta-recall worse. Conditioning and CFG are
# separable; only the first is wanted here.
#
# Arms 0 and 1 are the conditional runs; arm 2 is the unconditional ReMasker
# control that does not otherwise exist. The unconditional whitened control is
# the existing long run (drifting/result/adult/long_whitened_4x).
#
#   0  cond   whitened        vs existing long_whitened_4x
#   1  cond   remasker_flat   vs arm 2
#   2  uncond remasker_flat   control for arm 1
#
# ReMasker is included despite losing on density (0.9546 vs 0.9726 at 30k)
# because it has never been measured on coverage at all, and its one observed
# advantage points that way: on the DCR split its score sat closer to the ideal
# than whitened's (50.33 vs 50.90), consistent with a smoother feature space
# blurring the attraction rather than converging onto individual rows. Blurred
# attraction is bad for precision and plausibly good for coverage -- which is
# exactly the axis we never tested it on.
#
# alpha-precision / beta-recall are NOT produced here; they need the synthcity
# env. Run slurm/eval_quality.sbatch afterwards.
#
# Usage: sbatch --export=DATANAME=adult train_drift_cond.sbatch

DATANAME="${DATANAME:-adult}"
STEPS="${STEPS:-200000}"
C_GEN="${C_GEN:-2048}"
C_POS="${C_POS:-4096}"
R="${R:-0.08 0.2 0.8}"          # 4x, the sweep winner
ARCH="${ARCH:-e64_l4}"          # best ReMasker architecture

PHIS=(whitened remasker_flat remasker_flat)
# NOT `GROUPS` -- that is a bash built-in holding the current user's
# group IDs, so assigning to it is silently ignored and the array reads
# back real GIDs.
GROUP_BYS=(target target none)
TAGS=(cond_whitened_4x cond_remasker_4x uncond_remasker_4x)
PHI="${PHIS[$SLURM_ARRAY_TASK_ID]}"
GRP="${GROUP_BYS[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"

REPO=/playpen-shared/dinara/TabDiff
source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff
cd "$REPO"

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID on $(hostname): $TAG phi=$PHI group_by=$GRP"
nvidia-smi -L

EXTRA=()
if [ "$PHI" != "whitened" ]; then
    CKPT="$REPO/remasker/ckpt/${DATANAME}-${ARCH}/model.pt"
    [ -f "$CKPT" ] || { echo "ERROR: $CKPT missing" >&2; exit 1; }
    EXTRA+=(--remasker_ckpt "$CKPT")
fi

python -u -m drifting.train \
    --dataname "$DATANAME" --steps "$STEPS" \
    --phi "$PHI" --group_by "$GRP" --R $R "${EXTRA[@]}" \
    --c_gen "$C_GEN" --c_pos "$C_POS" --eval_every 20000 \
    --final_metrics density mle c2st \
    --out "drifting/result/$DATANAME/$TAG"

echo "[$(date)] Done: $TAG"
