#!/bin/bash
#SBATCH --job-name=bench-lownfe
#SBATCH --partition=ada,all,blackwell,a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%A_%a.out
set -euo pipefail

# How far down the NFE axis can TabDiff actually go?
#
# The frontier tables start at 5 steps because that is where the paper's own
# ablation (Table 12) stops. The obvious question is whether TabDiff has a
# 1-NFE configuration at all, so that drift can be compared against it at
# EQUAL cost rather than only at equal wall-clock.
#
# Two obstacles, both in tabdiff/models/unified_ctime_diffusion.py:
#   1. `sample()` builds its time grid as linspace(0, 1, T), so T=1 gives
#      t=[0.0] and the prior is drawn at sigma(0)=sigma_min=0.002 instead of
#      sigma_max=80 -- it starts from almost-clean data. T=1 is degenerate, not
#      a valid one-step sampler.
#   2. The second-order correction adds a call on every step except the last,
#      so NFE = 2T-1 with it on and T with it off. NFE is now MEASURED by
#      wrapping the denoiser, not computed, so the column cannot drift out of
#      sync with the code again.
#
# Array task 0 keeps the correction on (NFE 3, 5, 9 at T=2,3,5),
# task 1 turns it off (NFE 2, 3, 5) -- the genuine floor of this sampler.
#
# Usage: sbatch --export=DATANAME=adult slurm/bench_lownfe.sbatch

DATANAME="${DATANAME:-adult}"
RUNS="${RUNS:-20}"

FLAGS=("" "--no_second_order")
TAGS=(so2 so1)
F="${FLAGS[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"

REPO=/playpen-shared/dinara/TabDiff
source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff
cd "$REPO"
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID on $(hostname): $DATANAME low-NFE, tag=$TAG"
nvidia-smi -L

python -u -m drifting.bench_speed \
    --dataname "$DATANAME" --batch 4096 \
    --steps 2 3 5 10 \
    --repeats 1 --warmup 0 \
    --quality --metrics density mle c2st --quality_runs "$RUNS" \
    --stage_alpha_beta --stage_prefix "lownfe_${TAG}" \
    $F \
    --out "drifting/result/$DATANAME/bench_lownfe_${TAG}.csv"

echo "[$(date)] Stage A done: $TAG"

# alpha/beta are not part of TabMetrics and need the synthcity env. The prefix
# keeps the two array tasks -- and the main frontier's already-scored bench_*
# dirs -- from overwriting each other.
#
# !! STAGE B BELOW IS KNOWN BROKEN. It failed on all ten configs. Use
# !! slurm/rescore_lownfe_alphabeta.sbatch instead, which scores the samples
# !! stage A has already staged -- nothing needs regenerating.
# !!
# !! The comment that followed claimed "hiding the GPU makes this stage
# !! partition-independent". It does not. alpha-precision is built on a fitted
# !! one-class model that synthcity caches, and a cache HIT is returned without
# !! moving it to the active device (metrics/core/metric.py:118) while a fresh
# !! fit returns model.to(DEVICE) at :128. So a cache pickled on a GPU box gives
# !! "no kernel image is available for execution on the device" on blackwell AND
# !! "Attempting to deserialize object on a CUDA device but
# !! torch.cuda.is_available() is False" when the GPU is hidden -- one cause,
# !! two errors, and CPU-only is not the escape. The fix is a per-job
# !! SC_WORKSPACE (eval_quality.py reads it) plus a non-blackwell partition.
# !!
# !! Also: piping stage B through `tail -5` under `set -o pipefail` makes the
# !! first failure kill the loop AND truncate its own traceback, which is why
# !! the real cause stayed hidden for a month.
conda activate synthcity
export CUDA_VISIBLE_DEVICES=""
for d in eval/report_runs/lownfe_${TAG}_*/"$DATANAME"; do
    [ -d "$d" ] || continue
    EXP=$(basename "$(dirname "$d")")
    echo "[$(date)] Stage B: $EXP ($(ls "$d"/all_samples/*.csv | wc -l) samples)"
    python -u eval/eval_quality.py --dataname "$DATANAME" --exp_name "$EXP" 2>&1 | tail -5
done

echo "[$(date)] Done: $TAG"
