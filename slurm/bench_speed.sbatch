#!/bin/bash
#SBATCH --job-name=bench-speed
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%A_%a.out
set -euo pipefail

# Sampling latency: 1-NFE drift vs TabDiff across its step sweep, on one GPU,
# in one process, at a matched batch size (paper Appendix F.1, Table 11).
#
# Two array tasks because batch size dominates the measurement and a single
# batch size invites the objection that the ratio is an artifact of it:
#   0 -- batch 4096, the drift sampler's default; also scores Shape/Trend so
#        the run yields the latency-vs-quality frontier against paper Table 12
#   1 -- batch 32561, the whole table in one shot, the best case for the
#        multi-step model since it amortises per-batch overhead
#
# Timing needs an unshared GPU, which is the reason this goes through slurm
# rather than running on the head node.
#
# Usage: sbatch --export=DATANAME=adult slurm/bench_speed.sbatch

DATANAME="${DATANAME:-adult}"

BATCHES=(4096 32561)
QUALITY=("--quality" "")
BATCH="${BATCHES[$SLURM_ARRAY_TASK_ID]}"
Q="${QUALITY[$SLURM_ARRAY_TASK_ID]}"

REPO=/playpen-shared/dinara/TabDiff
source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff
cd "$REPO"

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID on $(hostname): $DATANAME batch=$BATCH"
nvidia-smi -L

python -u -m drifting.bench_speed \
    --dataname "$DATANAME" \
    --batch "$BATCH" \
    --steps 5 10 25 50 100 \
    --repeats 5 --warmup 2 $Q \
    --out "drifting/result/$DATANAME/bench_speed_b${BATCH}.csv"

echo "[$(date)] Done: batch=$BATCH"
