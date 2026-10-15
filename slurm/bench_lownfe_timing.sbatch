#!/bin/bash
#SBATCH --job-name=bench-lownfe-timing
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%j.out
set -euo pipefail

# Clean wall clock for TabDiff's FIRST-ORDER configs.
#
# bench_lownfe.sbatch measured their quality but not their speed: its gen_s
# column comes from a job that also ran metric evaluation on a shared GPU, and
# it clocks the second-order 5-step config at 1.685s against the 0.736s that
# bench_speed_b4096.csv measures for the same config on an unshared one. That
# is ~2x slow, so the low-NFE sweep's timings are not comparable to the
# report's frontier table and must not be quoted next to it.
#
# This job is the timing half only -- no --quality, repeats 5 after 2 warmup,
# batch 4096, pinned to `ada` -- i.e. byte-identical protocol to
# bench_speed.sbatch task 0, differing only in --no_second_order and the step
# list. That makes bench_speed_b4096_so1.csv joinable with
# bench_speed_b4096.csv.
#
# Why it matters: at a fixed 10-NFE budget, second-order buys 5 steps and
# first-order buys 10, and the first-order config is ~4.6x better on Shape
# (3.11 against 14.37). The report compares drift only against the second-order
# one, which is the weaker opponent at equal cost.
#
# Timing needs an unshared GPU, which is the reason this goes through slurm
# rather than running on the head node.
#
# Usage: sbatch --export=DATANAME=adult slurm/bench_lownfe_timing.sbatch

DATANAME="${DATANAME:-adult}"

REPO=/playpen-shared/dinara/TabDiff
source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff
cd "$REPO"

echo "[$(date)] $DATANAME first-order timings on $(hostname)"
nvidia-smi -L

python -u -m drifting.bench_speed \
    --dataname "$DATANAME" \
    --batch 4096 \
    --steps 2 3 5 10 \
    --repeats 5 --warmup 2 \
    --no_second_order \
    --out "drifting/result/$DATANAME/bench_speed_b4096_so1.csv"

echo "[$(date)] Done"
cat "drifting/result/$DATANAME/bench_speed_b4096_so1.csv"
