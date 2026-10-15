#!/bin/bash
#SBATCH --job-name=bench-quality
#SBATCH --partition=ada,all,blackwell,a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%A.out
set -euo pipefail

# The full seven-metric suite at every point on the latency frontier, so the
# efficiency table carries the same columns as the main standings table.
#
# bench_speed.sbatch measured latency with a SINGLE draw per point, which is not
# comparable to the report's 20-run rows (it lands ~0.15 high on Shape). This
# job re-scores every point over 20 draws, matching that protocol.
#
# Two stages because alpha-precision and beta-recall are not part of TabMetrics:
#   A (tabdiff env)   -- generate 20 draws per point, score Shape/Trend/MLE/C2ST,
#                        and stage the draws under eval/report_runs/bench_<slug>/
#   B (synthcity env) -- score alpha/beta over those same staged draws
#
# Timing columns are NOT re-measured here; bench_speed_b*.csv owns those.
#
# Runs on any partition: this job measures QUALITY ONLY, so GPU model does not
# affect a single number it produces. Timing jobs (bench_speed.sbatch) must stay
# pinned to `ada`, because the published latency table was measured on its
# A6000s and a Blackwell or A100 node would not be comparable.
#
# Usage: sbatch --export=DATANAME=adult slurm/bench_quality.sbatch

DATANAME="${DATANAME:-adult}"
RUNS="${RUNS:-20}"

REPO=/playpen-shared/dinara/TabDiff
source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
cd "$REPO"

# synthcity's AlphaPrecision trips OpenBLAS's 128-thread ceiling on these nodes.
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8

echo "[$(date)] Stage A on $(hostname): $DATANAME, $RUNS draws per point"
nvidia-smi -L
conda activate tabdiff

python -u -m drifting.bench_speed \
    --dataname "$DATANAME" \
    --batch 4096 \
    --steps 5 10 25 50 100 \
    --repeats 1 --warmup 0 \
    --quality --metrics density mle c2st \
    --quality_runs "$RUNS" \
    --stage_alpha_beta \
    --out "drifting/result/$DATANAME/bench_quality.csv"

echo "[$(date)] Stage A done"

conda activate synthcity
for d in eval/report_runs/bench_*/"$DATANAME"; do
    [ -d "$d" ] || continue
    EXP=$(basename "$(dirname "$d")")
    echo "[$(date)] Stage B: $EXP ($(ls "$d"/all_samples/*.csv | wc -l) samples)"
    python -u eval/eval_quality.py --dataname "$DATANAME" --exp_name "$EXP"
done

echo "[$(date)] Done. Collect with: python -m drifting.collect_bench --dataname $DATANAME"
