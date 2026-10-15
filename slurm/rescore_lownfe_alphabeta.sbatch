#!/bin/bash
#SBATCH --job-name=rescore-lownfe-ab
#SBATCH --partition=ada,a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --array=0-9
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%A_%a.out
set -euo pipefail

# Recover the alpha-precision / beta-recall column for the low-NFE sweep.
#
# Stage B of bench_lownfe.sbatch died on all ten configs with
#   RuntimeError: CUDA error: no kernel image is available for execution
# (see slurm/logs/bench-lownfe_151132_{0,1}.out), and that job piped stage B
# through `tail -5` under `set -o pipefail`, so the first failure both killed
# the loop and truncated its own traceback.
#
# The cause is NOT what that job's comment assumed, and CPU-only is not the fix.
# synthcity caches the fitted one-class model that alpha-precision is built on:
#   metrics/core/metric.py:110  _get_oneclass_model -> load_from_file(cache_file)
# keyed on a hash of the REAL data, so every run of every config shares one
# cached model -- workspace/sc_metric_cache_model_oneclass_*.bkp, ours written
# 2026-09-06 by the first alpha/beta job. It was pickled on a GPU box, so its
# tensors carry CUDA storage locations. That gives two failure modes from one
# cause:
#   on blackwell   -- it deserializes onto CUDA, where this torch (2.2.2) has
#                     no kernels for sm_120  -> "no kernel image available"
#   on CPU-only    -- it refuses to deserialize at all
#                     -> "Attempting to deserialize object on a CUDA device
#                         but torch.cuda.is_available() is False"
# The first attempt at this job (264433) set --gres=gpu:0 and
# CUDA_VISIBLE_DEVICES=-1 and hit the second mode in 12-18s on all ten tasks.
#
# So: run WITH a GPU, on a partition whose arch this torch has kernels for.
# Deleting the cache would also work but is worse -- refitting produces a
# different one-class model, and then these rows would not be comparable to the
# alpha/beta already published from the Sep 6 cache (bench_drift 98.80 / 34.32,
# bench_tabdiff_*). Reusing the cache is what keeps the new rows joinable.
#
# Each config is an independent array task, and output is unpiped, so one
# failure costs one row and arrives with its traceback intact.
#
# alpha/beta is CPU work on samples already staged by the Stage A run, so this
# regenerates nothing: it reads eval/report_runs/lownfe_*/<data>/all_samples.
#
# Usage: sbatch --export=DATANAME=adult slurm/rescore_lownfe_alphabeta.sbatch

DATANAME="${DATANAME:-adult}"

EXPS=(
  lownfe_so1_drift lownfe_so1_tabdiff_2 lownfe_so1_tabdiff_3
  lownfe_so1_tabdiff_5 lownfe_so1_tabdiff_10
  lownfe_so2_drift lownfe_so2_tabdiff_2 lownfe_so2_tabdiff_3
  lownfe_so2_tabdiff_5 lownfe_so2_tabdiff_10
)
EXP="${EXPS[$SLURM_ARRAY_TASK_ID]}"

REPO=/playpen-shared/dinara/TabDiff
source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
conda activate synthcity
cd "$REPO"
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8

# Per-task cache dir, SEEDED with the Sep 6 pooled-model cache.
#
# The shared workspace/ is not safe to read here. synthcity returns a cache HIT
# without moving the model to the active device (metrics/core/metric.py:118
# `return load_from_file(cache_file)`) while a fresh fit returns
# `model.to(DEVICE)` at :128, and _oneclass_predict always moves its input to
# DEVICE. workspace/ accumulates one cache per distinct real-data hash from
# whatever job ran last -- five were written there on 2026-10-14 by a
# concurrent run -- so a GPU reader that hits a CPU-written file dies with
#   "Expected all tensors to be on the same device, cuda:0 and cpu"
# which is exactly how attempt 264447 failed.
#
# Isolating the directory fixes that. But it must be seeded, not empty: the
# POOLED model is the one every published alpha/beta was scored against, its fit
# is unseeded, and refitting it would make these rows incomparable to
# bench_drift's 98.80 / 34.32. So copy in the 2026-09-06 pooled cache and let
# anything else fit fresh.
#
# Verification that the seed actually took: lownfe_so1_drift must come out at
# alpha 0.987868 / beta 0.343122, the values attempt 264447 logged from this
# same cache before it was interrupted.
export SC_WORKSPACE="$REPO/workspace_lownfe/task${SLURM_ARRAY_TASK_ID}"
rm -rf "$SC_WORKSPACE"
mkdir -p "$SC_WORKSPACE"
SEED_CACHE="$REPO/workspace_backup_20261014/sc_metric_cache_model_oneclass_10054514482153597817_3.10.20.bkp"
if [ -f "$SEED_CACHE" ]; then
    cp -p "$SEED_CACHE" "$SC_WORKSPACE/"
    echo "  seeded pooled cache: $(basename "$SEED_CACHE")"
else
    echo "  !! SEED CACHE MISSING -- pooled model would refit and these rows" \
         "would not be comparable to the published alpha/beta. Refusing." >&2
    exit 1
fi

D="eval/report_runs/$EXP/$DATANAME"
echo "[$(date)] $EXP on $(hostname): $(ls "$D"/all_samples/*.csv | wc -l) staged samples"
nvidia-smi -L

# Fail in 2s with a legible message if this node's arch is unsupported, rather
# than 40 lines deep inside synthcity's unpickler.
python - <<'SMOKE'
import sys, torch
assert torch.cuda.is_available(), "no GPU visible -- the oneclass cache cannot be loaded CPU-only"
try:
    (torch.ones(8, 8, device="cuda") @ torch.ones(8, 8, device="cuda")).cpu()
except RuntimeError as e:
    sys.exit(f"GPU present but unusable by this torch ({torch.__version__}) "
             f"on {torch.cuda.get_device_name(0)}: {e}")
print(f"gpu ok: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
SMOKE

# eval_quality.py APPENDS to avg_std.csv rather than creating it, so a reread of
# a file that already holds rows would double them. Stage A left a header-only
# file on every one of these configs (it writes the header, then stage B failed),
# which is exactly the state this needs -- but reset it anyway so a re-run of
# this script is idempotent.
printf ',avg,std\n' > "$D/avg_std.csv"

python -u eval/eval_quality.py --dataname "$DATANAME" --exp_name "$EXP"

echo "[$(date)] Done: $EXP"
cat "$D/avg_std.csv"
