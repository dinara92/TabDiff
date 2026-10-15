#!/bin/bash
#SBATCH --job-name=cache-refit-ctl
#SBATCH --partition=ada,a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%A_%a.out
set -euo pipefail

# Is synthcity's cached one-class model safe to delete?
#
# alpha-precision is built on a one-class model that synthcity fits once and
# caches under AlphaPrecision(workspace=...), keyed on a hash of the real data.
# Ours was fitted 2026-09-06 and every alpha/beta number in reports/ was scored
# against that one model. The fit is UNSEEDED -- OneClassLayer takes no seed and
# Metric._random_state is never passed to it -- so deleting the cache does not
# reproduce it, it refits a different model.
#
# The open question is whether that MATTERS, i.e. whether alpha/beta is stable
# across refits. If it is, the cache is a convenience and can be deleted with
# the published numbers left standing. If it is not, the cache is load-bearing:
# either it is kept, or every alpha/beta row is rescored together.
#
# This measures it. Three array tasks, each with its OWN empty SC_WORKSPACE, so
# each forces an independent fresh fit and none of them races the others on the
# cache path. All three score the SAME 20 staged draws -- cachectl_fit* symlink
# bench_drift's samples -- so the only difference between the three results, and
# between them and the published 98.80 / 34.32, is the one-class fit.
#
# Nothing here touches workspace/ or bench_drift/avg_std.csv. A backup of the
# Sep 6 cache is at workspace_backup_20261014/ regardless.
#
# Usage: sbatch --export=DATANAME=adult slurm/cache_refit_control.sbatch

DATANAME="${DATANAME:-adult}"
EXP="cachectl_fit${SLURM_ARRAY_TASK_ID}"

REPO=/playpen-shared/dinara/TabDiff
source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
conda activate synthcity
cd "$REPO"
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8

# Per-task, and empty -> guaranteed cache miss -> fresh fit.
export SC_WORKSPACE="$REPO/workspace_cachectl/fit${SLURM_ARRAY_TASK_ID}"
rm -rf "$SC_WORKSPACE"
mkdir -p "$SC_WORKSPACE"

D="eval/report_runs/$EXP/$DATANAME"
echo "[$(date)] $EXP on $(hostname)"
echo "  SC_WORKSPACE=$SC_WORKSPACE (empty: $(ls -A "$SC_WORKSPACE" | wc -l) files)"
echo "  samples: $(ls "$D"/all_samples/*.csv | wc -l)"
nvidia-smi -L

python - <<'SMOKE'
import sys, torch
assert torch.cuda.is_available(), "no GPU visible"
try:
    (torch.ones(8, 8, device="cuda") @ torch.ones(8, 8, device="cuda")).cpu()
except RuntimeError as e:
    sys.exit(f"GPU unusable by torch {torch.__version__} on "
             f"{torch.cuda.get_device_name(0)}: {e}")
print(f"gpu ok: {torch.cuda.get_device_name(0)}")
SMOKE

printf ',avg,std\n' > "$D/avg_std.csv"
python -u eval/eval_quality.py --dataname "$DATANAME" --exp_name "$EXP"

echo "[$(date)] fresh-fit result for $EXP (published reference: alpha 98.80, beta 34.32)"
cat "$D/avg_std.csv"
echo "  cache written by this fit:"
ls -la "$SC_WORKSPACE"/sc_metric_cache_model_oneclass_* 2>/dev/null || echo "  (none -- unexpected)"
