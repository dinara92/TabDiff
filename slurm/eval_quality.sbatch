#!/bin/bash
#SBATCH --job-name=quality
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --output=/playpen-shared/dinara/TabDiff/slurm/logs/%x_%A_%a.out
set -euo pipefail

# alpha-precision and beta-recall (README "Additional Fidelity Metrics").
#
# These are not part of TabMetrics and never ran in any earlier job: they live
# in eval/eval_quality.py and need the separate `synthcity` conda env, which is
# why no run so far has produced them.
#
# eval_quality.py reads a fixed path -- eval/report_runs/<exp_name>/<dataname>/
# all_samples/*.csv -- so drift samples are staged into that layout under their
# own exp_name rather than the script being modified.
#
# Usage: sbatch --export=DATANAME=adult eval_quality.sbatch

DATANAME="${DATANAME:-adult}"

EXPS=(learnable_schedule drift_whitened_4x drift_whitened_2x)
SRCS=("" "drifting/result/$DATANAME/long_whitened_4x/final_samples"
         "drifting/result/$DATANAME/long_whitened_2x/final_samples")
EXP="${EXPS[$SLURM_ARRAY_TASK_ID]}"
SRC="${SRCS[$SLURM_ARRAY_TASK_ID]}"

REPO=/playpen-shared/dinara/TabDiff
source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
cd "$REPO"

DEST="eval/report_runs/$EXP/$DATANAME/all_samples"
if [ -n "$SRC" ]; then
    # Stage the 20 raw draws only -- one model, one protocol, 20 samples, so
    # this matches the baseline's all_samples rather than mixing raw and ema.
    if [ ! -d "$SRC" ]; then
        echo "[$(date)] ERROR: $SRC missing. Run the long training job first." >&2
        exit 1
    fi
    mkdir -p "$DEST"
    rm -f "$DEST"/*.csv
    for i in $(seq 0 19); do
        [ -f "$SRC/raw_$i.csv" ] && cp "$SRC/raw_$i.csv" "$DEST/samples_$i.csv"
    done
    # eval_quality.py appends its rows to an existing avg_std.csv in the parent.
    PARENT="eval/report_runs/$EXP/$DATANAME"
    [ -f "$PARENT/avg_std.csv" ] || echo ",avg,std" > "$PARENT/avg_std.csv"
fi

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID on $(hostname): exp=$EXP dataset=$DATANAME"
echo "[$(date)] $(ls "$DEST"/*.csv 2>/dev/null | wc -l) sample files in $DEST"

conda activate synthcity
python -u eval/eval_quality.py --dataname "$DATANAME" --exp_name "$EXP"

echo "[$(date)] Done: $EXP"
