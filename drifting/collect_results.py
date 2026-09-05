"""Emit the markdown rows for reports/baseline_comparison_adult.md from source.

Every "ours" number in that report is generated here rather than typed. The one
error that has already happened in this project was a transcription: a per-run
all_results.json was copied into the report instead of the 20-run average.

  python -m drifting.collect_results --dataname adult
"""

import argparse
import json
import os

import pandas as pd


def tabdiff_repro(path):
    """TabDiff's report_test output. shape/trend are ERROR percentages,
    c2st is a percentage; mle is already a score."""
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, index_col=0)
    f = lambda c, s=1.0: (d[c].mean() / s, d[c].std(ddof=1) / s)
    return dict(n=len(d), shape=f("shape"), trend=f("trend"),
                mle=f("mle"), c2st=f("c2st", 100.0))


def drift(path):
    """drifting/train.py --final_metrics output. density sub-scores are SCORES,
    so they are converted to error percentages to match the paper's tables."""
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    tag = "raw"
    err = lambda k: ((1 - d[f"{tag}/density/{k}"]) * 100, d[f"{tag}/density/{k}/std"] * 100)
    out = dict(shape=err("Shape"), trend=err("Trend"),
               mle=(d[f"{tag}/mle"], d[f"{tag}/mle/std"]),
               c2st=(d[f"{tag}/c2st"], d[f"{tag}/c2st/std"]))
    if f"{tag}/dcr" in d:
        out["dcr"] = (d[f"{tag}/dcr"] * 100, d[f"{tag}/dcr/std"] * 100)
    return out


def row(name, r, nfe, source):
    if r is None:
        return f"| {name} | — | — | — | — | — | {nfe} | not yet run |"
    c = lambda k, p=3: (f"{r[k][0]:.{p}f} ±{r[k][1]:.{p}f}" if k in r else "—")
    return (f"| {name} | {c('shape')} | {c('trend')} | {c('mle',4)} | "
            f"{c('c2st',4)} | {c('dcr',2)} | {nfe} | {source} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataname", default="adult")
    a = ap.parse_args()
    d = a.dataname

    print("| Method | Shape ↓ | Trend ↓ | MLE ↑ | C2ST ↑ | DCR →50% | NFE | source |")
    print("|---|---|---|---|---|---|---|---|")
    print(row("TabDiff — our repro",
              tabdiff_repro(f"eval/report_runs/learnable_schedule/{d}/all_results.csv"),
              "~100", "ours"))
    for tag, label in [("long_whitened_4x", "Drift 4x whitened"),
                       ("long_whitened_2x", "Drift 2x whitened")]:
        print(row(label, drift(f"drifting/result/{d}/{tag}/final_metrics.json"), "1", "ours"))
    for tag, label in [("whitened_4x", "Drift 4x whitened (dcr split)"),
                       ("remasker_flat_4x", "Drift 4x remasker (dcr split)"),
                       ("remasker_flat_randinit_4x", "Drift 4x remasker rand-init (dcr split)")]:
        print(row(label, drift(f"drifting/result/{d}_dcr/{tag}/final_metrics.json"), "1", "ours"))


if __name__ == "__main__":
    main()
