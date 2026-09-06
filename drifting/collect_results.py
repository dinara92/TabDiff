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
    # Runs request different metric suites (the DCR arms ask for `dcr density`
    # only), so every field is optional -- a missing key means that metric was
    # not requested, not that it failed.
    out = {}
    for name, key in (("shape", "Shape"), ("trend", "Trend")):
        if f"{tag}/density/{key}" in d:
            out[name] = ((1 - d[f"{tag}/density/{key}"]) * 100,
                         d[f"{tag}/density/{key}/std"] * 100)
    for name in ("mle", "c2st"):
        if f"{tag}/{name}" in d:
            out[name] = (d[f"{tag}/{name}"], d[f"{tag}/{name}/std"])
    if f"{tag}/dcr" in d:
        out["dcr"] = (d[f"{tag}/dcr"] * 100, d[f"{tag}/dcr/std"] * 100)
    return out


def row(name, r, nfe, source, dcr=True):
    n = 6 if dcr else 5
    if r is None:
        return f"| {name} |" + " — |" * n + f" {nfe} | not yet run |"
    c = lambda k, p=3: (f"{r[k][0]:.{p}f} ±{r[k][1]:.{p}f}" if k in r else "—")
    cells = [c('shape'), c('trend'), c('mle', 4), c('c2st', 4)]
    if dcr:
        cells.append(c('dcr', 2))
    return f"| {name} | " + " | ".join(cells) + f" | {nfe} | {source} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataname", default="adult")
    a = ap.parse_args()
    d = a.dataname

    # Two tables, because DCR needs an equal 50/50 split and therefore a
    # separately trained model. Merging a DCR value into a 90/10 row would put
    # two different models on one line.
    print("## 90/10 split\n")
    print("| Method | Shape ↓ | Trend ↓ | MLE ↑ | C2ST ↑ | NFE | source |")
    print("|---|---|---|---|---|---|---|")
    print(row("TabDiff — our repro",
              tabdiff_repro(f"eval/report_runs/learnable_schedule/{d}/all_results.csv"),
              "~100", "ours", dcr=False))
    for tag, label in [("long_whitened_4x", "Drift 4x whitened"),
                       ("long_whitened_2x", "Drift 2x whitened")]:
        print(row(label, drift(f"drifting/result/{d}/{tag}/final_metrics.json"),
                  "1", "ours", dcr=False))

    print("\n## 50/50 split (DCR)\n")
    print("| Method | DCR →50% | NFE | source |")
    print("|---|---|---|---|")
    for tag, label in [("whitened_4x", "Drift 4x whitened"),
                       ("remasker_flat_4x", "Drift 4x remasker"),
                       ("remasker_flat_randinit_4x", "Drift 4x remasker rand-init")]:
        r = drift(f"drifting/result/{d}_dcr/{tag}/final_metrics.json")
        v = f"{r['dcr'][0]:.2f} ±{r['dcr'][1]:.2f}" if r and "dcr" in r else "not yet run"
        print(f"| {label} | {v} | 1 | ours |")


if __name__ == "__main__":
    main()
