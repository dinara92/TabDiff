"""Emit the markdown for the report's sampling-efficiency table.

Three sources, because no single job produces all seven columns:

  bench_speed_b<batch>.csv  timings          (slurm 145594, unshared GPU)
  bench_quality.csv         Shape/Trend/MLE/C2ST, 20-draw means
  eval/report_runs/bench_<slug>/<data>/avg_std.csv   alpha/beta, synthcity env

Timings and quality come from different runs on purpose: the timing job uses a
single draw so the clock is not polluted by metric evaluation, and the quality
job re-draws 20 times so its numbers are comparable to the report's other
tables. Do not merge the two jobs.

  python -m drifting.collect_bench --dataname adult
"""

import argparse
import os

import pandas as pd


def slug(name):
    return (name.lower().replace(" (ours)", "").replace(" steps", "")
            .replace(" ", "_"))


def alpha_beta(dataname, name, prefix="bench"):
    """alpha/beta land in a per-slug avg_std.csv written by eval_quality.py."""
    p = f"eval/report_runs/{prefix}_{slug(name)}/{dataname}/avg_std.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, index_col=0)
    if "alpha" not in d.index or "beta" not in d.index:
        return None
    return {k: (d.loc[k, "avg"], d.loc[k, "std"]) for k in ("alpha", "beta")}


def cell(mean, std, prec, pm=True):
    if mean is None or pd.isna(mean):
        return "—"
    if not pm or std is None or pd.isna(std) or std == 0:
        return f"{mean:.{prec}f}"
    return f"{mean:.{prec}f} ±{std:.{prec}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataname", default="adult")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--prefix", default="bench",
                    help="staging namespace the alpha/beta rows were written under")
    ap.add_argument("--source", default=None,
                    help="a single bench CSV carrying both timing and quality "
                         "columns (the low-NFE sweep). Default joins the "
                         "separate speed and quality CSVs instead")
    a = ap.parse_args()

    base = f"drifting/result/{a.dataname}"
    speed_path = f"{base}/bench_speed_b{a.batch}.csv"
    qual_path = f"{base}/bench_quality.csv"
    if not a.source and not os.path.exists(speed_path):
        raise SystemExit(f"{speed_path} missing -- run slurm/bench_speed.sbatch")
    if a.source:
        speed = pd.read_csv(a.source).set_index("method")
        qual = speed
    else:
        speed = pd.read_csv(speed_path).set_index("method")
        qual = (pd.read_csv(qual_path).set_index("method")
                if os.path.exists(qual_path) else None)
    if qual is None:
        print(f"note: {qual_path} missing, quality columns will be blank\n")

    runs = int(qual["quality_runs"].iloc[0]) if qual is not None else 0
    print(f"Batch {a.batch}, {int(speed['n'].iloc[0])} rows. Quality columns are "
          f"{runs}-draw means ±std.\n")
    print("| Method | NFE | gen (s) | rows/s | Shape ↓ | Trend ↓ | MLE ↑ | "
          "C2ST ↑ | α-prec ↑ | β-recall ↑ |")
    print("|---|---|---|---|---|---|---|---|---|---|")

    for name, s in speed.iterrows():
        q = qual.loc[name] if qual is not None and name in qual.index else None
        ab = alpha_beta(a.dataname, name, a.prefix)
        g = lambda k: (None if q is None or k not in q else q[k])
        cells = [
            f"**{name}**" if name.startswith("Drift") else name,
            f"{int(s['nfe'])}",
            cell(s["gen_s"], s["gen_std"], 3),
            f"{s['rows_per_s']:,.0f}",
            cell(g("shape_err"), g("shape_err_std"), 2),
            cell(g("trend_err"), g("trend_err_std"), 2),
            cell(g("mle"), g("mle_std"), 4),
            cell(g("c2st"), g("c2st_std"), 4),
            cell(*(ab["alpha"] if ab else (None, None)), 2),
            cell(*(ab["beta"] if ab else (None, None)), 2),
        ]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
