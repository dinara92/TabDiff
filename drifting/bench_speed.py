"""Sampling latency for the 1-NFE generator against multi-step TabDiff.

Reproduces the protocol of TabDiff's Appendix F.1 Table 11 -- time to generate
the same number of rows as the real table (32561 for adult) -- but on this
machine, with both models in the same process, on the same GPU, at a matched
batch size. The paper's own numbers (TabDDPM 125.1s, TabSyn 8.8s, TabDiff 15.2s)
are from different hardware and are quoted, never re-derived, in the report.

Two caveats the numbers are meaningless without:

  * Batch size dominates. Both samplers loop over batches, so a comparison at
    different batch sizes measures the batch size. `--batch` applies to both.
  * TabDiff's cost is linear in discretization steps, and it stays coherent at
    far fewer than its default 50 (paper Table 12). So a single speedup ratio
    against the default overstates the case; `--steps` sweeps the whole
    quality-latency frontier and `--quality` scores each point.

NFE here is TOTAL network calls per row, not the paper's per-step count:
TabDiff at 50 steps with second-order correction is 100, ours is 1.

  python -m drifting.bench_speed --dataname adult --quality
"""

import argparse
import contextlib
import io
import json
import os
import pickle
import statistics
import time

import numpy as np
import pandas as pd
import torch


from drifting.generator import DriftGenerator
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.modules.main_modules import Model, UniModMLP
from utils_train import TabDiffDataset
from tabdiff.metrics import TabMetrics
from tabdiff.trainer import recover_data, split_num_cat_target


def timed(fn, repeats, warmup, device):
    """Wall-clock over `repeats` calls, after `warmup` untimed ones.

    Every call is bracketed by a device sync: CUDA kernels are async, so
    without this the timer measures queue submission, not execution.
    """
    for _ in range(warmup):
        fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return out


def load_tabdiff(dataname, ckpt_dir, d_num, categories, device):
    """Rebuild the diffusion model from its cached config, as tabdiff/main.py
    does in test mode, and install the EMA weights the reported numbers use."""
    cfg = pickle.load(open(os.path.join(ckpt_dir, "config.pkl"), "rb"))
    ckpt = [f for f in os.listdir(ckpt_dir) if f.startswith("best_ema_model")]
    assert ckpt, f"no best_ema_model_* in {ckpt_dir}"
    backbone = UniModMLP(**cfg["unimodmlp_params"])
    model = Model(backbone, **cfg["diffusion_params"]["edm_params"]).to(device)
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories, num_numerical_features=d_num, denoise_fn=model,
        y_only_model=None, **cfg["diffusion_params"], device=device,
    ).to(device)
    state = torch.load(os.path.join(ckpt_dir, ckpt[0]), map_location=device)
    diffusion._denoise_fn.load_state_dict(state["denoise_fn"])
    diffusion.num_schedule.load_state_dict(state["num_schedule"])
    diffusion.cat_schedule.load_state_dict(state["cat_schedule"])
    diffusion.eval()
    return diffusion, cfg


def load_drift(path, d_num, categories, device, weights="gen"):
    gen = DriftGenerator(d_num, categories).to(device)
    ck = torch.load(path, map_location=device)
    gen.load_state_dict(ck[weights])
    gen.eval()
    return gen


def decode(syn, info, ds):
    """Tensor -> dataframe. Shared by both models, so it is a constant offset
    rather than a difference -- but the paper's Sample.T includes it."""
    syn_num, syn_cat, syn_target = split_num_cat_target(
        syn, info, ds.num_inverse, ds.int_inverse, ds.cat_inverse)
    df = recover_data(syn_num, syn_cat, syn_target, info)
    return df.rename(columns={int(k): v for k, v in info["idx_name_mapping"].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataname", default="adult")
    ap.add_argument("--n", type=int, default=None,
                    help="rows to generate; default is the real table size, "
                         "which is what the paper times")
    ap.add_argument("--batch", type=int, default=4096,
                    help="sampling batch size, applied to BOTH models")
    ap.add_argument("--steps", type=int, nargs="+",
                    default=[5, 10, 25, 50, 100],
                    help="TabDiff discretization steps to sweep")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--tabdiff_ckpt", default=None)
    ap.add_argument("--drift_ckpt", default=None)
    ap.add_argument("--quality", action="store_true",
                    help="also score Shape/Trend at each point, for the "
                         "latency-vs-quality frontier")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)

    info = json.load(open(f"data/{a.dataname}/info.json"))
    ds = TabDiffDataset(a.dataname, f"data/{a.dataname}", info, isTrain=True,
                        dequant_dist="none", int_dequant_factor=0)
    d_num, categories = ds.d_numerical, ds.categories
    n = a.n or len(pd.read_csv(f"synthetic/{a.dataname}/real.csv"))

    tabdiff_ckpt = a.tabdiff_ckpt or f"tabdiff/ckpt/{a.dataname}/learnable_schedule"
    drift_ckpt = a.drift_ckpt or f"drifting/result/{a.dataname}/long_whitened_4x/generator.pt"

    print(f"device: {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    print(f"{a.dataname}: generating {n} rows, batch {a.batch}, "
          f"{a.repeats} timed repeats after {a.warmup} warmup")

    diffusion, cfg = load_tabdiff(a.dataname, tabdiff_ckpt, d_num,
                                  categories, dev)
    gen = load_drift(drift_ckpt, d_num, categories.tolist(), dev)
    print(f"tabdiff: {sum(p.numel() for p in diffusion.parameters())/1e6:.2f}M params, "
          f"default {cfg['diffusion_params']['num_timesteps']} steps, "
          f"second_order={cfg['diffusion_params']['sampler_params']['second_order_correction']}")
    print(f"drift:   {sum(p.numel() for p in gen.parameters())/1e6:.2f}M params, 1 NFE")

    # tqdm's per-step redraw is real time inside the loop being measured, and
    # it is TabDiff-only, so it would show up as a difference. Silence it.
    import tabdiff.models.unified_ctime_diffusion as ucd

    class _NoBar:
        def __init__(self, it, *args, **kw):
            self.it = it

        def __iter__(self):
            return iter(self.it)

        def set_description(self, *args, **kw):
            pass

    ucd.tqdm = _NoBar

    metrics = None
    if a.quality:
        val = f"synthetic/{a.dataname}/val.csv"
        metrics = TabMetrics(f"synthetic/{a.dataname}/real.csv",
                             f"synthetic/{a.dataname}/test.csv",
                             val if os.path.exists(val) else None,
                             info, dev, metric_list=["density"])

    rows = []

    def record(name, nfe, sample_fn):
        holder = {}

        def once():
            holder["syn"] = sample_fn()

        with contextlib.redirect_stdout(io.StringIO()):
            gen_times = timed(once, a.repeats, a.warmup, dev)
            dec_times = timed(lambda: decode(holder["syn"], info, ds),
                              max(1, a.repeats // 2), 1, dev)
        gen_s = statistics.mean(gen_times)
        dec_s = statistics.mean(dec_times)
        sd = statistics.stdev(gen_times) if len(gen_times) > 1 else 0.0
        row = dict(method=name, nfe=nfe, n=n, batch=a.batch,
                   gen_s=gen_s, gen_std=sd, decode_s=dec_s,
                   total_s=gen_s + dec_s, rows_per_s=n / (gen_s + dec_s))
        if metrics is not None:
            with contextlib.redirect_stdout(io.StringIO()):
                m, _ = metrics.evaluate(decode(holder["syn"], info, ds))
            # The report's tables are error rates; the metric returns scores.
            row["shape_err"] = (1 - m["density/Shape"]) * 100
            row["trend_err"] = (1 - m["density/Trend"]) * 100
        rows.append(row)
        q = (f"  Shape {row['shape_err']:.2f}  Trend {row['trend_err']:.2f}"
             if metrics is not None else "")
        print(f"{name:<22} NFE {nfe:>4}  gen {gen_s:7.3f}s ±{sd:.3f}  "
              f"+decode {dec_s:5.3f}s  = {row['total_s']:7.3f}s  "
              f"{row['rows_per_s']:9.0f} rows/s{q}", flush=True)

    with torch.no_grad():
        second_order = cfg["diffusion_params"]["sampler_params"]["second_order_correction"]
        for s in a.steps:
            diffusion.num_timesteps = s
            record(f"TabDiff {s} steps", s * (2 if second_order else 1),
                   lambda: diffusion.sample_all(n, a.batch, keep_nan_samples=True))
        record("Drift (ours)", 1,
               lambda: torch.cat(
                   [t.float() for t in gen.sample(n, dev, batch_size=a.batch)],
                   dim=1))

    df = pd.DataFrame(rows)
    out = a.out or f"drifting/result/{a.dataname}/bench_speed.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    base = df[df.method == f"TabDiff {cfg['diffusion_params']['num_timesteps']} steps"]
    ours = df[df.method == "Drift (ours)"]
    if len(base) and len(ours):
        print(f"speedup vs TabDiff at its default "
              f"{cfg['diffusion_params']['num_timesteps']} steps: "
              f"{base.gen_s.iloc[0] / ours.gen_s.iloc[0]:.1f}x generation, "
              f"{base.total_s.iloc[0] / ours.total_s.iloc[0]:.1f}x including decode")


if __name__ == "__main__":
    main()
