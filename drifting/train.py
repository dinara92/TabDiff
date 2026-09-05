"""Train the one-step drift generator on a TabDiff dataset.

  python -m drifting.train --dataname adult --steps 20000

Reuses TabDiff's preprocessing, inverse transforms and metric harness, so the
numbers land on the same scale as the multi-step baseline. TabDiff itself is
never imported as a model -- only its dataset code, its column-token
transformer (via drifting.generator) and its evaluator.
"""

import argparse
import json
import os
import time
from copy import deepcopy
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from drifting.diagnostics import check, distance_report, suggest_R_list
from drifting.drift_loss import drift_loss
from drifting.features import ReMaskerFeatures, WhitenedIdentity
from drifting.generator import DriftGenerator
from tabdiff.metrics import TabMetrics
from tabdiff.trainer import recover_data, split_num_cat_target
from utils_train import TabDiffDataset


def to_onehot(x_cat: torch.Tensor, categories: List[int]) -> torch.Tensor:
    return torch.cat(
        [F.one_hot(x_cat[:, i], num_classes=k).float() for i, k in enumerate(categories)],
        dim=-1,
    )


def drift_objective(phi, gen_num, gen_oh, pos_num, pos_oh, R_list, report=False):
    """Sum the per-token drift losses over every block phi returns.

    phi gives ``[N, F, D]``; the token axis F is folded into the kernel's group
    axis, so each column (or each row token) gets its own kernel in its own
    embedding space -- the same fold the reference applies to patch tokens.
    """
    fg, fp = phi(gen_num, gen_oh), phi(pos_num, pos_oh)
    total, info = 0.0, {}
    for key in fg:
        g = fg[key].transpose(0, 1).contiguous()   # [F, C_g, D]
        p = fp[key].transpose(0, 1).contiguous()   # [F, C_p, D]
        loss, i = drift_loss(g, p, R_list=R_list)
        total = total + loss.mean()
        if report:
            info[key] = distance_report(g[:1], p[:1], R_list)
    return total, info


@torch.no_grad()
def synth_dataframe(gen, n, device, info, ds, groups=None):
    x_num, x_cat = gen.sample(n, device, groups=groups)
    syn = torch.cat([x_num, x_cat.float()], dim=1).numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(
        syn, info, ds.num_inverse, ds.int_inverse, ds.cat_inverse
    )
    df = recover_data(syn_num, syn_cat, syn_target, info)
    mapping = {int(k): v for k, v in info['idx_name_mapping'].items()}
    return df.rename(columns=mapping)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataname", default="adult")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--c_gen", type=int, default=512, help="generated particles per step")
    ap.add_argument("--c_pos", type=int, default=1024, help="real particles per step")
    ap.add_argument("--R", type=float, nargs="+", default=None,
                    help="kernel bandwidths; default is calibrated at step 0")
    # Optimiser settings follow the reference recipe (configs/gen/latent_sota_B.yaml):
    # lr 4e-4, linear warmup then constant, Adam b2=0.95, EMA 0.999.
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=10000)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--d_token", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--d_z", type=int, default=128)
    ap.add_argument("--group_by", choices=["none", "target"], default="none")
    ap.add_argument("--phi", choices=["whitened", "remasker_flat", "remasker_percol"],
                    default="whitened")
    ap.add_argument("--remasker_ckpt", default=None,
                    help="defaults to remasker/ckpt/<dataname>/model.pt")
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--final_metrics", nargs="*", default=["density"],
                    help="metric suite to run once at the end, e.g. density mle c2st")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = a.out or f"drifting/result/{a.dataname}/{a.phi}"
    os.makedirs(out, exist_ok=True)

    # ---- data ------------------------------------------------------------
    info = json.load(open(f"data/{a.dataname}/info.json"))
    ds = TabDiffDataset(a.dataname, f"data/{a.dataname}", info, isTrain=True,
                        dequant_dist="none", int_dequant_factor=0)
    d_num = ds.d_numerical
    cats = ds.categories.tolist()
    x_num = ds.X[:, :d_num].float().to(dev)
    x_cat = ds.X[:, d_num:].long().to(dev)
    x_oh = to_onehot(x_cat, cats).to(dev)
    n_rows = x_num.shape[0]
    print(f"{a.dataname}: {n_rows} rows, {d_num} numerical, {len(cats)} categorical {cats}")

    # The reference keeps a ring buffer because images stream from a loader and
    # re-encoding them is expensive. A table fits in memory, so the positive
    # pool is the whole training set -- strictly more diverse, and less
    # memorisation pressure than a small bank.
    if a.group_by == "target":
        # binclass/multiclass put the target at categorical column 0.
        assert info["task_type"] != "regression", "regression grouping needs quantile bins"
        group_of = x_cat[:, 0]
        num_groups = int(group_of.max().item()) + 1
        pools = [torch.nonzero(group_of == g, as_tuple=True)[0] for g in range(num_groups)]
        prior = torch.tensor([len(p) for p in pools], dtype=torch.float, device=dev)
        prior = prior / prior.sum()
        print(f"grouping by target: {num_groups} groups, sizes {[len(p) for p in pools]}")
    else:
        num_groups = 1
        pools = [torch.arange(n_rows, device=dev)]
        prior = torch.ones(1, device=dev)

    # ---- phi -------------------------------------------------------------
    if a.phi == "whitened":
        phi = WhitenedIdentity(cats).fit(x_num, x_oh).to(dev)
    else:
        ckpt = a.remasker_ckpt or f"remasker/ckpt/{a.dataname}/model.pt"
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"{ckpt} missing -- pretrain it with remasker/pretrain.py")
        readout = "flat" if a.phi == "remasker_flat" else "per_column"
        phi = ReMaskerFeatures(ckpt, readout=readout, device=dev).to(dev)
        assert phi.d_numerical == d_num and phi.categories == cats, (
            f"checkpoint schema {phi.d_numerical}/{phi.categories} does not match "
            f"dataset {d_num}/{cats}")
        print(f"phi = remasker[{readout}] {phi.num_tokens} tokens x {phi.embed_dim}")

    # ---- generator -------------------------------------------------------
    gen = DriftGenerator(d_num, cats, num_layers=a.num_layers, d_token=a.d_token,
                         n_head=a.n_head, d_z=a.d_z, tau=a.tau,
                         num_groups=num_groups).to(dev)
    n_par = sum(p.numel() for p in gen.parameters())
    print(f"generator: {n_par/1e6:.2f}M params, {num_groups} group(s), 1 NFE")
    opt = torch.optim.Adam(gen.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def lr_at(step):
        return a.lr * min(1.0, step / max(1, a.warmup))

    # EMA weights. TabDiff's own best adult result comes from its EMA model, and
    # the reference uses 0.999 -- evaluating the raw weights understates the
    # method.
    ema = deepcopy(gen).eval()
    for q in ema.parameters():
        q.requires_grad_(False)

    @torch.no_grad()
    def ema_update():
        d = a.ema_decay
        for qe, qg in zip(ema.parameters(), gen.parameters()):
            qe.mul_(d).add_(qg.detach(), alpha=1 - d)
        for be, bg in zip(ema.buffers(), gen.buffers()):
            be.copy_(bg)

    def draw_pos(gids, c):
        idx = torch.cat([pools[int(g)][torch.randint(0, len(pools[int(g)]), (c,), device=dev)]
                         for g in gids])
        return x_num[idx], x_oh[idx]

    # ---- bandwidth calibration ------------------------------------------
    R_list = tuple(a.R) if a.R else None
    if R_list is None:
        gids = torch.arange(num_groups)
        pn, po = draw_pos(gids, a.c_pos)
        with torch.no_grad():
            gn, go, _ = gen(a.c_gen * num_groups,
                            groups=gids.repeat_interleave(a.c_gen).to(dev) if num_groups > 1 else None)
            fg, fp = phi(gn, go), phi(pn, po)
        key = next(iter(fg))
        g1 = fg[key].transpose(0, 1)[:1]
        p1 = fp[key].transpose(0, 1)[:1]
        rep = distance_report(g1, p1, (0.02, 0.05, 0.2))
        R_list = tuple(suggest_R_list(g1, p1))
        print(f"calibrated R_list = {R_list}  (gen->pos p01={rep['gen_pos_p01']:.3f}, "
              f"p50={rep['gen_pos_p50']:.3f}, gen-gen p01={rep['gen_gen_p01']:.4f})")
        check(distance_report(g1, p1, R_list))

    # ---- metrics ---------------------------------------------------------
    metrics = TabMetrics(f"synthetic/{a.dataname}/real.csv",
                         f"synthetic/{a.dataname}/test.csv", None, info, dev,
                         metric_list=["density"])
    real_df = pd.read_csv(f"synthetic/{a.dataname}/real.csv")
    n_eval = len(real_df)

    def eval_now(model, tag):
        model.eval()
        g = None
        if num_groups > 1:
            g = torch.multinomial(prior, n_eval, replacement=True).cpu()
        df = synth_dataframe(model, n_eval, dev, info, ds, groups=g)
        path = os.path.join(out, f"samples_{tag}.csv")
        df.to_csv(path, index=False)
        m, _ = metrics.evaluate(pd.read_csv(path))
        model.train()
        return m

    # ---- train -----------------------------------------------------------
    print(f"training {a.steps} steps, C_g={a.c_gen} C_p={a.c_pos} R={R_list} "
          f"lr={a.lr} warmup={a.warmup} ema={a.ema_decay}")
    hist = []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        gids = torch.arange(num_groups)
        pn, po = draw_pos(gids, a.c_pos)
        gcond = gids.repeat_interleave(a.c_gen).to(dev) if num_groups > 1 else None
        gn, go, _ = gen(a.c_gen * num_groups, groups=gcond)

        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        loss, _ = drift_objective(phi, gn, go, pn, po, R_list)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 2.0)
        opt.step()
        ema_update()

        if step % 500 == 0:
            print(f"  step {step:6d}  loss {loss.item():7.3f}  "
                  f"{(time.time()-t0)/step*1000:.0f} ms/step", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            m_raw = eval_now(gen, "raw")
            m_ema = eval_now(ema, "ema")
            hist.append({"step": step,
                         **{f"raw/{k}": v for k, v in m_raw.items()},
                         **{f"ema/{k}": v for k, v in m_ema.items()}})
            print(f"  step {step:6d}  EVAL  "
                  f"raw {m_raw['density/Overall']:.4f}  ema {m_ema['density/Overall']:.4f}  "
                  f"(shape {m_ema['density/Shape']:.4f} trend {m_ema['density/Trend']:.4f})",
                  flush=True)
            pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
            torch.save({"gen": gen.state_dict(), "ema": ema.state_dict()},
                       os.path.join(out, "generator.pt"))

    # Final evaluation on the full suite -- mle and c2st are far too slow to
    # run every eval, but they are what the TabDiff baseline reports.
    if a.final_metrics != ["density"]:
        print(f"\nfinal evaluation: {a.final_metrics}", flush=True)
        full = TabMetrics(f"synthetic/{a.dataname}/real.csv",
                          f"synthetic/{a.dataname}/test.csv", None, info, dev,
                          metric_list=list(a.final_metrics))
        summary = {}
        for tag, model in (("raw", gen), ("ema", ema)):
            model.eval()
            g = torch.multinomial(prior, n_eval, replacement=True).cpu() if num_groups > 1 else None
            df = synth_dataframe(model, n_eval, dev, info, ds, groups=g)
            path = os.path.join(out, f"final_{tag}.csv")
            df.to_csv(path, index=False)
            m, _ = full.evaluate(pd.read_csv(path))
            summary.update({f"{tag}/{k}": v for k, v in m.items()})
            print(f"  {tag}: " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()), flush=True)
        with open(os.path.join(out, "final_metrics.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {out}")
    if hist:
        best = max(hist, key=lambda h: h.get("ema/density/Overall", 0))
        print("best:", {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in best.items()})


if __name__ == "__main__":
    main()
