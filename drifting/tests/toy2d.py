"""End-to-end gate for the drift loop on 2D data, before any tabular wiring.

Isolates JAX->PyTorch translation and training-loop bugs from data issues.
The loss value is uninformative by construction (each bandwidth is normalised
to unit RMS, so it sits at O(len(R_list)) forever) -- mode coverage and the
symmetric nearest-neighbour distance are the signals.

  python -m drifting.tests.toy2d --steps 3000
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

from drifting.diagnostics import check, distance_report
from drifting.drift_loss import drift_loss


def eight_gaussians(n, std=0.08, radius=2.0, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    ang = torch.arange(8, dtype=torch.float32) * (2 * np.pi / 8)
    centers = torch.stack([ang.cos(), ang.sin()], -1) * radius
    idx = torch.randint(0, 8, (n,), generator=g)
    return centers[idx] + torch.randn(n, 2, generator=g) * std, centers


class MLPGen(nn.Module):
    def __init__(self, d_z=16, h=128):
        super().__init__()
        self.d_z = d_z
        self.net = nn.Sequential(
            nn.Linear(d_z, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2),
        )

    def forward(self, n, device):
        return self.net(torch.randn(n, self.d_z, device=device))


def evaluate(samples, centers, real, std=0.08):
    d = torch.cdist(samples, centers)
    nearest = d.argmin(1)
    counts = torch.bincount(nearest, minlength=8).float() / len(samples)
    covered = int((counts > 0.01).sum())
    captured = float((d.min(1).values < 4 * std).float().mean())
    # Symmetric NN distance: low in both directions means matched supports.
    g2r = torch.cdist(samples, real).min(1).values.mean()
    r2g = torch.cdist(real, samples).min(1).values.mean()
    return dict(modes=covered, captured=captured, g2r=float(g2r), r2g=float(r2g),
                max_mode_share=float(counts.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--c_gen", type=int, default=256)
    ap.add_argument("--c_pos", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # NOT the paper default (0.02, 0.05, 0.2): those bandwidths sit far below
    # the gen->data gap of a freshly initialised generator, so attraction
    # underflows and training silently does nothing. See drifting/diagnostics.py.
    ap.add_argument("--R", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = a.device
    data, centers = eight_gaussians(20000, seed=a.seed)
    data, centers = data.to(dev), centers.to(dev)

    gen = MLPGen().to(dev)
    opt = torch.optim.Adam(gen.parameters(), lr=a.lr)

    for step in range(a.steps + 1):
        pos = data[torch.randint(0, len(data), (a.c_pos,), device=dev)][None]
        x = gen(a.c_gen, dev)[None]
        if step == 0:
            check(distance_report(x.detach(), pos, tuple(a.R)))
        loss, info = drift_loss(x, pos, R_list=tuple(a.R))
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 2.0)
        opt.step()

        if step % 500 == 0:
            with torch.no_grad():
                s = gen(4000, dev)
            m = evaluate(s, centers, data[:4000])
            print(f"step {step:5d}  loss {loss.item():6.3f}  scale {info['scale']:.3f}  "
                  f"modes {m['modes']}/8  captured {m['captured']:.3f}  "
                  f"gen->real {m['g2r']:.4f}  real->gen {m['r2g']:.4f}  "
                  f"max_share {m['max_mode_share']:.3f}")

    with torch.no_grad():
        s = gen(8000, dev)
    m = evaluate(s, centers, data[:8000])
    ok = m["modes"] == 8 and m["captured"] > 0.9 and m["r2g"] < 0.15
    print(f"\nFINAL {m}")
    print("GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
