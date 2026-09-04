"""Numerically pin drifting/drift_loss.py against the JAX reference.

Stage 1 (jax env):   python drifting/tests/crosscheck_jax.py --emit  <out.npz>
Stage 2 (tabdiff):   python drifting/tests/crosscheck_jax.py --check <out.npz>
"""
import argparse
import os
import sys

import numpy as np

REF_DIR = "/playpen-shared/dinara/drifting"
CASES = [
    dict(name="uncond", B=2, C_g=8, C_p=16, C_n=0, S=5, seed=0),
    dict(name="with_neg", B=2, C_g=8, C_p=16, C_n=4, S=5, seed=1),
    dict(name="single_gen", B=1, C_g=1, C_p=16, C_n=0, S=3, seed=2),
    dict(name="wide", B=1, C_g=32, C_p=64, C_n=0, S=17, seed=3),
]
R_LIST = (0.02, 0.05, 0.2)


def make_inputs(case):
    rng = np.random.default_rng(case["seed"])
    gen = rng.normal(size=(case["B"], case["C_g"], case["S"])).astype(np.float32)
    pos = (rng.normal(size=(case["B"], case["C_p"], case["S"])) + 2.0).astype(np.float32)
    neg = (
        rng.normal(size=(case["B"], case["C_n"], case["S"])).astype(np.float32)
        if case["C_n"] else None
    )
    w_neg = (
        rng.uniform(0.2, 1.5, size=(case["B"], case["C_n"])).astype(np.float32)
        if case["C_n"] else None
    )
    return gen, pos, neg, w_neg


def emit(path):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, REF_DIR)
    import jax
    import jax.numpy as jnp
    from drift_loss import drift_loss as jax_drift_loss

    out = {}
    for case in CASES:
        gen, pos, neg, w_neg = make_inputs(case)
        kw = dict(R_list=R_LIST)
        if neg is not None:
            kw.update(fixed_neg=jnp.asarray(neg), weight_neg=jnp.asarray(w_neg))
        loss, info = jax_drift_loss(jnp.asarray(gen), jnp.asarray(pos), **kw)
        grad = jax.grad(
            lambda g: jax_drift_loss(g, jnp.asarray(pos), **kw)[0].sum()
        )(jnp.asarray(gen))
        n = case["name"]
        out[f"{n}/loss"] = np.asarray(loss)
        out[f"{n}/grad"] = np.asarray(grad)
        out[f"{n}/scale"] = np.asarray(info["scale"])
        print(f"{n}: loss={np.asarray(loss)} scale={float(info['scale']):.5f}")
    np.savez(path, **out)
    print(f"wrote {path}")


def check(path):
    import torch

    sys.path.insert(0, "/playpen-shared/dinara/TabDiff")
    from drifting.drift_loss import drift_loss

    ref = np.load(path)
    failures = 0
    for case in CASES:
        n = case["name"]
        gen, pos, neg, w_neg = make_inputs(case)
        g = torch.tensor(gen, requires_grad=True)
        kw = dict(R_list=R_LIST)
        if neg is not None:
            kw.update(fixed_neg=torch.tensor(neg), weight_neg=torch.tensor(w_neg))
        loss, info = drift_loss(g, torch.tensor(pos), **kw)
        loss.sum().backward()

        for label, got, want, tol in [
            ("loss", loss.detach().numpy(), ref[f"{n}/loss"], 2e-4),
            ("grad", g.grad.numpy(), ref[f"{n}/grad"], 2e-4),
            ("scale", info["scale"].item(), float(ref[f"{n}/scale"]), 1e-5),
        ]:
            err = float(np.abs(np.asarray(got) - np.asarray(want)).max())
            ok = err < tol
            failures += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {n}/{label}  max|err|={err:.3e}")
    print(f"\n{failures} mismatch(es)")
    return failures


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit")
    ap.add_argument("--check")
    a = ap.parse_args()
    if a.emit:
        emit(a.emit)
    else:
        raise SystemExit(1 if check(a.check) else 0)
