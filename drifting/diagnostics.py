"""Pre-flight checks on the kernel geometry.

The drift force is only informative where the kernel bandwidth ``R`` actually
resolves the distances that matter. ``R`` applies to distances normalised by
``scale`` (the mean pairwise distance), so it is dimensionless -- but that does
NOT make it transferable: it has to bracket the gen<->data gap for the data and
initialisation at hand.

Two failure modes this catches, both observed:

1. ``R`` too small for the gen->data gap. Every R in the paper's default
   (0.02, 0.05, 0.2) sat 20-50x below the normalised gen->data distance of a
   freshly initialised generator, so ``exp(-d/R)`` underflowed, attraction
   vanished, and the surviving "force" was numerical noise -- which the per-R
   RMS normalisation then rescaled to full size, so nothing looked wrong.
   Symptom: training does nothing while the loss sits at its usual value.

2. ``R`` too small for tabular ties. Real tables contain duplicate rows and
   low-cardinality columns, so the positive bank has genuine zero distances
   that a peaked softmax collapses onto. Does not arise for image latents.
"""

from typing import Dict, Optional, Sequence

import torch

from drifting.drift_loss import cdist


@torch.no_grad()
def distance_report(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    R_list: Sequence[float] = (0.02, 0.05, 0.2),
    tie_atol: float = 1e-6,
) -> Dict[str, float]:
    """Geometry of one kernel group. ``gen``/``fixed_pos`` are ``[1, C, S]``."""
    gen, fixed_pos = gen.float(), fixed_pos.float()
    targets = torch.cat([gen, fixed_pos], dim=1)

    d_all = cdist(gen, targets)
    scale = d_all.mean()  # unit weights => same value drift_loss computes

    C_g = gen.shape[1]
    d_gg = cdist(gen, gen)
    off = ~torch.eye(C_g, dtype=torch.bool, device=gen.device)
    d_gg = d_gg[0][off] / scale
    d_gp = cdist(gen, fixed_pos)[0].flatten() / scale
    d_pp = cdist(fixed_pos, fixed_pos)[0]
    n_p = fixed_pos.shape[1]
    d_pp = d_pp[~torch.eye(n_p, dtype=torch.bool, device=gen.device)] / scale

    q = lambda t, p: float(torch.quantile(t, p))
    rep = {
        "scale": float(scale),
        "gen_gen_p01": q(d_gg, 0.01), "gen_gen_p50": q(d_gg, 0.50),
        "gen_pos_p01": q(d_gp, 0.01), "gen_pos_p50": q(d_gp, 0.50),
        "pos_pos_p01": q(d_pp, 0.01), "pos_pos_p50": q(d_pp, 0.50),
        "pos_tie_frac": float((d_pp * scale < tie_atol).float().mean()),
        "R_min": min(R_list), "R_max": max(R_list),
    }
    # Attraction is only informative if the widest bandwidth reaches the bulk
    # of the real data, not merely its nearest point.
    rep["reach"] = rep["R_max"] / max(rep["gen_pos_p01"], 1e-12)
    rep["resolve"] = rep["R_min"] / max(rep["gen_gen_p50"], 1e-12)
    return rep


# `reach` is R_max over the *nearest* normalised gen->pos distance: attraction
# acts locally, so what matters is whether the widest bandwidth reaches the
# closest real points, not the bulk. Anchoring on the median instead flags a
# perfectly healthy matched generator, since its typical gen->pos distance is
# the data's own spread.
#
# Calibrated on three observed configurations of the 2D gate:
#   free blob,      paper R      reach 0.38  -> stalls
#   network gen,    corrected R  reach 0.74  -> converges
#   matched gen,    paper R      reach 11.3  -> converges
# A smoke alarm, not a specification. Re-calibrate as datasets accumulate.
REACH_STALL = 0.5


def check(rep: Dict[str, float], verbose: bool = True) -> bool:
    """Warn on kernel geometry that cannot produce a useful gradient.

    Most informative at initialisation, where the geometry is at its most
    extreme -- but the generated cloud moves fast, so re-run this a few hundred
    steps in rather than treating step 0 as the verdict.
    """
    problems, notes = [], []

    if rep["reach"] < REACH_STALL:
        problems.append(
            f"reach={rep['reach']:.2f} (R_max={rep['R_max']:g} vs nearest normalised "
            f"gen->pos distance {rep['gen_pos_p01']:.2f}): attraction underflows and "
            f"training stalls silently -- the loss looks normal throughout. Add a "
            f"bandwidth near {rep['gen_pos_p50']:.2f}, or initialise the generator to "
            f"match the data's marginal moments."
        )
    elif rep["reach"] < 2.0:
        notes.append(
            f"reach={rep['reach']:.2f}: bandwidths only just span the gen->pos gap. "
            f"Workable, but re-check a few hundred steps in."
        )

    if rep["resolve"] > 5.0:
        notes.append(
            f"R_min={rep['R_min']:g} is {rep['resolve']:.0f}x the median gen-gen "
            f"distance: repulsion cannot resolve individual particles yet. Expected "
            f"for a freshly initialised generator whose output is still collapsed; "
            f"a problem only if it persists."
        )

    if rep["pos_tie_frac"] > 0.001 and rep["R_min"] < rep["pos_pos_p01"]:
        problems.append(
            f"{rep['pos_tie_frac']:.2%} of real pairs are exact duplicates while "
            f"R_min={rep['R_min']:g} is below the 1st-percentile real distance "
            f"{rep['pos_pos_p01']:.3f}: the softmax will collapse onto ties. "
            f"De-duplicate the positive bank or raise R_min."
        )

    if verbose:
        for x in problems:
            print(f"  [drift] WARNING: {x}")
        for x in notes:
            print(f"  [drift] note: {x}")
        if not problems and not notes:
            print(f"  [drift] kernel geometry OK (scale={rep['scale']:.3f}, "
                  f"reach={rep['reach']:.2f}, R={rep['R_min']:g}..{rep['R_max']:g})")
    return not problems


@torch.no_grad()
def suggest_R_list(gen: torch.Tensor, fixed_pos: torch.Tensor, n: int = 3) -> list:
    """Bandwidths spanning the observed geometry, log-spaced.

    Bottom: the 1st-percentile gen-gen distance, so repulsion can resolve
    individual particles. Top: the median gen->pos distance, so attraction
    reaches the data.

    Both ends are measurements. When they are close -- which happens in high
    dimension, where distances concentrate and an untrained generator sits
    almost as far from its own samples as from the data -- the range is widened
    only modestly, so the top bandwidth still reflects a real distance rather
    than a fabricated one.
    """
    rep = distance_report(gen, fixed_pos)
    lo = max(rep["gen_gen_p01"], 1e-3)
    hi = max(rep["gen_pos_p50"], lo * 1.5)
    return [round(float(x), 4) for x in torch.logspace(
        torch.log10(torch.tensor(lo)), torch.log10(torch.tensor(hi)), n)]
