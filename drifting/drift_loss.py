"""PyTorch port of the Drifting kernel loss.

Faithful translation of ``drift_loss.py`` from the official JAX release of
*Generative Modeling via Drifting* (arXiv:2602.04770). The reference lives at
``/playpen-shared/dinara/drifting/drift_loss.py``; ``tests/test_drift_loss.py``
pins this implementation against hand-derived invariants of that code.

The loss is a target-matching objective: a goal position is computed for every
generated particle under ``no_grad`` (the JAX original wraps the same block in
``jax.lax.stop_gradient``), and the returned loss is the squared distance from
the current particles to that frozen goal. Gradients therefore flow only
through ``gen``.
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise euclidean distance. ``[B, N, D] x [B, M, D] -> [B, N, M]``."""
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def drift_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: Optional[torch.Tensor] = None,
    weight_gen: Optional[torch.Tensor] = None,
    weight_pos: Optional[torch.Tensor] = None,
    weight_neg: Optional[torch.Tensor] = None,
    R_list: Sequence[float] = (0.02, 0.05, 0.2),
    legacy_affinity_clamp: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Multi-scale kernel drift loss.

    Args:
        gen: generated particles, ``[B, C_g, S]``. ``B`` indexes independent
            kernel groups -- one per class in the ImageNet setting, one per
            (column, layer) pair in the tabular port.
        fixed_pos: real-data particles that attract, ``[B, C_p, S]``.
        fixed_neg: extra particles that repel, ``[B, C_n, S]``. Used by the
            reference implementation only to realise training-time CFG, where
            it holds unconditional samples. Leave ``None`` for unconditional
            generation -- repulsion between the generated particles themselves
            is always active and does not depend on this argument.
        weight_gen / weight_pos / weight_neg: per-particle weights,
            ``[B, C_*]``. Default 1.
        R_list: kernel bandwidths, applied to distances that have already been
            normalised by the mean pairwise distance, so they are
            dimensionless -- which does NOT make them portable across datasets
            or feature spaces; see drifting/diagnostics.py.
        legacy_affinity_clamp: reproduce the reference's masked-self-pair
            behaviour bit-for-bit (see below). Only for the JAX cross-check --
            leave False for training.

    Returns:
        ``(loss, info)`` where ``loss`` is ``[B]`` and ``info`` carries the
        distance ``scale`` plus the per-bandwidth force norms.
    """
    B, C_g, S = gen.shape

    if fixed_neg is None:
        fixed_neg = gen.new_zeros((B, 0, S))
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    C_p = fixed_pos.shape[1]

    with torch.no_grad():
        info: Dict[str, torch.Tensor] = {}

        dist = cdist(old_gen, targets)
        weighted_dist = dist * targets_w[:, None, :]
        scale = weighted_dist.mean() / targets_w.mean()
        info["scale"] = scale

        # Normalise coordinates to order 1 so the kernel sees a fixed geometry.
        scale_inputs = torch.clamp(scale / (S ** 0.5), min=1e-3)
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs
        dist_normed = dist / torch.clamp(scale, min=1e-3)

        # Mask each particle against itself inside the ``old_gen`` block.
        mask_val = 100.0
        diag_mask = torch.eye(C_g, dtype=dist.dtype, device=dist.device)
        block_mask = F.pad(diag_mask, (0, C_n + C_p)).unsqueeze(0)
        dist_normed = dist_normed + block_mask * mask_val
        masked_pair = block_mask > 0

        force_across_R = torch.zeros_like(old_gen_scaled)

        for R in R_list:
            logits = -dist_normed / R

            # Symmetrised (Sinkhorn-flavoured) affinity.
            affinity = torch.softmax(logits, dim=-1)
            aff_transpose = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt(torch.clamp(affinity * aff_transpose, min=1e-6))
            # The 1e-6 floor gives sqrt a finite gradient on genuine underflow,
            # but applied uniformly it also floors masked self-pairs, which the
            # softmax has already driven to exact 0 (mask_val=100 pushes those
            # logits far past float32 underflow). That injects a constant fake
            # self-affinity which grows relatively larger as training sharpens
            # real affinities toward 0 -- worst at the smallest R. Same fix as
            # tabsyn commit 590d2e3.
            if not legacy_affinity_clamp:
                affinity = affinity.masked_fill(masked_pair, 0.0)
            affinity = affinity * targets_w[:, None, :]

            split_idx = C_g + C_n
            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]

            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            r_coeff_pos = aff_pos * sum_neg

            R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)

            total_force_R = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)
            # Coefficients sum to zero, so this makes the force translation
            # invariant rather than changing its value.
            total_coeffs = R_coeff.sum(dim=-1)
            total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled

            f_norm_val = (total_force_R ** 2).mean()
            info[f"loss_{R}"] = f_norm_val

            # Each bandwidth contributes at unit RMS, so no single R dominates.
            force_scale = torch.sqrt(torch.clamp(f_norm_val, min=1e-8))
            force_across_R = force_across_R + total_force_R / force_scale

        goal_scaled = old_gen_scaled + force_across_R

    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = (diff ** 2).mean(dim=(-1, -2))

    return loss, info
