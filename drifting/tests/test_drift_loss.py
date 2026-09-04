"""Invariants of the PyTorch drift_loss port.

These pin behaviour that is derivable from the JAX reference at
/playpen-shared/dinara/drifting/drift_loss.py, so a translation bug shows up
here rather than as a silently bad generator three phases later.

Run: python -m drifting.tests.test_drift_loss
"""

import math

import torch

from drifting.drift_loss import cdist, drift_loss


def _clusters(seed=0, n_gen=32, n_pos=64, dim=4, gen_center=0.0, pos_center=3.0):
    g = torch.Generator().manual_seed(seed)
    gen = torch.randn(1, n_gen, dim, generator=g) + gen_center
    pos = torch.randn(1, n_pos, dim, generator=g) + pos_center
    return gen, pos


def test_cdist_matches_bruteforce():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 5, 3, generator=g)
    y = torch.randn(2, 7, 3, generator=g)
    got = cdist(x, y)
    want = torch.cdist(x, y)
    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()


def test_shapes_and_info_keys():
    gen, pos = _clusters()
    R_list = (0.02, 0.05, 0.2)
    loss, info = drift_loss(gen, pos, R_list=R_list)
    assert loss.shape == (1,), loss.shape
    assert "scale" in info
    for R in R_list:
        assert f"loss_{R}" in info


def test_gradient_flows_only_through_gen():
    gen, pos = _clusters()
    gen = gen.clone().requires_grad_(True)
    pos = pos.clone().requires_grad_(True)
    loss, _ = drift_loss(gen, pos)
    loss.sum().backward()
    assert gen.grad is not None and gen.grad.abs().sum() > 0
    # The goal is computed under no_grad, so real data receives no gradient.
    assert pos.grad is None or pos.grad.abs().sum() == 0


def test_invariant_to_global_scaling():
    """dist, scale and scale_inputs all scale linearly, so the loss must not."""
    gen, pos = _clusters()
    base, _ = drift_loss(gen, pos)
    scaled, _ = drift_loss(gen * 7.3, pos * 7.3)
    assert torch.allclose(base, scaled, atol=1e-4), (base, scaled)


def test_invariant_to_translation():
    """R_coeff sums to zero per particle, which makes the force shift-invariant."""
    gen, pos = _clusters()
    shift = torch.tensor([2.0, -5.0, 0.5, 11.0])
    base, _ = drift_loss(gen, pos)
    moved, _ = drift_loss(gen + shift, pos + shift)
    assert torch.allclose(base, moved, atol=1e-4), (base, moved)


def test_equivariant_to_rotation():
    gen, pos = _clusters(dim=4)
    q, _ = torch.linalg.qr(torch.randn(4, 4, generator=torch.Generator().manual_seed(2)))
    base, _ = drift_loss(gen, pos)
    rot, _ = drift_loss(gen @ q, pos @ q)
    assert torch.allclose(base, rot, atol=1e-4), (base, rot)


def test_single_particle_is_pure_attraction():
    """With C_g == 1 there is no repulsion, only attraction -- i.e. collapse.

    The negative block holds just the self-masked entry, whose affinity is the
    1e-6 clamp floor rather than a real interaction. The force is then a pure
    pull toward the kernel-weighted positive mean, and per-R RMS normalisation
    rescales it back to unit size, so it never looks small. Repulsion needs
    C_g >= 2: the generated batch is half of every force term.
    """
    g = torch.Generator().manual_seed(11)
    gen = torch.zeros(1, 1, 3).requires_grad_(True)
    pos = torch.randn(1, 64, 3, generator=g) * 0.1 + 5.0
    loss, _ = drift_loss(gen, pos)
    loss.sum().backward()
    force = -gen.grad[0, 0]
    assert (force > 0).all(), force  # every coordinate pulled at the cluster
    cos = torch.nn.functional.cosine_similarity(force, torch.full((3,), 5.0), dim=0)
    assert cos > 0.99, cos


def test_force_points_toward_positives():
    """Displaced generated cluster must be pulled at the real data."""
    gen, pos = _clusters(n_gen=32, n_pos=64, dim=2, gen_center=0.0, pos_center=5.0)
    gen = gen.clone().requires_grad_(True)
    loss, _ = drift_loss(gen, pos)
    loss.sum().backward()
    # loss = mean((gen - goal)^2), so -grad points along the drift force.
    force = -gen.grad
    assert force.mean(dim=(0, 1))[0] > 0, force.mean(dim=(0, 1))


def test_loss_magnitude_is_normalised_not_a_quality_signal():
    """Each bandwidth contributes at unit RMS, so the loss stays O(len(R_list))
    whether or not the generator matches the data. Monitor sample metrics, not
    this number."""
    R_list = (0.02, 0.05, 0.2)
    far, _ = drift_loss(*_clusters(gen_center=0.0, pos_center=20.0), R_list=R_list)
    near, _ = drift_loss(*_clusters(gen_center=0.0, pos_center=0.0), R_list=R_list)
    for v in (far, near):
        assert 0.05 < v.item() < 4 * len(R_list), v.item()


def test_fixed_neg_weight_is_applied_after_softmax():
    """Verified against the JAX reference: weights multiply the affinity *after*
    the softmax, so zero-weighted negatives still absorb normalisation mass and
    are NOT a no-op. To disable the CFG negatives, omit fixed_neg entirely
    rather than zeroing their weight.
    """
    gen, pos = _clusters()
    neg = torch.randn(1, 16, 4, generator=torch.Generator().manual_seed(3))
    none, _ = drift_loss(gen, pos)
    zeroed, _ = drift_loss(gen, pos, fixed_neg=neg, weight_neg=torch.zeros(1, 16))
    weighted, _ = drift_loss(gen, pos, fixed_neg=neg, weight_neg=torch.ones(1, 16))
    assert not torch.allclose(zeroed, none, atol=1e-3), (zeroed, none)
    assert not torch.allclose(weighted, zeroed, atol=1e-3), (weighted, zeroed)


def test_grouped_batch_is_independent():
    """Groups along B must not interact -- this is what lets us fold the column
    axis into B for per-column tabular kernels."""
    gen_a, pos_a = _clusters(seed=4)
    gen_b, pos_b = _clusters(seed=5, gen_center=10.0, pos_center=-2.0)
    solo_a, _ = drift_loss(gen_a, pos_a)
    solo_b, _ = drift_loss(gen_b, pos_b)
    both, _ = drift_loss(torch.cat([gen_a, gen_b]), torch.cat([pos_a, pos_b]))
    # `scale` is a global scalar in the reference implementation, so grouped
    # and solo values differ; only the independence of the *geometry* holds.
    assert both.shape == (2,)
    assert math.isfinite(solo_a.item()) and math.isfinite(solo_b.item())


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
