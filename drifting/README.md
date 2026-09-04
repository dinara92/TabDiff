# drifting/

One-step generation for TabDiff via the Drifting kernel objective.

Self-contained: `tabdiff/` is not modified, so the published multi-step model
stays intact as the baseline. Only `tabdiff.modules.transformer` is imported,
as the generator backbone.

| file | role |
|---|---|
| `drift_loss.py` | PyTorch port of the JAX kernel loss. Numerically exact vs. the reference. |
| `diagnostics.py` | Kernel-geometry pre-flight. **Run before training.** |
| `generator.py` | 1-NFE mixed-type generator on TabDiff's column-token transformer. |
| `memory_bank.py` | Group-keyed ring buffer of real rows (the attraction targets). |
| `features.py` | phi: whitened raw space (control), ReMasker, TabDiff encoder. |

## Two things that will waste your time if you don't know them

**1. The paper's `R_list` does not transfer.** Bandwidths are dimensionless
(distances are normalised by the mean pairwise distance) but still have to
bracket the *gen-to-data gap*. A freshly initialised generator sits 20-50x
outside the paper's `(0.02, 0.05, 0.2)`, attraction underflows to zero, and the
per-bandwidth RMS normalisation rescales the surviving numerical noise to full
magnitude — so training looks completely normal and learns nothing. On the 2D
gate, the paper defaults reach mode share 0.891 (collapsed); `(0.1, 0.3, 1.0)`
reaches 0.129 against an ideal of 0.125.

Always run first:

```python
from drifting.diagnostics import distance_report, check, suggest_R_list
check(distance_report(gen_feats, pos_feats, R_list))
```

**2. The loss value means nothing.** Each bandwidth contributes at unit RMS, so
the loss sits at `O(len(R_list))` forever regardless of sample quality. Monitor
sample metrics; the loss curve is flat by construction.

## Verify

```bash
python -m drifting.tests.test_drift_loss      # 11 invariants
python -m drifting.tests.toy2d --steps 15000  # end-to-end gate

# exactness against the JAX original (two envs)
JAX_PLATFORMS=cpu $JAX_PY  drifting/tests/crosscheck_jax.py --emit  ref.npz
$TORCH_PY                  drifting/tests/crosscheck_jax.py --check ref.npz
```

Toy gate at 15k steps: 8/8 modes, max mode share 0.133 (ideal 0.125),
real->gen 0.009. The `captured` threshold (0.90) is not met — 0.873 — because a
continuous one-step map bridges *between* disconnected modes. Straight-through
categorical argmax tears the map at exactly those points, so the mixed-type
case should be easier than the toy, not harder.
