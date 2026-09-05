# Drifting on TabDiff — adult results

**Date:** 2026-09-05 · **Dataset:** adult (32561 rows, 6 numerical, 9 categorical)
**Code:** [`drifting/`](../drifting), [`remasker/`](../remasker) · **Jobs:** 143653, 144485, 144486, 144487

One-step (1 NFE) generation via the Drifting kernel objective, built on TabDiff's
column-token transformer with the diffusion process removed entirely. TabDiff is
untouched and serves as the multi-step baseline.

---

## Headline

| | density/Overall | MLE | C2ST | NFE |
|---|---|---|---|---|
| **TabDiff baseline** (multi-step) | **0.9876** | **0.9135** | **0.9876** | 50 |
| **Drift, whitened φ, R=4×** (EMA) | 0.9751 | 0.9025 | 0.9690 | **1** |
| Drift, whitened φ, R=4× (raw) | 0.9752 | 0.8999 | 0.9742 | 1 |
| Drift, whitened φ, R=2× (EMA) | 0.9704 | 0.8979 | 0.9489 | 1 |

One-step generation lands **1.2 density points, 1.1 MLE points and 1.3–1.9 C2ST
points** below the multi-step baseline. C2ST is the widest gap, which is the
expected shape: it is the metric most sensitive to whether a discriminator can
separate real from synthetic, and that is where single-step generation is
weakest.

EMA is not uniformly better — it helps MLE (+0.26), is neutral on density, and
*hurts* C2ST (−0.52). Both are reported rather than picking the flattering one.

---

## Setup

Generator: TabDiff's `Tokenizer → Transformer → Reconstructor`, timestep
conditioning and EDM preconditioning removed, driven by `z ~ N(0, I)`. Both of
TabDiff's noise processes (continuous SDE for numericals, absorbing/masking for
categoricals) and its feature-wise learnable schedules are gone — Drifting
replaces the diffusion; only the backbone survives. Categoricals use
straight-through Gumbel-softmax (`hard=True`) so φ sees the same support for
real and generated rows.

Long run: 200k steps, lr 4e-4 with 10k warmup then constant, Adam β₂=0.95, EMA
0.999, C_g=2048 / C_p=4096, unconditional (one kernel group). 24 ms/step, 1h19m
on one RTX 6000 Ada.

`drift_loss` is a PyTorch port of the JAX reference, verified numerically exact
against it (loss, gradient and scale across four configurations), plus 11
behavioural invariants and a 2D convergence gate.

---

## Result 1 — bandwidth (job 143653)

Best EMA density/Overall, 30k steps. R is a multiplier on the reference default
`[0.02, 0.05, 0.2]`.

| phi | 1× | 2× | 4× | 8× | 16× | 32× |
|---|---|---|---|---|---|---|
| whitened | 0.9690 | 0.9718 | **0.9726** | 0.9680 | 0.9052 | 0.8415 |
| remasker_flat | 0.8895 | 0.9412 | **0.9487** | 0.9463 | 0.9374 | 0.9373 |
| remasker_percol | 0.7830 | 0.8492 | 0.8740 | 0.8865 | 0.9368 | **0.9380** |

Clean inverted-U for whitened φ, peaking at 2–4× and collapsing by 32×. **This
reproduces the TabSyn port's finding** (peak at 2×, collapse at both extremes),
including the observation that whitening does not move the optimum. Two ports,
two feature spaces, same answer — so the multiplier parameterisation looks
transferable, and a full per-dataset sweep is probably unnecessary.

`remasker_percol`'s optimum runs the opposite direction — still improving at 32×
where the others have collapsed. Consistent with per-column kernels living in a
64-dim space with a different distance scale, so the shared multiplier grid is
mis-centred for that readout rather than the readout being inherently bad.

At 200k steps 4× beat 2× on every metric, resolving a gap (0.0008) that the 30k
sweep could not separate.

---

## Result 2 — ReMasker architecture (jobs 144485, 144487)

Every pre-existing ReMasker checkpoint uses `embed_dim=64, encoder_layers=8` —
`pretrain.py`'s argparse defaults, applied uniformly across all datasets and all
`pretrain_data` ablations. The architecture had never been varied, so the
first sweep's ReMasker result confounded "ReMasker doesn't help here" with "this
one architecture doesn't help". Six variants were pretrained (500 epochs each,
~11 min) and swept.

Best EMA density/Overall, 30k steps:

| readout / R | e16_l8 | e32_l4 | e32_l8 | e64_l2 | e64_l4 | e64_l8 |
|---|---|---|---|---|---|---|
| flat / 4× | 0.9043 | 0.9312 | 0.8743 | 0.9541 | **0.9546** | 0.9269 |
| flat / 16× | 0.8375 | 0.8930 | 0.8252 | 0.9415 | 0.9423 | 0.9394 |
| percol / 4× | 0.7829 | 0.8526 | 0.9149 | 0.9448 | 0.9111 | 0.9204 |
| percol / 16× | 0.8537 | 0.9017 | 0.9089 | 0.9464 | 0.9300 | 0.9274 |

Best ReMasker configuration: `e64_l4`, flat readout, 4× → **0.9546**, against
whitened's **0.9726** at the same budget.

**Across 24 configurations — 6 architectures × 2 readouts × 2 bandwidths —
nothing came within 1.8 points of whitened raw space.** That is a reasonably
well-supported negative result rather than a single-config artifact.

The likely reason is structural, not architectural. In the TabSyn port ReMasker
sat on decoded VAE output because raw table space was not directly available. In
TabDiff raw space *is* the decoded space, so a per-column-equalised identity —
each one-hot block scaled by `1/√(1−Σp²)` so every column contributes the same
expected squared distance — is simply a strong φ, and a much stronger control
than TabSyn had.

---

## Corrections to earlier claims

Recorded because each was stated with more confidence than the evidence
supported, and each was wrong.

1. **"ReMasker is the best-motivated piece of the port."** The opposite held:
   it lost to whitened raw space at every bandwidth and every architecture
   tested. The argument was sound for TabSyn's substrate and does not carry to
   TabDiff's.

2. **"Per-column kernels are ReMasker's payoff."** `remasker_percol` was the
   worst readout throughout. It was also a departure from the TabSyn port's
   validated flat readout, which was not flagged as such at the time.

3. **"Adult wants a very wide bandwidth (~34×)."** Based on one wide run
   beating one near-degenerate run — wide-vs-broken, not wide-vs-tuned. With a
   proper grid, 32× is the *worst* whitened arm.

4. **"Lower embed_dim will help by reducing distance concentration."** Backwards:
   `embed_dim=16` is the worst column, and 64 the best. Depth moved mildly in
   the predicted direction (4 layers > 8) but non-monotonically.

5. **"The ReMasker architecture was tuned for TabSyn."** It was never tuned;
   those are argparse defaults.

6. **"The affinity-clamp bug distorts the small-R arms."** Measured effect on
   gradient direction is 1e-7 to 1e-5 — float32 noise. The fix is correct and
   retained (matching tabsyn `590d2e3`), but it justified no reordering of work.

---

## Two properties worth knowing before touching this code

### 1. The loss measures agreement between bandwidths, not distance to the data

Inside `drift_loss`, each bandwidth's force is divided by its own RMS before the
bandwidths are summed:

```python
f_norm_val  = (total_force_R ** 2).mean()
force_scale = sqrt(clamp(f_norm_val, min=1e-8))
force_across_R += total_force_R / force_scale        # every R contributes at RMS 1
goal_scaled = old_gen_scaled + force_across_R
```

Because `old_gen` is `gen.detach()`, `gen_scaled` and `old_gen_scaled` are
numerically the same tensor, so

```
diff = gen_scaled - goal_scaled = -force_across_R
loss = mean(force_across_R²)
```

The loss is therefore the mean square of a sum of `k` unit-RMS vectors. Its value
depends only on how **correlated** those `k` per-bandwidth forces are:

| bandwidths agree | loss (k=3) | meaning |
|---|---|---|
| perfectly aligned | **9.0** = k² | all three R produce the same force — the multi-scale kernel has degenerated to a single scale |
| mutually orthogonal | **3.0** = k | the three scales are contributing genuinely different information |

Nothing in that expression references how far the generator is from the data.

**Measured across the 18 sweep arms:** correlation between final loss and final
density/Overall is **r = +0.05** — no relationship. `remasker_percol 1×` has the
*second-lowest* loss (6.63) and the *worst* density (0.761); `whitened 8×` has a
high loss (8.38) and near-best density (0.968).

Within a single φ the picture is different — for whitened alone, **r = −0.76** —
but that is the degenerate-regime detector, not a quality measure: whitened 32×
sits at loss 8.999 (≈ k², bandwidths collapsed onto each other) and is the worst
whitened arm at 0.842.

**Practical rule:** loss ≈ k² means the bandwidths have collapsed and the grid is
mis-centred. Any other value tells you nothing. Never compare loss across φ, and
never read convergence off the loss curve.

### 2. Bandwidth failure is silent

The softmax runs over **all** targets jointly — the generator's own samples and
the real rows in one row-wise normalisation. So the mass split between attraction
and repulsion is set by the *ratio* of gen-gen to gen→data distances, not by
either alone.

With a tight generator far from the data (normalised gen-gen median 0.029,
gen→data 1.484):

| R | softmax mass on other generated | mass on real data | \|force\| before normalisation |
|---|---|---|---|
| 0.02 | 1.0000 | **1.4e-26** | 0.0027 |
| 0.1 | 1.0000 | 1.8e-06 | 0.0027 |
| 0.5 | 0.8989 | 1.0e-01 | 0.0027 |
| 2.0 | 0.5070 | 4.9e-01 | 0.0027 |

At R=0.02 the real data receives ~1e-26 of the mass. Since every force
coefficient is a product across the partition
(`r_coeff_neg = -aff_neg · Σaff_pos`), `Σaff_pos ≈ 0` collapses the *entire*
force, attraction and repulsion alike.

What makes it silent is the last column: the gradient magnitude reaching the
generator is **identical (0.0027) at every bandwidth**, because the per-R RMS
normalisation rescales whatever survives — real signal or float32 noise — to unit
RMS. On the 2D gate this produced a force pointing the *wrong way* for 91% of
particles while training proceeded normally.

So: training runs, gradients look healthy, the loss sits in its usual range, and
nothing is learned. `drifting/diagnostics.py` checks the geometry before training;
its threshold is calibrated on few observations and should be treated as a smoke
alarm, not a specification.

## Caveats

- **adult only.** Nothing has been run on the other six datasets.
- **Single seed per arm.** Differences under ~0.01 are not meaningful; the
  whitened 4× dip at the 15k checkpoint was noise that resolved by 30k.
- **No DCR.** `adult_dcr` is not processed. This matters more than usual here:
  the attraction term is a soft nearest-neighbour match onto real rows, so
  memorisation is a live risk that density/MLE/C2ST will not reveal.
- Architecture sweep ran at 30k steps, not 200k.
- `remasker_percol` was swept at 4×/16×, but its optimum was still rising at 32×
  in sweep 1 — its bandwidth range was never properly bracketed.

---

## Reproduce

```bash
sbatch --export=DATANAME=adult slurm/train_drift_long.sbatch     # 200k, full metrics
sbatch --export=DATANAME=adult slurm/sweep_drift.sbatch          # phi x bandwidth
sbatch --export=DATANAME=adult slurm/pretrain_remasker.sbatch    # architecture variants
sbatch --dependency=afterok:<id> --export=DATANAME=adult slurm/sweep_remasker_arch.sbatch

python -m drifting.tests.test_drift_loss    # 11 invariants
python -m drifting.tests.toy2d              # 2D convergence gate
```

Results land in `drifting/result/adult/` (gitignored).

---

## Next

1. **DCR on adult** — process `adult_dcr` and measure. The highest-value missing
   number, and the one most likely to complicate the headline.
2. **Second dataset** — magic (no categoricals) and diabetes (27) bracket the
   geometry. Tests whether the 2–4× optimum transfers, which is the claim the
   two-port agreement rests on. diabetes needs a ReMasker checkpoint pretrained
   first; the other six exist.
3. **Seeds** on the winning configuration, to put an error bar on the 1.2-point
   gap to baseline.
4. **Bracket `remasker_percol` properly** (32×–128×) before concluding anything
   about that readout, or drop it.
