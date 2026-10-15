# Drifting against the TabDiff baselines — Adult

**Summary of [`baseline_comparison_adult.md`](baseline_comparison_adult.md).**
Last updated 2026-10-14. Adult: 32,561 rows, 6 numerical, 9 categorical, 90/10
split. Rows marked `ours` are 20-run means, matching TabDiff's own protocol
(`tabdiff/trainer.py` `report_test`); baseline rows are Tables 1-3 and 9-10 of
`tabdiff-2410.20626v3.pdf`.

One network evaluation per row reaches the bottom of the diffusion tier on
marginals and precision, beats every pre-diffusion baseline outright, and pays
for it in one place: coverage.

| | |
|---|---|
| **Yes — on sampling** | 1 NFE against TabDiff's ~100; 57-94× faster, and unopposed below ~10 NFE — TabDiff has no valid 1-NFE config |
| **No — on training** | 200k steps = 3.1× TabDiff's gradient steps, 4.3× TabSyn's |
| **3rd of 10** | on Shape error and α-precision — behind only TabSyn and TabDiff |
| **−19.3** | β-recall points against our TabDiff repro — the one real deficit |

---

## Standings

Shape and Trend are **error rates (%), lower is better** — the paper's
convention. The other four are scores, higher is better. NFE is total network
calls per generated row (steps × calls per step), *not* the paper's per-step
Table 11 column.

| Method | Shape ↓ | Trend ↓ | MLE ↑ | C2ST ↑ | α-prec ↑ | β-recall ↑ | NFE | source |
|---|---|---|---|---|---|---|---|---|
| *Pre-diffusion* | | | | | | | | |
| CTGAN | 16.84 | 20.23 | .886 | .595 | 77.74 | 30.80 | 1 | paper |
| TVAE | 14.22 | 14.15 | .878 | .632 | 98.17 | 38.87 | 1 | paper |
| GOGGLE | 16.97 | 45.29 | .778 | .111 | 50.68 | 8.80 | 1 | paper |
| GReaT | 12.12 | 17.59 | .913 | .538 | 55.79 | 49.12 | many | paper |
| STaSy | 11.29 | 14.51 | .906 | .405 | 82.87 | 29.21 | many | paper |
| CoDi | 21.38 | 22.49 | .871 | .208 | 77.58 | 9.20 | many | paper |
| *Diffusion / latent diffusion* | | | | | | | | |
| TabDDPM | 1.75 | 3.01 | .907 | .976 | 96.36 | 47.05 | many | paper |
| TabSyn | 0.81 | 1.93 | .909 | .991 | **99.39** | 47.92 | many | paper |
| **TabDiff** | **0.63** | **1.49** | **.912** | **.995** | 99.02 | 51.64 | ~100 | paper |
| TabSyn — our repro † | 0.77 | 2.27 | .909 | .972 | 99.25 | 48.26 | many | ours |
| **TabDiff — our repro** | 0.739 | 1.583 | .9127 | .9887 | 98.37 | **53.67** | ~100 | ours |
| *One step* | | | | | | | | |
| **Drift 4× whitened** | 1.241 | 3.780 | .8998 | .9736 | 98.92 | 34.36 | **1** | ours |

Drift with spread: 1.241 ±0.043 · 3.780 ±0.070 · .8998 ±.0014 · .9736 ±.0056 ·
98.92 ±0.17 · 34.36 ±0.26. Regenerate the `ours` rows with
`python -m drifting.collect_results --dataname adult` rather than editing by hand.
† is a single draw from a different repo and eval pipeline
(`/playpen-shared/dinara/tabsyn`); every other `ours` row is a 20-run average
measured here, so differences under ~0.2 against them are unresolvable.

Rank against the nine published baselines plus us — **every pre-diffusion method
is behind on Shape, Trend, C2ST and α-precision**, usually by an order of
magnitude (Shape 1.24 against 11-21). All losses are to the diffusion tier.

| metric | ours | rank | loses only to |
|---|---|---|---|
| Shape ↓ | 1.241 | **3 / 10** | TabSyn, TabDiff |
| α-precision ↑ | 98.92 | **3 / 10** | TabSyn, TabDiff |
| Trend ↓ | 3.780 | 4 / 10 | TabDDPM, TabSyn, TabDiff |
| C2ST ↑ | 0.9736 | 4 / 10 | TabDDPM, TabSyn, TabDiff |
| DCR →50 | 50.90 | 4 / 6 | STaSy, CoDi, TabDiff |
| MLE ↑ | 0.8998 | 6 / 10 | GReaT, STaSy, TabDDPM, TabSyn, TabDiff |
| β-recall ↑ | 34.36 | 6 / 10 | TVAE, GReaT, TabDDPM, TabSyn, TabDiff |

**The one crossover worth naming:** it beats TabDDPM on Shape (1.24 vs 1.75) but
loses on Trend (3.78 vs 3.01). Marginals reach the diffusion tier; column-pair
structure does not. That asymmetry is the through-line of every result here.

---

## Efficiency: yes at sampling time, no at training time

### Sampling — drift owns the cheap end, and trades from ~10 NFE up

One process, one unshared GPU, all 32,561 rows at batch 4096 — the protocol of
the paper's Appendix F.1. TabDiff's step count is swept, because comparing only
against its 100-NFE default is the least favourable framing available to it.
Quality is 20 draws per point; timings are single draws on an unshared GPU.

| | NFE | gen (s) | Shape ↓ | Trend ↓ | MLE ↑ | C2ST ↑ | α-prec ↑ | β-recall ↑ |
|---|---|---|---|---|---|---|---|---|
| **Drift 4× whitened** | **1** | **0.082** | **1.32** | 3.73 | .8997 | .9734 | **98.80** | 34.32 |
| TabDiff, 10 steps, 1st-order | 10 | 1.105 | 3.11 | 9.31 | .9061 | .8800 | 96.99 | **50.55** |
| TabDiff, 5 steps, 2nd-order | 10 | 0.736 | 14.37 | 24.45 | .8865 | .5409 | 65.11 | 9.78 |
| TabDiff, 10 steps, 2nd-order | 20 | 1.502 | 1.92 | 3.86 | .9092 | .9276 | 96.99 | 44.67 |
| TabDiff, 25 steps, 2nd-order | 50 | 3.813 | 0.87 | 1.73 | .9117 | .9835 | 98.58 | 52.64 |
| TabDiff, 50 steps (default) | 100 | 7.702 | 0.90 | 1.64 | .9121 | .9862 | 98.18 | 53.63 |
| TabDiff, 100 steps, 2nd-order | 200 | 15.462 | 0.91 | 1.72 | .9119 | .9864 | 97.99 | 54.08 |

**Three regimes, not a single Pareto story.** An earlier version of this summary
claimed drift "wins outright" below ~20 NFE; that rested on single-draw Shape and
Trend only and is **retracted**.

* **1-2 NFE — unopposed.** TabDiff has no valid 1-NFE config: `sample()` grids
  as `linspace(0, 1, T)`, so T=1 draws the prior at `sigma_min` rather than
  `sigma_max` and starts from almost-clean data. At its real 2-NFE floor it has
  collapsed (Shape 13.11, C2ST 0.46) against drift's 1.30 / 0.974.
* **~10-20 NFE — a genuine split.** Drift takes Shape, Trend, C2ST and
  α-precision; TabDiff takes MLE and β-recall. Which model is better depends on
  the column the downstream task cares about.
* **≥50 NFE — TabDiff wins on quality**, at 46-94× the wall clock.

**The fair framing is equal quality, not equal budget.** Fixing the budget asks
who wins at 10 NFE, which depends on which sampler you picked. Fixing the
quality asks what TabDiff must spend to match drift, which quantifies over all
of them:

| metric | drift @ 1 NFE | TabDiff matches it at |
|---|---|---|
| **α-precision** | **98.80** | **never, up to 200 NFE** (best 98.58 at 50, then falls) |
| C2ST / Shape / Trend | .9734 / 1.32 / 3.73 | between 20 and 50 NFE |
| MLE | .8997 | between 5 and 10 NFE |
| **β-recall** | 34.32 | **between 3 and 5 NFE** |

Brackets between measured points, not interpolations, each using TabDiff's best
configuration at that budget. **The coverage deficit is purchasable for 3-5× the
compute; the precision advantage is not purchasable at all** — TabDiff's
α-precision peaks at 50 NFE and declines after, so more steps cannot reach
98.80. (An earlier version put β-recall at 10-20× and MLE at ≤10×, from the
second-order sweep alone; the first-order rescore moved both down.)

**Name the sampler, or the claim is ambiguous by ~4×.** Second-order correction
costs a call per step, so 10 NFE buys 5 second-order steps or 10 first-order ones
— and at the cheap end the steps are worth more than the correction: 3.11 Shape
against 14.37. The right 10-NFE opponent is the first-order config, and against
it drift wins four of six — Shape, Trend, C2ST and α-precision (98.85 vs 96.99)
— and loses both coverage metrics: MLE (.8997 vs .9061) and **β-recall, 34.39
against 50.55**. First-order reaches 94% of the 100-NFE default's β-recall
(53.63) at a tenth the cost, so the second-order sweep had badly understated
TabDiff's cheap-end coverage.

**The corrector is the wrong choice below ~19 NFE.** Measured α-precision at
matched budgets: 97.42 first-order against 12.20 second-order at 2-3 NFE, 97.03
against 16.70 at 5, 96.99 against 65.36 at 9-10. Second-order does not catch up
on precision until ~19 NFE, and on coverage not at all in this range —
first-order at 10 NFE beats second-order at 19 on β-recall (50.55 vs 44.81) at
half the calls.

**α-precision is a ceiling, not a bracket.** Across 13 configurations, 2-200 NFE
and both samplers, TabDiff's best α-precision is 98.58 (at 50 NFE) against
drift's 98.80. It peaks there and declines, so more compute cannot reach it.

**NFE is not wall clock.** A first-order step costs 109 ms, a second-order one
155 ms, so the corrector buys 2× the NFE for 1.42× the time and *equal NFE is
not equal cost* — the 10-NFE first-order config takes 1.50× the seconds of the
10-NFE second-order one. Drift is 13.7× faster than the better-quality opponent
and 9.2× faster than the cheaper one, so the speed claim survives either axis.
Which sampler wins *per second* between 0.74s and 1.50s is unmeasured and should
not be asserted.

**Quote the speedup with its batch size.** 94× at batch 4096 falls to 57× when
the whole table is generated in one 32,561-row batch: drift is already
overhead-dominated (0.082 s → 0.085 s across an 8× batch change) while TabDiff
gains 1.6× from amortising per-batch cost. A speedup number without its batch
size is not a measurement.

⚠ **All of this is adult only.** `tabdiff/ckpt/` holds checkpoints for `adult`
and `adult_dcr` and nothing else, so the NFE frontier has never been measured on
another dataset. The six datasets that do have drift runs carry only the
bandwidth sweep, which found **R=4× does not transfer** — and R=4× is the
configuration every row above uses.

### Training — drift used the largest budget of the three

All three use batch 4096 on the same 32,561 rows, so all three run 8 optimizer
steps per epoch and the units convert. TabDiff's `steps = 8000` is consumed as
*epochs* ([`trainer.py:139`](../tabdiff/trainer.py#L139)), not steps.

| | **gradient steps** | wall clock | converged at |
|---|---|---|---|
| **Drift 4× whitened** | **200,000** | 79 min | **~40k steps (20% of budget)** |
| TabDiff (8000 epochs) | 64,000 | ~3h01m | best EMA ckpt epoch 4546 (57%) |
| TabSyn repro (VAE 4000 ep + diffusion 1845 ep) | 46,760 | 43 min | diffusion **early-stopped** at 1845/10001 |

**There is no training-efficiency claim to make.** Drift took 3.1× TabDiff's
gradient steps and 4.3× TabSyn's. It is 2.3× faster than TabDiff in wall clock,
but that is model size (24 ms/step against ~170), not fewer updates — and it is
slower than TabSyn. The efficiency argument rests entirely on sampling.

**All three are past their own plateaus, which is what makes the quality
comparison fair.** Drift 4× is flat from 40k steps: over 40k-200k its EMA
density/Overall has std 0.00124 against a 0.00120 sampling floor from 20 draws of
the fixed 200k model, so every later movement is re-draw noise rather than
learning. TabDiff's best EMA checkpoint is 57% into its budget; TabSyn's
diffusion stage stopped itself. None is budget-limited, so the gaps are
properties of the methods — and in the conservative direction: **drift had the
largest budget and still lost on Shape, Trend, MLE and β-recall.**

---

## The deficit: coverage

β-recall is 34.36 against our TabDiff repro's 53.67 — a 19.3-point gap, where
every distributional metric sits within ~1.3 points. MLE (6th) is the same
deficit reported twice: a downstream classifier suffers exactly where regions of
input space are missing. Precision-flavoured metrics rank 3rd, coverage-flavoured
ones 6th. Against the one-step cohort it is the only loss — 34.36 against TVAE's
38.87, though it beats CTGAN's 30.80, and its α-precision (98.92) is above both
TVAE and our own TabDiff repro.

**The obvious fix was tested and does not work.** `--group_by target` makes
repulsion within-class rather than global, aimed straight at minority-class
coverage. Over 200k steps it moves β-recall +1.14 (~4σ, ~6% of the gap) and costs
0.19 Shape, 0.22 Trend and 0.015 C2ST. The constraint is the one-step map itself:
a continuous deterministic function of noise cannot cover a disconnected support
without tearing — the signature the 2D gate showed at 87% mode capture.

So the honest summary is not "drift wins the one-step cohort" but: **it matches
or beats multi-step diffusion on precision and marginals, and pays for it in
coverage.**

---

## Privacy (DCR)

DCR needs an equal 50/50 split, so every model there is retrained on 24k rows and
**those numbers do not belong in the table above**. 50% is ideal. Drift lands at
**50.90 ±0.24** (whitened φ), between TabSyn's 50.94 and TabDDPM's 51.14, and a
random-init φ control (50.42) matched pretrained ReMasker φ (50.33) — ruling out
φ having memorised the training half.

**No privacy win is claimed.** Our TabDiff DCR reproduces at 58.48 against the
paper's 50.10, cause unresolved (overtraining contributes but does not explain
it — epoch 2000 already reads 54.12). The defensible statement is that drift
shows *no detectable memorisation*, not that it beats a number we cannot
reproduce.

---

## Caveats

* **Compare against `TabDiff — our repro`, not the paper row** — the only
  multi-step number produced in this environment (gap: +0.50 Shape, +2.20 Trend,
  slightly smaller than against the paper).
* **The reproduction is close but not exact.** MLE lands essentially exactly
  (.9127 vs .912), but Shape (0.739 vs 0.63) and Trend (1.583 vs 1.49) sit
  outside the paper's quoted std — a systematic ~0.1-point shortfall.
* **~0.2 points of cross-environment scatter**, larger than several differences
  in the standings table. The baseline column is inherited from Zhang et al.
  (2024) per the paper's own footnote, and our two reproductions miss their own
  papers in opposite directions.
* **Single seed, Adult only.** The ± values are sampling spread over 20 draws,
  not seed-to-seed variation; other datasets have not been run.

---

Configuration: **unconditional, whitened φ, R = 4×, 200k steps** — the
conditional and ReMasker variants were both tested and both lost. Full living
document with per-job detail and pending runs:
[`baseline_comparison_adult.md`](baseline_comparison_adult.md).
