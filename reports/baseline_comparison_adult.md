# Adult — comparison against the TabDiff paper's baselines

**Living document.** Updated as runs land. Last updated 2026-10-14.

Shape and Trend are **error rates (%), lower is better** — the paper's convention.
MLE (AUC) and C2ST are **scores, higher is better**. DCR is a probability where
**50% is ideal**; higher means synthetic rows sit closer to the training set than
a real holdout row does.

Rows marked `ours` are 20-run means ± std, matching TabDiff's own reporting
protocol (`tabdiff/trainer.py` `report_test`). Rows marked `paper` are Table 1,
2, 3, 9 and 10 of `tabdiff-2410.20626v3.pdf`.

**NFE below is the total network calls per generated row**, i.e. discretization
steps × calls per step. The paper's Table 11 uses the same header for the
*per-step* count (TabDiff 2, TabDDPM 1) — the two columns are not the same
quantity and must not be read across. TabDiff's 50 steps × 2 (second-order
correction) is the ~100 here.

**Every TabDiff row outside the low-NFE section uses the second-order sampler**,
its default, so NFE = 2T−1 (the final step takes no corrector; the ~100 above is
the rounding of 99). With `--no_second_order` it is NFE = T, and at the cheap end
that is the better configuration — see
[The NFE floor, and the second-order confound](#the-nfe-floor-and-the-second-order-confound).
A low-NFE claim that does not name its sampler is ambiguous by a factor of ~4 on
Shape.

---

## Adult

| Method | Shape ↓ | Trend ↓ | MLE ↑ | C2ST ↑ | α-prec ↑ | β-recall ↑ | NFE | source |
|---|---|---|---|---|---|---|---|---|
| CTGAN | 16.84 ±0.03 | 20.23 ±1.20 | .886 ±.002 | .595 | 77.74 ±0.15 | 30.80 ±0.20 | 1 | paper |
| TVAE | 14.22 ±0.08 | 14.15 ±0.88 | .878 ±.004 | .632 | 98.17 ±0.17 | 38.87 ±0.31 | 1 | paper |
| GOGGLE | 16.97 | 45.29 | .778 ±.012 | .111 | 50.68 | 8.80 | 1 | paper |
| GReaT | 12.12 ±0.04 | 17.59 ±0.22 | .913 ±.003 | .538 | 55.79 ±0.03 | 49.12 ±0.18 | many | paper |
| STaSy | 11.29 ±0.06 | 14.51 ±0.25 | .906 ±.001 | .405 | 82.87 ±0.26 | 29.21 ±0.34 | many | paper |
| CoDi | 21.38 ±0.06 | 22.49 ±0.08 | .871 ±.006 | .208 | 77.58 ±0.45 | 9.20 ±0.15 | many | paper |
| TabDDPM | 1.75 ±0.03 | 3.01 ±0.25 | .907 ±.001 | .976 | 96.36 ±0.20 | 47.05 ±0.25 | many | paper |
| TabSyn | 0.81 ±0.05 | 1.93 ±0.07 | .909 ±.001 | .991 | **99.39** ±0.18 | 47.92 ±0.23 | many | paper |
| **TabDiff** | **0.63** ±0.05 | **1.49** ±0.16 | **.912** ±.002 | **.995** | 99.02 ±0.20 | **51.64** ±0.20 | ~100 | paper |
| TabSyn — our repro | 0.77 | 2.27 | .909 | .972 | 99.25 | 48.26 | many | ours¹ |
| **TabDiff — our repro** | 0.739 ±0.047 | 1.583 ±0.062 | .9127 ±.0012 | .9887 ±.0048 | 98.37 ±0.39 | **53.67** ±0.33 | ~100 | ours |
| **Drift 4× whitened** | 1.241 ±0.043 | 3.780 ±0.070 | .8998 ±.0014 | .9736 ±.0056 | 98.92 ±0.17 | 34.36 ±0.26 | **1** | ours |
| Drift 2× whitened | 1.443 ±0.082 | 4.186 ±0.173 | .8974 ±.0010 | .9559 ±.0059 | **99.26** ±0.20 | 34.17 ±0.30 | 1 | ours |

All rows above are the **90/10 split**. DCR needs a different split and
therefore a different model, so it has its own table below.

¹ From `/playpen-shared/dinara/tabsyn`, `eval/reports/full_benchmark_drifting_vs_
  tabsyn_2026-08-18.md` and the raw files under `eval/{density,mle,quality}/adult/`.
  **Single run, no std** — every other `ours` row is a 20-run average — and a
  different repo, conda env and eval pipeline. Treat differences under ~0.2 against
  the other `ours` rows as unresolvable.

Regenerate the `ours` rows from source rather than editing them by hand:

```bash
python -m drifting.collect_results --dataname adult
```

---

## Adult — DCR (privacy)

Separate table because DCR requires an **equal 50/50 train/holdout split**
(`adult_dcr`), so every model here is retrained from scratch on 24k rows rather
than the 32k of the table above. The two are not comparable and must not be
merged into one row. **50% is ideal**; higher means synthetic rows sit closer to
the training set than a real holdout row does.

| Method | DCR →50% | NFE | source |
|---|---|---|---|
| CoDi | 49.92 ±0.18 | many | paper |
| **TabDiff** | **50.10** ±0.32 | ~100 | paper |
| STaSy | 50.33 ±0.19 | many | paper |
| TabSyn | 50.94 ±0.17 | many | paper |
| TabDDPM | 51.14 ±0.18 | many | paper |
| TabDiff — our repro | 58.48 ±0.32 ⚠ | ~100 | ours |
| Drift 4× ReMasker φ | 50.33 ±0.35 | 1 | ours |
| Drift 4× ReMasker rand-init | 50.42 ±0.35 | 1 | ours |
| Drift 4× whitened φ | 50.90 ±0.24 | **1** | ours |

⚠ **Our TabDiff DCR does not reproduce the paper's, and the cause is unresolved.**
We measure 58.48 ±0.32 against the paper's 50.10 ±0.32.

Training was healthy (all 8000 epochs, final loss 1.5893 against 1.5875 on the
main split). The overtraining hypothesis — `adult_dcr` has 24421 training rows
against the main split's 32561 on an unchanged epoch budget — was tested by
scoring DCR across checkpoints (job 145952):

| checkpoint | DCR →50% |
|---|---|
| epoch 2000 | 54.12 ±0.27 |
| epoch 4000 | 58.28 ±0.29 |
| epoch 6000 | 58.47 ±0.23 |
| epoch 8000 | 58.57 ±0.40 |
| best (7722) | 58.48 ±0.32 |

DCR does rise with training and plateaus by epoch 4000, so overtraining is real
and contributes — but it is **not** the explanation: the earliest checkpoint is
already 54.12, and no stopping point reaches 50.10.

What this licenses, and what it does not:

* **The within-environment comparison is sound.** Both models used the same
  split, metric and reference files. A broken split would shift both, and the
  drift arms sit almost exactly on the ideal.
* **Do not claim a privacy win over TabDiff.** The defensible statement is that
  drift shows no detectable memorisation (50.3-50.9 against an ideal of 50),
  not that it beats a baseline whose published number we cannot reproduce.

TabSyn's own repo also reports a DCR for Adult (0.674) but on the **original ~2:1
split**, where the no-leakage target is ~0.674 rather than 50%. It scored exactly
its target, but the number is a different quantity on a different scale and is
deliberately excluded rather than converted.

---

## Adult — sampling efficiency

Measured in one process on one unshared A6000 (slurm 145594), generating 32,561
rows — the protocol of the paper's Appendix F.1, Table 11. Batch size is applied
to **both** models, because it dominates the ratio.

```bash
python -m drifting.bench_speed --dataname adult --quality      # batch 4096
sbatch --export=DATANAME=adult slurm/bench_speed.sbatch        # both batches
```

### Batch 4096

Timing columns come from slurm 145594 (single draw, unshared GPU, so the clock
is not polluted by metric evaluation). Quality columns come from slurm 151071:
**20 independent draws per point, scored separately and averaged**, which is the
protocol of every other table in this report. Regenerate with:

```bash
sbatch --export=DATANAME=adult slurm/bench_quality.sbatch
python -m drifting.collect_bench --dataname adult
```

| Method | NFE | gen (s) | rows/s | Shape ↓ | Trend ↓ | MLE ↑ | C2ST ↑ | α-prec ↑ | β-recall ↑ |
|---|---|---|---|---|---|---|---|---|---|
| **Drift 4× whitened** | **1** | **0.082** ±.000 | **295,503** | 1.32 ±0.04 | 3.73 ±0.17 | .8997 ±.0013 | .9734 ±.0058 | **98.80** ±0.20 | 34.32 ±0.25 |
| TabDiff, 5 steps | 10 | 0.736 ±.009 | 42,644 | 14.37 ±0.08 | 24.45 ±0.10 | .8865 ±.0047 | .5409 ±.0081 | 65.11 ±0.27 | 9.78 ±0.16 |
| TabDiff, 10 steps | 20 | 1.502 ±.011 | 21,299 | 1.92 ±0.07 | 3.86 ±0.08 | .9092 ±.0016 | .9276 ±.0073 | 96.99 ±0.33 | 44.67 ±0.29 |
| TabDiff, 25 steps | 50 | 3.813 ±.009 | 8,481 | 0.87 ±0.04 | 1.73 ±0.24 | .9117 ±.0014 | .9835 ±.0047 | 98.58 ±0.31 | 52.64 ±0.27 |
| TabDiff, 50 steps (default) | 100 | 7.702 ±.030 | 4,213 | 0.90 ±0.04 | 1.64 ±0.15 | .9121 ±.0010 | .9862 ±.0051 | 98.18 ±0.19 | 53.63 ±0.25 |
| TabDiff, 100 steps | 200 | 15.462 ±.089 | 2,102 | 0.91 ±0.05 | 1.72 ±0.21 | .9119 ±.0008 | .9864 ±.0052 | 97.99 ±0.30 | 54.08 ±0.38 |

### Batch 32,561 — the whole table in one shot

The multi-step model's best case: one batch amortises all per-batch overhead.

| Method | NFE | gen (s) | rows/s |
|---|---|---|---|
| **Drift 4× whitened** | **1** | **0.085** ±.000 | **287,266** |
| TabDiff, 5 steps | 10 | 0.439 ±.002 | 69,812 |
| TabDiff, 10 steps | 20 | 0.914 ±.007 | 34,598 |
| TabDiff, 25 steps | 50 | 2.359 ±.049 | 13,647 |
| TabDiff, 50 steps (default) | 100 | 4.840 ±.016 | 6,690 |
| TabDiff, 100 steps | 200 | 9.898 ±.046 | 3,281 |

Decoding to a dataframe adds 0.026–0.028s to every row in both tables — the same
code path for both models, a shared constant rather than a difference. The
paper's Sample.T includes it; the gen column does not.

### The NFE floor, and the second-order confound

The table above starts at 5 steps because that is where the paper's own ablation
(Table 12) stops. Two questions it leaves open: does TabDiff have a 1-NFE
configuration at all, so the comparison can be at **equal cost** rather than
only equal wall clock — and at a fixed budget, is the second-order correction
worth the steps it costs?

**TabDiff has no valid 1-NFE configuration.** `sample()` in
[`unified_ctime_diffusion.py`](../tabdiff/models/unified_ctime_diffusion.py)
builds its time grid as `linspace(0, 1, T)`, so `T=1` gives `t=[0.0]` and draws
the prior at `sigma(0) = sigma_min = 0.002` instead of `sigma_max = 80`: it
starts from almost-clean data and "generates" by denoising noise that was never
added. T=1 is degenerate, not a one-step sampler. Its genuine floor is 2 NFE.

**NFE is now measured, not computed.** `bench_speed.py` wraps the denoiser and
counts calls, so the column cannot silently desync from the code again. With the
correction on, NFE = 2T−1 (the last step takes no corrector); with
`--no_second_order`, NFE = T.

20 draws per point, batch 4096, from
[`bench_lownfe_so1.csv`](../drifting/result/adult/bench_lownfe_so1.csv) and
[`bench_lownfe_so2.csv`](../drifting/result/adult/bench_lownfe_so2.csv):

| Method | sampler | NFE | gen (s) | Shape ↓ | Trend ↓ | MLE ↑ | C2ST ↑ | α-prec ↑ | β-recall ↑ |
|---|---|---|---|---|---|---|---|---|---|
| **Drift 4× whitened** | — | **1** | **0.080** ±.003 | **1.30** ±0.04 | **3.78** ±0.13 | .8997 ±.0017 | **.9741** ±.0049 | **98.85** ±0.16 | 34.39 ±0.32 |
| TabDiff, 2 steps | 1st-order | 2 | 0.231 ±.001 | 13.11 ±0.04 | 57.70 ±0.66 | .4291 ±.0391 | .4603 ±.0023 | 97.42 ±0.11 | 5.45 ±0.07 |
| TabDiff, 2 steps | 2nd-order | 3 | — | 40.78 ±0.04 | 59.13 ±0.02 | — ⚠ | .0000 ±.0000 | **12.20** ±0.03 | **0.01** ±0.00 |
| TabDiff, 3 steps | 1st-order | 3 | 0.343 ±.007 | 6.10 ±0.06 | 27.95 ±5.54 | .7895 ±.0067 | .7831 ±.0048 | 96.53 ±0.30 | 25.38 ±0.18 |
| TabDiff, 3 steps | 2nd-order | 5 | — | 35.99 ±0.07 | 51.00 ±0.06 | .6706 ±.0367 | .0015 ±.0002 | 16.70 ±0.08 | 0.55 ±0.03 |
| TabDiff, 5 steps | 1st-order | 5 | 0.567 ±.002 | 5.01 ±0.06 | 16.47 ±1.79 | .8862 ±.0047 | .7849 ±.0035 | 97.03 ±0.29 | 40.67 ±0.22 |
| TabDiff, 5 steps | 2nd-order | 9 | 0.736 ±.009 | 14.34 ±0.08 | 24.41 ±0.09 | .8858 ±.0045 | .5405 ±.0127 | 65.36 ±0.33 | 9.77 ±0.18 |
| TabDiff, 10 steps | 1st-order | 10 | 1.105 ±.012 | 3.11 ±0.04 | 9.31 ±1.34 | **.9061** ±.0012 | .8800 ±.0052 | 96.99 ±0.25 | **50.55** ±0.20 |
| TabDiff, 10 steps | 2nd-order | 19 | 1.502 ±.011 | 1.97 ±0.09 | 3.89 ±0.08 | .9093 ±.0012 | .9286 ±.0077 | 97.16 ±0.25 | 44.81 ±0.32 |

**First-order is not just better on marginals at the cheap end — it is 5× better
on coverage.** At 10 NFE, first-order scores β-recall 50.55 against the
second-order config's 9.78, and 50.55 is **94% of what TabDiff's 100-NFE default
achieves (53.63) at a tenth of the cost**. The second-order-only sweep therefore
understated TabDiff's cheap-end coverage severely, and every statement in this
report about "what the extra NFE buys" was drawn from that understated curve.

**The corrector's damage at small step counts is on α-precision, and it is
enormous.** At matched NFE:

| budget | 1st-order α / β | 2nd-order α / β | α cost of the corrector |
|---|---|---|---|
| 2-3 NFE | 97.42 / 5.45 | 12.20 / 0.01 | **−85.2** |
| 5 NFE | 97.03 / 40.67 | 16.70 / 0.55 | **−80.3** |
| 9-10 NFE | 96.99 / 50.55 | 65.36 / 9.77 | −31.6 |
| 19 NFE | not measured | 97.16 / 44.81 | — |

First-order α-precision is essentially flat across 2→10 NFE (97.42, 96.53,
97.03, 96.99) while second-order has to climb from 12.20 and does not catch up
until ~19 NFE. And on coverage it never catches up in this range at all:
**first-order at 10 NFE beats second-order at 19 NFE on β-recall, 50.55 against
44.81, at half the network calls.** For the cheap regime, first-order dominates
the corrector on both axes.

Timings are 5 timed repeats after 2 warmup on an unshared RTX 6000 Ada, batch
4096 — `bench_speed_b4096_so1.csv` (first-order, slurm 264434) and
`bench_speed_b4096.csv` (second-order, slurm 145594), the same protocol for
both. The two 2nd-order rows at 2 and 3 steps were never clocked; their quality
already rules them out.

⚠ MLE failed on all 20 draws at 2 steps / 2nd-order: the target column collapses
to a single class and XGBoost raises `NotFittedError`. That is the metric
reporting total collapse, not a harness bug — the guard exists so one dead cell
costs that cell and not the sweep.

**The correction is a bad trade below ~10 steps, and a good one above.** At
matched step counts the second-order rows are catastrophically worse (2 steps:
40.78 against 13.11 Shape) — the corrector amplifies rather than corrects when
the step is enormous. By 10 steps it has flipped: 1.97 at 19 NFE against 3.11 at
10 NFE. So **at a fixed NFE budget of 10, the right opponent is 10 first-order
steps, not 5 second-order ones**: 3.11 Shape against 14.34, a 4.6× difference
that earlier text in this report never saw because it only ever swept the
second-order sampler.

**But NFE is not a faithful proxy for wall clock, and the two axes disagree
here.** With the first-order timings in hand the sweeps can be differenced: a
first-order step costs 109 ms and a second-order step 155 ms, so the corrector
buys 2× the NFE for 1.42× the time — second-order delivers ~1.41 NFE per unit
time, and *equal NFE is not equal cost*. At the fixed 10-NFE budget above, the
first-order config takes **1.50× the wall clock** of the second-order one
(1.105s against 0.736s). So:

| cost axis | the better TabDiff config at ~10 NFE / ~0.7-1.1s |
|---|---|
| **network calls** | 10 steps first-order — 3.11 Shape against 14.34 |
| **seconds** | ambiguous, and partly unmeasured (below) |

At equal wall clock the ordering flips with the budget: at ~0.6-0.7s first-order
still wins (5 steps, 0.567s, Shape 5.01, against second-order 5 steps, 0.736s,
Shape 14.34), but by ~1.1-1.5s second-order wins (10 steps, 1.502s, Shape 1.97,
against first-order 10 steps, 1.105s, Shape 3.11).

⚠ **The 0.74-1.50s window is unresolved**, and it is the interesting one: the
second-order quality curve collapses from 14.34 to 1.97 Shape across it and
there is no measured point in between (6, 7, 8 steps were never run). Any claim
about which sampler is better *per second* in that window is unsupported. The
NFE-axis claim above does not depend on it.

Drift is unaffected by all of this — it has one configuration — and is **13.7×
faster than the 10-NFE first-order config** and 9.2× faster than the 10-NFE
second-order one. The speed claim gets stronger against the better-quality
opponent, not weaker.

**Against that corrected opponent, drift wins four of six at 10 NFE** — Shape
1.30 vs 3.11, Trend 3.78 vs 9.31, C2ST .9741 vs .8800, α-precision 98.85 vs
96.99 — and loses the two coverage-flavoured ones: MLE .8997 vs .9061 (~4σ) and
**β-recall 34.39 vs 50.55, a 16-point gap**. The β-recall loss is much the larger
of the two and was invisible until the first-order α/β landed: against the
second-order config at the same budget drift *wins* β-recall 34.39 to 9.78. Same
split as everywhere else, but at 10 NFE the coverage side of it is now decisively
TabDiff's.

**Below 10 NFE there is no contest.** TabDiff's best 5-NFE config scores 5.01
Shape and .785 C2ST; its 2-NFE floor is 13.11 and .460, which is a coin flip.
Drift at 1 NFE is 1.30 and .974. This is the one regime where the one-step map
is not trading anything away.

Both columns that were missing from this section are now measured; the
provenance of each is below.

* ~~**α-precision / β-recall are missing for every row.**~~ **Done** (slurm
  264534; the history is kept because the failure mode is instructive). Stage B of
  `bench_lownfe.sbatch` died on all ten configs with `CUDA error: no kernel
  image is available for execution on the device`, and it piped stage B through
  `tail -5` under `set -o pipefail`, so the first failure both killed the loop
  and truncated its own traceback.

  The cause is **not** the one that job's comment assumed, and it is worth
  recording because the obvious fix makes it worse. α-precision is built on a
  one-class model that synthcity **caches to disk**
  ([`metrics/core/metric.py:110`](../../miniconda3/envs/synthcity/lib/python3.10/site-packages/synthcity/metrics/core/metric.py))
  keyed on a hash of the *real* data — so one fitted model is shared by every
  config and every run, ours written 2026-09-06 as
  `workspace/sc_metric_cache_model_oneclass_*.bkp`. It was pickled on a GPU box,
  so its tensors carry CUDA storage locations, and that single fact produces two
  different errors: on blackwell it deserializes onto CUDA where this torch
  (2.2.2) has no `sm_120` kernels; **CPU-only it refuses to deserialize at
  all** (`Attempting to deserialize object on a CUDA device but
  torch.cuda.is_available() is False`). A CPU-only retry of this job confirmed
  the second mode on all ten tasks in 12-18s.

  So `rescore_lownfe_alphabeta.sbatch` redoes stage B **with** a GPU, pinned off
  blackwell, one array task per config, with a 2-second arch smoke test so an
  unsupported node fails legibly. Deleting the cache would also work but is the
  wrong call: refitting yields a different one-class model, and these rows would
  then not be comparable to the α/β already published off the Sep 6 cache
  (`bench_drift` 98.80 / 34.32). Reusing it is what keeps them joinable. The 20
  draws per config are already staged, so nothing is regenerated.
* ~~**Timings.**~~ **Done** (slurm 264434). The `gen (s)` column of the low-NFE
  *quality* CSVs comes from a shared GPU and reads ~2× slow (it clocks 5 steps /
  2nd-order at 1.685s against the clean 0.736s), so it is never quoted;
  `bench_lownfe_timing.sbatch` re-measured the first-order configs under
  byte-identical protocol to `bench_speed.sbatch` task 0. Drift re-clocked at
  0.080 ±.003 against the original run's 0.082, which is the protocol check: the
  two sweeps are joinable. Outstanding from this: second-order at 6-8 steps, to
  close the wall-clock window above.
### Equal-quality crossover — what TabDiff must spend to match drift

Every comparison above fixes the *budget* and asks who wins. Fixing the
*quality* and asking what the budget must be is the fairer question, and it is
the one a practitioner actually faces. Read off the 20-draw frontier: the first
NFE at which TabDiff matches or beats drift's 1-NFE value, per metric.

Each bracket uses **the best TabDiff configuration available at that budget**,
first- or second-order, so it is the fairest form of the question rather than the
most flattering one.

| metric | drift @ 1 NFE | TabDiff matches it at | cost ratio |
|---|---|---|---|
| **α-precision ↑** | **98.80** | **never, up to 200 NFE** (its best is 98.58 at 50) | **∞** |
| C2ST ↑ | .9734 | between 20 and 50 NFE | 20-50× |
| Shape ↓ | 1.32 | between 20 and 50 NFE | 20-50× |
| Trend ↓ | 3.73 | between 20 and 50 NFE | 20-50× |
| MLE ↑ | .8997 | between 5 and 10 NFE (1st-order: .8862 → .9061) | 5-10× |
| **β-recall ↑** | 34.32 | **between 3 and 5 NFE** (1st-order: 25.38 → 40.67) | **3-5×** |

Brackets, not interpolations — the crossover is located between measured points,
never fitted between them.

**An earlier version of this table put β-recall at 10-20 NFE and MLE at ≤10.**
Both were computed from the second-order sweep alone, where the cheapest point
is 10 NFE and scores β-recall 9.78. The first-order α/β rescore (slurm 264534)
shows TabDiff reaching 25.38 at 3 NFE and 40.67 at 5 — so it overtakes drift on
coverage at **three to five network calls**, not ten to twenty. The coverage
deficit is far cheaper to buy than this report previously stated.

**α-precision is not a bracket, it is a ceiling.** TabDiff does not reach 98.80
at any configuration tested — **13 of them**, spanning 2 to 200 NFE and both
samplers. Its global maximum is 98.58, at 50 NFE. This is not an extrapolation
artifact: α-precision *peaks* there and then falls — 98.18 at 100 NFE, 97.99 at
200 — because more steps buy coverage and cost precision, so spending more
cannot get there. Nor is it a sampler artifact: first-order tops out at 97.42.
Drift's precision is unmatched at every budget on the diffusion model's own step
axis, in either sampler.

**The split is the same one as everywhere else, now priced — and the price gap
is far wider than it first looked.** The two metrics TabDiff matches cheaply
(β-recall 3-5×, MLE 5-10×) are the two coverage-flavoured ones. The three it
needs 20-50× for, plus the one it never reaches, are the precision- and
fidelity-flavoured ones. So the one-step map's deficit is purchasable for **3-5×
the compute** while its advantage is not purchasable at all — an order of
magnitude separates the two sides of the trade, not the ~2× the second-order-only
numbers implied.

**This is the framing to lead with**, in preference to either of the two used
earlier in this report's history:

* *best vs best* (drift 1.24 at 1 NFE against TabDiff 0.739 at 100) compares at
  100× unequal cost and is the convention, but it answers only "which model is
  best if compute is free".
* *equal budget* answers "who wins at 10 NFE", which depends on the sampler and
  turned out to be ambiguous by 4.6× on Shape until the first-order sweep ran.
* *equal quality*, above, is the one that needs no caveat about which
  configuration was picked, because it quantifies over all of them.
### Reading

**Speedup against TabDiff's default is batch-dependent: 94× at batch 4096, 57×
at batch 32,561.** Quote the conservative figure. The gap narrows because drift
is already overhead-dominated — 0.082s to 0.085s across an 8× batch change,
essentially flat — while TabDiff gains 1.6× from the larger batch. A single
speedup number without its batch size is not a measurement.

**Drift is Pareto-optimal but does NOT dominate TabDiff's cheap regime.** Nothing
in the table is both faster and better, because nothing is faster. But the
stronger claim made before the full metric suite existed — that TabDiff's whole
sub-25-step regime is dominated — is **false**, and the four added columns are
what falsified it:

* **vs 5 steps, second-order (10 NFE): genuine dominance.** Drift wins all six
  metrics and is 9× faster. TabDiff at 5 second-order steps is not a usable
  configuration — C2ST 0.54 is coin-flip, β-recall 9.78 is near-total mode
  collapse. **But this is not TabDiff's best use of 10 NFE** — 10 first-order
  steps score 3.11 Shape against 14.37, and against *that* opponent drift wins
  four of six rather than all of them — it loses MLE and, by 16 points,
  β-recall. See the low-NFE section above.
* **vs 10 steps (20 NFE): incomparable, not a win.** Drift takes Shape
  (1.32 vs 1.92), C2ST (.9734 vs .9276) and α-precision (98.80 vs 96.99);
  TabDiff takes MLE (.9092 vs .8997, a ~6σ gap) and β-recall (44.67 vs 34.32).
  Trend is a tie — 3.73 vs 3.86 against a combined spread of ~0.19. Drift is
  18× faster, but "faster and better on everything" is not available here.

**The precision/recall split is the real finding.** Drift has the **highest
α-precision in the table** — 98.80, above every TabDiff setting including the
100-step one (97.99) — while its β-recall (34.32) sits between TabDiff's 5-step
(9.78) and 10-step (44.67) values. Its marginals are worth ~25 second-order
steps; its coverage is worth **3 to 5 first-order NFE** — the first-order sweep
reaches 40.67 β-recall at 5 NFE and 50.55 at 10, so against TabDiff's *best*
cheap configuration the coverage deficit is far starker than the second-order
numbers in this paragraph suggest. A one-step deterministic map that produces
highly in-distribution samples over too little of the support is exactly the
signature the toy-2D gate showed (87% mode capture, samples bridging between
modes), now confirmed on real data.

**More sampling steps buy coverage, and cost precision.** Along TabDiff's own
step axis β-recall rises monotonically — 9.78 → 44.67 → 52.64 → 53.63 → 54.08 —
while α-precision peaks at 25 steps and falls after (98.58 → 98.18 → 97.99).
That is what the extra NFE is purchasing, and it is the axis drift cannot buy
into at one step.

**MLE and C2ST behave differently from each other.** MLE saturates almost
immediately (.9092 by 10 steps, .912 thereafter) and barely discriminates. C2ST
separates the cheap end sharply (0.54 → 0.93 → 0.98) and is the more useful of
the two on this frontier.

**TabSyn is not in these tables.** Different repo and env, so it cannot be timed
in-process. The paper's Table 12 puts it at 1.99 Shape / 3.92 Trend at 10 steps,
which drift beats on both at 1 NFE — but that is a cross-environment claim and
weaker than everything above.

### What these numbers are and are not

**Quality columns are 20-draw means, comparable to the tables above.** An earlier
version of this section carried single draws, which read ~0.15 high on Shape
(drift 1.36 against its 1.241, TabDiff 0.82 against 0.739). Those are gone;
`bench_quality.sbatch` now re-scores every point over 20 draws. Drift's 1.32 here
against 1.241 in the standings table is a genuine residual difference — a
different checkpoint load path and no EMA selection — not a protocol mismatch.

**Timing columns are single draws, deliberately.** A clock should not be
measuring metric evaluation. The two jobs use two protocols on purpose and must
not be merged.

**Timings require the `ada` partition.** They were measured on its A6000s;
`bench_speed.sbatch` stays pinned there. `bench_quality.sbatch` runs on any
partition because GPU model cannot change a metric value.

The NaN path was ruled out as a cause of the old single-draw gap:
`sample_all(keep_nan_samples=True)` zeroed 0 of 32,561 rows, so TabDiff is not
penalised by the zero-fill.

**Cross-environment against the paper's Table 12**: TabDiff scores 14.37 here at
5 steps against their 12.51, and 1.92 at 10 steps against their 1.55 — the same
direction as the repro gap documented under Caveats.

---

## Reading

**Drift beats six of the nine published baselines.** CTGAN, TVAE, GOGGLE, GReaT,
STaSy and CoDi are all behind it on Shape, Trend and C2ST — usually by an order
of magnitude (Shape 1.24 against 11–21). On MLE it beats CTGAN, TVAE, GOGGLE and
CoDi but trails GReaT, STaSy and the three diffusion models.

Placement per metric: **Shape 3rd of 10, Trend 4th, C2ST 4th, MLE 5th.**

**Against the one-step cohort it wins on distributional fidelity — but not,
probably, on coverage.** CTGAN, TVAE and GOGGLE are also single-pass generators;
a GAN generator or VAE decoder is one forward pass, the same cost as ours.

| | Shape ↓ | Trend ↓ | MLE ↑ | C2ST ↑ | α-prec ↑ | β-recall ↑ |
|---|---|---|---|---|---|---|
| CTGAN | 16.84 | 20.23 | .886 | .595 | 77.74 | 30.80 |
| TVAE | 14.22 | 14.15 | .878 | .632 | **98.17** | **38.87** |
| GOGGLE | 16.97 | 45.29 | .778 | .111 | 50.68 | 8.80 |
| **Drift 4× whitened** | **1.24** | **3.78** | **.900** | **.974** | **98.92** | 34.36 |

On the four measured metrics it wins outright: 11× lower Shape error, 4× lower
Trend, C2ST from ~0.6 to 0.97. That is the defensible claim — drift closes most
of the distance between the one-step and multi-step tiers at one-step cost.

**Measured: it wins five of six, and loses β-recall.** α-precision came in at
98.92, above TVAE's 98.17 — and above our own TabDiff repro's 98.37. β-recall is
the single loss: 34.36 against TVAE's 38.87 (it does beat CTGAN's 30.80).

This was predicted the wrong way round. The TabSyn port's numbers (α 0.898,
β 0.286) suggested drift would lose *both*; in this port it wins α-precision
outright. Coverage is the real and only weakness, and it is large: 34.36 against
TabDiff's 53.67, a 19-point gap, far bigger than the ~1.3 point gaps on the
distributional metrics.

So the honest one-line summary of the method is not "drift wins the one-step
cohort" but: **it matches or beats multi-step diffusion on precision and
marginals, and pays for it in coverage.** A one-step deterministic map onto a
disconnected support is structurally bad at coverage; the 2D gate showed the same
signature (87% mode capture, samples bridging between modes). Conditional
generation is the untested lever aimed straight at this — within-class repulsion
is what should improve coverage — and `--group_by target` has never been run.

**DCR admits no comparison here at all.** The paper reports DCR only for STaSy,
CoDi, TabDDPM, TabSyn and TabDiff — none of the three one-step baselines have a
published number.

**Only the whitened φ arm appears in this table.** ReMasker was never given a
200k run on the 90/10 split (only 30k in the sweep, 0.9546 against whitened's
0.9726) and its full-length numbers exist only on the DCR split. Whitened is the
right representative because it won, but no ReMasker row is available.

**The one crossover.** Drift beats TabDDPM on Shape (1.24 vs 1.75) but loses on
Trend (3.78 vs 3.01) — marginals reach the diffusion tier, column-pair structure
does not. That asymmetry is the through-line of every result so far.

---

## Where Adult stands

Final configuration: **unconditional, whitened φ, R=4×, 200k steps** — the
conditional and ReMasker variants were both tested and both lost (see below).
Rank is against the nine published baselines plus us; DCR is ranked by distance
from the ideal 50%, and only five baselines report it.

| metric | ours | rank | beats | loses only to |
|---|---|---|---|---|
| Shape ↓ | 1.241 | **3 / 10** | 7 of 9 | TabSyn, TabDiff |
| α-precision ↑ | 98.92 | **3 / 10** | 7 of 9 | TabSyn, TabDiff |
| Trend ↓ | 3.780 | 4 / 10 | 6 of 9 | TabDDPM, TabSyn, TabDiff |
| C2ST ↑ | 0.9736 | 4 / 10 | 6 of 9 | TabDDPM, TabSyn, TabDiff |
| DCR →50 | 50.90 | 4 / 6 | 2 of 5 | STaSy, CoDi, TabDiff |
| MLE ↑ | 0.8998 | 6 / 10 | 4 of 9 | GReaT, STaSy, TabDDPM, TabSyn, TabDiff |
| β-recall ↑ | 34.36 | 6 / 10 | 4 of 9 | TVAE, GReaT, TabDDPM, TabSyn, TabDiff |

**It beats every pre-diffusion baseline** — CTGAN, TVAE, GOGGLE, GReaT, STaSy,
CoDi — on Shape, Trend, C2ST and α-precision, and sits at the bottom of the
diffusion tier on those. Against our own TabDiff reproduction it is behind by
0.50 Shape, 2.20 Trend, 1.3 MLE points and 1.5 C2ST points, **ahead** on
α-precision (98.92 vs 98.37), and behind by 19.3 points on β-recall.

**The two weak metrics are the same weakness.** MLE and β-recall are both 6th,
and both are coverage-sensitive: β-recall measures support coverage directly,
and a downstream classifier trained on synthetic data suffers when regions of the
input space are missing. Precision-flavoured metrics (Shape, α-precision) rank
3rd; coverage-flavoured ones rank 6th. That is one deficit reported twice.

**On efficiency it wins the cheap end, but "no trade-off" was too strong.**
Comparing only against TabDiff's default (100 NFE) is the least favourable
framing, because TabDiff has a step knob. Sweeping it, 20 draws per point at
batch 4096 (quality from `bench_quality.csv`, clean timings from
`bench_speed_b4096.csv`):

| | NFE | gen (s) | Shape ↓ | Trend ↓ | MLE ↑ | C2ST ↑ | vs drift |
|---|---|---|---|---|---|---|---|
| **Drift 4×** | **1** | **0.082** | **1.32** | **3.73** | .8997 | **.9734** | — |
| TabDiff, 10 steps, 1st-order | 10 | 1.105 | 3.11 | 9.31 | **.9061** | .8800 | 14× slower; better MLE, worse on the other three |
| TabDiff, 10 steps, 2nd-order | 20 | 1.502 | 1.92 | 3.86 | .9092 | .9276 | 18× slower; splits the metrics |
| TabDiff, 25 steps, 2nd-order | 50 | 3.813 | 0.87 | 1.73 | .9117 | .9835 | 46× slower, better quality |
| TabDiff, 50 steps, 2nd-order | 100 | 7.702 | 0.90 | 1.64 | .9121 | .9862 | 94× slower, better quality |

An earlier version of this section claimed that at 20 NFE "there is no dimension
on which TabDiff is the better choice" and that drift "wins outright" below
~20 NFE. **Both statements are retracted.** They were computed from single draws
on Shape and Trend alone; the 20-draw rescore with MLE, C2ST, α-precision and
β-recall contradicts them. See
[Adult — sampling efficiency](#adult--sampling-efficiency), which is the
authoritative version. In short:

* **10 NFE, second-order (5 steps)** — genuine dominance, all six metrics.
* **10 NFE, first-order (10 steps)** — drift wins Shape (1.32 vs 3.11), Trend
  (3.73 vs 9.31) and C2ST (.9734 vs .8800); **loses MLE** (.8997 vs .9061, ~4σ).
  Three of four, not four of four.
* **20 NFE** — split, not a win: drift takes Shape, C2ST and α-precision,
  TabDiff takes MLE and β-recall, Trend ties.

**The second-order flag is a confound in every earlier comparison.** At a fixed
NFE budget, second-order correction buys half as many steps, and at the cheap end
the steps are worth more than the correction: at 10 NFE, first-order is 4.6×
better on Shape (3.11 against 14.37). Earlier text compared drift against the
second-order config at 10 NFE, which is the weaker of TabDiff's two options at
equal cost. Any low-NFE claim must now say which sampler it ran.

So the shape of the result is three regimes, not two:

* **1-2 NFE** — drift is unopposed. TabDiff has no valid 1-NFE configuration
  (`sample()` builds its grid as `linspace(0, 1, T)`, so T=1 draws the prior at
  `sigma_min`, not `sigma_max` — it starts from almost-clean data), and at its
  genuine 2-NFE floor it has collapsed: Shape 13.11, C2ST 0.46.
* **~10-20 NFE** — drift wins the distributional metrics and loses the
  coverage-flavoured ones. Which model is "better" depends on which column the
  downstream use cares about.
* **at or above ~50 NFE** — TabDiff wins on quality, at 46-94× the time.

The honest efficiency claim is therefore "**drift owns the sub-10-NFE regime
where TabDiff is unusable, trades metric-for-metric from ~10-20 NFE, and
TabDiff buys the remaining ~0.4 Shape points at 46× the cost**".

**One-line summary.** At ~100× fewer network evaluations, drift matches or beats
multi-step diffusion on precision and marginals, and gives up coverage. Below
~10 NFE there is no competing configuration at all; from ~10 NFE up it is a
genuine trade, not a free win.

⚠ **Single dataset.** Every number in this section is adult. There is no TabDiff
checkpoint for any other dataset in this repo (`tabdiff/ckpt/` holds only
`adult` and `adult_dcr`), so the NFE frontier has never been measured elsewhere.
The bandwidth sweep that does cover six other datasets found that **R=4× — the
configuration all of these rows use — does not transfer** (8× wins magic and
diabetes while ranking second-worst on adult). The 1-NFE cost is architectural
and would carry over; the quality at 1 NFE has not been shown to.

---

## Conditional generation does not fix coverage

β-recall was the method's one large deficit (34.36 against TabDiff's 53.67, a
19.3-point gap, while every distributional metric sat within ~1.3 points).
Conditioning was the lever aimed at it: `--group_by target` makes the kernel
group axis the target column, so repulsion is *within class* rather than global,
which is the structure Drifting was designed around and should help exactly where
coverage fails first — the minority class (24% of adult).

Measured over 200k steps, both phi variants, with the unconditional runs as
controls:

| run | Shape ↓ | Trend ↓ | MLE ↑ | C2ST ↑ | α-prec ↑ | β-recall ↑ |
|---|---|---|---|---|---|---|
| TabDiff — our repro | 0.739 | 1.583 | .9127 | .9887 | 98.37 | **53.67** |
| uncond whitened | **1.241** | **3.780** | .8998 | **.9736** | **98.92** | 34.36 |
| cond whitened | 1.429 | 3.999 | **.9004** | .9585 | 98.77 | 35.50 |
| uncond remasker | 2.865 | 6.346 | .8903 | .8179 | 96.37 | 29.22 |
| cond remasker | 2.669 | 5.740 | .8950 | .8572 | 94.37 | 32.15 |

**The hypothesis is wrong.** Conditioning moves β-recall +1.14 for whitened —
real at ~4σ, and about **6% of the 19.3-point gap**. It also costs 0.19 Shape,
0.22 Trend and 0.015 C2ST. For whitened that is a bad trade, and unconditional
remains the recommended setting.

Within-class repulsion is therefore not what limits coverage. The remaining
explanation is the one-step map itself: a continuous deterministic function of
noise cannot cover a disconnected support without tearing, and the 2D gate showed
the same signature (87% mode capture, samples bridging between modes). This
matches the TabSyn port's own conclusion that "single-step generation, not the
loss bug, is the binding constraint" — reached there by a different route.

**This also closes the ReMasker question.** At the full 200k budget it loses on
every metric measured, coverage included: α-precision 96.37 against whitened's
98.92, β-recall 29.22 against 34.36, Shape 2.865 against 1.241. The 30k sweep
result was not a budget artefact. Conditioning helps ReMasker more than it helps
whitened (+2.93 β-recall, and it improves every density metric), but from so far
behind that it never becomes competitive.

---

## Caveats

**Compare against `TabDiff — our repro`, not the paper row.** It is the only
multi-step number produced in this environment.

**The reproduction is close but not exact.** MLE reproduces essentially exactly
(.9127 vs .912), but Shape (0.739 vs 0.63) and Trend (1.583 vs 1.49) both sit
outside the paper's quoted std, and C2ST is 0.989 vs 0.995 — a systematic
shortfall of roughly 0.1 points on the density metrics. Not a failed
reproduction, but not within noise either. Measured against our own repro the
drift gap is +0.50 Shape and +2.20 Trend, slightly smaller than against the
paper row.

**The baseline column is inherited.** Per the paper's own footnote, every
baseline except TabSyn (and everything on Diabetes) is taken from
Zhang et al. (2024) — a different environment and different runs. Cross-row
differences of ~0.1 should not be read as meaningful.

**The scored summaries are in the repo; the raw draws are not.** `eval/` and
`drifting/result/` are blanket-ignored, which previously left every table here
unverifiable from a clean checkout. The per-point summaries are now tracked —
`drifting/result/<data>/bench_*.csv` and
`eval/report_runs/<exp>/<data>/{avg_std,quality}.csv`, ~250 KB total — so any
number in this document can be checked against its source without rerunning
anything, and `quality.csv` carries the per-draw values behind each mean ± std.
The generated tables themselves (`all_samples/`, 2.3 GB) and every `generator.pt`
stay ignored. The `.gitignore` negation rules that carve this out are order- and
slash-sensitive; the comment there says how to verify a change to them.

**Single training seed** for every `ours` row. The ±values are sampling spread
across 20 draws from one trained model, not seed-to-seed variation. The TabSyn
repro is a single draw with no spread at all.

**The two reproductions disagree with their own papers in opposite directions.**
Our TabDiff repro is *worse* than the paper on Shape (0.739 vs 0.63); our TabSyn
repro is *better* than TabDiff's reported TabSyn on Shape (0.77 vs 0.81) but
worse on Trend (2.27 vs 1.93). The TabSyn paper itself reports 0.58/1.54 for the
same model. So there is roughly 0.2 points of environment-to-environment scatter
on these metrics, which is larger than several of the differences in the table.

---

## Pending

| what | job | status |
|---|---|---|
| TabDiff DCR on the 50/50 split | 145542 → 145543 | training (144920 died on a missing `kaleido`) |
| Drift DCR, whitened φ | 144925_0 | done — 50.90 ±0.24 |
| Drift DCR, ReMasker φ | 144925_1 | done — 50.33 ±0.35 |
| Drift DCR, ReMasker φ random-init (control) | 144925_2 | done — 50.42 ±0.35 |

DCR needs every model retrained on an equal 50/50 split (`adult_dcr`), since the
default split is 90/10 — it cannot be computed from the runs above.

The three drift arms differ only in how much training-set information φ carries:
`WhitenedIdentity` holds ~110 parameters and cannot encode individual rows;
pretrained ReMasker is a transformer fit on the training half and can; the
random-init arm has the same architecture and output dimensionality with no
training at all. If the ReMasker arm shows worse DCR, the control says whether
that came from the generator or from φ.

| α-precision / β-recall, TabDiff repro + drift | 145555 | running |
| Sampling latency sweep, both batch sizes | 145594 | done — see Sampling efficiency |
| Low-NFE sweep, quality (both samplers) | 151132 | done — see The NFE floor |
| Low-NFE α-precision / β-recall (stage B redo) | 264433 → 264447 → 264534 | done — 264433 failed CPU-only on the GPU-pickled oneclass cache, 264447 on a cache poisoned by a concurrent job; 264534 isolated the cache dir and scored all 10 |
| Low-NFE first-order clean timings | 264434 | done — `bench_speed_b4096_so1.csv`; 109 ms/step vs 155 ms 2nd-order |
| 2nd-order at 6-8 steps, to close the 0.74-1.50s window | not submitted | — |

α-precision and β-recall need the separate `synthcity` env (`eval/eval_quality.py`)
and are not part of `TabMetrics`, which is why no earlier job produced them.

**DCR is in and the memorisation risk did not materialise.** Drift's 50.90 sits
between TabSyn (50.94) and TabDDPM (51.14) and well inside the published range.
The random-init control answers its question: pretrained ReMasker φ (50.33) and
random-init φ (50.42) are statistically identical, so φ's pretraining is not what
moves DCR — the shift relative to whitened comes from the feature space's
geometry, not from φ having memorised the training half.

**Not yet run: other datasets, training seeds.** Worth stating plainly, because
it bounds every claim in this document: there is no TabDiff checkpoint outside
`adult` / `adult_dcr`, so the entire NFE and latency frontier is a one-dataset
result. The six datasets with drift runs have only the bandwidth sweep, and that
sweep found R=4× does not transfer — so porting the frontier means re-sweeping R
per dataset *and* training a TabDiff baseline there, not just rerunning
`bench_speed`.
