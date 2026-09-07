# Adult — comparison against the TabDiff paper's baselines

**Living document.** Updated as runs land. Last updated 2026-09-06.

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

| Method | NFE | gen (s) | rows/s | Shape ↓ | Trend ↓ |
|---|---|---|---|---|---|
| **Drift 4× whitened** | **1** | **0.082** ±.000 | **295,503** | 1.36 | 3.79 |
| TabDiff, 5 steps | 10 | 0.736 ±.009 | 42,644 | 14.28 | 24.36 |
| TabDiff, 10 steps | 20 | 1.502 ±.011 | 21,299 | 1.88 | 3.80 |
| TabDiff, 25 steps | 50 | 3.813 ±.009 | 8,481 | 0.86 | 1.60 |
| TabDiff, 50 steps (default) | 100 | 7.702 ±.030 | 4,213 | 0.82 | 1.45 |
| TabDiff, 100 steps | 200 | 15.462 ±.089 | 2,102 | 0.93 | 1.60 |

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

### Reading

**Speedup against TabDiff's default is batch-dependent: 94× at batch 4096, 57×
at batch 32,561.** Quote the conservative figure. The gap narrows because drift
is already overhead-dominated — 0.082s to 0.085s across an 8× batch change,
essentially flat — while TabDiff gains 1.6× from the larger batch. A single
speedup number without its batch size is not a measurement.

**Drift sits on the Pareto frontier: no TabDiff setting is both faster and
better.** At batch 4096, TabDiff's 5-step config is 9× slower and far worse
(14.28 Shape); its 10-step config is 18× slower for a worse Shape and a tied
Trend; it first genuinely beats drift at 25 steps, where it is 47× slower. The
same ordering holds at batch 32,561 (5× / 11× / 28×).

**At 20 NFE the win is Shape only, and Trend is a tie.** Drift 1.36 / 3.79
against TabDiff 1.88 / 3.80. The Shape margin (0.52) is ~10× the 20-run
sampling spread and is real; the Trend margin (0.01) is noise. An earlier
head-node run showed 3.81 vs 3.88 and that 0.07 was noise too — this is a tie,
not a win, and should not be written up as one.

**TabSyn is not in these tables.** Different repo and env, so it cannot be timed
in-process. The paper's Table 12 puts it at 1.99 Shape / 3.92 Trend at 10 steps,
which drift beats on both at 1 NFE — but that is a cross-environment claim and
weaker than everything above.

### What these Shape/Trend numbers are not

**Single draws, not the 20-run averages of the tables above.** Both models were
measured through the identical code path, so the relative ordering is sound, but
the absolutes must not be quoted as either model's quality — drift reads 1.36
here against its reported 1.241, TabDiff 0.82 against 0.739. TabDiff's
non-monotonicity across steps (0.86 → 0.82 → 0.93) is the size of that same
single-draw noise.

The NaN path was ruled out as a cause: `sample_all(keep_nan_samples=True)`
zeroed 0 of 32,561 rows, so TabDiff is not penalised by the zero-fill.

**Quality is environment-independent, timing is not.** A preliminary head-node
run under GPU contention gave the same quality to within 0.08 on every point
(TabDiff@10 1.90/3.88, drift 1.35/3.81) while its timings ran ~1.5× slow. Kept
at `bench_speed_b4096_headnode.csv`; the slurm numbers above supersede it.

**Cross-environment against the paper's Table 12**: TabDiff scores 14.28 here at
5 steps against their 12.51, and 1.88 at 10 steps against their 1.55 — the same
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

**On efficiency it does more than win — it dominates an operating point.**
Comparing only against TabDiff's default (100 NFE) is the least favourable
framing, because TabDiff has a step knob. Sweeping it (single draws, same code
path, batch 4096):

| | NFE | gen (s) | Shape ↓ | Trend ↓ | vs drift |
|---|---|---|---|---|---|
| **Drift 4×** | **1** | **0.082** | **1.36** | 3.79 | — |
| TabDiff, 10 steps | 20 | 1.502 | 1.88 | 3.80 | **18× slower and worse on Shape** |
| TabDiff, 25 steps | 50 | 3.813 | 0.86 | 1.60 | 46× slower, better quality |
| TabDiff, 50 steps | 100 | 7.702 | 0.82 | 1.45 | 94× slower, better quality |

At 20 NFE drift is **strictly better on Shape, tied on Trend, and 18× faster** —
Pareto-dominant, not a trade. TabDiff has to spend 50 NFE, 46× the wall clock,
before it beats drift on quality at all. The honest efficiency claim is therefore
not "94× faster and somewhat worse" but "**dominates the whole low-NFE regime,
and buys the remaining ~1.3 quality points at 46× the cost**".

**One-line summary.** At ~100× fewer network evaluations, drift matches or beats
multi-step diffusion on precision and marginals, and gives up coverage. Against
TabDiff's own low-step configurations it is Pareto-dominant.

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

α-precision and β-recall need the separate `synthcity` env (`eval/eval_quality.py`)
and are not part of `TabMetrics`, which is why no earlier job produced them.

**DCR is in and the memorisation risk did not materialise.** Drift's 50.90 sits
between TabSyn (50.94) and TabDDPM (51.14) and well inside the published range.
The random-init control answers its question: pretrained ReMasker φ (50.33) and
random-init φ (50.42) are statistically identical, so φ's pretraining is not what
moves DCR — the shift relative to whitened comes from the feature space's
geometry, not from φ having memorised the training half.

Not yet run: other datasets, training seeds.
