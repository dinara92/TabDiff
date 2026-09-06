# Adult — comparison against the TabDiff paper's baselines

**Living document.** Updated as runs land. Last updated 2026-09-06.

Shape and Trend are **error rates (%), lower is better** — the paper's convention.
MLE (AUC) and C2ST are **scores, higher is better**. DCR is a probability where
**50% is ideal**; higher means synthetic rows sit closer to the training set than
a real holdout row does.

Rows marked `ours` are 20-run means ± std, matching TabDiff's own reporting
protocol (`tabdiff/trainer.py` `report_test`). Rows marked `paper` are Table 1,
2, 3, 9 and 10 of `tabdiff-2410.20626v3.pdf`.

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
| **TabDiff — our repro** | 0.739 ±0.047 | 1.583 ±0.062 | .9127 ±.0012 | .9887 ±.0048 | pending | pending | ~100 | ours |
| **Drift 4× whitened** | 1.241 ±0.043 | 3.780 ±0.070 | .8998 ±.0014 | .9736 ±.0056 | pending | pending | **1** | ours |
| Drift 2× whitened | 1.443 ±0.082 | 4.186 ±0.173 | .8974 ±.0010 | .9559 ±.0059 | pending | — | 1 | ours |

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
| TabDiff — our repro | pending (145543) | ~100 | ours |
| Drift 4× ReMasker φ | 50.33 ±0.35 | 1 | ours |
| Drift 4× ReMasker rand-init | 50.42 ±0.35 | 1 | ours |
| Drift 4× whitened φ | 50.90 ±0.24 | **1** | ours |

TabSyn's own repo also reports a DCR for Adult (0.674) but on the **original ~2:1
split**, where the no-leakage target is ~0.674 rather than 50%. It scored exactly
its target, but the number is a different quantity on a different scale and is
deliberately excluded rather than converted.

---

## Reading

**Drift beats six of the nine published baselines.** CTGAN, TVAE, GOGGLE, GReaT,
STaSy and CoDi are all behind it on Shape, Trend and C2ST — usually by an order
of magnitude (Shape 1.24 against 11–21). On MLE it beats CTGAN, TVAE, GOGGLE and
CoDi but trails GReaT, STaSy and the three diffusion models.

Placement per metric: **Shape 3rd of 10, Trend 4th, C2ST 4th, MLE 5th.**

**Against the one-step cohort it wins on everything.** CTGAN, TVAE and GOGGLE are
also single-pass generators — a GAN generator or VAE decoder is one forward
pass, the same cost as ours.

| | Shape ↓ | Trend ↓ | MLE ↑ | C2ST ↑ |
|---|---|---|---|---|
| CTGAN | 16.84 | 20.23 | .886 | .595 |
| TVAE | 14.22 | 14.15 | .878 | .632 |
| GOGGLE | 16.97 | 45.29 | .778 | .111 |
| **Drift 4×** | **1.24** | **3.78** | **.900** | **.974** |

11× lower Shape error, 4× lower Trend, and C2ST from ~0.6 to 0.97. This is the
defensible claim: drift closes most of the distance between the one-step and
multi-step tiers at one-step cost.

**The one crossover.** Drift beats TabDDPM on Shape (1.24 vs 1.75) but loses on
Trend (3.78 vs 3.01) — marginals reach the diffusion tier, column-pair structure
does not. That asymmetry is the through-line of every result so far.

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

α-precision and β-recall need the separate `synthcity` env (`eval/eval_quality.py`)
and are not part of `TabMetrics`, which is why no earlier job produced them.

**DCR is in and the memorisation risk did not materialise.** Drift's 50.90 sits
between TabSyn (50.94) and TabDDPM (51.14) and well inside the published range.
The random-init control answers its question: pretrained ReMasker φ (50.33) and
random-init φ (50.42) are statistically identical, so φ's pretraining is not what
moves DCR — the shift relative to whitened comes from the feature space's
geometry, not from φ having memorised the training half.

Not yet run: other datasets, training seeds.
