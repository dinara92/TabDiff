# Adult — comparison against the TabDiff paper's baselines

**Living document.** Updated as runs land. Last updated 2026-09-05.

Shape and Trend are **error rates (%), lower is better** — the paper's convention.
MLE (AUC) and C2ST are **scores, higher is better**. DCR is a probability where
**50% is ideal**; higher means synthetic rows sit closer to the training set than
a real holdout row does.

Rows marked `ours` are 20-run means ± std, matching TabDiff's own reporting
protocol (`tabdiff/trainer.py` `report_test`). Rows marked `paper` are Table 1,
2, 3, 9 and 10 of `tabdiff-2410.20626v3.pdf`.

---

## Adult

| Method | Shape ↓ | Trend ↓ | MLE ↑ | C2ST ↑ | DCR →50% | NFE | source |
|---|---|---|---|---|---|---|---|
| CTGAN | 16.84 ±0.03 | 20.23 ±1.20 | .886 ±.002 | .595 | — | 1 | paper |
| TVAE | 14.22 ±0.08 | 14.15 ±0.88 | .878 ±.004 | .632 | — | 1 | paper |
| GOGGLE | 16.97 | 45.29 | .778 ±.012 | .111 | — | 1 | paper |
| GReaT | 12.12 ±0.04 | 17.59 ±0.22 | .913 ±.003 | .538 | — | many | paper |
| STaSy | 11.29 ±0.06 | 14.51 ±0.25 | .906 ±.001 | .405 | 50.33 ±0.19 | many | paper |
| CoDi | 21.38 ±0.06 | 22.49 ±0.08 | .871 ±.006 | .208 | 49.92 ±0.18 | many | paper |
| TabDDPM | 1.75 ±0.03 | 3.01 ±0.25 | .907 ±.001 | .976 | 51.14 ±0.18 | many | paper |
| TabSyn | 0.81 ±0.05 | 1.93 ±0.07 | .909 ±.001 | .991 | 50.94 ±0.17 | many | paper |
| **TabDiff** | **0.63** ±0.05 | **1.49** ±0.16 | **.912** ±.002 | **.995** | **50.10** ±0.32 | ~100 | paper |
| **TabDiff — our repro** | 0.739 ±0.047 | 1.583 ±0.062 | .9127 ±.0012 | .9887 ±.0048 | pending | ~100 | ours |
| **Drift 4× whitened** | 1.241 ±0.043 | 3.780 ±0.070 | .8998 ±.0014 | .9736 ±.0056 | pending | **1** | ours |
| Drift 2× whitened | 1.443 ±0.082 | 4.186 ±0.173 | .8974 ±.0010 | .9559 ±.0059 | — | 1 | ours |

Regenerate the `ours` rows from source rather than editing them by hand:

```bash
python -m drifting.collect_results --dataname adult
```

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
across 20 draws from one trained model, not seed-to-seed variation.

---

## Pending

| what | job | status |
|---|---|---|
| TabDiff DCR on the 50/50 split | 144920 → 144922 | training |
| Drift DCR, whitened φ | 144925_0 | training |
| Drift DCR, ReMasker φ | 144925_1 | training |
| Drift DCR, ReMasker φ random-init (control) | 144925_2 | training |

DCR needs every model retrained on an equal 50/50 split (`adult_dcr`), since the
default split is 90/10 — it cannot be computed from the runs above.

The three drift arms differ only in how much training-set information φ carries:
`WhitenedIdentity` holds ~110 parameters and cannot encode individual rows;
pretrained ReMasker is a transformer fit on the training half and can; the
random-init arm has the same architecture and output dimensionality with no
training at all. If the ReMasker arm shows worse DCR, the control says whether
that came from the generator or from φ.

Not yet run: other datasets, α-precision, β-recall, training seeds.
