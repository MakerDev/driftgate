# Integrated Results — R150 (full-horizon, primary)

**Adaptive-SplitOMC v4 · all phases at the R150 training horizon · auto-generated from result JSONs (no hand-transcribed numbers).** Companion: `Integrated_R100.md`.

## Signal-mode provenance

| Phase | adaptive signal |
|---|---|
| E1 static | entropy |
| E2 temporal | disagreement (mu=0.31) |
| E3 spatial | disagreement (mu=0.31) |
| E4 mobility | disagreement (mu=0.31) + entropy backup |
| E5 signal ablation | 3-way: entropy / disagreement / disagreement+margin (β=0.5), 150R |

Calibration (disagreement): `mu_drift=0.31, tau_drift=0.045, lam[0.15,0.7], Lam[0.4,0.7]`; entropy: `mu_H=2.0, tau_H=0.5`.

---

## E1 — Static Pareto (no drift, ρ=0.4 eval)  [adaptive = ENTROPY]

| Rank | Method | acc_total | main | oop | oor | worst_cell |
|---:|---|---:|---:|---:|---:|---:|
| 1 | splitomcplus_lam0.2_Lam0.5 | 0.7031 | 0.779 | 0.643 | 0.271 | 0.656 |
| 2 | splitomcplus_lam0.4_Lam0.5 | 0.7010 | 0.835 | 0.518 | 0.188 | 0.664 |
| 3 | **adaptive_splitomc** | 0.6994 | 0.763 | 0.638 | 0.371 | 0.656 |
| 4 | splitomc_lam0.2 | 0.6813 | 0.787 | 0.598 | 0.078 | 0.610 |
| 5 | splitfed | 0.6805 | 0.696 | 0.678 | 0.555 | 0.636 |
| 6 | fedavg | 0.6797 | 0.696 | 0.678 | 0.550 | 0.635 |
| 7 | fedprox | 0.6787 | 0.694 | 0.677 | 0.553 | 0.635 |
| 8 | splitomc_lam0.4 | 0.6783 | 0.836 | 0.471 | 0.052 | 0.620 |
| 9 | splitgp_lam0.5 | 0.6730 | 0.838 | 0.387 | 0.246 | 0.652 |
| 10 | splitgp_lam0.2 | 0.6719 | 0.727 | 0.602 | 0.448 | 0.621 |
| 11 | splitomc_lam0.0 | 0.6668 | 0.737 | 0.663 | 0.092 | 0.596 |
| 12 | splitomcplus_lam0.6_Lam0.5 | 0.6554 | 0.877 | 0.273 | 0.081 | 0.625 |
| 13 | splitomc_lam0.6 | 0.6501 | 0.879 | 0.265 | 0.024 | 0.610 |
| 14 | fedmes | 0.6217 | 0.684 | 0.639 | 0.042 | 0.481 |
| 15 | splitomc_lam0.8 | 0.6125 | 0.898 | 0.080 | 0.005 | 0.582 |
| 16 | splitgp_lam0.8 | 0.6065 | 0.895 | 0.059 | 0.022 | 0.582 |

## E2 — Temporal drift (Schedule A)  [adaptive = DISAGREEMENT mu=0.31]

| Method | integrated | ρ=0 | ρ=0.4 | ρ=0.8 |
|---|---:|---:|---:|---:|
| adaptive (disagree mu=0.31) | 0.6671 | 0.6905 | 0.6614 | 0.6240 |
| adaptive (disagree mu=0.35) | 0.6667 | 0.6921 | 0.6602 | 0.6205 |
| adaptive (disagree mu=0.392) | 0.6609 | 0.6990 | 0.6430 | 0.6076 |
| adaptive (entropy) | 0.6427 | 0.6952 | 0.6303 | 0.5450 |
| fixed λ=0.2 | 0.6435 | 0.6350 | 0.6606 | 0.6294 |
| fixed λ=0.4 | 0.6539 | 0.6687 | 0.6576 | 0.6118 |
| fixed λ=0.6 | 0.6453 | 0.7019 | 0.6273 | 0.5494 |

- best fixed = fixed λ=0.4 (0.6539); adaptive(mu=0.31) **+1.33 pp = PARTIAL** · vs entropy **+2.44 pp**

## E3 — Spatial heterogeneity (equal_spread: cell0 ρ=0 … cell4 ρ=0.8)  [adaptive = DISAGREEMENT mu=0.31]

| Method | acc_total | worst_cell | best_cell | cell_gap |
|---|---:|---:|---:|---:|
| adaptive (disagree mu=0.31) | 0.7258 | 0.6319 | 0.8681 | 0.2362 |
| adaptive (disagree mu=0.35) | 0.7148 | 0.6117 | 0.8780 | 0.2663 |
| adaptive (entropy) | 0.6878 | 0.5800 | 0.8487 | 0.2687 |
| fixed λ=0.2 | 0.7230 | 0.6812 | 0.7669 | 0.0857 |
| fixed λ=0.4 | 0.7290 | 0.6648 | 0.8172 | 0.1524 |
| fixed λ=0.6 | 0.6889 | 0.5711 | 0.8371 | 0.2660 |

- acc_total: adaptive 0.7258 vs best fixed 0.7290 (fixed λ=0.4) → **-0.32 pp**
- worst_cell: adaptive 0.6319 vs best fixed 0.6812 (fixed λ=0.2) → **-4.93 pp = NOT MET**
- λ stratification spread @R150: disagreement **0.227** vs entropy **0.009** (FLAT)

## E4 — Mobility (Gauss-Markov rewiring)  [adaptive = DISAGREEMENT mu=0.31]

| Method | acc_total | main | oop | oor | worst_cell |
|---|---:|---:|---:|---:|---:|
| adaptive (disagreement) | 0.6813 | 0.767 | 0.547 | 0.413 | 0.6505 |
| adaptive (entropy) | 0.6445 | 0.880 | 0.205 | 0.145 | 0.6321 |
| fixed λ=0.2 | 0.6737 | 0.743 | 0.574 | 0.428 | 0.6475 |
| fixed λ=0.4 | 0.6825 | 0.817 | 0.455 | 0.324 | 0.6752 |
| fixed λ=0.6 | 0.6509 | 0.883 | 0.224 | 0.142 | 0.6394 |

- disagreement vs entropy: **acc +3.68 pp, worst +1.84 pp**
- disagreement vs **best fixed acc (fixed λ=0.4, 0.6825)**: **-0.12 pp**
- disagreement vs **best fixed worst (fixed λ=0.4, 0.6752)**: **-2.47 pp = NOT MET**

## E5 — 3-way SIGNAL ablation (Schedule A, 150R, same controller — only signal source changes)

| Signal | integrated | ρ=0 | ρ=0.4 | ρ=0.8 |
|---|---:|---:|---:|---:|
| entropy | 0.6426 | 0.6934 | 0.6315 | 0.5463 |
| disagreement (S1) | 0.6669 | 0.6902 | 0.6613 | 0.6238 |
| disagreement+margin (β=0.5) | 0.6397 | 0.6285 | 0.6582 | 0.6288 |

- **disagreement − entropy = +2.43 pp** (same controller, only the signal swapped) — direct C1 isolation.
- **(disagreement+margin) − disagreement = -2.72 pp** — margin HURTS → S1 alone best, β=0 confirmed.

---

## Headline summary (R150)

- **C1 (signal): disagreement > entropy everywhere** — E5 signal-swap +2.43 pp; E2 +2.44; E3 acc +3.80; E4 acc +3.68.
- **vs best hand-tuned fixed λ:** E2 PARTIAL (+1.33 pp); E3 worst_cell -4.93 pp; E4 acc -0.12 pp / worst -2.47 pp. Disagreement does NOT beat best fixed on these aggregates at R150.
- **S1 sufficiency:** adding S2 margin hurts (β=0 is correct).

_Generated by `scripts/gen_integrated.py`. All values are real eval points at the R150 cutoff (eval_every=10/20)._
