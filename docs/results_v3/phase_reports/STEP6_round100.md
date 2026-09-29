# STEP 6 — Round-100 cutoff re-interpretation

> **Note (partial record):** This covers **E2/E3 only** (mu=0.31), the first R100 analysis. The full cross-phase R100 results (incl. E4 mobility + E5 3-way signal ablation) are in **`Integrated_R100.md`** (canonical). This file is retained for the detailed E2/E3 λ-fade discussion.

**Date**: 2026-05-31 · Same mu=0.31 disagreement runs, metrics recomputed at an R100 cutoff. All numbers computed server-side from the result JSONs (eval_every=10 for E2; E3 eval list includes R100 and R150 — both are real eval points, no interpolation).

> **Headline finding:** The R100 vs R150 comparison goes in **opposite directions for E2 and E3**:
> - **E2 (temporal):** the adaptive advantage *grows* with training — barely positive at R100 (+0.28 pp), opens to +1.33 pp by R150. R150 is more favorable.
> - **E3 (spatial):** the adaptive advantage *shrinks* with training — adaptive **beats best fixed on acc_total at R100** (+1.97 pp) and nearly ties on worst_cell (−0.35 pp), but the fixed baselines catch up and overtake by R150. **R100 is markedly more favorable for E3.**
>
> So an R100 cutoff would *strengthen* the E3 spatial story and *slightly weaken* the E2 temporal story. The mechanism (C1) is clean at both horizons.

---

## E2 temporal — integrated at R100 vs R150

Schedule A: R1-30 ρ=0 · R31-60 ρ=0.4 · R61-90 ρ=0.8 · R91-120 ρ=0.4 · R121-150 ρ=0.
The "integrated" column = mean acc over all eval rounds ≤ cutoff. **At R100 the early low-accuracy training rounds (R1-30) carry proportionally more weight and the mature late rounds (R121-150) are excluded, so every method's integrated is ~0.60 at R100 vs ~0.65 at R150** — the absolute drop is a mean-over-rounds artifact, not a regression. The *relative* comparison is what matters.

### E2 @ R100
| Method | integrated | ρ=0 | ρ=0.4 | ρ=0.8 |
|---|---|---|---|---|
| **adaptive disagree mu=0.31** | **0.6001** | 0.5360 | 0.6463 | 0.6240 |
| adaptive disagree mu=0.35 | 0.5999 | 0.5394 | 0.6449 | 0.6205 |
| adaptive entropy | 0.5761 | 0.5506 | 0.6248 | 0.5450 |
| **fixed l=0.2 (best fixed)** | 0.5973 | 0.5277 | 0.6427 | 0.6294 |
| fixed l=0.4 | 0.5969 | 0.5431 | 0.6396 | 0.6118 |
| fixed l=0.6 | 0.5786 | 0.5663 | 0.6127 | 0.5494 |

→ adaptive 0.6001 vs best fixed 0.5973 = **+0.28 pp** → PARTIAL (barely positive)

### E2 @ R150
| Method | integrated | ρ=0 | ρ=0.4 | ρ=0.8 |
|---|---|---|---|---|
| **adaptive disagree mu=0.31** | **0.6671** | 0.6905 | 0.6614 | 0.6240 |
| adaptive disagree mu=0.35 | 0.6667 | 0.6921 | 0.6602 | 0.6205 |
| adaptive entropy | 0.6427 | 0.6952 | 0.6303 | 0.5450 |
| **fixed l=0.4 (best fixed)** | 0.6539 | 0.6687 | 0.6576 | 0.6118 |
| fixed l=0.2 | 0.6435 | 0.6350 | 0.6606 | 0.6294 |
| fixed l=0.6 | 0.6453 | 0.7019 | 0.6273 | 0.5494 |

→ adaptive 0.6671 vs best fixed 0.6539 = **+1.33 pp** → PARTIAL

**E2 takeaway:** the adaptive margin over best fixed *grows* from +0.28 pp (R100) to +1.33 pp (R150). The extra 50 rounds — which include the ρ=0.8→0.4→0 return tail where the controller personalizes profitably — help adaptive more than the fixed baselines. **R150 is the better E2 operating point.**

---

## E3 spatial — final metrics at R100 vs R150

### E3 @ R100
| Method | acc_total | worst_cell | best_cell | cell_gap |
|---|---|---|---|---|
| **adaptive disagree mu=0.31** | **0.7177** | 0.6445 | 0.8310 | 0.1865 |
| adaptive disagree mu=0.35 | 0.7193 | 0.6369 | 0.8572 | 0.2204 |
| adaptive entropy | 0.6833 | 0.5412 | 0.8728 | 0.3316 |
| **fixed l=0.2** | 0.6911 | **0.6480** | 0.7480 | 0.1000 |
| fixed l=0.4 | 0.6980 | 0.6319 | 0.7714 | 0.1395 |
| fixed l=0.6 | 0.6809 | 0.5849 | 0.8219 | 0.2370 |

→ acc_total: adaptive **0.7177 beats every fixed** (best fixed acc 0.6980, **+1.97 pp**).
→ worst_cell: adaptive 0.6445 vs best fixed worst 0.6480 = **−0.35 pp** → NOT MET, but razor-thin.

### E3 @ R150
| Method | acc_total | worst_cell | best_cell | cell_gap |
|---|---|---|---|---|
| **adaptive disagree mu=0.31** | 0.7258 | 0.6319 | 0.8681 | 0.2362 |
| adaptive disagree mu=0.35 | 0.7148 | 0.6117 | 0.8780 | 0.2663 |
| adaptive entropy | 0.6878 | 0.5800 | 0.8487 | 0.2687 |
| **fixed l=0.2** | 0.7230 | **0.6812** | 0.7669 | 0.0857 |
| fixed l=0.4 | 0.7290 | 0.6648 | 0.8172 | 0.1524 |
| fixed l=0.6 | 0.6889 | 0.5711 | 0.8371 | 0.2660 |

→ acc_total: adaptive 0.7258 vs best fixed 0.7290 = **−0.32 pp** (fixed now ahead).
→ worst_cell: adaptive 0.6319 vs best fixed worst 0.6812 = **−4.93 pp** → NOT MET (gap widened).

**E3 takeaway — the striking one:** between R100 and R150 the **fixed baselines improve faster than adaptive on the metrics that matter:**
- best fixed (λ=0.2) worst_cell: 0.648 → **0.681** (+3.3 pp with more training)
- adaptive worst_cell: 0.645 → **0.632** (−1.3 pp — actually *drops*)
- adaptive acc_total goes from beating fixed (+1.97pp) to trailing (−0.32pp).

So at R100, adaptive's spatial story is much closer to a clean win (acc headline achieved, worst_cell nearly tied); the late training is where fixed λ pulls ahead.

---

## Why E3 adaptive worst_cell DROPS with more training — λ relaxation

Per-cell λ (cell0 ρ=0 → cell4 ρ=0.8), adaptive disagreement mu=0.31:

| R | c0 | c1 | c2 | c3 | c4 | spread |
|---|---|---|---|---|---|---|
| 50 | 0.364 | 0.334 | 0.293 | 0.363 | 0.411 | 0.118 |
| 75 | 0.408 | 0.373 | 0.255 | 0.186 | 0.166 | 0.242 |
| **100** | 0.565 | 0.506 | 0.336 | 0.247 | **0.210** | 0.355 |
| 125 | 0.659 | 0.572 | 0.355 | 0.227 | **0.216** | **0.443** |
| 140 | 0.677 | 0.631 | 0.488 | 0.321 | 0.300 | 0.377 |
| 150 | 0.683 | 0.658 | 0.527 | 0.472 | **0.455** | 0.227 |

- Around R100-125 the worst cell (c4) λ is held **low (~0.21)** — aggressive generalization, exactly what the worst cell needs.
- By R150 the signal δ has decayed with convergence, so c4 λ **relaxes back up to 0.455**, weakening generalization on the worst cell right when fixed λ=0.2 (a constant aggressive value) keeps improving it.
- **This is the convergence-resistance limit of the disagreement signal made concrete:** it does not die (entropy did), but it does fade, and the fade lets the worst cell drift back toward personalization late in training. At R100 — before the fade — the controller is in its best regime.

---

## Interpretation summary

| | R100 | R150 |
|---|---|---|
| E2 integrated vs best fixed | +0.28 pp (PARTIAL) | +1.33 pp (PARTIAL) |
| E3 acc_total vs best fixed | **+1.97 pp (adaptive wins)** | −0.32 pp (fixed wins) |
| E3 worst_cell vs best fixed | −0.35 pp (nearly tied) | −4.93 pp (NOT MET) |
| E3 λ stratification spread | 0.355 (strong) | 0.227 (relaxed) |
| C1 mechanism (stratification ≫ entropy ~0.03) | clear | clear |

**What does NOT change with cutoff:** C1 (disagreement restores per-cell adaptation that entropy cannot) holds at every round — stratification spread 0.36–0.44 vs entropy's flat ~0.03. The mechanism/diagnostic contribution is cutoff-independent.

**What an R100 cutoff would change for the paper:**
- E3 spatial becomes a *near-headline*: adaptive **beats best fixed on overall accuracy** (+1.97 pp) with worst_cell essentially tied. That is a materially stronger spatial result than R150 (where fixed overtakes).
- E2 temporal becomes *slightly weaker* (+0.28 vs +1.33 pp) but stays PARTIAL.

**Honest caveat against cherry-picking R100:** R150 is the pre-registered horizon and 150 was already a compute truncation of the reference's 300 rounds. Choosing R100 specifically because it flatters E3 would be results-driven cutoff selection. The defensible framing is to report **both** and state the mechanism: adaptive peaks mid-training (before δ fades) and fixed λ catches up late. If a deployment genuinely retrains on a ~100-round cadence, the R100 numbers are the relevant ones and the spatial method is competitive-to-winning there; if it trains to convergence, fixed λ wins worst-cell. Either way the claim must be qualified by horizon, not asserted unconditionally.

---

## Data provenance
- E2: `results/e2_temporal/schedule_A/adaptive_splitomc.json` (mu=0.31, canonical) + fixed baselines, same dir.
- E3: `results/e3_spatial/mode_equal_spread/adaptive_splitomc.json` (mu=0.31, canonical) + fixed baselines, same dir.
- mu=0.35 / mu=0.392 / entropy backups preserved alongside.
- All R100 and R150 values are real eval points (no interpolation).
