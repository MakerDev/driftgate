# Phase D — Final-DV Causal Baseline Comparison (RQ4, re-run with DV-2)

**Date**: 2026-07-23 · **Data**: `runs/phaseD_final/` (DV-2 = TV) + D1 baselines (fixed
grid, legacy, labeled oracle, 5 seeds) + Phase-4 bandits (A/abrupt). Table:
`tables/final_causal_baselines.csv`. DV-2 = 3 seeds (pilot); baselines 5 seeds.

## 1. Ranking per unseen schedule (integrated accuracy, ↓)

| gradual_sigmoid | asym_return | piecewise_random |
|---|---|---|
| labeled oracle 0.6393 | labeled oracle 0.6535 | labeled oracle 0.6389 |
| fixed λ0.2 0.6323 | legacy 0.6479 | fixed λ0.2 0.6284 |
| **DV-2 0.6272** | **DV-2 0.6472** | fixed λ0.4 0.6259 |
| fixed λ0.4 0.6233 | fixed λ0.4 0.6426 | legacy 0.6251 |
| legacy 0.6162 | fixed λ0.2 0.6423 | **DV-2 0.6231** |

DV-2 vs entropy-DV (gsig, 3 seeds): **+2.18 pp (3/3)** — signal advantage holds in-loop
under the final controller.

## 2. Findings (RQ4, honest)

1. **Ordering vs the labels-only upper bound is unchanged from the v3c study**: the
   labeled causal oracle leads every schedule by +1.1–1.6 pp; no label-free method reaches
   it. DV-2 sits mid-pack among deployable methods — ahead of legacy and fixed λ0.4 on
   2/3 schedules, behind fixed λ0.2 on 2/3.
2. **DV-2 does NOT dominate the fixed grid** (consistent with the SVHN holdout, C-13):
   it is within ±0.5 pp of the best deployable fixed on gsig/asym and −0.5 pp on pwr.
   Reported as *"DV-2 matches the deployable fixed grid to within ±0.5 pp on unseen
   temporal schedules"* — NOT "best".
3. **Where DV-2 clearly wins: churn and no-tuning.** Parameter switching is
   0.006–0.012 |Δλ|/round vs bandits' 0.16–0.22 (Phase 4) and legacy's 0.039–0.042; it
   needs no per-deployment λ choice, whereas each fixed row is a different hindsight pick
   (the best fixed flips between λ0.2 and λ0.4 across schedules). This is the accuracy–
   stability trade-off framing (C-4 fallback), which the evidence supports; the stronger
   "best label-free controller" claim is NOT supported and is dropped.
4. **Signal effect (C1) reconfirmed in-loop** under the final TV controller (+2.18 pp vs
   entropy-DV, 3/3) — the robust, portable result.

## 3. Verdict for RQ4

**DV-2 provides the best accuracy–stability trade-off among evaluated label-free
controllers** (lowest churn, no per-deployment tuning, near-fixed accuracy, guaranteed
signal advantage over entropy) — but it does **not** achieve the lowest regret vs the
labeled oracle, and it does not beat the best hindsight fixed λ. Both stated plainly.
The 5-seed completion of the DV-2 arms is queued (Phase D used 3-seed pilots for the new
schedules); it will tighten the CIs but is not expected to change the ordering.
