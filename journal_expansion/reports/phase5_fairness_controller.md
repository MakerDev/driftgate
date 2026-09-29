# Phase 5 — Worst-Cluster Mechanisms (Gate C)

**Date**: 2026-07-13 · **Data**: `runs/gatec_fairness/` (23 runs, CIFAR-10 spatial
equal_spread cell0 ρ=0…cell4 ρ=0.8, 100R, 3 seeds unless noted, 0 failures).
Table: `tables/fairness_results.csv`. Metrics averaged over eval rounds ≥ 50.

## 1. Results (3-seed means; Δ = paired per-seed vs temporal-only selfcal)

| method | acc | worst-cell | Δworst | Δacc | paired worst (3 seeds) |
|---|---:|---:|---:|---:|---|
| fixed λ=0.2 (hand-tuned ref) | 0.6679 | **0.5841** | +6.15 pp | +2.16 pp | 3/3 + |
| fixed λ=0.4 | 0.6652 | 0.5738 | +5.13 pp | +1.89 pp | 3/3 + |
| **selfcal-ST** (spatial-z, ours) | 0.6573 | 0.5450 | **+2.25 pp** | **+1.10 pp** | 3/3 + |
| fair-deploy + ST (F1/F2) | 0.6565 | 0.5469 | +2.44 pp | +1.02 pp | vs ST: +0.19 pp, mixed signs |
| selfcal temporal-only (v3b) | 0.6463 | 0.5225 | — | — | baseline |
| fair-deploy, temporal risk | 0.6437 | 0.5142 | −0.83 pp | −0.26 pp | |
| fair-ORACLE donors | 0.6431 | 0.5050 | −1.75 pp | −0.32 pp | |
| fair-random donors (s0) | 0.6435 | 0.5160 | −3.43 pp | −0.66 pp | |

## 2. Findings

### 2.1 The spatial reference frame is the real fix (main positive result)
selfcal-ST lifts worst-cell **+2.25 pp AND avg +1.10 pp simultaneously**, 3/3 seeds — the
pre-registered Gate-C bar (worst +2 pp at ≤0.5 pp avg cost) is passed with an avg *gain*.
Mechanism: temporal self-calibration is structurally blind to STATIC heterogeneity
(nothing ever changes ⇒ z_t≡0 ⇒ no stratification); the cross-cell robust-z restores a
label-free, dimensionless reference frame. This also *explains* the v4 legacy behavior:
the absolute threshold worked on E3 because it measured "is my traffic hard", not "did it
change" — its leak-calibrated constant was doing the job of a reference frame.

### 2.2 Donor-selected fairness aggregation is a NEGATIVE result (F1/F2)
- On top of ST, deployable donor selection adds **+0.19 pp worst (signs mixed)** — no value.
- **Oracle donors are WORSE than deployable (−1.75 pp vs baseline)** and random donors
  worst of all: cross-cell parameter inflow, however well targeted, mostly imports other
  cells' personalization, which is off-distribution for the worst cell. This reproduces
  the v5 η-weighting failure at 3-seed rigor and extends it: the deficit survives donor
  QUALITY, donor RANDOMIZATION, and absorption DIRECTION. The worst-cell problem is not
  a knowledge-routing problem.
- Classification: **failed mechanism** (reported, dropped from Gate D).

### 2.3 The ceiling vs hand-tuned fixed λ stands (boundary condition)
fixed λ=0.2 worst 0.5841 vs best adaptive 0.5469 (**−3.9 pp, 3/3 seeds**). Uniform heavy
sharing wins BOTH worst and average on static spatial heterogeneity at this horizon. Two
non-exclusive explanations to test at Gate D: (a) λ_max=0.7 personalizes low-ρ cells too
hard (bounds inherited from legacy; a bounds ablation is queued for Gate D); (b) with
static composition there is nothing to adapt TO — a well-chosen constant is the right
answer, and its cost is precisely the hindsight tuning our method avoids. Claim
discipline: we do NOT claim to beat hand-tuned fixed λ on static spatial settings.

### 2.4 F3 routing: exact tie (negative/neutral)
Pooled over Gate-C runs: entropy routing 0.6535 vs disagreement-confidence routing 0.6535.
On converged CIFAR-10 models the entropy threshold and the disagreement rule route almost
identically (the mock-model advantage does not materialize — client confidence and
client-server agreement are strongly correlated here). Kept as an honest null; may matter
under stronger shift types (Gate D corruption arm).

## 3. Gate-C verdict

- **Mechanism selected for Gate D: selfcal-ST** (v3c + spatial-z). PASS vs pre-registered
  bar against the adaptive baseline; the fixed-λ ceiling is documented as a boundary
  condition, not hidden.
- F1/F2 fairness arms: dropped (negative result, §2.2). F3 routing: reported as null.
- The v4/v5 "structural ceiling" narrative is REVISED: part of it was a reference-frame
  problem (fixed by ST, +2.25 pp); the remainder (−3.9 pp to hand-tuned fixed) persists
  across all five tested mechanism families (λ-temporal, λ-spatial, aggregation-weight,
  donor-selection, routing).
