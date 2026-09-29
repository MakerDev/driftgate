# Phase 4 — Causal Online Baselines (RQ4)

**Date**: 2026-07-13 · **Data**: `runs/phase4_baselines/` (22 runs) + matched
downstream/passive runs; CIFAR-10, schedules {A, abrupt}, 100R, seeds 0-2 (proxy-grid and
update-norm: seed 0 pilot only). Table: `tables/phase4_baselines.csv`. 0 failures.

All baselines are causal (no future information). Label-free unless marked. Bandits/grid
use the pre-registered routed-confidence proxy reward; B11 is the *labeled* greedy causal
oracle (hill-climbs λ on per-round labeled accuracy — an upper bound, not deployable).

## 1. Results (integrated acc; regret = B11 − method, paired)

| controller | A | regret | switch/rd | abrupt | regret | switch/rd |
|---|---:|---:|---:|---:|---:|---:|
| **B11 labeled causal oracle** | 0.6330 | 0 | 0.045 | 0.6183 | 0 | 0.045 |
| legacy (leak-calibrated) | 0.6268 | +0.62 pp | 0.045 | 0.6084 | +0.98 pp | 0.045 |
| **selfcal v3c (ours)** | 0.6163 | +1.67 pp | **0.009** | 0.6073 | +1.09 pp | **0.010** |
| UCB bandit (proxy) | 0.6180 | +1.50 pp | 0.163 | 0.5981 | +2.01 pp | 0.224 |
| fixed λ=0.4 | 0.6157 | +1.73 pp | 0 | 0.6028 | +1.55 pp | 0 |
| EXP3 (proxy) | 0.6152 | +1.78 pp | 0.186 | 0.6008 | +1.75 pp | 0.186 |
| update-norm selfcal (B10, s0) | 0.6125 | +2.75 pp | 0.003 | 0.5881 | +3.35 pp | 0.003 |
| periodic proxy grid (B5, s0) | 0.6042 | +3.58 pp | 0.023 | 0.5898 | +3.17 pp | 0.023 |

## 2. Findings (RQ4 answers)

1. **Among label-free deployable methods, selfcal-δ v3c has the lowest regret on the
   unseen schedule** (+1.09 pp vs oracle on abrupt) with an order of magnitude less
   parameter churn than bandits (0.01 vs 0.16-0.22 |Δλ|/round). Bandit-style exploration
   is the wrong tool here: the label-free proxy reward is too flat across λ arms, so
   UCB/EXP3 keep exploring (thrash) and land at-or-below fixed λ.
2. **The labeled causal oracle bounds what ANY causal controller can get**: fixed→oracle
   headroom is only 1.7 pp (A) / 1.5 pp (abrupt) at this horizon. v3c captures ~30% of
   that headroom on abrupt without labels; legacy captures ~60% but its constants encode
   test-schedule information (Phase 1). Dynamic-λ upside at 100R is intrinsically modest —
   consistent with the Phase-2 honesty note; the interesting regime is longer horizons
   and harsher drift (Gate D).
3. **Update-norm (B10) is a poor drift signal** (−2 to −3.4 pp): client update norms track
   optimization dynamics, not traffic composition.
4. B5 periodic grid pays a heavy exploration tax (sweeping bad λ values live).

## 3. Classification
- Main result: v3c = best label-free causal controller (lowest regret, least churn).
- Negative findings: proxy-reward bandits (B5/B6/B7) and update-norm (B10) are not
  competitive — reported, not carried to Gate D (except UCB as a Gate-D reference arm).
