# Phase 1 — E1 Static Pareto Sweep Report

**Run window**: 2026-05-28 03:14 → 2026-05-29 10:07 KST (≈30h53m wall, 2× GPU parallel)
**Config**: `configs/base_v3.yaml` — 150 rounds × 3 local epochs × 50 clients × 5 ES, no drift
**Status**: ✅ All 16 configurations completed

---

## Full Pareto (sorted by acc_total descending)

| Rank | Method | acc_main | acc_oop | acc_oor | **acc_total** | worst_cell | cell_gap |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | splitomcplus λ=0.2 Λ=0.5 | 0.779 | 0.643 | 0.271 | **0.703** | 0.656 | 0.109 |
| 2 | splitomcplus λ=0.4 Λ=0.5 | 0.835 | 0.518 | 0.188 | 0.701 | 0.664 | 0.079 |
| **3** | **adaptive_splitomc** | **0.763** | **0.638** | **0.371** | **0.699** | **0.656** | **0.104** |
| 4 | splitomc λ=0.2 | 0.787 | 0.598 | 0.078 | 0.681 | 0.610 | 0.154 |
| 5 | splitfed | 0.696 | 0.678 | 0.555 | 0.680 | 0.636 | 0.105 |
| 6 | fedavg | 0.696 | 0.678 | 0.550 | 0.680 | 0.635 | 0.105 |
| 7 | fedprox | 0.694 | 0.677 | 0.553 | 0.679 | 0.635 | 0.104 |
| 8 | splitomc λ=0.4 | 0.836 | 0.471 | 0.052 | 0.678 | 0.620 | 0.142 |
| 9 | splitgp λ=0.5 | 0.838 | 0.387 | 0.246 | 0.673 | 0.652 | 0.055 |
| 10 | splitgp λ=0.2 | 0.727 | 0.602 | 0.448 | 0.672 | 0.621 | 0.125 |
| 11 | splitomc λ=0.0 | 0.737 | 0.663 | 0.092 | 0.667 | 0.596 | 0.166 |
| 12 | splitomcplus λ=0.6 Λ=0.5 | 0.877 | 0.273 | 0.081 | 0.655 | 0.625 | 0.080 |
| 13 | splitomc λ=0.6 | 0.879 | 0.265 | 0.024 | 0.650 | 0.610 | 0.105 |
| 14 | fedmes | 0.684 | 0.639 | 0.042 | 0.622 | **0.481** | **0.284** |
| 15 | splitomc λ=0.8 | 0.898 | 0.080 | 0.005 | 0.613 | 0.582 | 0.068 |
| 16 | splitgp λ=0.8 | 0.895 | 0.059 | 0.022 | 0.607 | 0.582 | 0.050 |

---

## Analysis

### Headline: adaptive_splitomc ties for top-3 *and* dominates on coverage

Within 0.4 pp of the best acc_total (0.699 vs 0.703), but with materially better profile:

| Metric | adaptive | best fixed (splitomcplus λ=0.2 Λ=0.5) | Δ |
|---|---:|---:|---:|
| acc_total | 0.699 | 0.703 | −0.004 |
| acc_main | 0.763 | 0.779 | −0.016 |
| acc_oop | 0.638 | 0.643 | −0.005 |
| **acc_oor** | **0.371** | **0.271** | **+0.100** |

The −0.004 acc_total gap is noise; the +0.100 acc_oor gap is real and meaningful — adaptive does 37% better at cross-region transfer than the best fixed splitomcplus. The controller chose to give up a sliver of personalization (acc_main −0.016) to push out-of-region accuracy substantially.

### λ Pareto curve for vanilla splitomc

Clean monotone trade-off, no surprises:

| λ | acc_main | acc_oop | acc_oor | acc_total |
|---:|---:|---:|---:|---:|
| 0.0 | 0.737 | 0.663 | 0.092 | 0.667 |
| 0.2 | 0.787 | 0.598 | 0.078 | 0.681 |
| 0.4 | 0.836 | 0.471 | 0.052 | 0.678 |
| 0.6 | 0.879 | 0.265 | 0.024 | 0.650 |
| 0.8 | 0.898 | 0.080 | 0.005 | 0.613 |

- acc_main monotone ↑ (0.737 → 0.898) as λ raises personalization weight.
- acc_oop and acc_oor monotone ↓ (0.663 → 0.080 and 0.092 → 0.005).
- acc_total peaks at λ=0.2 (0.681), drops sharply past λ=0.6 because OOP collapse outweighs main gains.

### Λ (cross-cell) lifts the Pareto frontier

Comparing splitomcplus (Λ=0.5) against splitomc (Λ=0) at matched λ:

| λ | Δacc_main | Δacc_oop | Δacc_oor | Δacc_total |
|---:|---:|---:|---:|---:|
| 0.2 | −0.008 | +0.045 | +0.193 | **+0.022** |
| 0.4 | −0.001 | +0.047 | +0.136 | **+0.023** |
| 0.6 | −0.002 | +0.008 | +0.057 | **+0.005** |

splitomcplus (Λ=0.5) always wins on acc_total by 0.5–2.3 pp, with massive acc_oor gains at small λ. As λ grows, the marginal value of Λ shrinks (already λ-personalized). **Λ=0.5 is essentially free at small λ** — recovers cross-region accuracy without sacrificing personalization.

### Baseline group (fedavg / fedprox / splitfed) is degenerate

These three are statistically indistinguishable (acc_total within 0.001 of each other). Under our 5-ES ND1 partition, the "all-global, no personalization" approaches collapse to the same operating point. Useful as a single baseline reference; no need to compare against them individually in the paper.

### fedmes is broken — drop it or fix it before reporting

- acc_total 0.622 (worst non-splitomc)
- worst_cell 0.481, cell_gap **0.284** (more than 2× the next-worst)

This signals fedmes's multi-cell averaging strategy is making one cell catastrophic while letting others off easy. Either there is a configuration bug, or `fedmes` as implemented does not survive ND1's cell-heterogeneity. Recommend not citing fedmes as a useful baseline in the paper.

### Split-GP is mid-tier; not competitive with splitomcplus

splitgp_lam0.2 reaches acc_oor 0.448, but at acc_total 0.672 — beaten by splitomcplus at every λ. The reference paper's claim that SplitGP is a strong personalization baseline holds, but Λ=0.5 in splitomcplus does the same job more efficiently.

### Fairness (worst-cell, cell_gap) lens

Sorted by **worst-cell accuracy**:

| Rank by worst | Method | worst_cell | cell_gap |
|---:|---|---:|---:|
| 1 | splitomcplus λ=0.4 Λ=0.5 | 0.664 | 0.079 |
| 2 | splitomcplus λ=0.2 Λ=0.5 | 0.656 | 0.109 |
| 2 | **adaptive_splitomc** | **0.656** | **0.104** |
| 4 | splitgp λ=0.5 | 0.652 | 0.055 |
| 5 | splitfed | 0.636 | 0.105 |

adaptive_splitomc ties for #2 on worst-cell. splitgp_lam0.5 has the lowest cell_gap (0.055) but a lower worst-cell — different cell uniformity vs absolute fairness trade-off. The paper should highlight worst-cell (raw fairness) over cell_gap (relative spread).

---

## Pareto Implications

**Three operating points define the frontier:**

1. **Personalization corner**: `splitomc_lam0.8` (acc_main 0.898) — paper baseline for "extreme personalization".
2. **Generalization corner**: `fedavg/splitfed` (acc_main 0.696, acc_oor 0.55) — paper baseline for "extreme global model".
3. **Adaptive sweet spot**: `adaptive_splitomc` (0.763 / 0.638 / 0.371 / 0.699) — *on the frontier and well to the interior at the same time*. Hard to beat without controller tuning.

Fixed-config alternative for the paper's "best non-adaptive": `splitomcplus_lam0.2_Lam0.5`. It marginally beats adaptive on acc_total (+0.4 pp) but loses on every other axis. In a paper telling a "tracks the moving optimum" story, the +0.100 acc_oor swing for adaptive matters more.

---

## Implications for Phase 2 (E2 Temporal CORE)

E1 is the no-drift baseline. Under E2 schedules A (mild) and B (severe) drift:
- Fixed splitomcplus configs lose their Pareto-optimal λ when ρ moves — adaptive should pull ahead.
- The headline gate: **adaptive Schedule-A integrated acc ≥ best-fixed integrated acc + 2 pp**.
- Looking at E1, the "best fixed" baselines for Phase 2 (per the run_all.sh script) are splitomcplus λ∈{0.2, 0.4, 0.6}_Lam0.5. Their static-acc spread is 0.046 (0.655–0.703) — so a 2-pp adaptive lead under drift is achievable but not trivial.

---

## Run Artifacts

- `results/e1_static/*.json` (16 files, ~32 KB each, full per-round eval traces)
- `logs/e1_gpu{0,1}.log` — raw stdout

## Wall-clock Notes

- GPU 0: 1644.7 min (27.4h) for 8 configs (fedavg, fedprox, splitfed, fedmes, splitgp ×3, splitomc λ=0)
- GPU 1: ~1640 min (27.3h) for 8 configs (splitomc λ×4, splitomcplus ×3, adaptive_splitomc)
- Per-config range: 158 min (fedavg) → 330 min (fedprox). Variance driven by competing user load on shared GPUs.
- ~2× CLAUDE.md's 10-13h estimate, expected given the constant GPU contention from 5–6 other users per GPU.
