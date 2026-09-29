# STEP 6 — E2 re-run (mu=0.35) + E3 disagreement

> **⚠ SUPERSEDED (historical record).** These numbers are the **mu=0.35** calibration, later replaced by the final **mu=0.31** and extended with E4/E5 disagreement. **Canonical final results: `Integrated_R150.md` (primary) and `Integrated_R100.md`.** Kept for the mu-calibration audit trail.

**Date**: 2026-05-30 20:13 KST · **Config**: signal_mode=disagreement, mu_drift=0.35 (final, re-centered), tau_drift=0.045, lam[0.15,0.7] Lam[0.4,0.7]
**Both runs complete** (E2 150R, E3 150R, ~204 min each, detached PPID=1). Numbers computed server-side from result JSONs.

---

## E2 — temporal Schedule A — VERDICT: **PARTIAL** (improved)

| Method | integrated | ρ=0 | ρ=0.4 | ρ=0.8 |
|---|---|---|---|---|
| **adaptive disagree mu=0.35** | **0.6667** | 0.6921 | 0.6602 | 0.6205 |
| adaptive disagree mu=0.392 | 0.6609 | 0.6990 | 0.6430 | 0.6076 |
| adaptive entropy | 0.6427 | 0.6952 | 0.6303 | 0.5450 |
| fixed λ=0.2 | 0.6435 | 0.6350 | 0.6606 | 0.6294 |
| **fixed λ=0.4 (best fixed)** | 0.6539 | 0.6687 | 0.6576 | 0.6118 |
| fixed λ=0.6 | 0.6453 | 0.7019 | 0.6273 | 0.5494 |

- integrated **+1.29 pp vs best fixed** (was +0.70 at mu=0.392) → **PARTIAL** (PASS needs +2)
- **+2.40 pp vs entropy**
- ρ=0.8 segment **0.6205**: +1.3pp vs mu=0.392 (0.608), **+7.6pp vs entropy** (0.545); now between fixed λ=0.4 (0.612) and fixed λ=0.2 (0.629)
- mu re-centering helped both ρ=0.4 (+1.7pp) and ρ=0.8 (+1.3pp) vs mu=0.392, at small cost to ρ=0 (−0.7pp)

### E2 λ-trace (mu=0.35)
| R | ρ | δ_avg | λ_avg |
|---|---|---|---|
| 30 | 0.0 | 0.374 | 0.373 |
| 65 | 0.8 | 0.467 | 0.189 |
| 75 | 0.8 | 0.376 | 0.352 |
| 90 | 0.8 | 0.391 | 0.310 |
| 135 | 0.0 | 0.131 | 0.695 |

λ now dips lower through sustained ρ=0.8 (R90 0.31 vs mu=0.392's 0.47) — the re-centering worked. Still relaxes mid-segment because sustained δ (~0.38) ≈ mu (0.35).

---

## E3 — spatial equal_spread — ★ MECHANISM FIXED, headline NOT met

cell0 ρ=0 → cell4 ρ=0.8 (fixed per cell).

### Per-cell λ stratification — DISAGREEMENT (the key result)
| R | c0 | c1 | c2 | c3 | c4 | spread |
|---|---|---|---|---|---|---|
| 25 | 0.406 | 0.358 | 0.260 | 0.313 | 0.333 | 0.146 |
| 75 | 0.613 | 0.576 | 0.425 | 0.247 | 0.192 | **0.421** |
| 100 | 0.685 | 0.675 | 0.601 | 0.404 | 0.272 | 0.412 |
| 125 | 0.690 | 0.679 | 0.606 | 0.412 | 0.306 | 0.384 |
| **150** | 0.697 | 0.690 | 0.645 | 0.547 | 0.494 | **0.203** |

### Per-cell λ stratification — ENTROPY (the failure being fixed)
| R | c0 | c1 | c2 | c3 | c4 | spread |
|---|---|---|---|---|---|---|
| 25 | 0.542 | 0.533 | 0.513 | 0.510 | 0.508 | 0.034 |
| 75 | 0.633 | 0.627 | 0.614 | 0.609 | 0.607 | 0.026 |
| **150** | 0.651 | 0.651 | 0.648 | 0.645 | 0.642 | **0.009** |

→ **Disagreement maintains per-cell stratification to R150 (spread 0.203, monotone in ρ_z); entropy collapses to 0.009 (flat).** Root-cause failure is fixed: low-ρ cells keep λ high (0.70), high-ρ cells push λ low (0.49). This is the qualitative paper figure.

### E3 final-round metrics (R150)
| Method | acc_total | worst_cell | best_cell | cell_gap |
|---|---|---|---|---|
| adaptive disagree | 0.7148 | 0.6117 | 0.8780 | 0.2663 |
| adaptive entropy | 0.6878 | 0.5800 | 0.8487 | 0.2687 |
| fixed λ=0.2 | 0.7230 | **0.6812** | 0.7669 | 0.0857 |
| fixed λ=0.4 | **0.7290** | 0.6648 | 0.8172 | 0.1524 |
| fixed λ=0.6 | 0.6889 | 0.5711 | 0.8371 | 0.2660 |

- adaptive disagree beats entropy on every metric (acc +2.7pp, worst +3.2pp).
- **BUT worst_cell 0.612 < best fixed 0.681 (λ=0.2); cell_gap 0.266 WIDER than fixed λ=0.2 (0.086).**
- Criterion "adaptive worst_cell ≥ best fixed" = **NOT MET**.

---

## Honest bottom line

**What worked (the contribution):** The disagreement signal definitively fixes the convergence-death root cause.
- E2 gate G1/G2/G3 passed; G2 re-confirmed on full run (δ alive at ρ=0.8).
- E3 λ stratification held to R150 (0.203) where entropy collapsed (0.009).
- Beats the entropy controller everywhere: E2 +2.4pp integrated / +7.6pp at ρ=0.8; E3 +2.7pp acc / +3.2pp worst_cell.

**What did NOT clear the bar:**
- E2: PARTIAL (+1.29pp vs best fixed, target +2).
- E3: adaptive worst_cell (0.612) below best fixed (0.681); cell_gap wider, not tighter.

**Why (single shared cause):** sustained high-ρ δ (~0.31–0.38) sits near/below mu=0.35, so the high-ρ cell's λ relaxes to ~0.49 instead of approaching lam_min — generalization is not aggressive enough exactly where it's needed most (worst cell / sustained drift). This is a calibration-center mismatch, **not** signal death. Per guardrail, mu is locked (last adjustment used); not tuned further.

**Paper implication:** C1 (disagreement signal solves entropy's convergence-death) is strongly supported as a mechanism/diagnostic contribution — the stratification figure and entropy-vs-disagreement contrast are clean. But neither E2 nor E3 delivers a "beats best fixed" headline at this calibration. The framing should lead with C2 (entropy failure analysis) + C1 (disagreement as the fix that restores adaptation), with the accuracy results reported honestly as: matches/slightly-improves integrated (E2 PARTIAL), large improvement over the prior adaptive controller, does not yet beat hand-tuned fixed λ on worst-cell (E3).

## Backups preserved (C2 ablation)
- E2: `adaptive_splitomc_entropy.json`, `adaptive_splitomc_disagree_mu392.json`
- E3: `mode_equal_spread/adaptive_splitomc_entropy.json`
- Canonical (mu=0.35 disagreement): `adaptive_splitomc.json` in both E2/E3 dirs.
