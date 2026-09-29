# DriftGate TMC Supplementary Experiments Report

**Date**: 2026-08-05 · **Scope**: Task A (Λ-confound ablation), Task B (Schedule-A 5-seed +
significance), Task C (deployment-view aggregate). Method frozen — no controller parameter
changed; the Task-A1 Λ=0.5 freeze is a **diagnostic ablation**, not a method change.
New training: 69 runs (`runs/phaseS_supp/`, manifest in Appendix); reused runs identified
inline. Numbers are real eval points; framing sentences are deliberately omitted.

---

## 1. Executive summary

- **Task A (Λ confound) — closed.** Freezing Λ at 0.5 while keeping λ adaptive reproduces
  the joint DriftGate result almost exactly (Schedule A **+0.04 pp**, p=0.59, CI crosses 0;
  mobility −0.05 pp). And even the best point of a fixed **(λ,Λ)** 2-D grid does not beat
  DriftGate (Schedule A DriftGate **+0.78 pp**, mobility **+0.56 pp**). Both point the same
  way: the advantage comes from λ-adaptation timing, not from Λ sitting at a particular value.
- **Task B (5-seed + stats).** With Schedule A now at 5 paired seeds, adaptive − best fixed λ
  = **+0.73 pp (5/5 seeds, paired t p=0.002, 95% CI [+0.46,+1.00] excludes 0)** — smaller than
  the earlier 2-seed +0.93 pp but statistically supported. The full significance table (t-test
  + Wilcoxon + CI) is in §3.2; small-n limits are flagged.
- **Task C (deployment view).** As a single blind policy applied to all 12 environments,
  DriftGate averages **+0.11 pp** over the deployable robust fixed (λ0.2, Λ0.5) and is not
  worse on the worst environment; it trades static-optimum losses for temporal/mobility gains.

---

## 2. Task A — Λ-confound ablation

### 2.1 A1: adaptive λ + fixed Λ=0.5 (diagnostic)

| environment | A1 (adaptive λ, Λ=0.5) | joint DriftGate | A1 − joint (paired) |
|---|---:|---:|---:|
| Schedule A (5 seeds) | 0.6598 | 0.6594 | **+0.04 pp** (3/5+, per-seed [−0.07,+0.15,−0.14,+0.17,+0.07]×10⁻²) |
| mobility-med (3 seeds) | 0.5920 | 0.5925 | −0.05 pp (per-seed [−0.14,0.00,−0.02]×10⁻²) |

Freezing Λ removes essentially none of the joint benefit. (Joint = existing `dvsig_tv_A` +
new B1 seeds; A1 = new runs.) λ/Λ trajectories, churn (0.0092), and bounds were sanity-checked
(§4).

### 2.2 A2: fixed (λ, Λ) 2-D grid (new), best point vs DriftGate

Schedule A (3 seeds/point; Λ=0.5 column reused from the existing grid, not re-run):

| | Λ=0.4 | Λ=0.6 | Λ=0.7 |
|---|---:|---:|---:|
| λ=0.3 | 0.6474 | 0.6497 | 0.6486 |
| λ=0.4 | 0.6495 | **0.6517** | 0.6507 |
| λ=0.5 | 0.6473 | 0.6504 | 0.6498 |

Best fixed (λ,Λ) = (0.4, 0.6) = **0.6517**; joint DriftGate 0.6594 → **DriftGate +0.78 pp**.
Mobility-med: best fixed (0.4, 0.6) = **0.5870**; DriftGate 0.5925 → **+0.56 pp**.
(Best Λ for the fixed policy is 0.6, ~0.2 pp above Λ=0.5 — so the original Λ=0.5-only grid
slightly understated the fixed baseline; DriftGate's margin narrows from +0.93 to +0.78 pp on
A but stays positive.)

### 2.3 Judgment (per the pre-stated frame)
A1 retains the benefit AND no fixed (λ,Λ) point beats DriftGate → **the interpretation "the
gain comes from λ-adaptation timing" is supported**; the Λ confound does not explain it.
Λ's own adaptation contributes ≈0 here (A1 vs joint +0.04 pp, not significant).

---

## 3. Task B — Schedule-A 5 seeds + significance

### 3.1 B1: Schedule A at 5 paired seeds (new seeds s3, s4)

adaptive A (5 seeds) = 0.6594; best fixed λ over the grid at 5 seeds = fx40 = 0.6521
(fx50 = 0.6490). Paired **adaptive − fx40 = +0.73 pp, 5/5 seeds**. New seed s3 is a harder
partition (adaptive 0.6294 vs the s0–s2 cluster 0.657–0.662; see §4 anomaly) — but the
FIXED arm drops on s3 too, so the **paired** difference stays positive; that is why the
5-seed mean difference (+0.73) is lower than the 2-seed (+0.93) yet now significant.

### 3.2 B2: significance table (new + reused runs; no new training)

| comparison | n | mean diff | std | +/n | paired-t (p) | Wilcoxon p | 95% CI | 0? |
|---|--:|--:|--:|--:|--:|--:|--:|:--:|
| (1) signal swap TV−entropy @A 150R | 5 | +1.06 pp | 0.87 | 5/5 | 2.74 (0.052) | 0.062 | [−0.02,+2.14] | crosses |
| (2) adaptive − best fixed λ @A | 5 | +0.73 pp | 0.22 | 5/5 | 7.56 (**0.002**) | 0.062 | [+0.46,+1.00] | excl |
| (3) adaptive − best fixed @asym | 3 | +0.60 pp | 0.08 | 3/3 | 13.8 (**0.005**) | 0.250 | [+0.42,+0.79] | excl |
| (4a) mobility slow DG−fixed | 3 | +0.76 pp | 0.17 | 3/3 | 7.83 (**0.016**) | 0.250 | [+0.34,+1.17] | excl |
| (4b) mobility med DG−fixed | 3 | +0.65 pp | 0.17 | 3/3 | 6.48 (**0.023**) | 0.250 | [+0.22,+1.09] | excl |
| (4c) mobility fast DG−fixed | 3 | +0.56 pp | 0.21 | 3/3 | 4.61 (**0.044**) | 0.250 | [+0.04,+1.08] | excl |
| (5a) SVHN temporal signal | 5 | +2.40 pp | 0.91 | 5/5 | 5.91 (**0.004**) | 0.062 | [+1.27,+3.52] | excl |
| (5b) SVHN spatial signal | 5 | +0.76 pp | 0.28 | 5/5 | 5.99 (**0.004**) | 0.062 | [+0.41,+1.11] | excl |
| (6a) A1 fixed-Λ − joint @A | 5 | +0.04 pp | 0.14 | 3/5 | 0.59 (0.587) | 0.625 | [−0.13,+0.21] | crosses |
| (6b) DG − best fixed(λ,Λ) @A | 3 | +0.78 pp | 0.46 | 3/3 | 2.93 (0.100) | 0.250 | [−0.37,+1.94] | crosses |

**Small-n caveats (binding):** for n=3 the Wilcoxon signed-rank cannot fall below p≈0.25
regardless of effect, so those rows rely on the paired-t and CI (low power — noted). The
signal-swap row (1) is only borderline at 5 seeds (p=0.052, CI touches 0) due to high seed
variance (std 0.87); reported as "positive 5/5, borderline". (6b) is n=3 and crosses 0
(directional only). Rows (2),(3),(4a–c),(5a,b) have CIs excluding 0.

---

## 4. Anomaly and sanity-check log

- **Gate (1 seed/family) sanity — passed**: a1_A_s0 = 0.6590 ≈ joint 0.6595; a2 points match
  the corresponding fixed grid; λ∈[0.16,0.59], Λ frozen at 0.50 for A1 (verified), churn
  0.007–0.009 (existing range 0.006–0.012); all bounds respected.
- **Anomaly (recorded, not hidden)**: new Schedule-A seed **s3** gives adaptive 0.6294 vs the
  tight s0–s2 cluster (0.657–0.662), a −3 pp outlier. Cause diagnosis: seed-0 new runs are
  normal (a1_A_s0 = 0.6590), the run is complete (150 rounds, 16 evals, correct ρ trace,
  ρ=0 acc 0.664), so this is a **harder seed-3 partition/init, not an implementation bug**.
  Confirmed by the paired design: the fixed arm drops on s3 too, so the paired difference is
  preserved; only the absolute level shifts. s4 is within the normal range.
- Absolute accuracies elsewhere are within the expected ranges for each dataset (no other
  anomalies).

---

## 5. Task C — deployment-view aggregate (no new training)

Single deployable policy applied to every environment (arms: robust fixed λ0.2/Λ0.5;
DriftGate; entropy where available). Hindsight best fixed (λ at Λ0.5) is an **upper
reference**, not deployable.

| environment | DriftGate | robust fixed λ0.2 | entropy | DG−robust | hindsight best (ref) |
|---|---:|---:|---:|---:|---:|
| Schedule A | 0.6595 | 0.6445 | 0.6436 | +1.50 | 0.6500 |
| asym-return | 0.6472 | 0.6423 | 0.6288 | +0.49 | 0.6426 |
| gradual-sigmoid | 0.6272 | 0.6323 | 0.6003 | −0.51 | 0.6323 |
| static spatial | 0.6243 | 0.6328 | — | −0.85 | 0.6328 |
| CIFAR-100 gsig | 0.3540 | 0.3605 | — | −0.65 | 0.3605 |
| CIFAR-100 spatial | 0.3606 | 0.3657 | — | −0.51 | 0.3629 |
| Tiny-ImageNet | 0.2409 | 0.2380 | — | +0.30 | 0.2380 |
| SVHN temporal | 0.7937 | 0.8042 | 0.7697 | −1.05 | 0.8042 |
| SVHN spatial | 0.7865 | 0.7965 | 0.7789 | −1.01 | 0.7965 |
| mobility slow | 0.5926 | 0.5787 | 0.5721 | +1.39 | 0.5850 |
| mobility med | 0.5925 | 0.5803 | 0.5725 | +1.22 | 0.5860 |
| mobility fast | 0.5888 | 0.5788 | 0.5688 | +1.00 | 0.5832 |

- **Mean over the 12 environments**: DriftGate **0.5723**, robust fixed **0.5712** → **+0.11 pp**.
- **Worst environment**: DriftGate 0.2409, robust fixed 0.2380 (DriftGate not worse).
- All cells filled from existing runs (CIFAR-100 robust fixed reused from the grid's λ0.2
  points); no missing cells required new training. Worst-cluster where available is in
  `tables/deployment_view_aggregate.csv`.

---

## 6. Appendix

### 6.1 New-run manifest (`tables/supp_run_manifest.csv`)
69 runs under `runs/phaseS_supp/`; git commit `e82b96b76a5a`; controller frozen (`--mode
selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --spatial_norm --abs_cap`; A1 adds
`--fixed_Lambda 0.5`; A2 uses `--mode fixed --lambda_val · --big_lambda_val ·`). Each row:
run_id, run_name, mode, seed, λ, Λ, wall_min. Provenance JSONs in `provenance/`.

### 6.2 Reused existing runs (identifiers)
- adaptive Schedule A (3 seeds): `dvsig_tv_A_s0..s2`.
- fixed grid Schedule A (Λ=0.5): `phaseB_grid/fx{30,40,50}_A_s*`.
- mobility joint/fixed/entropy: `phaseH_mobility/mob_{slow,med,fast}_{dv2,fx40,fx20,dve}_s*`.
- asym: `phaseD_final/dv2_asym_return_s*`, `gated/d1_unseen/d1_fixed04_asym_return_s*`.
- SVHN holdout: `phaseE_holdout/ho_{dv2,fx20,dve}_{gsig,sp}_s*`.
- signal swap: `gated/d2_horizon/d2_{main,entswap}_A_s*`.
- CIFAR-100 robust fixed: `phaseB_grid/fx20_c100{gsig,sp}_s*`.
- Tiny-ImageNet: `gated/d5_tinyimagenet/d5_{dual,fixed04}_gsig_s*`.

### 6.3 Open questions (assumptions avoided; flagged for the user)
- Task-B1 "best fixed λ" was taken as the grid argmax at 5 seeds (fx40); if the paper wants
  the full 9-λ grid re-run to 5 seeds (45 runs) rather than the peak-region check, that is a
  larger job not performed here.
- Task-A2 grid used the pre-registered best±1 λ window {0.3,0.4,0.5}; a wider window was not
  explored (would be tuning-adjacent and is not needed for the confound judgment).
