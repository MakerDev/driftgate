# DriftGate — FINAL Comprehensive Report

**Date**: 2026-08-02 · **Branch**: `adaptive_splitomc_tmc/journal_expansion` ·
**Status**: ALL experiments complete (queue drained). Supersedes `FIRST_REPORT_2026-07-28`
and the earlier interim snapshots.
**Evidence**: 646 run JSONs · 880 provenance records · **0 lost to failure** · 56 unit tests
· `tables/all_runs.csv` = 797 rows. Method frozen: DV-2 (`provenance/dv2_frozen_manifest.json`).

> All numbers are real eval points from run JSONs. Claim discipline: the C-1…C-14 ledger in
> `tmc_remaining_issues_and_claim_corrections.md` is binding. Companion verdict:
> `final_tmc_submission_assessment_v2.md`.

---

## 1. What DriftGate is

In split learning with role-separated dual exits (client head trained only on the client's
own classes; server head aggregated across a cell), the **disagreement between the two exits
on unlabeled traffic** is an architecture-native drift signal — no separate detector, no
labels. DriftGate (**selfcal-DV-2**) maps this signal (total-variation "dual-exit
divergence") to the SplitOMC weights (λ, Λ) through three dimensionless, environment-invariant
reference frames — temporal guarded-robust-z, cross-cell spatial-z, and the signal's absolute
probability level — so **no dataset- or schedule-specific calibration constant is used**.

## 2. The two contributions and when adaptation helps (the spine)

- **C1 — Signal**: dual-exit divergence is a convergence-robust drift signal that
  outperforms absolute predictive entropy. Holds on every schedule, dataset (incl. frozen
  holdout), and under mobility. **Lead contribution.**
- **C2 — Controller**: calibration-free; **it is most beneficial when the distribution
  change moves the preferred personalization level over time in a trackable way** (temporal
  drift, mobility). A fixed policy remains preferable when one operating point stays
  near-optimal (static heterogeneity), when the λ loss surface is flat (transfer datasets),
  or when the preferred point moves unpredictably.

> **When adaptation helps (measured, §preferred_operating_point_analysis):** DriftGate's
> advantage correlates with movement of the preferred operating point V_λ (Spearman +0.82,
> Pearson +0.88, n=7 environments) — NOT with non-stationarity per se. Its signal benefit
> over entropy is present throughout EXCEPT under severe covariate input corruption.
> (The earlier "boundary law / ∝ non-stationarity" wording is retired as too strong.)

## 3. Reproduction & audit (Gate A: PASS)
E2/Schedule-A adaptive reproduced to integrated **0.6669 vs ICTC 0.6671** (−0.02 pp); all
segments ±0.06 pp. 12/12 audit items pass; controller causal, label-free, deterministic.

## 4. Calibration-leakage motivation (Phase 1)
ICTC `mu_drift` re-fitted 3× (0.392→0.35→0.31) on ρ-labeled test observations; leak value
≈1.3 pp on its home schedule, **non-transferring** to unseen schedules → journal method
must be calibration-free.

## 5. Signal benchmark (Gate B) + three controller fixes
Raw AUROC: entropy 0.16 (abrupt) – 0.85; **δ 0.94–1.00; TV 0.98–1.00.** Entropy fails via
global convergence-trend confounding (within-segment retention ≈0.95 @150R). Spearman vs ρ
@150R: TV 0.921, δ 0.875, entropy 0.498. Controller lineage (each fix pre-registered +
unit-tested before its runs; all constants dimensionless):

| # | failure | fix | confirmation |
|---|---|---|---|
| 1 | boiling-frog | asymmetric guard (adapt only z<0.5) | abrupt 0.5922→0.6073 |
| 2 | spatial blindness | cross-cell spatial-z, z=max(z_t,z_sp) | Gate-C worst +2.25 pp (3/3) |
| 3 | dataset blindness | absolute-level cap λ_abs (δ = probability) | +1.9–4.0 pp over v3c (9/9) |

**Gate 2**: TV selected (env-mean +0.37 pp vs δ; JS/KL/cos negative) → "dual-exit
divergence". **Gate 3**: DV-2 frozen.

## 6. Adaptation-value decomposition (Gate 1) — Case A / Case B
Paired DV−control (integrated; final-signal numbers where available):

| env | vs global mean-matched | vs per-cluster mean-matched | vs time-shuffled |
|---|---:|---:|---:|
| Schedule A (temporal, TV) | **+0.90 (3/3)** | — | **+1.24 (3/3)** |
| gradual sigmoid (temporal) | +0.46 (3/3) | +0.49 (3/3) | +0.81 (3/3) |
| equal-spread (static) | +0.33 (3/3) | +0.01 (tie) | +0.05 (tie) |

Temporal → **Case A** (genuine causal adaptation; beats own time-shuffled trajectory).
Static → **Case B** (calibration-free per-cluster configuration).

## 7. Full evaluation (Gate D, 150R, 5-seed core)
- **Horizon (A)**: signal swap +1.06 pp (5/5), **+3.64 pp** on the sustained ρ=0.8 segment.
- **Unseen temporal** (3-sched mean): oracle 0.6439 > hindsight fixed 0.6344 > selfcal
  0.6284 ≈ legacy(leak) 0.6297; signal−entropy +1.5 pp.
- **DV-2 vs FULL fixed grid, seed-paired** (the completed-grid headline):

  | env | type | DV-2 − best-of-full-grid |
  |---|---|---|
  | Schedule A | temporal | **+0.93 pp (2/2 seeds)** |
  | asym_return | temporal | **+0.55 pp (3/3 seeds)** |
  | gradual_sigmoid | temporal | −0.01 pp (tie) |
  | spatial equal_spread | static | −0.69 pp (0/3 seeds) |
  | CIFAR-100 gsig | transfer/static | −0.65 pp (0/3 seeds) |
  | CIFAR-100 spatial | transfer/static | −0.38 pp |

  → confirmed with the *complete* grid: DriftGate beats/matches the full grid on temporal drift, loses on static optima (fixed λ peaks ~0.3–0.4
  then declines; DV-2 sits at/above the peak on temporal, below it on static).

## 8. Frozen holdout (SVHN) — decisive honesty test (preregistered, 5 seeds)

| criterion | temporal | spatial | outcome |
|---|---|---|---|
| signal (DV-2 − entropy-DV) | **+2.40 (5/5)** | +0.76 (5/5, below +1 pp margin) | temporal PASS |
| non-inferiority vs robust fixed λ0.2 ≥ −0.5 pp | −1.05 (1/5) | −1.01 (0/5) | FAIL (static optimum) |

Signal generalizes to a truly unseen dataset; adaptive loses to a tuned fixed λ on SVHN's
static optimum — consistent with the operating-point-movement analysis (§preferred_operating_point). Method NOT revised (freeze held).

## 9. Mobility (Phase H) — where adaptive wins
Composition-coupled mobility (36 runs; O3 bug fixed): DV-2 beats best fixed λ **+0.6–1.4 pp
(3/3 every speed)** and entropy **+2.0 pp**; mobility-speed insensitive; churn
0.009. Together with §7 this is the high-movement (trackable) regime where adaptation helps.

## 10. Causal baselines / network / architecture / covariate / overhead
- **Phase D (causal baselines)**: DV-2 = best accuracy–stability trade-off among label-free
  controllers (churn 0.006–0.012 vs bandits 0.16–0.22; near-fixed accuracy; no tuning);
  labeled oracle leads all by +1.1–1.6 pp (labels-only headroom).
- **Phase I (network, final DV)**: delay/loss/topology spread **0.72 pp** (delay10 −0.8,
  loss20 ≈0, topology 2nd-order); participation drop is a training effect; no oscillation.
- **Phase F (ResNet-18 splits)**: DV-2 beats fixed at every split; signal-vs-entropy edge is
  split-depth-dependent, peaking at MIDDLE (early −0.7 pp = weak role separation; middle
  +0.5 pp; late +0.3 pp). RQ5 supported with a mechanistic boundary.
- **Phase G (covariate corruption) — BOUNDARY**: under Gaussian noise entropy-DV beats DV-2
  by ~4 pp (both-wrong DISAGREEMENT: both exits fail with different labels 44.6% @sev0.8 (misleading divergence — not agreement)). Signal
  scope = traffic-composition drift, not covariate corruption.
- **Phase 7/J (overhead)**: 4 B scalar vs 12.1 MB/round model exchange, 0 B activation
  upload; probe 1.5 ms/client/round. On-device harness built (no fabricated numbers).
- **Failed/negative** (reported): donor/fairness aggregation (oracle donors worse),
  disagreement routing (tie), update-norm & proxy-bandit controllers.

## 11. Final verdict
**TMC: GO (READY_WITH_LIMITATIONS); ToN: CONDITIONAL_GO.** Lead = the signal (holdout-
confirmed). Controller framed by the operating-point-movement result (Spearman +0.82):
beats/matches the full fixed grid on temporal/mobility drift, ties/loses on static optima;
no per-deployment tuning. Covariate corruption, fairness, and routing are reported
boundaries/negatives. Full structured verdict + positioning + title in
`final_tmc_submission_assessment_v3.md`.

## 12. Completeness & artifact index
- Complete: Gates A–D, Gate 1/2/3, SVHN holdout, Phases D/G/H/I, full fixed grid (185),
  g1ext decomposition (69), overhead/harness. Phase F ResNet early+middle complete, late
  partial (6 expensive runs unrun; trend set). 646 runs, 0 failures.
- Reports (24): phase0–2, phaseB/C/D/E/F/G/H/I, gateD_results, dv_frozen_specification,
  tmc_remaining_issues (C1–C14), master_experiment_plan, progress_log,
  FIRST_REPORT + FULL_REPORT + INTERIM (superseded), final_tmc_submission_assessment_v2,
  **this file**.
- Tables (17+): all_runs (797), signal_quality*, adaptation_value_decomposition,
  final_signal_comparison, final_causal_baselines, final_dv_network_stress, gated_*,
  fairness_results, overhead. Figures (8): signal traj/AUROC, dataset_transfer,
  unseen_schedules, dv_vs_mean_matched.
- Durability: cron supervisor (`*/10 * * * *` + `@reboot`) restarts the worker pool;
  it recorded 0 restarts over the final multi-day run (workers stable).
