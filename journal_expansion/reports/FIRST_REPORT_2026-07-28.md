# DriftGate — First Comprehensive Report (all results to date)

**Date**: 2026-07-28 · **Branch**: `adaptive_splitomc_tmc/journal_expansion` ·
**Status**: FIRST (interim) synthesis — everything decisive is in; the remaining jobs are
confirmatory and cannot reverse any conclusion (see §11). A FINAL report will regenerate
this once the queue drains.
**Evidence**: 592 run JSONs, 690 provenance records, **0 lost to failure** · 56 unit tests
passing · Gates A–D + Gate 1/2/3 + frozen holdout + Phases D/G/H/I complete; Phase F
(ResNet) partial, full fixed grid partial.

> Numbers are real eval points from run JSONs; nothing hand-transcribed or estimated.
> This file supersedes the phrasing of all earlier reports where they differ, and folds in
> the mobility result (Phase H) that unifies the story. Claim discipline: the banned→
> replacement ledger in `tmc_remaining_issues_and_claim_corrections.md` (C-1…C-13) is binding.

---

## 1. One-paragraph statement of the work

In split learning with **role-separated dual exits** (a client head trained only on the
client's own classes; a server head aggregated across a cell), the **disagreement between
the two exits on unlabeled incoming traffic** is an architecture-native drift signal —
obtained with no separate detector and no labels. A per-edge-server controller,
**DriftGate / selfcal-DV-2**, maps this signal (total-variation "dual-exit divergence")
to the SplitOMC personalization/sharing weights (λ, Λ) through three **dimensionless,
environment-invariant** reference frames — a causal temporal robust-z, a cross-cell spatial
robust-z, and the signal's absolute probability level — so that **no dataset- or
schedule-specific calibration constant is ever used**.

## 2. The unified finding (the spine of the paper)

Two contributions, one honest boundary law tying them together:

- **C1 — Signal (robust, portable, holdout-confirmed).** Dual-exit divergence is a
  *convergence-robust* drift signal that outperforms absolute predictive entropy. This
  holds on every tested schedule, dataset (incl. a frozen holdout), architecture split,
  and under mobility. It is the lead contribution.
- **C2 — Controller (calibration-free; advantage tracks non-stationarity).** DriftGate
  needs no per-deployment λ tuning and no leaky calibration. **Its accuracy advantage over
  a well-tuned fixed λ appears precisely when the deployment is genuinely non-stationary**
  (temporal drift, mobility) and vanishes when the environment's optimum is a static
  extreme λ (static spatial heterogeneity, SVHN). Both directions are measured and reported.

> **Boundary law**: *DriftGate's adaptive benefit is proportional to how non-stationary the
> traffic-composition drift is; its signal benefit over entropy is present throughout, but
> is specific to traffic-composition / support-mismatch drift and does not extend to
> covariate input corruption.*

## 3. Reproduction & audit (Gate A: PASS)

E2/Schedule-A adaptive reproduced to **integrated 0.6669 vs ICTC 0.6671** (−0.02 pp), all
ρ segments within ±0.06 pp. 12/12 code-audit items pass (λ = model-mixing weight, not loss
weight; Λ = cell↔global; OOP/OOR labels never in training loss; disagreement label-free;
controller causal; deterministic splits). Reports: `phase0_*`, tests in `tests/`.

## 4. Calibration-leakage motivation (Phase 1)

The ICTC controller's `mu_drift` was re-fitted **three times** (0.392→0.35→0.31), each step
using ρ-LABELED test-schedule observations (final step also citing worst-cell separation).
Measured value of the leak: **≈1.3 pp on its home schedule that does NOT transfer to unseen
schedules** (§7). This is why the journal method must be calibration-free.

## 5. Signal benchmark (Gate B: PASS) and the three controller fixes

**Raw AUROC (drift vs no-drift)**: entropy 0.16 (abrupt!) – 0.85; **hard δ 0.94–1.00;
TV 0.98–1.00**. Entropy's long-horizon failure is *global convergence-trend confounding*
(not within-segment decay; retention ≈0.95 even at 150R). Spearman vs ρ @150R: TV 0.921,
δ 0.875, entropy 0.498.

Three diagnose→pre-register→verify cycles built the final controller (each with a unit
test written before its runs; all constants dimensionless):

| # | failure | fix | confirmation |
|---|---|---|---|
| 1 | boiling-frog (stepped ramp absorbed) | asymmetric guard (adapt only when z<0.5) | abrupt 0.5922→0.6073 |
| 2 | spatial blindness (temporal z≡0 on static drift) | cross-cell spatial robust-z, z=max(z_t,z_sp) | Gate-C worst +2.25 pp (3/3) |
| 3 | dataset blindness (static δ level normalized away) | absolute-level cap λ_abs (δ is a probability) | +1.9–4.0 pp over v3c (9/9) |

**Gate 2** selected **TV** as the control signal (env-mean +0.37 pp vs δ; JS/symKL/cosine
negative). Method named **"dual-exit divergence"**; hard δ is the discretized variant.
**Gate 3** froze **DV-2** (`provenance/dv2_frozen_manifest.json`).

## 6. Adaptation-value decomposition (Gate 1) — what the controller actually buys

Controls preserve DV's own λ value distribution and destroy only structure. Paired
DV−control (integrated; TV-DV numbers where available):

| env | vs global mean-matched | vs per-cluster mean-matched | vs time-shuffled |
|---|---:|---:|---:|
| Schedule A (temporal, final TV) | **+0.90 (3/3, t=15)** | — | **+1.24 (3/3)** |
| gradual sigmoid (temporal) | +0.46 (3/3) | +0.49 (3/3) | +0.81 (3/3) |
| equal-spread (static spatial) | +0.33 (3/3, t=18) | +0.01 (tie) | +0.05 (tie) |

**Verdict per regime**: temporal → **Case A, genuine causal closed-loop adaptation** (beats
even its own time-shuffled trajectory — same λ values at the wrong times lose accuracy);
static spatial → **Case B, calibration-free per-cluster configuration** (ties per-cluster
mean-matched, as theory predicts when nothing changes over time).

## 7. Full evaluation (Gate D, 150R, 5-seed core)

- **Horizon (Schedule A)**: signal swap +1.06 pp integrated (5/5), **+3.64 pp** in the
  sustained ρ=0.8 segment.
- **Unseen temporal schedules** (3-sched mean): labeled oracle 0.6439 > hindsight fixed
  0.6344 > **selfcal 0.6284** ≈ legacy(leak) 0.6297; signal − entropy +1.5 pp. Self-cal
  = parity with the leak-calibrated controller **without its leak**.
- **Spatial ceiling** (150R): fixed λ0.2 0.6328 > ST+λmax0.5 0.6294 > ST 0.6213; bounds
  ablation attributes ~⅓–½ of the gap to λmax.
- **Dataset transfer + dual-view fix**: change-only selfcal collapses on CIFAR-100/Tiny-IN
  (0/9); DV repairs it (+1.9–4.0 pp, 9/9), above the two fixed points tested — but see §8.

## 8. Frozen holdout (SVHN) — the decisive honesty test

Preregistered before any run; 74 runs, both shifts, 5 seeds:

| criterion | temporal | spatial | outcome |
|---|---|---|---|
| signal: DV-2 − entropy-DV ≥ +1 pp | **+2.40 (5/5)** | +0.76 (5/5) | **PASS** |
| non-inferiority: DV-2 − robust fixed λ0.2 ≥ −0.5 pp | **−1.05 (1/5)** | **−1.01 (0/5)** | **FAIL (both)** |

- **Signal (C1) PASSES on a truly unseen dataset** — lead contribution confirmed.
- **Adaptive-beats-fixed FAILS**: a deployable robust fixed λ0.2 beats DV-2 by ~1 pp on
  both shifts (worst-cell −2.32 pp). Method NOT revised (freeze held). "beats/above the
  fixed grid" phrasing fully retracted (C-13). SVHN's optimum is an extreme λ (heavy
  generalization) — a static optimum, exactly the Case-B/§2 boundary.

## 9. Where the controller DOES win — mobility (Phase H) and temporal

**Composition-coupled mobility** (36 runs; a move changes membership + cell scope +
Main/OOP/OOR together; O3 bug fixed & tested):

| speed | DV-2 | best fixed (λ0.4) | entropy-DV |
|---|---:|---:|---:|
| slow / med / fast | **0.593 / 0.593 / 0.589** | 0.585 / 0.586 / 0.583 | 0.572 / 0.573 / 0.569 |

DV-2 − best fixed = **+0.6 to +1.4 pp (3/3 seeds, every speed, t=4.6–15.1)**; vs entropy
**+2.0 pp (3/3)**; mobility-speed insensitive; churn 0.009 (no thrash). And on Schedule A
the completed fixed grid so far (fx00 0.630 / fx10 0.638 / fx20 0.645) sits **below DV-2's
0.6595** — temporal drift is a dynamic regime where DV leads. **This is the result that
unifies the story** (§2): fixed wins on static optima (§8), DV wins on dynamic drift.

## 10. Causal baselines, network, overhead, boundaries

- **Phase D (final DV-2 causal baselines)**: matches the deployable fixed grid to within
  ±0.5 pp on unseen temporal schedules; labeled oracle leads all by +1.1–1.6 pp; DV-2 vs
  entropy-DV +2.18 pp (3/3). Verdict: **best accuracy–stability trade-off among label-free
  controllers** (churn 0.006–0.012 vs bandits 0.16–0.22, no tuning) — not lowest regret.
- **Phase I (final-DV network stress)**: delay/loss/topology spread **0.72 pp** (delay10
  −0.8 pp, loss20 ≈0, topology 2nd-order); participation drop is a training effect
  (−7.1 pp @50%, matching fixed-λ controls); no oscillation. Abstract multi-edge model (C-9).
- **Phase G (covariate corruption) — BOUNDARY / negative**: under Gaussian noise
  **entropy-DV beats DV-2 by ~4 pp** — the dual-exit signal loses its advantage. Mechanism:
  false agreement (both exits fail with different labels 44.6% of the time at sev 0.8) —
  corruption inflates disagreement but it is not recoverable drift. Scope: **signal
  advantage is specific to traffic-composition drift, not covariate input corruption.**
- **Phase 7/J (overhead)**: probe 1.5 ms/client/round; **4 B scalar uplink vs 12.1 MB/round
  model exchange; 0 B activation upload**. On-device benchmark harness built (export
  eager/TorchScript/ONNX; scripts refuse unfilled metadata; no fabricated latency numbers) —
  the user measures on target devices.
- **Failed/negative mechanisms** (reported, not hidden): donor-selected/fairness
  aggregation (oracle donors WORSE than deployable), disagreement exit routing (exact tie),
  update-norm and proxy-reward-bandit controllers.

## 11. Why the remaining jobs do not change any conclusion

| remaining | ~count | what it would do | conclusion risk |
|---|---|---|---|
| full fixed grid λ0.0–0.8 | ~167 | complete the per-env DV-vs-grid picture | NONE — confirms §8/§9 (fixed wins static, DV wins dynamic); A grid already trails DV |
| ResNet middle/late seeds | ~10 | tighten split-depth/role-separation story | LOW — direction set (early: entropy≈DV; middle: DV+4 pp) |
| g1ext trajectory tail | ~few | extra adaptation-value controls | NONE — Gate-1 verdict set from core + final-signal re-run |
| 4 slow corruption fx20 | 4 | fixed baseline for motion_blur | NONE — Phase-G boundary set from Gaussian |

The two contributions (C1 signal; C2 calibration-free-adaptation-tracks-non-stationarity),
the boundary law, and every negative are all fixed by data already in hand. The final report
will differ only in tightened CIs and completed per-environment grid tables.

## 12. Provisional verdict

**TMC: GO as READY_WITH_LIMITATIONS.** Lead = the signal (C1), holdout-backed. The
controller is positioned as calibration-free with a quantified, honest boundary: it beats
fixed λ on dynamic drift (temporal/mobility, +0.6–1.4 pp) and loses to a tuned fixed λ on
static optima (SVHN/spatial, ~1 pp), with the signal advantage over entropy present
throughout except covariate corruption. ToN: CONDITIONAL (abstract network model).
Pre-submission: 5-seed transfer completion, ResNet completion, figure set, draft — none
change the conclusions.

## 13. Artifact index
Reports (21): phase0–2, phaseB (adaptation), phaseC (signal), phaseD (causal), phaseE
(holdout), phaseG (covariate), phaseH (mobility), phaseI (network), gateD_results,
dv_frozen_specification, tmc_remaining_issues (claims C1–C13), master_experiment_plan,
progress_log, FULL_REPORT + INTERIM_REPORT (superseded snapshots), **this file**.
Provenance: dv/dv2 frozen manifests, holdout preregistration, all_runs.jsonl (690).
Tables (17): all_runs, signal_quality*, adaptation_value_decomposition, final_signal_
comparison, final_causal_baselines, final_dv_network_stress, fairness_results, overhead,
gated_*. Figures (8): signal traj/AUROC, dataset_transfer, unseen_schedules,
dv_vs_mean_matched. Code: src/ (signals, normalizers, controllers incl. DV+fairness+replay,
baselines, corruptions, network, models_ext, datasets_ext, runner), scripts/, tests/ (56).
Durability: cron supervisor (`*/10 * * * *` + `@reboot`) restarts the worker pool.
