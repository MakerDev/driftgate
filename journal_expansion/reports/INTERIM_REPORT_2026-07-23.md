# DriftGate / Adaptive-SplitOMC — Comprehensive Interim Report

**Snapshot date**: 2026-07-23 · **Branch**: `adaptive_splitomc_tmc/journal_expansion`
**Status**: TMC-closure experiments in progress (Phases F/G/H/I + full grid running;
Gates A–3 + holdout complete). **459 run JSONs, 511 provenance records, 0 lost to failure**
(32 ResNet OOMs isolated to a self-healing queue, no data loss).
**56 unit tests passing.** Hardware: 2× RTX 3090 Ti (shared with other users).

> This is a self-contained snapshot of the entire program to date. It supersedes the
> phrasing of earlier reports where they differ (see the claim-correction ledger, §12).
> Per-phase canonical files are listed in §14. Numbers are real eval points from the run
> JSONs; nothing is hand-transcribed or estimated.

---

## 1. What DriftGate is (one paragraph)

In split learning with role-separated dual exits (a client head trained only on the
client's own classes; a server head aggregated across a cell), the **disagreement between
the two exits on unlabeled incoming traffic** is an architecture-native drift signal — it
requires no separate drift detector and no labels. A per-edge-server controller maps this
signal to the SplitOMC personalization/sharing weights (λ, Λ). The final controller,
**selfcal-DV**, combines three dimensionless reference frames — a causal temporal
robust-z, a cross-cell spatial robust-z, and the signal's absolute probability level —
so that **no dataset- or schedule-specific calibration constant is ever used**.

## 2. The two contributions, and their evidence status

| # | Contribution | Status | Strongest evidence |
|---|---|---|---|
| **C1** | Dual-exit divergence is a convergence-robust drift signal that outperforms absolute predictive entropy | **CONFIRMED, incl. frozen holdout** | raw AUROC 0.94–1.00 vs entropy 0.16–0.85; 150R signal-swap +1.06 pp (5/5 seeds); SVHN holdout +2.40 pp temporal / +0.76 pp spatial (5/5) |
| **C2** | A calibration-free controller (no per-deployment tuning) built on that signal | **CONFIRMED with a bounded limitation** | ties/leads legacy on unseen schedules without its leak; adaptation has genuine temporal value (Gate 1); but does NOT beat a well-chosen deployable fixed λ (holdout −1 pp) |

The program's honest headline: **the signal is the robust, portable win; the controller
is a no-tuning convenience that approaches — but does not beat — an oracle-tuned fixed λ.**

---

## 3. Phase 0 — Reproduction & audit (Gate A: PASS)

Reproduced the ICTC E2/Schedule-A adaptive run bit-for-method (v4 code, same seeds, new
torch): integrated **0.6669 vs 0.6671** (−0.02 pp); all ρ segments within ±0.06 pp.
12/12 code-audit items pass (λ = model-mixing weight not loss weight; Λ = cell↔global
mix; OOP/OOR labels never in the training loss; disagreement label-free; controller
causal; deterministic splits). Two caveats documented (probe/eval pool overlap; dormant
mobility index bug O3 — later fixed, §10).

## 4. Phase 2/Gate B — Signal benchmark (PASS)

12 passive runs (4 schedules × 3 seeds, 16 raw signals). **Raw AUROC** (drift vs no-drift):

| signal | A | abrupt | recurring | burst |
|---|---:|---:|---:|---:|
| entropy | 0.643 | **0.161** | 0.852 | 0.819 |
| hard δ | 0.941 | 0.992 | 0.999 | 1.000 |
| TV distance | 0.978 | 1.000 | 1.000 | 1.000 |

RQ1 mechanism (refined at 150R, §7): entropy's failure is **global convergence-trend
confounding**, not within-segment decay (within-segment retention ≈0.95 even at 150R).
Causal normalization partly rescues entropy for onset detection (honest caveat), but it
never beats δ/TV. Signal Spearman vs ρ @150R: TV 0.921, δ 0.875, entropy 0.498.

## 5. Controller lineage — three diagnose→pre-register→verify cycles

Each revision was diagnosed on development traces, its dimensionless fix pre-registered
with a unit test **before** its runs, then confirmed. This IS the methodological
contribution.

| # | failure diagnosed | fix (no dataset constants) | confirmation |
|---|---|---|---|
| 1 | **boiling-frog**: guarded baseline absorbs a stepped ρ ramp stage-by-stage → z≈0 through ρ=0.8 | asymmetric guard: adapt baseline only when z<0.5 (convergence=down, drift=up) | abrupt v1 0.5922 → v3c 0.6073; unit test |
| 2 | **spatial blindness**: temporal z≡0 under STATIC heterogeneity | cross-cell robust-z; z=max(z_temporal, z_spatial) | Gate-C worst +2.25 pp (3/3); unit test |
| 3 | **dataset blindness**: static δ level (CIFAR-100 ≈0.6) normalized away → over-personalization | absolute-level cap λ_abs=λ_max−(λ_max−λ_min)·δ̂ (δ is a probability) | +1.9–4.0 pp over v3c (9/9 seeds); unit test |

Final method **selfcal-DV** = all three views; **DV-2** = DV with signal TV (Gate 2).
Frozen: `provenance/dv2_frozen_manifest.json`, spec `reports/dv_frozen_specification.md`.

## 6. Calibration-leakage finding (the motivation)

The ICTC controller's `mu_drift` was re-fitted **three times** (0.392→0.35→0.31), each step
using ρ-LABELED observations of the final test schedule (final step also citing worst-cell
separation). Measured value of the leak: **≈1.3 pp on its home schedule, which does NOT
transfer to unseen schedules** (§7, §9). This is why the journal method must be
calibration-free. Full ledger: `reports/phase1_calibration.md`.

---

## 7. Gate D — Full evaluation (150R, 5-seed core)

### 7.1 Horizon (Schedule A) — C1 at scale
Signal swap in the same controller, 5 seeds paired: **+1.06 pp integrated (5/5)**,
**+3.64 pp in the sustained ρ=0.8 segment**.

### 7.2 Unseen temporal schedules (3-schedule mean, 5 seeds)
| labeled oracle | hindsight best fixed | selfcal (ours) | legacy(leak) | entropy |
|---:|---:|---:|---:|---:|
| 0.6439 | 0.6344 | 0.6284 | 0.6297 | 0.6132 |

C1 generalizes (signal − entropy +1.5 pp mean); self-cal = parity with the leak-calibrated
controller **without its leak**; no deployable adaptive beats the hindsight fixed; the
labeled oracle keeps +1.1–1.6 pp (labels-only headroom).

### 7.3 Spatial ceiling + bounds ablation (150R, 5 seeds)
fixed λ0.2 (0.6328) > ST+λmax0.5 (0.6294) > ST (0.6213). The worst-cell ceiling persists;
λmax=0.7 over-personalization explains ~⅓–½ of it (bounds ablation).

### 7.4 Dataset transfer & the dual-view fix
Change-only selfcal collapses on CIFAR-100/Tiny-IN (0/9 seeds, −1.6 to −3.8 pp) because the
static δ level is normalized away; **DV (absolute-level view) repairs it**: +4.0/+3.4/+1.9 pp
over change-only (9/9 seeds), and above the two fixed points tested (λ0.2/0.4) — but see
§9: the SVHN holdout shows this "above fixed" edge does not survive a full comparison.

## 8. Gate 1 — Adaptation-value decomposition (the key writing-risk answer)

Controls preserve DV's own λ value distribution and destroy only structure (global/
per-cluster mean-matched fixed; time-shuffled replay). Paired DV−control, 3 seeds:

| env | vs global mean-match | vs per-cluster mean-match | vs time-shuffled |
|---|---:|---:|---:|
| Schedule A (temporal) | +0.41 (3/3) | +0.44 (3/3) | +0.42 (3/3) |
| gradual sigmoid (temporal) | +0.46 (3/3) | +0.49 (3/3) | +0.81 (3/3) |
| equal-spread (static spatial) | +0.33 (3/3, t=18) | +0.01 (tie) | +0.05 (tie) |
| **A, under FINAL TV signal** | **+0.90 (3/3)** | — | **+1.24 (3/3)** |

**Verdict (per-regime, locked)**: temporal drift → **Case A, genuine causal closed-loop
adaptation** (beats even its own time-shuffled trajectory — the same λ values at the wrong
times lose accuracy); static spatial → **Case B, calibration-free per-cluster
configuration** (ties per-cluster mean-matched, as theory predicts when nothing changes).

## 9. Gate 2 + Frozen Holdout (SVHN) — the honesty tests

**Gate 2 (signal family in DV, 3 seeds)**: TV env-mean +0.37 pp vs δ (abrupt +0.68, C100
+0.85, both 3/3; worst-case −0.16 = noise); JS/symKL/cosine negative → excluded. Final
signal = **TV ("dual-exit divergence")**; δ = hard-variant ablation.

**SVHN frozen holdout (preregistered before any run; 74 runs, both shifts, 5 seeds)** —
the decisive test, and it splits:

| preregistered criterion | temporal | spatial | outcome |
|---|---|---|---|
| signal: dv2 − entropy-DV ≥ +1 pp | **+2.40 (5/5)** | +0.76 (5/5) | **PASS** |
| non-inferiority: dv2 − robust fixed λ0.2 ≥ −0.5 pp | **−1.05 (1/5)** | **−1.01 (0/5)** | **FAIL (both)** |

- **Signal (C1) PASSES on a truly unseen dataset.** Lead contribution confirmed.
- **Adaptive-beats-fixed FAILS**: a deployable robust fixed λ0.2 beats DV-2 by ~1 pp on
  both shifts (worst-cell −2.32 pp). Method NOT revised (freeze held).
- **Reconciliation with Gate 1**: DV beats its OWN mean-matched fixed (temporal, +0.9–1.2 pp)
  yet loses to the BEST tunable fixed (λ0.2). Both true, both reported. SVHN's optimum is
  an extreme λ (heavy generalization); DV's self-selected average point is ~1 pp short of it.

## 10. Phase D/H/I — final-DV re-runs, mobility fix

- **Phase D (RQ4, final DV-2 causal baselines)**: DV-2 matches the deployable fixed grid
  to within ±0.5 pp on unseen temporal schedules, ahead of legacy/fixed0.4 on 2/3;
  labeled oracle leads all by +1.1–1.6 pp; DV-2 vs entropy-DV +2.18 pp (3/3). **Verdict:
  best accuracy–STABILITY trade-off** (churn 0.006–0.012 vs bandits 0.16–0.22, no tuning)
  — NOT lowest regret, NOT beats-fixed.
- **Mobility bug O3 FIXED** (cid-map instead of list-index) + regression test + dormancy
  evidence; **composition-coupled mobility** implemented in the v2 runner (moving changes
  membership + cell scope + Main/OOP/OOR together; smoke: 48/50 rewired at R5). Phase H
  runs queued.
- **Phase I** (final-DV network stress: delay/loss/topology/participation) queued.

## 11. Phase 7/J — Overhead & on-device harness

Measured (RTX 3090 Ti): probe path 1.5 ms/client/round (+0.7 ms vs entropy-only), full
16-signal library 6.9 ms; controller uplink **4 B scalar** vs **12.1 MB/round model
exchange**, **0 B activation upload** (server copies are client-local in SplitOMC).
**Phase J on-device package built** (export eager/TorchScript/ONNX; benchmark scripts;
schema/protocol/templates; build/plot tools that **refuse unfilled metadata**) — the user
runs it on target devices; no latency numbers are fabricated. Host-CPU reference recorded.

## 12. Claim-correction ledger (13 items; binding for the draft)

Banned → replacement (full list in `reports/tmc_remaining_issues_and_claim_corrections.md`):
- "DV strictly dominates v3c everywhere" → non-inferior on CIFAR-10 temporal; substantially
  improves cross-dataset transfer.
- "zero-constant/parameter-free" → no dataset-/schedule-specific calibration constants.
- **"beats/above the fixed grid" → RETRACTED (C-13)**: holdout shows deployable fixed λ0.2
  beats DV by ~1 pp; positioning = calibration-free controller close to (and on some
  datasets ~1 pp below) an oracle-tuned fixed.
- "best label-free method" → best accuracy–stability trade-off among evaluated label-free
  controllers.
- "entropy always anti-predictive" → anti-predictive under abrupt continued training; less
  consistent across schedules.
- "convergence-invariant" → convergence-robust under evaluated role-separated settings.
- "unseen transfer (C100/TinyIN)" → development transfer datasets; SVHN = frozen holdout.
- "cell-free 6G validated" → abstract multi-edge impairment model.

## 13. Current running work (in progress; partial numbers, not final)

| phase | purpose | status |
|---|---|---|
| F ResNet-18 splits (RQ5 architecture) | role-separation vs split depth | 6/36 (self-healing OOM queue; early-split DV-2 0.562 vs fixed0.2 0.499, n=2 — indicative only) |
| G covariate corruption (RQ + boundary) | false-agreement decomposition under p(x) shift | 16/54 running |
| H composition-coupled mobility (RQ6) | membership+scope+composition coupled | queued (36) |
| I final-DV network stress (RQ6) | delay/loss/topology/participation | queued (30) |
| full fixed grid λ0.0–0.8 (RQ3 confirm) | settle "vs full grid" per environment | 9/180 |

Compute note: the shared machine is loaded by other users; ResNet runs opportunistically
via a dedicated self-healing worker (re-queue on OOM, no data loss). Main CNN queue healthy
(0 non-ResNet failures after the fix).

## 14. Artifact index

- **Reports** (17): phase0–6, phaseB (adaptation), phaseC (signal), phaseD (final causal),
  phaseE (holdout), gateD_results, dv_frozen_specification, tmc_remaining_issues (claims),
  master_experiment_plan, progress_log, final_tmc_ton_assessment (pre-directive),
  FULL_REPORT (pre-directive consolidation), **this file**.
- **Provenance**: `dv_frozen_manifest.json`, `dv2_frozen_manifest.json`,
  `holdout_preregistration.md`, `all_runs.jsonl` (511 records).
- **Tables** (16): all_runs, signal_quality*, adaptation_value_decomposition,
  final_signal_comparison, final_causal_baselines, gated_*, fairness_results, overhead, …
- **Figures** (8+ PDF/PNG): signal trajectories/AUROC, dataset_transfer, unseen_schedules,
  dv_vs_mean_matched, …
- **Code**: `src/` (signals, normalizers, controllers incl. DV + fairness + replay,
  baselines, corruptions, network, models_ext, datasets_ext, runner), `scripts/`
  (run_v2, enqueue_*, analyze_*, queue_worker, resnet_worker, export/benchmark), `tests/`
  (56 tests).

## 15. Provisional final verdict (to be finalized after F/G/H/I)

**TMC: GO as READY_WITH_LIMITATIONS.** Lead = the signal (C1), now holdout-backed. The
controller is positioned as **calibration-free** (no per-deployment λ search, guaranteed
signal advantage over entropy, genuine temporal adaptation over its own operating point)
with an explicitly quantified limitation: it does not beat a well-chosen deployable fixed
λ, and is ~1 pp below it on datasets whose optimum is an extreme λ (SVHN). Fairness/donor
aggregation (failed), disagreement routing (null), and bandit/update-norm baselines
(negative) are reported as boundaries/negatives, not hidden. ToN: CONDITIONAL (network
plane is an abstract model). Remaining pre-submission items: F/G/H/I completion, 5-seed
transfer completion, figure set, draft.
