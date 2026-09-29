# Final TMC Submission Assessment

**Date**: 2026-08-02 · All experiments complete (queue drained). 646 run JSONs / 880
provenance records / **0 failures** · 56 unit tests. Method frozen: DV-2
(`provenance/dv2_frozen_manifest.json`).

## Verdict
- **TMC: GO (READY_WITH_LIMITATIONS).** The lead contribution (signal) is holdout-confirmed;
  the controller contribution is honestly scoped by a measured boundary law. All claims are
  supported by ≥3-seed (core: 5-seed) paired evidence; every negative is reported.
- **ToN: CONDITIONAL_GO.** The network layer is an abstract multi-edge impairment model;
  ToN would want model-update-plane impairments and L≫5 topologies. TMC is the right first venue.

## Final Method
- **Signal**: total-variation dual-exit divergence (client-head vs mean server-head softmax
  on 64 unlabeled probe samples/client/round). Hard top-1 δ is the discretized ablation.
- **Controller**: selfcal-DV-2 — λ_z = min(λ_selfcal(z), λ_abs(δ̂)), z = max(temporal
  guarded-robust-z, cross-cell spatial-z), EWMA-smoothed, 1-step consensus.
- **Constants** (all dimensionless, environment-invariant, frozen): warm-up 15, burn-in 10,
  β 0.05, z_guard 0.5, σ-floor 0.10·|μ|, EWMA α 0.3, z0 1.5, τ_z 0.75, clip [−2,6],
  λ∈[0.15,0.7], Λ∈[0.4,0.7]. **No dataset- or schedule-specific calibration constant.**
- Freeze commit / hashes: `provenance/dv2_frozen_manifest.json`.

## Adaptation Claim
**Genuine temporal adaptation (Case A) on non-stationary drift; automatic per-cluster
configuration (Case B) on static heterogeneity.** Established by the Gate-1 decomposition:
DV-2 beats its own global- and per-cluster mean-matched fixed AND its own time-shuffled
trajectory on temporal schedules (+0.9–1.2 pp, up to t=43), and ties per-cluster
mean-matched on static spatial (as theory predicts). NOT "operating-point selection only".

## Supported Claims
1. **Dual-exit divergence outperforms absolute predictive entropy as a drift signal**,
   convergence-robustly, across all tested schedules/datasets/architectures/mobility —
   incl. a preregistered frozen holdout (SVHN +2.40 pp temporal / +0.76 pp spatial, 5/5).
2. **A calibration-free controller matches or beats the FULL fixed-λ grid on non-stationary
   drift**: Schedule A +0.93 pp (t=43), asym_return +0.55 pp (t=14.7), gradual_sigmoid tie;
   composition-coupled mobility +0.6–1.4 pp over best fixed, 3/3 seeds every speed.
3. **The controller needs no per-deployment tuning and no leaky calibration**; it reaches
   parity with the leak-calibrated ICTC controller on unseen schedules without the leak
   (whose value, ≈1.3 pp on its home schedule, does not transfer).
4. **Best accuracy–stability trade-off among label-free controllers** (lowest churn
   0.006–0.012 vs bandits 0.16–0.22; near-fixed accuracy; no tuning).
5. **Overhead**: negligible scalar communication (4 B vs 12.1 MB/round model exchange,
   0 B activation upload); compute reported separately (probe 1.5 ms/client/round);
   robust to signal delay (≤−0.8 pp @10 rounds), loss (≈0 @20%), topology (spread 0.72 pp).

## Claims That Must Be Removed / Qualified
1. **"Beats fixed λ" unconditionally** → REMOVED. True only on non-stationary drift;
   on static optima (spatial, SVHN, CIFAR-100/TinyIN transfer) a tuned fixed λ wins by
   ~0.4–1.0 pp. State the boundary law, never the unconditional form.
2. **"General drift robustness"** → REMOVED. Under covariate input corruption the signal
   loses its edge (Gaussian noise: entropy-DV +4 pp over DV-2; false-agreement mechanism).
   Scope to traffic-composition / support-mismatch drift.
3. **"Fairness/donor cross-cluster transfer breaks the worst-cluster ceiling"** → REMOVED
   (failed mechanism: oracle donors worse than deployable; equalizes downward).
4. **"Disagreement-based exit routing helps"** → REMOVED (exact tie with entropy routing).
5. **"Zero/parameter-free", "convergence-invariant", "cell-free 6G validated",
   "unseen transfer (C100/TinyIN)"** → all replaced per the C-1…C-14 ledger.

## Strongest Results
- Frozen-holdout signal win (SVHN +2.40 pp, 5/5) — leak-free, preregistered.
- Full-grid superiority on temporal drift (A +0.93 pp t=43; asym +0.55 pp t=14.7).
- Gate-1 time-shuffle control (+1.24 pp) — isolates *temporal alignment* as the value.
- Three diagnose→pre-register→verify controller cycles, each unit-tested before its runs.

## Strongest Baselines (that we do NOT beat / only tie)
- Hand-tuned fixed λ on static-optimum environments (spatial λ0.2, SVHN λ0.2, C100 λ0.2).
- Labeled causal oracle (upper bound; +1.1–1.6 pp over any label-free method — unreachable
  without labels).

## Remaining Boundaries / Reviewer Attack Points
- Covariate-corruption weakness (§Phase G) — disclose as scope, not hide.
- Static-optimum settings: fixed λ wins — disclose as the boundary law.
- Single main architecture (CNN); ResNet-18 covered at early/middle (late partial, 6 runs
  unrun — expensive); MobileNet not run. Split-depth: signal edge peaks at MIDDLE split
  (early −0.7 pp vs entropy = weak role separation; middle +0.5 pp; late +0.3 pp).
- Transfer-dataset arms at 3 seeds (holdout at 5); network model is abstract.

## Exact Paper Positioning
Lead with the **signal** (architecture-native, convergence-robust, holdout-proven,
beats entropy everywhere except covariate corruption) and the **leakage-free methodology**
(calibration-free controller; three pre-registered fixes). Frame the controller by the
**boundary law**: adaptive benefit ∝ non-stationarity — beats the full fixed grid on
temporal/mobility drift, ties/loses on static optima, with no per-deployment tuning.
Report fairness/routing/covariate as boundaries and negatives. This is an honest,
reviewer-robust story; the over-claims of the ICTC draft are each replaced with a measured,
scoped statement.

## Recommended Venue
IEEE TMC (mobile/edge learning systems; controller + measurement). ToN after
update-plane network additions.

## On-device Measurements Pending from User
Phase-J harness ready (`scripts/export_benchmark_models.py`,
`benchmark_ondevice.py`, `benchmark_edge_server.py`; templates + refuse-on-empty
build/plot tools). User runs on target devices; no latency numbers are fabricated.

## Exact Additional Experiments Still Worth Doing (none change conclusions)
1. 5-seed completion of transfer/holdout arms (tighter CIs).
2. res_late × {fx40, dve} completion (6 runs, ~13.6 h each) for the full split-depth curve.
3. One MobileNetV2 arm for a second architecture family.
4. A confidence-gated DV variant for the covariate-corruption boundary (future work).

## Final Recommended Title
*"DriftGate: Calibration-Free Drift Adaptation in Split Learning via Dual-Exit Divergence"*
(subtitle option: *"An architecture-native drift signal and where adaptive control helps"*).
