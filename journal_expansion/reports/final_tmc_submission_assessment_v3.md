# Final TMC Submission Assessment v3

**Date**: 2026-08-03 · All closure items complete (§2–§14). Frozen method: DV-2 (unchanged).
Companion clean report: `DRIFTGATE_FINAL_CLOSURE_REPORT.md`.

## Verdict
**READY_PENDING_ONDEVICE.** Every closure item is closed with measured evidence — consistency
audit, boundary-claim correction, SVHN criterion correction, role-separation ablation, ResNet
completion, communication accounting, full-grid claim correction, paper result selection.
The only remaining input is the user's on-device latency/energy numbers (harness ready and
verified). Method is not retuned; results that came in weaker were scoped, not hidden.

## Final Scientific Claim
In split learning with role-separated dual exits, the divergence between the personalized
client exit and the generalized server exit is an architecture-native drift signal that
(a) outperforms absolute predictive entropy on traffic-composition / class-support drift,
and (b) drives a calibration-free cluster-level personalization controller that helps when
the preferred operating point moves over time in a trackable way.

## Final Method
DV-2: TV dual-exit divergence → λ_z = min(λ_selfcal(z), λ_abs(δ̂)), z = max(temporal
guarded-robust-z, cross-cell spatial-z); dimensionless constants frozen; no dataset/schedule
calibration. (`provenance/dv2_frozen_manifest.json`.)

## Supported Scope
Traffic-composition shift, class-support mismatch, temporal drift, spatial heterogeneity,
composition-coupled mobility, and settings with sufficient client/server role separation.

## Unsupported Scope
Severe covariate input corruption (signal reverses vs entropy); static-optimum environments
where a tuned fixed λ wins; arbitrary "any distribution/concept drift" generality.

## Evidence for Signal Novelty (role separation, not diversity)
Role-separation ablation (§7): TV–ρ Spearman R1 role-separated **+0.844** vs R2 same-role
−0.44, **R3 same-role+independent-init +0.08 (dead)**, R4 weak-server +0.21. Ensemble
diversity does not create the signal; the roles do. Differentiates from same-role
dual-classifier methods (e.g. MCD).

## Evidence for Controller Value
Gate-1 decomposition (beats own mean-matched fixed + time-shuffled trajectory on temporal,
+0.9–1.2 pp); operating-point-movement correlation (Spearman +0.82, n=7); full-grid wins on
temporal (A +0.93, asym +0.55) and mobility (+0.6–1.4 pp, 3/3).

## Full-Grid Result
DriftGate beats or matches the best fixed setting in several CIFAR-10 temporal and mobility
environments (A +0.93 pp, asym +0.55 pp, gsig tie); a fixed policy remains preferable when
one operating point stays near-optimal (spatial −0.69, SVHN −1.0, CIFAR-100 transfer −0.4 to
−0.65). "best fixed per test env" is a hindsight reference (selection bias); the deployable
baseline is the development-selected robust λ0.2.

## Frozen Holdout Result
SVHN (preregistered, 5 seeds): signal improves over entropy by +2.40 pp (temporal) and
+0.76 pp (spatial), positive all 5 seeds (temporal meets the +1 pp criterion; spatial is
positive but below it). Fixed-policy non-inferiority fails on this static-optimum dataset.

## Role-Separation Result
See Signal-Novelty above — the decisive new experiment; R1 clearly strongest.

## Architecture Result
ResNet-18 (§F, complete): DV-2 beats fixed λ at every split (+2.1–2.7 pp, 3/3); signal edge
over entropy peaks at the middle split (early −0.7, middle +0.5, late +0.2) — most
informative at balanced role separation.

## Mobility Result
Composition-coupled mobility: DV-2 > best fixed +0.6–1.4 pp and > entropy +2.0 pp, 3/3 every
speed; mobility-speed insensitive.

## Communication Accounting
DriftGate adds one 4-byte scalar per client per round (plus a per-edge scalar for consensus)
on top of the underlying SplitOMC traffic (activation 32 KB/sample, model exchange 12.1 MB/
round here); no additional activations or model weights. (NOT "total 4 B".)

## Measurements Pending from User
On-device device name(s)/chipset/RAM/OS/runtime; per-stage latency mean/median/p95/p99;
peak memory; energy (optional). Harness + exact commands in `ondevice_benchmark_handoff.md`.

## Main Reviewer Attack Points Remaining
Single main architecture (CNN; ResNet pilot at 16 clients/100R); covariate-corruption
boundary; static-optimum losses; 3-seed ablations (main claims are 5-seed); abstract network
model. All disclosed.

## Recommended Paper Title
*"DriftGate: Calibration-Free Drift Adaptation in Split Learning via Dual-Exit Divergence."*

## Recommended Contribution Bullets
1. An architecture-native drift signal: the divergence between role-separated split-learning
   exits, needing no separate detector and no runtime labels; shown to arise from the roles,
   not classifier diversity, and to beat absolute entropy (incl. a frozen holdout).
2. A calibration-free cluster-level controller (temporal + spatial + absolute reference
   views) with no dataset/schedule-specific constants; it helps when the preferred
   personalization level moves trackably (temporal drift, mobility) and matches a fixed
   policy otherwise.
3. A measurement-grounded evaluation (full fixed-grid, frozen holdout, mobility, role and
   architecture ablations, communication accounting, on-device harness) with honestly
   scoped boundaries (covariate corruption; static optima).
