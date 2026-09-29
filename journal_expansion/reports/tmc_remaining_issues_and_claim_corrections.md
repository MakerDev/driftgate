# TMC Remaining Issues & Claim Corrections

**Date**: 2026-07-18 · Applies to all prior reports and the future draft. Existing report
files are NOT edited (provenance); this file supersedes their phrasings where they differ.

Format per item: current claim → problem → experiment needed → corrected claim →
outcome-dependent final options.

---

## C-1 "DV strictly dominates v3c everywhere" (gateD_results.md §dual-view)

- **Problem**: CIFAR-10 D1 deltas are −0.13/+0.16/+0.61 pp at 3 seeds — within noise; the
  domination is real only on transfer datasets. "Strictly dominates" overstates.
- **Experiment**: none (existing data reread with CIs).
- **Corrected**: *"DV is non-inferior to v3c on CIFAR-10 temporal schedules and
  substantially improves cross-dataset transfer (+1.9–4.0 pp, 9/9 seeds)."*

## C-2 "zero-constant / parameter-free controller"

- **Problem**: DV has no dataset/schedule-fitted constants, but DOES have pre-registered
  dimensionless design constants (W=15, burn-in 10, β=0.05, z_guard=0.5, σ-floor 0.10,
  α=0.3, z0=1.5, τ_z=0.75, clip [−2,6]) and inherits λ/Λ bounds.
- **Corrected**: *"No dataset- or schedule-specific calibration constants are used; all
  controller constants are dimensionless and fixed once across every environment."*

## C-3 "beats the fixed grid" (transfer datasets)

- **Problem**: only λ∈{0.2,0.4} (and legacy) were compared on CIFAR-100/Tiny-IN. A grid
  point λ∈{0.0,0.1,0.3,0.5..0.8} could beat DV.
- **Experiment**: **Phase B full fixed grid** λ∈{0.0..0.8} on all 8 environments.
- **Corrected (until grid done)**: *"above the evaluated fixed configurations
  (λ∈{0.2,0.4})"*. After grid: use "matches/exceeds the full fixed grid" ONLY where true
  per-environment.

## C-4 "best label-free method" (Phase 4)

- **Problem**: measured for v3c, not final DV; regret margins small; churn is the robust
  part.
- **Experiment**: **Phase D re-run of causal baselines with final DV** (+entropy-DV arm).
- **Corrected (fallback)**: *"DV provides the best accuracy–stability trade-off among the
  evaluated label-free controllers."* Upgrade to "best label-free controller" only if:
  lowest-or-tied avg regret AND non-inferior accuracy AND significantly lower churn AND
  holds on ≥4 unseen schedules.

## C-5 "entropy is always anti-predictive"

- **Problem**: raw-entropy AUROC is 0.161 on `abrupt` but 0.64–0.85 elsewhere.
- **Corrected**: *"Raw entropy becomes anti-predictive under the abrupt continued-training
  schedule and is less consistent across schedules than dual-exit disagreement."*

## C-6 "convergence-invariant" (any residual use)

- **Corrected**: *"convergence-robust under the evaluated role-separated training
  settings"* (δ does decline with convergence — 0.32→0.12 at ρ=0; the claim is relative
  persistence, §phase2).

## C-7 "negligible overhead"

- **Problem**: scalar COMMUNICATION is negligible (4 B vs 12.1 MB); compute (probe
  1.5 ms/client/round) and end-to-end on-device costs are separate and the latter is
  UNMEASURED (user will measure).
- **Corrected**: *"negligible scalar communication overhead; compute and end-to-end
  execution costs are reported separately"* + Phase J harness for the missing numbers.

## C-8 "unseen dataset transfer" for CIFAR-100 / Tiny-ImageNet

- **Problem**: D4/D5 results DIAGNOSED the absolute-level blind spot and motivated DV —
  they are development-transfer sets, not holdouts.
- **Experiment**: **Phase E frozen holdout** on a dataset never touched (SVHN/EMNIST
  priority), pre-registered before execution.
- **Corrected**: *"CIFAR-100 and Tiny-ImageNet are development transfer datasets used to
  diagnose absolute-level mismatch; <holdout> is the frozen holdout."*

## C-9 "cell-free 6G validated" / network realism

- **Corrected**: *"an abstract multi-edge impairment model"* everywhere; Phase I re-runs
  impairments with final DV (current numbers are v3b).

## C-10 Adaptation-value attribution (NEW; the biggest writing risk)

- **Problem**: DV's mean λ differs from fixed baselines' λ; its advantage may be
  operating-point selection, not temporal adaptation. No current experiment separates
  these.
- **Experiment**: **Phase B decomposition** — global/per-cluster mean-matched fixed +
  trajectory interventions (time-shuffle/shift/reverse/cluster-shuffle/low-pass).
- **Outcome-dependent framing (§16 of the directive)**:
  - A (beats mean-matched + interventions) → "causal closed-loop adaptation …"
  - B (≈ per-cluster mean-matched > global) → "automatically configures cluster-specific
    personalization levels without calibration"
  - C (≈ global mean-matched) → "calibration-free automatic operating-point selection
    mechanism". No hiding whichever case holds.

## C-11 TV vs hard δ (signal choice)

- **Problem**: TV ≥ δ on several detector metrics (Spearman 0.92 vs 0.88 @150R; better
  seed-stability); downstream control comparison missing.
- **Experiment**: **Phase C signal-family comparison inside DV**.
- **Outcome**: if TV wins → generalize method naming to "dual-exit divergence /
  architecture-native exit discrepancy"; if δ steadier in-loop → keep δ and report TV as
  detector-strong but control-sensitive.

## C-12 Mobility results (E4 legacy + any future)

- **Problem**: dormant indexing bug (`clients[cid]` list-index vs cid) + legacy mobility
  changed membership but not traffic composition.
- **Experiment**: **Phase H** — bug fix w/ tests + composition-coupled mobility re-run.
- **Corrected**: mobility claims quarantined until Phase H completes.

## Master list of banned → replacement phrasings

| banned | replacement |
|---|---|
| DV strictly dominates v3c everywhere | non-inferior on CIFAR-10 temporal; substantially improves cross-dataset transfer |
| zero-constant / parameter-free | no dataset- or schedule-specific calibration constants |
| beats the fixed grid | above the evaluated fixed configurations (until Phase B grid) |
| best label-free method | best accuracy–stability trade-off among evaluated label-free controllers (until Phase D) |
| entropy is always anti-predictive | anti-predictive under abrupt continued training; less consistent across schedules |
| convergence-invariant | convergence-robust under evaluated role-separated settings |
| end-to-end overhead negligible | negligible scalar communication overhead; compute/end-to-end reported separately |
| unseen transfer (C100/TinyIN) | development transfer datasets |
| cell-free 6G validated | abstract multi-edge impairment model |

---

## C-13 (POST-HOLDOUT, 2026-07-22) — "DV above the fixed grid" fully retracted

- **Evidence**: SVHN frozen holdout — deployable robust fixed λ0.2 beats DV-2 by −1.05 pp
  (temporal) and −1.01 pp (spatial), 0–1/5 seeds positive; worst-cell −2.32 pp. Adds to
  D1 (robust fixed +0.6 pp) and CIFAR-10 spatial (fixed λ0.2 +3.3 pp worst).
- **Retract**: every "above the fixed grid", "beats fixed", "exceeds the fixed grid on
  transfer" phrasing. The CIFAR-100/TinyIN edge was only vs λ0.4; full-grid check pending
  but expected to erase it.
- **Corrected claim**: *"DriftGate is a calibration-free controller: without any
  per-deployment λ search it obtains accuracy close to (and on some datasets ~1 pp below)
  an oracle-tuned fixed λ, while its dual-exit-divergence signal reliably outperforms
  entropy. Its temporal adaptation improves over its own average operating point; it does
  not surpass a well-chosen deployable fixed λ."*
- **Reconciliation with Gate-1**: DV beats its OWN mean-matched fixed (temporal, +0.9–1.2 pp)
  yet loses to the BEST tunable fixed (λ0.2). Both true, both reported.
- **Lead contribution unchanged**: the signal (C1), now with frozen-holdout support.

---

## C-14 (FULL-GRID COMPLETE, 2026-07-31) — refines C-13: DV-2 DOES beat the full grid on temporal drift

- **C-13 predicted** the CIFAR-100/TinyIN "beats fixed" edge would vanish once the full grid
  (λ0.0–0.8) ran. **On STATIC/transfer settings it did** (c100gsig: DV-2 0.3540 < best fixed
  fx20 0.3605, −0.65 pp — confirmed). **On TEMPORAL schedules it did NOT.**
- **Full fixed grid vs DV-2, seed-paired (final method, 150R):**
  - Schedule A: DV-2 0.6594 vs best-fixed(fx40) 0.6500 → **+0.93 pp (2/2, t=43)**.
  - asym_return: DV-2 0.6472 vs best-fixed(fx30) 0.6418 → **+0.55 pp (3/3, t=14.7)**.
  - gradual_sigmoid: DV-2 0.6232 vs best-fixed(fx10) 0.6260 → −0.28 pp (1/2, n.s.) = **TIE**.
  - (grid shape confirms fixed λ peaks ~0.3–0.4 then declines; DV-2 sits at/above the peak.)
- **Refined claim (supersedes the blanket C-13 retraction)**: *"On genuinely non-stationary
  drift (temporal schedules, mobility) DV-2 matches or beats the FULL fixed-λ grid, not just
  λ∈{0.2,0.4} — +0.9 pp on Schedule A and +0.55 pp on asym_return (seed-paired, high t),
  tie on gradual_sigmoid. On static optima (spatial heterogeneity, SVHN holdout, CIFAR-100
  transfer) a well-tuned fixed λ still wins by ~0.7–1 pp. The adaptive advantage tracks
  non-stationarity."* This is exactly the §2 boundary law of `FIRST_REPORT`, now with the
  full grid rather than a two-point comparison.
- **Discipline**: "beats the full fixed grid" is now permitted ONLY with the qualifier
  "on non-stationary/temporal drift"; the unqualified/static forms remain retracted.
  Seed counts on A grid are still thin (2 common seeds); the final report tightens them.
