# Final TMC/ToN Expansion Assessment

**Date**: 2026-07-17 · **Evidence base**: 265+ provenance-tracked runs (0 failures),
Gates A-D all executed; reports phase0-6 + gateD_results; tables/figures under
`journal_expansion/`. All controllers label-free and ρ-free at runtime (audited, tested).

## Verdict
- **TMC: GO** (conditional on the writing following the positioning below and the two
  remaining experiment items being added during drafting).
- **ToN: CONDITIONAL_GO** — the network layer is an abstract model (delay/loss/topology/
  participation on the signal plane); ToN would want model-update-plane impairments and
  a larger-L topology study. TMC is the better first target.

## Claims Supported (each with its strongest evidence)
1. **Dual-exit disagreement is an architecture-native drift signal that is more
   persistent than absolute predictive entropy under continued training.**
   Raw entropy AUROC 0.161 on post-convergence drift vs raw δ 0.992 (12 runs, 4
   schedules × 3 seeds); Spearman at 150R: δ 0.875 vs entropy 0.498; signal-swap inside
   the SAME self-calibrated controller: +1.06 pp (5/5 seeds, 150R) and +3.64 pp on the
   sustained high-drift segment; generalizes to 3 unseen schedules (+1.5 pp mean) —
   entirely leak-free.
2. **Entropy's long-horizon failure is global convergence-trend confounding, not
   within-segment decay** (within-segment retention ≈0.95 even at 150R) — a more precise
   mechanism than the ICTC framing, and normalization partially rescues entropy for
   onset detection (honest caveat, included).
3. **A constant-free controller matching hand-calibration.** The ICTC controller's
   mu_drift was re-fitted 3× on ρ-labeled test observations (documented leak; its home-
   schedule value ≈1.3 pp, which does NOT transfer to unseen schedules). The final
   **dual-view controller (selfcal-DV)** — absolute δ level + temporal guarded robust-z +
   spatial cross-cell z, all dimensionless — reaches parity with the leak-calibrated
   controller on unseen schedules AND beats the fixed grid on unseen datasets
   (CIFAR-100 +0.2-0.3 pp, Tiny-ImageNet +0.3 pp over best tested fixed; +1.9-4.0 pp
   over the change-only controller, 9/9 seeds), with zero regression on CIFAR-10.
4. **Best label-free causal method**: lowest regret vs a labeled causal oracle among
   deployable baselines (UCB/EXP3/proxy-grid/update-norm all worse, with 20× more
   parameter churn for bandits).
5. **Robust controller plane**: signal delay 10 rounds ≤ −0.7 pp; 20% loss ≈ 0;
   topology second-order; participation drop fully attributed to training (controller
   share +0.8 pp favorable). Overhead measured: probe 1.5 ms/client/round, 4 B scalar
   uplink vs 12.1 MB/round model exchange, zero activation upload (server copies are
   client-local in SplitOMC).

## Claims Not Supported (report as boundary conditions / negative findings)
1. **"Adaptive beats hindsight-tuned fixed λ" — NOT supported.** On unseen temporal
   schedules the per-schedule best fixed keeps +0.5-1 pp; on CIFAR-10 static spatial,
   fixed λ0.2's worst-cell lead (−3.3 pp vs selfcal-DV) survives every mechanism tested.
   Positioning: "approaches hindsight-tuned fixed without per-deployment tuning; exceeds
   the fixed grid on transfer datasets".
2. **Donor-selected / fairness-weighted cross-cell aggregation (F1/F2) — failed
   mechanism** (oracle donors WORSE than deployable; random worst; 3-seed rigor).
   The worst-cell deficit is not a knowledge-routing problem.
3. **Disagreement-based exit routing (F3) — null** on converged models (exact tie with
   entropy routing).
4. Update-norm and proxy-reward-bandit controllers — negative baselines.

## Strongest Evidence
- The three diagnose→pre-register→verify cycles (boiling-frog → asymmetric guard;
  spatial blindness → cross-cell z; dataset blindness → absolute-level view), each with
  its own unit test, pre-registered constants, and confirming runs — a methodological
  narrative reviewers can audit end-to-end via provenance.
- 5-seed paired signal-swap at 150R (+1.06 pp, 5/5) with the raw-separability mechanism.

## Remaining Reviewer Attack Points
- Single model family for the main line (CNN); ResNet/MobileNet split-point runs exist
  as infrastructure but were not run at Gate-D scale → add 1 dataset × 1 alt-model arm.
- Main-line seeds = 5 (CIFAR-10) / 3 (transfer datasets); bump transfer arms to 5 during
  drafting.
- Traffic-composition drift only (Main/OOP/OOR mixture); no covariate-corruption arm ran
  → one corruption schedule recommended pre-submission.
- Fixed-λ0.2 spatial worst-cell gap needs its "why" fully told (bounds ablation covers
  ~half; the rest is the personalization-generalization frontier itself).
- ToN-grade network realism (update-plane loss/delay, L≫5 topologies) if ToN is chosen.

## Recommended Paper Positioning
Lead with the signal (C1) + the constant-free dual-view controller (C3) as the two
contributions; the leakage audit (Phase 1) becomes the motivation section; fairness
negatives and the fixed-λ boundary are §Limits — the honesty is a feature, the ICTC
narrative's over-claims are all repaired with measured replacements.

## Recommended Venue
**IEEE TMC first** (mobile/edge learning systems, controller + measurement focus).
ToN viable after the network-plane additions.

## Exact Additional Experiments Still Needed (pre-submission)
1. 5-seed completion for D4/D5 dual-view arms (+2 seeds × 9 runs ≈ 1 day).
2. One covariate-corruption drift schedule (CIFAR-10-C style) × main/entropy/fixed ×
   3 seeds (~12 runs).
3. One alt-architecture arm (ResNet-18 middle split, CIFAR-10, schedule A + gsig,
   3 seeds × 3 arms ≈ 18 runs).
4. Multi-seed E4-style mobility with composition-coupled trajectories (runner supports
   it; 6-9 runs).
5. Figure set completion + paper draft (no new compute).
