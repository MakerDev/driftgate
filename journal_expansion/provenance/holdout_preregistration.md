# Frozen Holdout Preregistration — SVHN

**Created BEFORE any holdout run**: 2026-07-20 (see file mtime + provenance)
**Frozen method**: selfcal-DV-2 per `provenance/dv2_frozen_manifest.json`
(signal tv_dist; burn-in 10, warm-up 15, z_guard 0.5, spatial-z, abs-cap; all
constants dimensionless, fixed before this document).

## Dataset
- **SVHN** (73,257 train / 26,032 test, 10 classes, 32×32). Never used in ANY
  controller design, calibration, or diagnosis (verified against
  dv_frozen_manifest.design_contaminated_datasets).
- Reason: real-world sensing imagery (street numbers), zero-friction torchvision
  loading, same input geometry as the frozen CNN (no architecture confound).
- Partition: ND1 (classes_per_client=2, ES scope 40–70%, overlap 50%) — identical
  generator as CIFAR-10; Main/OOP/OOR defined exactly as in the audited pipeline.

## Runs (150R, eval_every 10; model CNN; partition/model seeds (s,100+s))
- Shifts: (a) temporal traffic-composition, schedule gradual_sigmoid;
  (b) spatial heterogeneity, equal_spread (schedule shapes are reused infrastructure,
  documented — the HOLDOUT dimension is the dataset).
- Arms:
  - dv2 (tv), dv_delta (hard variant), dv_entropy (signal ablation) — 5 seeds each
  - fixed λ = 0.2 and 0.4 — 5 seeds; pilot grid λ ∈ {0.0, 0.1, 0.3, 0.6} — 3 seeds
  - robust fixed = λ0.2 (selected on development schedules D1: best 3-schedule mean)
- Total: 2 shifts × (3×5 + 2×5 + 4×3) = 74 runs.

## Success / failure criteria (binding)
- **Success**: dv2 non-inferior (≥ −0.5 pp paired) to the best DEPLOYABLE fixed
  (robust fixed λ0.2) on both shifts AND ≥ +1 pp over dv_entropy on the temporal shift.
- **Partial**: non-inferior on one shift only → reported as boundary.
- **Failure**: dv2 < robust fixed − 0.5 pp on both → reported as failure; method NOT
  revised (freeze holds); paper claims scoped accordingly.
- No method/constant changes in response to these results, in any case.
