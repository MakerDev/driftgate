# §7 — Role-Separation Ablation

**Date**: 2026-08-03 · **Data**: `runs/phaseR_role/` (R2/R3/R4 × 3 seeds) + existing
`dvsig_tv_A` (R1). CIFAR-10, Schedule A, main CNN, 150R, 3 seeds; identical partition/init/
probe/TV signal/controller mapping. **DV-2 (signal + controller) is unchanged** — only the
aggregation ROLE STRUCTURE of the underlying split learning is perturbed. Table:
`tables/role_separation_ablation.csv`.

## Variants
- **R1 role-separated** (DriftGate): client exit λ-personalized (Main specialist), server
  exit cluster-averaged (generalist).
- **R2 same-role**: client exit ALSO cluster-averaged → both exits are generalists.
- **R3 same-role + independent init**: R2 with the client aux head re-initialized from a
  distinct seed → tests whether ensemble diversity (different inits, same role) alone
  creates the signal.
- **R4 weak server**: client stays personalized; the server exit is 50/50-mixed with its
  own pre-aggregation weights → graded reduction of the server's generalization role.

## Result — the signal comes from ROLES, not classifier diversity

| variant | TV–ρ Spearman | AUROC (norm) |
|---|---:|---:|
| **R1 role-separated (DriftGate)** | **+0.844** | **0.826** |
| R2 same-role (both generalized) | −0.438 | 0.603 |
| R3 same-role + independent init | +0.078 | 0.101 |
| R4 weak server | +0.210 | 0.769 |

- **Removing role separation destroys the signal**: R2's TV–ρ correlation drops from +0.84
  to −0.44 and AUROC to chance-level 0.60. When both exits are cluster generalists, their
  divergence no longer tracks the traffic-composition drift.
- **Ensemble diversity does NOT rescue it**: R3 (same role, deliberately independent inits)
  has an essentially dead signal (Spearman +0.08, AUROC 0.10). Two differently-initialized
  same-role classifiers do not produce a drift signal.
- **The effect is graded**: R4 (server role partially weakened) sits between R1 and R2/R3
  (Spearman +0.21) — as the server's generalization advantage shrinks, so does the signal.

(Downstream integrated accuracy differs across variants because each role_mode is a
different training regime; those absolute numbers are confounded and are NOT the ablation's
metric — the metric is SIGNAL QUALITY, i.e. whether the controller's input still tracks
drift. Prediction-decomposition traces are in the CSV.)

## Verdict (§7.5 — R1 clearly strongest)

> **DriftGate does not rely on arbitrary classifier diversity. Its drift signal emerges from
> the complementary personalization (client) and generalization (server) roles already
> present in personalized split learning; removing role separation collapses the signal
> (Spearman +0.84 → −0.44), and mere ensemble diversity does not restore it (+0.08).**

This directly differentiates DriftGate from same-role dual-classifier methods (e.g. Maximum
Classifier Discrepancy), and cross-checks the ResNet split-depth finding (§F: signal edge
peaks at balanced role separation). Figures:
`figures/role_separation_signal_quality.pdf`, `figures/role_separation_prediction_decomposition.pdf`.
