# Phase C — Dual-Exit Signal Family Inside the Final Controller (Gate 2)

**Date**: 2026-07-20 · **Data**: `runs/phaseC_signals/` (36 runs, 0 failures) vs frozen
DV-δ references; 3 seeds, paired. Table: `tables/final_signal_comparison.csv`.

All arms share the IDENTICAL DV pipeline (temporal guarded-z + spatial z + absolute
view); only the signal source changes. Absolute views use each signal's MATHEMATICAL
range (TV, δ ∈ [0,1]; JS ≤ ln2; cosine ≤ 2) — no dataset constants. Symmetric KL is
unbounded, so its absolute view is undefined by construction (documented weakness).
Entropy's absolute level has no competence interpretation → entropy-DV runs
temporal+spatial views only (its Phase-D/E arms are labeled accordingly).

## 1. Results (paired vs DV-δ, integrated accuracy, 3 seeds)

| signal | A | abrupt | spatial | C100-gsig | TinyIN | env mean | worst env |
|---|---:|---:|---:|---:|---:|---:|---:|
| **TV distance** | +0.59 (2/3) | +0.68 (3/3) | −0.09 (0/3) | +0.85 (3/3) | −0.16 (0/3) | **+0.37** | −0.16 |
| Jensen–Shannon | −0.92 (0/3) | +0.34 (2/3) | — | — | — | −0.29 | −0.92 |
| symmetric KL | −0.90 (0/3) | +0.26 (2/3) | — | — | — | −0.32 | −0.90 |
| logit cosine | −0.51 (0/3) | −1.08 (1/3) | — | — | — | −0.80 | −1.08 |

Churn: TV ≤ δ in 4/5 environments (0.001–0.009 both; no over-reactivity in-loop).

## 2. Selection (pre-registered §5.3 principles)

**Final control signal = total-variation distance** ("dual-exit divergence"; hard
disagreement is its top-1 discretization and is retained as the hard-variant ablation).

- Environment-mean best (+0.37 pp) with NO meaningful worst-case loss (min −0.16 pp,
  within seed noise); wins are concentrated where control matters (abrupt +0.68,
  C100 +0.85, 3/3 seeds each).
- Per-seed tally is 8/15 positive — the selection rests on the mean/worst-case
  dominance and the asymmetric magnitudes (wins +0.6–0.9 pp vs losses ≤0.16 pp),
  stated as such (not as "consistently improves").
- Detector-side evidence agrees (Phase 2: TV Spearman 0.921 vs δ 0.875 at 150R; seed
  stability 0.92–0.97 vs 0.56–0.77): the soft average is simply a lower-variance
  estimator of the same exit discrepancy; in-loop it is NOT over-sensitive (churn ≤ δ).
- JS/symKL/cosine: negative results (convergence-coupled scales and, for KL,
  an undefined absolute view) — excluded from mains, reported here.

## 3. Consequences

- **Gate-3 freeze → DV-2** (`provenance/dv2_frozen_manifest.json`): DV with
  signal=tv_dist. Method naming in the paper generalizes to **dual-exit divergence**.
- Gate-1 confirmation arms for TV-DV queued (gmm/shuffle on Schedule A from the TV
  traces) — the adaptation-value verdict is re-checked under the final signal.
- SVHN holdout preregistered AFTER this freeze (`provenance/holdout_preregistration.md`)
  with dv2/dvδ/dv-entropy/fixed-grid arms, 5 seeds, binding success criteria.
- All Phase-D final-baseline arms use DV-2; δ-DV rows kept for continuity.
