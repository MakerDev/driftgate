# §14 — Related-Work Positioning Memo

**Date**: 2026-08-02 · Table: `tables/related_work_comparison.csv`. Positions DriftGate
without over-claimed "first" statements. Metadata for prior methods is stated only where we
are confident from their primary descriptions; uncertain specifics are hedged.

## Axes and where DriftGate differs
See the CSV for the full matrix (base architecture · signal source · extra detector/model ·
runtime labels · control target · temporal/spatial adaptation · cluster-level control ·
role-separated exits · env-specific calibration · mobility evaluation).

## Closest threats and the distinction
- **SplitGP / SplitOMC / PGFedSplit**: same role-separated split-learning family, but use a
  **fixed** personalization/sharing policy (no drift signal, no adaptation). DriftGate adds
  a control loop driven by the exits they already have.
- **Maximum Classifier Discrepancy (MCD)** and dual-classifier UDA: also use classifier
  discrepancy, but between **two same-role classifiers trained adversarially** on target
  data for feature alignment — a training objective, requiring target data and an extra
  trained head. DriftGate's exits are **not adversarial and not same-role**: the discrepancy
  is a read-only control signal from a personalized vs a generalized exit. Our role-
  separation ablation (§7) directly tests that the roles — not mere two-classifier diversity
  — are what make the signal useful.
- **Drift/representation detectors (Fed-ADE, entropy/feature-distance)**: add a separate
  detector and often need validation labels or reference statistics. DriftGate uses no
  separately trained detector, no runtime labels, and no environment-specific calibration.
- **early-exit personalized SFL under label shift**: uses exit confidence for routing, not a
  cluster-level personalization controller across temporal/spatial/absolute reference frames.

## Recommended novelty sentence (do NOT use "first …")
> DriftGate is distinguished by using the divergence between existing role-separated exits to
> control cluster-level personalization, without a separately trained detector, runtime
> labels, or environment-specific calibration.

Conditioning on the §7 result: if same-role exits (R2/R3) match role-separated exits, drop
the role-separation emphasis and narrow novelty to "existing dual-exit predictions + no
detector + cluster-level temporal/spatial/absolute control." (Resolved once R2/R3/R4 land.)

## Avoided over-claims
"first drift-aware split learning", "first adaptive personalized SFL", "first use of
classifier discrepancy" — all avoided (MCD etc. precede classifier-discrepancy use).
