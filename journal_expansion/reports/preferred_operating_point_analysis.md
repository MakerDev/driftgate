# §3 — Preferred-Operating-Point Variation vs Adaptive Gain

**Date**: 2026-08-02 · No new training — computed from the completed full fixed-λ grid.
Table: `tables/preferred_operating_point_variation.csv`. Purpose: test whether DriftGate's
advantage is driven by *distribution non-stationarity per se* or by **temporal movement of
the preferred personalization operating point**.

## Method
For each environment, from the full grid, per eval window t compute λ*_t = argmax_λ A_t(λ)
(fixed-λ accuracy). Then V_λ = Σ_t |λ*_t − λ*_{t−1}| (movement of the preferred point),
plus its range, #switches, and the best−2nd-best fixed gap (how much λ choice matters).
Correlate V_λ with DV-2's advantage over the best full-grid fixed (integrated, seed-paired).

## Result

| env | V_λ | range λ* | #switch | best−2nd gap | DV-2 adv vs grid (pp) |
|---|---:|---:|---:|---:|---:|
| Schedule A (temporal) | **1.8** | 0.8 | 6 | 0.0156 | **+0.93** |
| asym_return (temporal) | **1.5** | 0.8 | 5 | 0.0174 | **+0.55** |
| gradual_sigmoid (temporal) | 0.9 | 0.8 | 3 | 0.0175 | −0.01 |
| piecewise_random (temporal) | 1.5 | 0.7 | 5 | 0.0087 | −0.16 |
| spatial equal_spread (static) | 0.5 | 0.5 | 1 | 0.0067 | −0.69 |
| CIFAR-100 gsig (transfer) | 0.8 | 0.8 | 6 | **0.0014** | −0.65 |
| CIFAR-100 spatial (transfer) | 0.9 | 0.7 | 6 | **0.0007** | −0.51 |

**Correlation V_λ ↔ DV advantage: Spearman +0.82, Pearson +0.88 (n=7).**

## Reading (honest, n=7)
1. **The relationship is real and directional**: the two clear DV wins (A, asym) have the
   highest V_λ *and* a meaningful best−2nd gap (fixed choice matters and the best choice
   moves). The clear losses have either a static preferred point (spatial, V_λ=0.5) or a
   near-flat loss surface where λ barely matters (CIFAR-100: best−2nd gap ≈0.001 → nothing
   for adaptation to win).
2. **Two informative exceptions sharpen the claim** (movement is necessary but not
   sufficient): piecewise_random has high V_λ (1.5) yet DV loses (−0.16) — its preferred
   point moves *unpredictably*, so a causal signal cannot track it. CIFAR-100 has moderate
   V_λ but a flat surface — movement exists but yields no accuracy to capture.
   → DV helps when the preferred point moves **and that movement is causally trackable and
   accuracy-relevant.**

## Claim correction (supersedes "boundary law / ∝ non-stationarity")
Per §3.3 (strong relationship), the corrected wording — to be used in the final report,
assessment, and ledger — is:

> **DriftGate is most beneficial when the distribution change moves the preferred
> personalization level over time in a trackable way; a fixed policy remains preferable
> when one operating point stays near-optimal (static heterogeneity), when the loss surface
> in λ is flat (transfer datasets here), or when the preferred point moves unpredictably.**

The phrases "boundary law" and "adaptive benefit ∝ non-stationarity" are RETIRED (too
strong: SVHN/CIFAR-100 are non-stationary in ρ yet fixed wins because the preferred λ does
not move usefully). Figure: `figures/adaptive_gain_vs_operating_point_variation.pdf`.
