# Phase G — Covariate-Corruption Shift (RQ boundary) — near-complete

**Date**: 2026-07-25 · **Data**: `runs/phaseG_corrupt/` (50/54 done). CIFAR-10, static
ND1, time-varying image-corruption severity (tensor-space; p(x) changes, labels fixed —
COVARIATE drift, NOT concept or traffic-composition drift). 120R, 3 seeds. Arms:
DV-2 (TV) / fixed λ0.2 / entropy-DV. Exit-agreement decomposition recorded every eval.

## 1. Result — the dual-exit signal LOSES its advantage under covariate corruption

Integrated accuracy (3 seeds):

| corruption | schedule | DV-2 | fixed λ0.2 | entropy-DV |
|---|---|---:|---:|---:|
| gaussian_noise | gradual | 0.393 | 0.392 | **0.435** |
| gaussian_noise | abrupt | 0.402 | 0.405 | **0.441** |
| motion_blur | abrupt | **0.512** | 0.485 | 0.510 |
| contrast | gradual | **0.504** | 0.493 | 0.490 |
| contrast | abrupt | 0.493 | 0.480 | 0.493 |

- **On Gaussian noise, entropy-DV beats DV-2 by ~4 pp** — the signal advantage **reverses**.
- On blur/contrast, DV-2 ≈ entropy-DV (±1 pp); DV-2 edges fixed by ~1–2 pp.
- This is the opposite of the traffic-composition regime, where DV-2 beats entropy by
  +2.4 pp (holdout) — a genuine, honest boundary.

## 2. Mechanism — both-wrong disagreement / misleading divergence (why the signal misfires)

Exit-agreement decomposition at Gaussian noise, severity 0.8 (final round):

| both correct | server-only | client-only | both wrong, SAME label | both wrong, DIFF label |
|---:|---:|---:|---:|---:|
| 0.092 | 0.145 | (small) | 0.259 | **0.446** |

Under strong corruption both exits fail on ~70% of inputs; on **44.6%** they fail with
**different** predictions (high disagreement) while **both being wrong**. So corruption
inflates the disagreement signal, but the elevated disagreement is NOT recoverable drift —
generalizing (low λ) cannot fix an input that is itself destroyed. Entropy, a confidence
measure, degrades more gracefully and its controller keeps a better operating point.
The dual-exit signal assumes disagreement ⇒ "server can help"; under covariate corruption
that assumption breaks (both heads are broken), which is the misleading-divergence failure mode
anticipated in the design.

## 3. Claim consequence (scope the contribution honestly)

- **DriftGate's signal advantage is specific to traffic-composition / class-support-mismatch
  drift** (its designed regime: the classes exist, the mixture/exposure changes). It does
  **not** extend to covariate input-corruption shift, where it can underperform entropy
  (Gaussian noise −4 pp).
- Positioning: state the regime explicitly ("distribution / traffic-composition drift")
  and report covariate corruption as a **boundary condition / negative finding**, with the
  misleading-divergence mechanism as the explanation. Do NOT claim general drift robustness.
- A confidence-gated variant (fall back toward entropy/confidence when BOTH exits are
  low-confidence, i.e. suspected corruption rather than drift) is a natural future
  direction — noted, not implemented (freeze holds).

_4 runs pending; numbers may shift by <0.5 pp but the boundary is unambiguous._
