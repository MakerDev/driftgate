# Phase F — Architecture & Split-Point Generalization (RQ5) — COMPLETE

**Date**: 2026-08-03 · **Data**: `runs/phaseF_arch/` (36 runs; ResNet-18, 16 clients, 100R,
schedule A, 3 seeds; early/middle/late all complete incl. late fixed+entropy). Arms:
DV-2 (TV) / fixed λ0.2 / fixed λ0.4 / entropy-DV × split {early, middle, late}.

## 1. Results (integrated acc; DV-2 shows integrated/worst)

| split | DV-2 | fixed λ0.2 | fixed λ0.4 | entropy-DV |
|---|---:|---:|---:|---:|
| early  | 0.577 / 0.495 | 0.533 | 0.555 | 0.584 |
| middle | 0.597 / 0.518 | 0.544 | 0.570 | 0.592 |
| late   | 0.619 / 0.550 | 0.567 | 0.597 | 0.617 |

Paired (3 seeds): **DV-2 − best fixed = +2.16 / +2.67 / +2.12 pp (3/3 at every split).**
DV-2 − entropy-DV = **−0.72 (early) / +0.54 (middle) / +0.18 (late)**.

## 2. Findings

1. **DV-2 beats fixed λ at every split point** (+2.1–2.7 pp, 3/3 seeds) — the controller
   transfers cleanly to a ResNet-18 architecture, confirming RQ5 (architecture
   generalization) beyond the CNN.
2. **The signal's edge over entropy depends on role-separation strength, and is
   non-monotonic in split depth**: entropy is ahead at the EARLY split (−0.72 pp), DV-2 wins
   at MIDDLE (+0.54 pp), and they converge at LATE (+0.18 pp).
   - Early → the client head is a shallow, weak own-task specialist → the two exits carry
     little complementary information → disagreement is a weaker signal than entropy.
   - Middle → best role separation → the dual-exit signal is most useful.
   - Late → both heads are competent (deep client block), so they agree more and the edge
     shrinks toward entropy.
   → **The signal is most informative at balanced role separation** — a mechanistic boundary
   consistent with the design premise (cross-checked by the §7 role-separation ablation).
3. Absolute ResNet accuracies (0.58–0.62) are below the CNN's Schedule-A level; this is the
   16-client / 100-round pilot configuration, reported as an architecture-generalization
   pilot — NOT mixed into the main CNN accuracy claims. Relative comparisons are within-config.

## 3. Verdict
RQ5 supported: DriftGate carries to ResNet-18 and beats fixed λ across all split depths
(+2.1–2.7 pp, 3/3). The signal advantage over entropy is split-depth-dependent, peaking at
the middle (balanced role separation) — reported honestly and used (with §7) to support
"the signal benefit depends on the degree of role separation between the exits."
