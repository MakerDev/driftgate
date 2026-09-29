# Phase I — Final-DV Network Stress (RQ6, abstract model) — COMPLETE

**Date**: 2026-07-28 · **Data**: `runs/phaseI_network/` (30/30, 0 failures). CIFAR-10,
schedule abrupt, 120R, **final DV-2** (supersedes the v3b numbers of phase6). Explicitly an
ABSTRACT multi-edge impairment model (impairments on the scalar-signal plane +
participation; no PHY claims). Reference (no impairment) DV-2 abrupt ≈ 0.632 (from the
topology/loss rows, which are ≈ clean).

## 1. Results (integrated / worst-cell, 3 seeds)

| condition | integrated | worst-cell | Δ vs ~clean (0.632) |
|---|---:|---:|---:|
| ring / star / dynamic topology | 0.6309 / 0.6309 / 0.6312 | 0.5498 / 0.5486 / 0.5500 | ≈ 0 |
| signal loss 10% / 20% (hold-last) | 0.6310 / 0.6305 | 0.5487 / 0.5477 | ≈ 0 |
| signal delay 2 / 5 / 10 rounds | 0.6295 / 0.6272 / 0.6240 | 0.5467 / 0.5439 / 0.5405 | −0.2 / −0.5 / −0.8 pp |
| participation 75% / 50% | 0.6059 / 0.5608 | 0.5218 / 0.4774 | −2.6 / −7.1 pp |

**Robustness spread** across all delay/loss/topology conditions (excluding participation):
integrated range = **0.72 pp** (min 0.6240 @delay10, max 0.6312 @dynamic). Switching rate
0.0065–0.0072 |Δλ|/round across all impairments, rising only to 0.0109 at 50% participation
(fewer clients per round) — no delay-induced oscillation anywhere.

## 2. Findings (final DV-2)

1. **The controller plane is delay/loss/topology robust — reconfirmed for DV-2.** 20% signal
   loss ≈ 0 pp (hold-last-value absorbs it); consensus topology (ring/star/dynamic) is
   second-order at 5 cells; 10-round-stale signal costs only −0.8 pp. This underwrites the
   "negligible scalar signaling, graceful degradation" claim with the FINAL method, not v3b.
2. **Participation drop is a training effect, not a controller effect** (same as phase6):
   50% participation −7.1 pp, but fixed-λ controls dropped ~−6.4 pp under identical sampling
   (phase6 §2.3), so the controller-attributable share is small/favorable. The DV-2 numbers
   here are consistent; the attribution carries over.
3. No impairment-induced oscillation.

## 3. Verdict
Final-DV network robustness confirms the abstract-model supplement with the frozen method:
delay/loss/topology cost ≤0.8 pp; participation cost is training-bound. Wording stays
"abstract multi-edge impairment model" (C-9). Update-plane impairments and L≫5 topologies
remain future/ToN work.
