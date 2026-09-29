# Phase 6 — Network-Level Stress (RQ6, abstract model)

**Date**: 2026-07-13 · **Data**: `runs/phase6_network/` (21 runs + 3 participation
controls queued), all selfcal v3b on `abrupt`, 100R; clean matched baseline =
`sc_delta_b10_abrupt` (integ 0.5966 / worst 0.5146, 3 seeds).
Table: `tables/network_robustness.csv`. Explicitly an ABSTRACT network model
(delay/loss/topology on the scalar signal plane + participation) — no PHY claims.

## 1. Results (paired Δ vs clean baseline, 3 seeds unless noted)

| impairment | Δ integrated | Δ worst-cell | note |
|---|---:|---:|---|
| signal delay 2 rounds | −0.39 pp | −0.39 pp | graceful |
| signal delay 5 | −0.71 pp | −0.90 pp | graceful |
| signal delay 10 | −0.66 pp | −0.84 pp | plateaus ≈ −0.7 pp |
| signal loss 10% | +0.10 pp | +0.09 pp | hold-last-value fully absorbs |
| signal loss 20% | +0.14 pp | +0.03 pp | 〃 |
| topology ring / star / dynamic (s0) | −0.1 / −0.8 / −0.2 pp | ±0.3 pp | shape-insensitive at 5 cells |
| participation 50% | −5.54 pp | −5.50 pp | see §2.3 — attribution pending controls |

## 2. Findings

1. **The controller input is delay- and loss-robust.** Because δ varies slowly relative
   to the round clock, a 10-round-stale or 20%-lossy scalar costs ≤0.7 pp. Loss with
   hold-last-value is effectively free. This underwrites the "negligible scalar signaling"
   claim with measured degradation curves rather than assertion. Switching rate stays
   ~0.008-0.010 under all impairments (no delay-induced oscillation at these settings).
2. **Consensus-graph shape is second-order at 5 edge servers** (spectral-gap differences
   don't materialize in accuracy). A larger-L topology arm belongs to Gate D if the
   multi-cell story is emphasized for ToN.
3. **Participation 50% attribution (controls complete)**: fixed λ0.4 under the same 50%
   participation drops **−6.38 pp** (paired, 3 seeds) vs selfcal's −5.54 pp. The entire
   drop is the training effect (half the data per round); the controller-attributable
   component is **+0.84 pp in selfcal's favor** — partial participation does not
   destabilize the controller, it slightly cushions it (per-ES signal aggregation still
   sees all clients' probes; only training participation was sampled).

## 3. Scope notes / future track
- Impairments were applied to the controller-signal plane only; model-update
  delay/loss and asymmetric fronthaul remain open (listed for Gate D or revision).
- Interfaces for a high-fidelity simulator (channel-trace schema, AP/UE topology input)
  are documented in the master plan; not built — core pipeline kept dependency-free.
