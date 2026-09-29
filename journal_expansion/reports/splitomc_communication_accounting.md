# §9 — Communication Accounting (logical SplitOMC vs DriftGate additional)

**Date**: 2026-08-02 · Corrects the misleading "0 B activation upload" phrasing. Separates
the underlying SplitOMC protocol's traffic from DriftGate's *additional* traffic.

## Execution semantics (from the code)
- **Algorithmic implementation** (this repo): each client owns its client block AND a local
  copy of its |Z_k| server block(s); forward/backward run in one process on one GPU. So in
  THIS implementation no smashed activation crosses a network — but that is an
  implementation choice, not the protocol's communication cost.
- **Logical SplitOMC protocol** (what the paper's system model assumes): the client block
  runs on-device and the server block on the edge; the smashed activation is uploaded each
  forward and its gradient returned each backward; server/client blocks are exchanged for
  cell aggregation each round.

## Quantities (CNN, cut = [4c,8,8] = 8192 floats/sample)

| Component | Underlying SplitOMC (logical) | DriftGate additional |
|---|---|---|
| Smashed activation (fwd) | 8192 floats = **32 KB/sample** (×batch×steps×epochs) | 0 |
| Activation-gradient (bwd) | 32 KB/sample | 0 |
| Model exchange / round | client 1.12 MB + 2× server 5.51 MB = **12.1 MB** (overlap-2) | 0 |
| Client drift signal / round | 0 | **4 B/client** (one TV scalar) |
| Edge-server consensus / round | existing ES graph traffic | **≤ 4 B × degree/ES** (scalar z) |

## Correct claim (replaces "0 B activation upload" and "total 4 B")
> DriftGate adds **one 4-byte scalar per client per round** (plus a per-edge scalar for the
> 1-step consensus) on top of whatever communication the underlying SplitOMC protocol
> already requires. It uploads **no additional activations and exchanges no additional model
> weights**; the probe forward reuses the client's local blocks. The probe's COMPUTE cost is
> reported separately (host-RTX reference: 1.5 ms/client/round; on-device pending, §10).

Do NOT state or imply that DriftGate's total split-learning traffic is 4 B — the 4 B is the
*increment over the SplitOMC baseline*, whose activation and model-exchange traffic
(32 KB/sample and 12.1 MB/round here) is unchanged by DriftGate. `tables/communication_accounting.csv`.
