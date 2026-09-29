"""Phase 7 — probe/controller overhead accounting (measured, not asserted).

The SplitOMC architecture keeps each client's |Z_k| server-model COPIES on the
client (trained locally, aggregated at the ES). The disagreement probe therefore
runs entirely locally; the controller adds NO activation upload. What it adds:
  - local compute: client forward + |Z_k| server-tail forwards on n probe samples
  - uplink: one 4-byte scalar per client per probe round
  - ES<->ES: one scalar per neighbor per consensus step
  - controller math: negligible CPU

This script MEASURES the compute pieces (GPU + CPU) and tabulates bytes, per
probe size and per signal set. Output: tables/overhead.csv + printed summary.
"""
import sys
import time
import csv
from pathlib import Path

import numpy as np
import torch

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = JOURNAL_ROOT.parent
for p in (str(PROJECT_ROOT), str(JOURNAL_ROOT)):
    sys.path.insert(0, p)

from models.architectures import ModelFactory, count_params
from src.signals.library import compute_client_signals, probe_forward, RepReference


def bench(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000  # ms


def main(device="cuda:1"):  # [SERVER-GPU]
    if not torch.cuda.is_available():
        device = "cpu"
    mf = ModelFactory()
    client = mf.make_client().to(device).eval()
    servers = [mf.make_server().to(device).eval() for _ in range(2)]  # |Z_k|=2 overlap
    cp, sp = count_params(client), count_params(servers[0])

    rows = []
    for n_probe in (16, 64, 128):
        x = torch.randn(n_probe, 3, 32, 32, device=device)

        t_client = bench(lambda: client(x))
        t_full = bench(lambda: probe_forward(client, servers, x, device))
        ref = RepReference(warmup_rounds=1)
        with torch.no_grad():
            _, _, rp = probe_forward(client, servers, x, device)
        ref.accumulate(rp)
        t_allsig = bench(lambda: compute_client_signals(client, servers, x, device, ref))

        rows.append({
            "probe_n": n_probe,
            "client_fwd_ms": round(t_client, 3),
            "probe_fwd_ms": round(t_full, 3),           # client + 2 server tails
            "all16_signals_ms": round(t_allsig, 3),      # full library incl. MMD
            "entropy_only_extra_ms": 0.0,                 # entropy uses client fwd only
            "delta_extra_vs_entropy_ms": round(t_full - t_client, 3),
            "scalar_uplink_bytes": 4,
            "consensus_bytes_per_es_per_step": 4 * 2,     # scalar to <=2 neighbors
            "activation_upload_bytes": 0,                 # probe is fully local
        })

    # model-exchange baseline traffic (for perspective; happens WITHOUT controller)
    model_bytes = {"client_model_bytes": cp * 4, "server_model_bytes": sp * 4,
                   "per_round_update_bytes_overlap2": (cp + 2 * sp) * 4}

    out = JOURNAL_ROOT / "tables/overhead.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"device={device}  client_params={cp:,} server_params={sp:,}")
    for r in rows:
        print(r)
    print(model_bytes)
    print(f"-> per-round probe compute at n=64 is {rows[1]['probe_fwd_ms']:.1f} ms/client "
          f"vs 3 local epochs of training (~seconds); scalar traffic 4 B vs "
          f"{model_bytes['per_round_update_bytes_overlap2']/1e6:.1f} MB model exchange.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")  # [SERVER-GPU]
