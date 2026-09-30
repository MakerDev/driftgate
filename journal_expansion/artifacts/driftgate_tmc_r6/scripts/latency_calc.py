"""§6.2 end-to-end latency and DriftGate overhead from the Round-6 offload logs.

Per request (A = 32,768 bytes, C = 10 classes):
  ends at the client exit : t_c(1)
  offloaded               : t_c(1) + 8A/B + RTT + t_s_edge(1)
                            (with --net-csv: t_c(1) + network_ms sample + t_s_edge(1); the samples come from
                             offload_client.py and already exclude the server's compute time)
A client in two cells sends to both edges at once, so the same formula applies.
Offload counts per client and evaluation round are read from each run's *_trace.npz
(eval_n_off / eval_n_total). Mean and p95 pool every request of every evaluation round, client and
seed of a method; the p95 is the exact quantile of the resulting mixture.

DriftGate overhead per round (probe of 64 requests):
  (a) server-block replica on the device : t_c(64) + t_s_device(64)
  (b) server block on the edge           : upload 64 (A + 4C) bytes at B + t_s_edge(64)
reported as a share of the local training round (t_train_round) and of the 6-min round.

Inputs: --device-csv (§6.2 device CSV; default = the host placeholder written by r6_server_timing.py),
        --edge-csv (edge GPU timing, T9a_server_timing.csv), --net-csv (optional measured samples).
Outputs: tables/T9_latency.csv, tables/T9_latency_by_slot.csv, tables/T9_overhead.csv
"""
import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parent.parent
TAB = HERE / "tables"
A_BYTES = 128 * 8 * 8 * 4
C = 10
PROBE = 64
ROUND_S = 360.0
SLOTS = [("pre-commute 05:00-07:30", 1, 25), ("commute 07:30-09:30", 26, 45), ("daytime 09:30-16:00", 46, 110),
         ("return 16:00-19:00", 111, 140), ("evening 19:00-20:00", 141, 150)]
SCEN_DIRS = {"S1": "phaseT6_S1", "S2": "phaseT6_S2", "S1fast": "phaseT6_S1fast", "S3_K200": "phaseT6_S3",
             "S3_K500": "phaseT6_S3", "S4": "phaseT6_S4"}


def read_rows(path):
    with open(path) as f:
        return [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]


def mixture_quantile(values, weights, q):
    o = np.argsort(values)
    v, w = np.asarray(values)[o], np.asarray(weights, float)[o]
    cw = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(cw, q - 1e-12)])


def runs():
    """yield (scenario, arm, seed, trace path) for every finished Round-6 run."""
    for jf in sorted(glob.glob(str(JR / "runs" / "phaseT6_*" / "*.json"))):
        if "phaseT6_env" in jf:
            continue
        tr = jf[:-5] + "_trace.npz"
        if not Path(tr).exists():
            continue
        h = json.load(open(jf))
        cfg = h["config"]
        seed = int(re.search(r"_s(\d+)$", Path(jf).stem).group(1))
        if cfg["arm"] in ("driftgate", "entropy"):      # retired (neighbour score average), 2026-10-01
            continue
        yield cfg["scenario"], cfg["arm"], seed, tr, h["eval_rounds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-csv", default=str(TAB / "T9a_host_placeholder_device.csv"))
    ap.add_argument("--device", default=None, help="device name in the device CSV (default: first row)")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--edge-csv", default=str(TAB / "T9a_server_timing.csv"))
    ap.add_argument("--net-csv", default=None)
    ap.add_argument("--bandwidths", nargs="+", type=float, default=[10, 50, 100])
    ap.add_argument("--rtts", nargs="+", type=float, default=[20, 50])
    ap.add_argument("--scenarios", nargs="+", default=["S1", "S2"])
    a = ap.parse_args()

    dev_rows = read_rows(a.device_csv)
    dev = next(r for r in dev_rows if (a.device is None or r["device"] == a.device)
               and (a.backend is None or r["backend"] == a.backend))
    placeholder = dev.get("source", "") != "measured"
    t_c1, t_c64 = float(dev["t_c_b1_ms"]), float(dev["t_c_b64_ms"])
    t_sd64 = float(dev["t_s_b64_ms"])
    t_train = float(dev["t_train_round_s"])
    edge = {r["quantity"]: r for r in read_rows(a.edge_csv)}
    t_se1 = float(edge["t_s_edge_gpu_b1"]["median_ms"])
    t_se64 = float(edge["t_s_edge_gpu_b64"]["median_ms"])
    net = None
    if a.net_csv:
        net = np.array([float(r["network_ms"]) for r in read_rows(a.net_csv)])
    tag = f"{dev['device']} / {dev['backend']}" + (" (host placeholder)" if placeholder else "")

    # gather offload counts per method
    per = {}
    for scen, arm, seed, tr, erounds in runs():
        if scen not in a.scenarios:
            continue
        z = np.load(tr)
        off, tot = z["eval_n_off"].astype(float), z["eval_n_total"].astype(float)
        d = per.setdefault((scen, arm), {"seeds": [], "off": 0.0, "tot": 0.0, "slot": {}})
        d["seeds"].append(seed)
        d["off"] += off.sum()
        d["tot"] += tot.sum()
        for name, lo, hi in SLOTS:
            m = np.array([lo <= r <= hi for r in erounds])
            s = d["slot"].setdefault(name, [0.0, 0.0])
            s[0] += off[m].sum()
            s[1] += tot[m].sum()

    out, out_slot = [], []
    for (scen, arm), d in sorted(per.items()):
        f = d["off"] / d["tot"]
        for B in a.bandwidths:
            for rtt in ([None] if net is not None else a.rtts):
                if net is None:
                    netv = np.array([8 * A_BYTES / (B * 1e6) * 1000 + rtt])
                else:
                    netv = net
                lc = t_c1
                lo = t_c1 + netv + t_se1
                vals = np.concatenate([[lc], lo])
                w = np.concatenate([[d["tot"] - d["off"]], np.full(len(lo), d["off"] / len(lo))])
                mean = float((vals * w).sum() / w.sum())
                out.append(dict(scenario=scen, arm=arm, seeds=" ".join(map(str, sorted(d["seeds"]))),
                                bandwidth_mbps=B if net is None else "measured",
                                rtt_ms=rtt if net is None else "measured", offload_rate=f"{f:.4f}",
                                mean_e2e_ms=f"{mean:.3f}", p95_e2e_ms=f"{mixture_quantile(vals, w, 0.95):.3f}",
                                device_timing=tag, edge_timing=edge["t_s_edge_gpu_b1"]["hardware"]))
                for name, (so, st) in d["slot"].items():
                    if st == 0:
                        continue
                    out_slot.append(dict(scenario=scen, arm=arm, bandwidth_mbps=out[-1]["bandwidth_mbps"],
                                         rtt_ms=out[-1]["rtt_ms"], slot=name, offload_rate=f"{so / st:.4f}",
                                         mean_e2e_ms=f"{(t_c1 * (st - so) + (t_c1 + netv.mean() + t_se1) * so) / st:.3f}"))
    TAB.mkdir(parents=True, exist_ok=True)
    for name, rows in (("T9_latency.csv", out), ("T9_latency_by_slot.csv", out_slot)):
        if rows:
            with open(TAB / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    over = []
    upload_bytes = PROBE * (A_BYTES + 4 * C)
    ta = t_c64 + t_sd64
    over.append(dict(placement="(a) server-block replica on the device", bandwidth_mbps="",
                     probe_bytes_uploaded=0, overhead_ms=f"{ta:.3f}",
                     share_of_local_training=f"{ta / 1000 / t_train:.5f}", share_of_6min_round=f"{ta / 1000 / ROUND_S:.6f}",
                     signal_bytes_per_client=4, timing=tag))
    for B in a.bandwidths:
        tb = 8 * upload_bytes / (B * 1e6) * 1000 + t_se64
        over.append(dict(placement="(b) server block on the edge", bandwidth_mbps=B, probe_bytes_uploaded=upload_bytes,
                         overhead_ms=f"{tb:.3f}", share_of_local_training=f"{tb / 1000 / t_train:.5f}",
                         share_of_6min_round=f"{tb / 1000 / ROUND_S:.6f}", signal_bytes_per_client=4,
                         timing=f"upload at B; edge {edge['t_s_edge_gpu_b64']['hardware']}; training round: {tag}"))
    with open(TAB / "T9_overhead.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(over[0]))
        w.writeheader()
        w.writerows(over)
    print(f"{len(out)} latency rows, {len(over)} overhead rows; device timing: {tag}")


if __name__ == "__main__":
    main()
