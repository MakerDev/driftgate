"""Round 15 (task H): accuracy-latency and accuracy-energy figure from a cost input JSON.

Usage:
  python r15_mobile_cost.py INPUT.json OUT_PREFIX

The input follows mobile/mobile_cost_schema.json. measurement_type is one of
  measured  - stream values measured on a device (streams[*]: mean/p95 latency, stream energy); these are plotted as is;
  modelled  - per-branch costs (per_branch) combined as (1 - beta) * local + beta * offloaded[rule] for each rule's own
              offloaded cost; labelled "modelled" in the figure;
  demo      - like modelled, with assumed values (for layout only); the figure is stamped DEMO and must not be used as a
              result. Demo outputs go to figures/demo/.
Accuracy and offloading ratios of the operating points come from the Round 15 tables (S1 by default, whole day):
Device only (beta 0), Confidence-based offloading at tau 0.8, Logit sum and Probability average (full offloading),
DriftGate (full offloading), DriftGate under the online controller (beta 0.5), and the curves of Confidence-based
offloading and of DriftGate over the offloading ratio. Missing energy values stay null and the energy panel says so.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
TAB = HERE / "tables"


def parse(cell):
    return float(str(cell).split(" (")[0])


def operating_points(setting="S1"):
    rows = {r["rule (display)"]: r for r in csv.DictReader(open(TAB / "R15_A1_rules_full.csv"))}
    col = f"{setting} mean (SD)"
    ev = {r["setting"]: r for r in csv.DictReader(open(TAB / "R15_C1_offload_evidence.csv")) if r["window"] == "full"}
    on = {r["answer for offloaded requests"]: r for r in csv.DictReader(open(TAB / "R15_C2_online_controller.csv")) if r["setting"] == setting}
    cur = {r["curve"]: r for r in csv.DictReader(open(TAB / "R15_C1_curves.csv")) if r["setting"] == setting and r["window"] == "full"}
    pts = [dict(rule="Device only", beta=0.0, acc=parse(rows["Device only"][col])),
           dict(rule="Confidence-based offloading", beta=float(ev[setting]["B0 offloading"]), acc=parse(rows["Confidence-based offloading"][col])),
           dict(rule="Probability average", beta=1.0, acc=parse(rows["Probability average"][col])),
           dict(rule="Logit sum", beta=1.0, acc=parse(rows["Logit sum"][col])),
           dict(rule="DriftGate", beta=1.0, acc=parse(rows["DriftGate"][col]))]
    if "DriftGate" in on:
        pts.append(dict(rule="DriftGate (online, beta 0.5)", beta=parse(on["DriftGate"]["realized offloading mean (SD)"]),
                        acc=parse(on["DriftGate"]["accuracy full mean (SD)"]), cost_rule="DriftGate"))
    opts = [float(c.split("_")[1]) for c in cur["SplitGP"] if c.startswith("offloading_")]
    curves = {name: (np.array(opts), np.array([parse(cur[name][f"offloading_{o}"]) for o in opts]))
              for name in ("SplitGP", "DriftGate-P") if name in cur}
    return pts, curves


def cost(inp, rule, beta):
    """modelled per-request cost of a rule at offloading ratio beta: (1 - beta) local + beta offloaded(rule)."""
    pb = inp["per_branch"]
    loc = pb["local"]
    off = pb["offloaded"].get(rule)
    out = {}
    for key in ("latency_ms", "energy_mJ"):
        if loc.get(key) is None or off is None or off.get(key) is None:
            out[key] = None
        else:
            out[key] = (1 - beta) * loc[key] + beta * off[key]
    return out


def main(inp_path, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    inp = json.load(open(inp_path))
    mtype = inp["measurement_type"]
    assert mtype in ("measured", "modelled", "demo"), mtype
    pts, curves = operating_points(inp.get("setting", "S1"))
    rows = []
    if mtype == "measured":
        meas = {(s["rule"], s.get("operating_point", "")): s for s in inp["streams"]}
        for p in pts:
            s = meas.get((p["rule"], p.get("operating_point", "")))
            if s is None:
                continue
            rows.append(dict(p, latency_ms=s.get("mean_latency_ms"), p95_ms=s.get("p95_latency_ms"),
                             energy_mJ=s.get("energy_per_request_mJ"), measured_beta=s.get("offload_ratio")))
    else:
        for p in pts:
            rows.append(dict(p, **cost(inp, p.get("cost_rule", p["rule"]), p["beta"])))
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.2), layout="constrained")
    for ax, key, lab in ((axs[0], "latency_ms", "mean latency per request (ms)"), (axs[1], "energy_mJ", "energy per request (mJ)")):
        have = [r for r in rows if r.get(key) is not None]
        for r in have:
            ax.plot(r[key], r["acc"], "o", ms=5, color="#d7263d" if r["rule"].startswith("DriftGate") else "#555555")
            ax.annotate(r["rule"], (r[key], r["acc"]), fontsize=5.5, xytext=(3, 2), textcoords="offset points")
        if mtype != "measured":
            for name, (b, acc) in curves.items():
                rule = "DriftGate" if name.startswith("DriftGate") else "Confidence-based offloading"
                xs = [cost(inp, rule, float(x))[key] for x in b]
                if all(x is not None for x in xs):
                    ax.plot(xs, acc, "-", lw=1.0, color="#d7263d" if rule == "DriftGate" else "#555555", alpha=0.6)
        if not have:
            ax.text(0.5, 0.5, "no values (null)", transform=ax.transAxes, ha="center", va="center", color="#888888")
        ax.set_xlabel(lab)
        ax.set_ylabel("accuracy (%)")
    stamp = {"measured": "measured", "modelled": "modelled (per-branch costs combined linearly)", "demo": "DEMO: assumed values, not a result"}[mtype]
    fig.suptitle(stamp, fontsize=7, color="#c00000" if mtype == "demo" else "#333333")
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_prefix}.pdf")
    fig.savefig(f"{out_prefix}.png", dpi=300)
    json.dump(dict(measurement_type=mtype, input=str(inp_path), points=rows), open(f"{out_prefix}.json", "w"), indent=1)
    print("written", out_prefix)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
