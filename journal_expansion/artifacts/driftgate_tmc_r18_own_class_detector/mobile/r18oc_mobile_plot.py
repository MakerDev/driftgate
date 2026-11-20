"""Round 18, stage 4: accuracy versus mean latency and mean energy per request from a measurement JSON.

  python r18oc_mobile_plot.py MEASUREMENT.json --curves ../tables/R18oc_stage2_curves.csv --reference ../tables/R18oc_stage2_reference.csv
         --setting S1 --config pretrained-finetuned-block --condition frozen-replay --variant 'concat|mlp64|both' --out ../figures/mobile_S1

Single-edge sequential path, as in the v32 cost model: every request runs the device side (block, exit, detector); an
offloaded request adds uplink of the smashed data, the network round trip, edge execution and downlink of the logits:
  T(rho) = T_device_side + rho * (8 * uplink_bytes / uplink_bps + rtt + T_edge + 8 * downlink_bytes / downlink_bps)
  E(rho) = E_device_side + rho * E_offload_extra            (only when both energies are given; otherwise "not measured")
Overlapping cells (two edge calls) and queueing are not modelled. Accuracy at each offloading ratio comes from the seed-mean
curves of the analysis tables: the new method (detector + edge-corrected-by-detector), the v32 method (device entropy +
average-corrected-even) and the upper envelope of the 25 comparators; device only is the ratio-0 point. If
measurement_type is not "measured", the figure is stamped "가정값 (assumed values, not measurements)".
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

SCORES = ["device-entropy", "device-max-prob", "device-energy", "own-mahalanobis", "own-knn"]
ANSWERS = ["edge-raw", "edge-corrected-even", "average-corrected-even", "probability-average", "logit-sum"]


def font():
    for f in font_manager.findSystemFonts():
        if "NanumGothic" in f and "Coding" not in f:
            return font_manager.FontProperties(fname=f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("measurement")
    ap.add_argument("--curves", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    M = json.loads(Path(a.measurement).read_text())
    rows = [r for r in csv.DictReader(open(a.curves)) if r["setting"] == a.setting and r["model config"] == a.config
            and r["condition"] == a.condition]
    ratio_cols = [c for c in rows[0] if c.startswith("ratio ")]
    rho = np.array([float(c.split()[1]) for c in ratio_cols])
    get = lambda s, ans: np.array([float(r[c]) for c in ratio_cols for r in rows if r["score or detector"] == s and r["answer rule"] == ans])
    new = get(a.variant, "edge-corrected-by-detector")
    v32 = get("device-entropy", "average-corrected-even")
    env = np.max(np.stack([get(s, x) for s in SCORES for x in ANSWERS]), axis=0)
    ref = {(r["rule"], r["answer rule"]): r for r in csv.DictReader(open(a.reference)) if r["setting"] == a.setting
           and r["model config"] == a.config and r["condition"] == a.condition}
    dev_only = float(ref[("Device only", "")]["accuracy mean (SD)"].split()[0])
    p = M["per_request"]
    t_dev = p["device_side_ms"]["mean"]
    net = M["network"]
    t_off = (8 * p["uplink_bytes"] / (net["uplink_mbps"] * 1e6) * 1e3 + net["rtt_ms"] + p["edge_ms"]["mean"]
             + 8 * p["downlink_bytes"] / (net["downlink_mbps"] * 1e6) * 1e3)
    R = np.r_[0.0, rho]
    lat = t_dev + R * t_off
    e_dev, e_off = p.get("energy_device_side_mJ"), p.get("energy_offload_extra_mJ")
    fp = font()
    assumed = M.get("measurement_type") != "measured"
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    for ax, xs, xl in ((axes[0], lat, "mean latency per request (ms)"),
                       (axes[1], None if (e_dev is None or e_off is None) else e_dev + R * e_off, "mean energy per request (mJ)")):
        if xs is None:
            ax.text(0.5, 0.5, "energy not measured\n(에너지 미측정)", ha="center", va="center", transform=ax.transAxes, fontproperties=fp)
            ax.set_xlabel(xl)
            continue
        for y, lab, st in ((np.r_[dev_only, new], "detector + edge-corrected-by-detector", "-"),
                           (np.r_[dev_only, v32], "v32 (entropy + average-corrected-even)", "--"),
                           (np.r_[dev_only, env], "upper envelope of 25 comparators", ":")):
            ax.plot(xs, y, st, lw=1.4, label=lab)
        ax.set_xlabel(xl)
        ax.set_ylabel("accuracy (%)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, loc="lower right")
    if assumed:
        fig.text(0.5, 0.5, "가정값 (assumed values, not measurements)", ha="center", va="center", fontsize=14, color="red",
                 alpha=0.35, rotation=20, fontproperties=fp)
    fig.suptitle(f"{a.setting}, {a.config}, {a.condition}", fontsize=8)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=300)
    json.dump(dict(ratio=R.tolist(), latency_ms=lat.tolist(), offload_path_ms=t_off, device_side_ms=t_dev, assumed=assumed,
                   accuracy_new=np.r_[dev_only, new].tolist(), accuracy_v32=np.r_[dev_only, v32].tolist(),
                   accuracy_envelope=np.r_[dev_only, env].tolist()), open(f"{a.out}.json", "w"), indent=1)
    print(f"{a.out}.pdf (assumed={assumed})")


if __name__ == "__main__":
    main()
