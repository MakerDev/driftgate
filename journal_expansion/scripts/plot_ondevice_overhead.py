"""Plot on-device overhead from user-measured JSONs. Refuses to plot without
valid measurements (no placeholders)."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
STAGES = ["client_forward", "probe_path", "signal_computation",
          "controller_update", "e2e_local_probe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="*")
    ap.add_argument("--out", default=str(JOURNAL_ROOT / "figures/ondevice_overhead"))
    args = ap.parse_args()

    devices, data = [], []
    for p in args.results:
        try:
            r = json.load(open(p))
        except Exception:
            continue
        md, st = r.get("metadata", {}), r.get("stages", {})
        if not md.get("device_model") or not st:
            print(f"SKIP {p}: unfilled metadata or no stages")
            continue
        devices.append(f"{md['device_model']}\n({md.get('precision')},"
                       f"{md.get('thread_count')}thr)")
        data.append([st.get(s, {}).get("mean_ms", np.nan) for s in STAGES])

    if not data:
        print("ERROR: no valid measurements — nothing plotted (no fabricated values).")
        sys.exit(1)

    data = np.array(data)
    fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(devices), 4))
    x = np.arange(len(devices))
    w = 0.15
    for i, s in enumerate(STAGES):
        ax.bar(x + i * w, data[:, i], w, label=s.replace("_", " "),
               hatch=["", "//", "..", "xx", "\\\\"][i % 5], edgecolor="k", linewidth=0.4)
    ax.set_xticks(x + 2 * w)
    ax.set_xticklabels(devices, fontsize=9)
    ax.set_ylabel("latency (ms)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Measured on-device probe/controller overhead")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=150)
    print(f"plotted {len(devices)} device(s) -> {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
