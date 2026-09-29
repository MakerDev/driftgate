"""Build the paper-ready on-device table from user-measured result JSONs.
REFUSES to produce anything without at least one valid measurement — no dummy
or estimated values, ever.

Usage: python build_ondevice_table.py results/*.json [--out tables/ondevice.csv]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

JOURNAL_ROOT = Path(__file__).resolve().parent.parent

STAGES = ["client_forward", "client_exit", "probe_path", "signal_computation",
          "controller_update", "e2e_local_probe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="*")
    ap.add_argument("--out", default=str(JOURNAL_ROOT / "tables/ondevice.csv"))
    args = ap.parse_args()

    rows = []
    for p in args.results:
        try:
            r = json.load(open(p))
        except Exception as e:
            print(f"SKIP {p}: unreadable ({e})")
            continue
        md, st = r.get("metadata", {}), r.get("stages", {})
        missing_meta = [k for k in ("device_model", "chipset", "ram_gb", "power_mode")
                        if md.get(k) in (None, "")]
        if missing_meta:
            print(f"SKIP {p}: metadata not filled in: {missing_meta} "
                  f"(fill by hand per the protocol; no defaults are assumed)")
            continue
        if not st:
            print(f"SKIP {p}: no measured stages")
            continue
        row = {"device_model": md["device_model"], "chipset": md["chipset"],
               "precision": md.get("precision"), "model_format": md.get("model_format"),
               "batch_size": md.get("batch_size"), "probe_size": md.get("probe_size")}
        for s in STAGES:
            row[f"{s}_ms"] = round(st[s]["mean_ms"], 3) if s in st else ""
        row["p95_e2e_ms"] = round(st["e2e_local_probe"]["p95_ms"], 3) \
            if "e2e_local_probe" in st else ""
        row["peak_rss_mb"] = round(r.get("peak_rss_mb") or 0, 1) or ""
        row["energy_j"] = r.get("energy_j") if r.get("energy_j") is not None else ""
        rows.append(row)

    if not rows:
        print("ERROR: no valid measurements. Nothing written — this tool never "
              "fabricates numbers. Run benchmark_ondevice.py on the target device "
              "and fill the metadata first.")
        sys.exit(1)

    cols = list(rows[0].keys())
    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"{len(rows)} measurement(s) -> {args.out}")


if __name__ == "__main__":
    main()
