"""Generate tables/all_runs.csv (task-spec §12.4 schema) from every run JSON
under journal_expansion/runs/ plus its provenance record."""
import csv
import json
from pathlib import Path
import numpy as np

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
RUNS = JOURNAL_ROOT / "runs"
PROV = JOURNAL_ROOT / "provenance"

COLS = ["run_id", "method", "dataset", "model", "split_point", "seed", "drift_type",
        "schedule", "rho_setting", "topology", "delay", "packet_loss", "mobility",
        "lambda_mode", "Lambda_mode", "signal", "controller", "avg_accuracy",
        "main_accuracy", "oop_accuracy", "oor_accuracy", "worst_cluster",
        "p10_client", "cluster_gap", "detection_auroc", "detection_delay",
        "communication_bytes", "runtime", "status"]


def prov_index():
    idx = {}
    f = PROV / "all_runs.jsonl"
    if f.exists():
        for line in f.read_text().strip().split("\n"):
            try:
                r = json.loads(line)
                idx[r.get("config", {}).get("run_name", "")] = r
            except Exception:
                pass
    return idx


def main():
    prov = prov_index()
    rows = []
    for f in sorted(RUNS.rglob("*.json")):
        if f.name in ("manifest.json", "e0_summary.json") or "provenance" in f.parts:
            continue
        try:
            h = json.load(open(f))
        except Exception:
            continue
        if "eval" not in h or not h.get("eval"):
            continue
        cfg = h.get("config", {})
        evals = h["eval"]
        final = evals[-1]
        p10s = [np.percentile(list(e["per_client_acc"].values()), 10)
                for e in evals if e.get("per_client_acc")]
        pr = prov.get(f.stem, {})
        rows.append({
            "run_id": cfg.get("run_id", pr.get("run_id", f.stem)),
            "method": cfg.get("method", ""),
            "dataset": pr.get("config", {}).get("cfg", {}).get("dataset", "cifar10"),
            "model": pr.get("config", {}).get("cfg", {}).get("model_family", "cnn"),
            "split_point": pr.get("config", {}).get("cfg", {}).get("split_point", "middle"),
            "seed": cfg.get("partition_seed", ""),
            "drift_type": "traffic-composition",
            "schedule": cfg.get("schedule", ""),
            "rho_setting": cfg.get("spatial_mode") or cfg.get("schedule", ""),
            "topology": pr.get("topology", "line"),
            "delay": pr.get("config", {}).get("cfg", {}).get("signal_delay", 0),
            "packet_loss": pr.get("config", {}).get("cfg", {}).get("signal_loss", 0),
            "mobility": bool(cfg.get("use_mobility", False)),
            "lambda_mode": cfg.get("controller_mode", "fixed"),
            "Lambda_mode": cfg.get("controller_mode", "fixed"),
            "signal": cfg.get("controller_signal", ""),
            "controller": cfg.get("controller_mode", ""),
            "avg_accuracy": round(float(np.mean([e["acc_total"] for e in evals])), 4),
            "main_accuracy": round(float(np.mean([e.get("acc_main", np.nan) for e in evals])), 4),
            "oop_accuracy": round(float(np.mean([e.get("acc_oop", np.nan) for e in evals])), 4),
            "oor_accuracy": round(float(np.mean([e.get("acc_oor", np.nan) for e in evals])), 4),
            "worst_cluster": round(float(np.mean([e.get("worst_cell_acc", np.nan)
                                                  for e in evals])), 4),
            "p10_client": round(float(np.mean(p10s)), 4) if p10s else "",
            "cluster_gap": round(float(np.mean([e.get("cell_gap", np.nan) for e in evals])), 4),
            "detection_auroc": "",
            "detection_delay": "",
            "communication_bytes": "",
            "runtime": round(h.get("total_time_sec", 0) / 60, 1),
            "status": pr.get("status", "completed"),
        })
    out = JOURNAL_ROOT / "tables/all_runs.csv"
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
