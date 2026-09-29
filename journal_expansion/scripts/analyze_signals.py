"""Gate-B analysis: signal-quality metrics from passive recording runs.

Reads journal_expansion/runs/signal_benchmark/rec_<schedule>_s<seed>.json
Writes  tables/signal_quality.csv          (one row per run x signal)
        tables/signal_quality_summary.csv  (mean over seeds per schedule x signal)
        tables/normalizer_comparison.csv   (delta_hard under all S4 normalizers)
        figures/signal_traj_<schedule>.{png,pdf}
        figures/signal_auroc.{png,pdf}
"""
import sys
import json
import csv
from pathlib import Path
import numpy as np

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JOURNAL_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.signals.library import RAW_SIGNAL_NAMES
from src.evaluation.signal_metrics import (
    evaluate_signal, causal_normalize, seed_stability, spearman,
)
from src.controllers.normalizers import NORMALIZERS

RUNS = JOURNAL_ROOT / "runs/signal_benchmark"
TABLES = JOURNAL_ROOT / "tables"
FIGS = JOURNAL_ROOT / "figures"
WARMUP = 15
# rep_* signals are identically 0 until the representation reference freezes
# (R15); normalizing over those zeros clips z at the rail and voids the
# metrics. Their normalizer window starts after the reference exists.
REP_OFFSET = 15
BURN_IN = 10  # v3b: skip the untrained-model transient before warm-up


def load_run(path):
    h = json.load(open(path))
    rho = np.array([r if not isinstance(r, dict) else np.mean(list(r.values()))
                    for r in h["rho_trace"]], dtype=float)
    series = {}
    for name in RAW_SIGNAL_NAMES:
        per_round = h["signals_per_es"].get(name, [])
        series[name] = np.array([np.mean(list(d.values())) if d else np.nan
                                 for d in per_round])
    return h, rho, series


def main():
    TABLES.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    files = sorted(RUNS.glob("rec_*.json"))
    if not files:
        print("no runs found")
        return

    rows = []
    trajs = {}  # (schedule) -> {signal: [series per seed]}, plus rho
    for f in files:
        name = f.stem  # rec_<schedule>_s<seed>
        parts = name.split("_")
        seed = int(parts[-1][1:])
        sched = "_".join(parts[1:-1])
        h, rho, series = load_run(f)
        trajs.setdefault(sched, {"rho": rho, "sig": {}})
        for sig in RAW_SIGNAL_NAMES:
            x = series[sig]
            if np.isnan(x).all():
                continue
            off = REP_OFFSET if sig.startswith("rep_") else BURN_IN
            m = evaluate_signal(x[off:], rho[off:], warmup=WARMUP)
            m.update(run=name, schedule=sched, seed=seed, signal=sig, offset=off)
            rows.append(m)
            trajs[sched]["sig"].setdefault(sig, []).append(x)

    cols = ["run", "schedule", "seed", "signal", "spearman_rho", "pearson_rho",
            "auroc_raw", "auroc_norm", "auprc_norm", "false_alarm_rate",
            "retention_raw", "signal_std_nodrift", "detection_delay", "recovery_delay"]
    with open(TABLES / "signal_quality.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- summary: mean over seeds + seed stability
    summary = {}
    for r in rows:
        key = (r["schedule"], r["signal"])
        summary.setdefault(key, []).append(r)
    with open(TABLES / "signal_quality_summary.csv", "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["schedule", "signal", "n_seeds",
                    "spearman_mean", "spearman_std",
                    "auroc_norm_mean", "auroc_norm_std",
                    "retention_mean", "detection_delay_mean",
                    "false_alarm_mean", "seed_stability"])
        for (sched, sig), rs in sorted(summary.items()):
            stab = seed_stability(trajs[sched]["sig"].get(sig, []))
            def m(k):
                vals = [x[k] for x in rs if np.isfinite(x.get(k, np.nan))]
                return (float(np.mean(vals)) if vals else float("nan"),
                        float(np.std(vals)) if vals else float("nan"))
            sp, sps = m("spearman_rho")
            au, aus = m("auroc_norm")
            ret, _ = m("retention_raw")
            dd, _ = m("detection_delay")
            fa, _ = m("false_alarm_rate")
            w.writerow([sched, sig, len(rs), f"{sp:.3f}", f"{sps:.3f}",
                        f"{au:.3f}", f"{aus:.3f}", f"{ret:.3f}", f"{dd:.1f}",
                        f"{fa:.3f}", f"{stab:.3f}"])

    # ---- normalizer comparison on delta_hard
    with open(TABLES / "normalizer_comparison.csv", "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["schedule", "seed", "normalizer", "auroc", "detection_delay",
                    "false_alarm_rate"])
        for f in files:
            name = f.stem
            parts = name.split("_")
            seed = int(parts[-1][1:])
            sched = "_".join(parts[1:-1])
            _, rho, series = load_run(f)
            x = series["delta_hard"][BURN_IN:]
            rho_c = rho[BURN_IN:]
            for nk in NORMALIZERS:
                m = evaluate_signal(x, rho_c, warmup=WARMUP, normalizer=nk)
                w.writerow([sched, seed, nk, f"{m['auroc_norm']:.3f}",
                            m.get("detection_delay", ""),
                            f"{m['false_alarm_rate']:.3f}"])

    # ---- figures: trajectories (entropy vs delta_hard vs rho)
    for sched, d in trajs.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax2 = ax.twinx()
        rho = d["rho"]
        T = len(rho)
        ax2.fill_between(range(1, T + 1), rho, color="0.85", label=r"true $\rho$")
        ax2.set_ylabel(r"$\rho$")
        ax2.set_ylim(0, 1.6)
        styles = {"ent_client": ("C0", "--", "o"), "delta_hard": ("C3", "-", "s"),
                  "kl_sym": ("C2", "-.", "^"), "rep_centroid": ("C4", ":", "v")}
        for sig, (c, ls, mk) in styles.items():
            if sig not in d["sig"]:
                continue
            arr = np.stack([s[:T] for s in d["sig"][sig]])
            mean = arr.mean(0)
            mean = mean / (np.nanmax(np.abs(mean)) + 1e-12)
            ax.plot(range(1, T + 1), mean, ls, color=c, marker=mk, markevery=10,
                    markersize=4, label=sig, linewidth=1.4)
        ax.set_xlabel("round")
        ax.set_ylabel("signal (max-normalized)")
        ax.legend(loc="upper left", fontsize=9, ncol=2)
        ax.set_title(f"Raw signal trajectories — schedule {sched} (mean of seeds)")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIGS / f"signal_traj_{sched}.{ext}", dpi=150)
        plt.close(fig)

    # ---- AUROC bars
    scheds = sorted(trajs)
    key_sigs = ["ent_client", "ent_server", "delta_hard", "kl_sym", "js_div",
                "tv_dist", "rep_centroid", "rep_mmd"]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.1
    xs = np.arange(len(scheds))
    for i, sig in enumerate(key_sigs):
        vals = []
        for sched in scheds:
            rs = [r["auroc_norm"] for r in rows
                  if r["schedule"] == sched and r["signal"] == sig
                  and np.isfinite(r["auroc_norm"])]
            vals.append(np.mean(rs) if rs else np.nan)
        ax.bar(xs + i * width, vals, width, label=sig,
               hatch=["", "//", "..", "xx"][i % 4])
    ax.set_xticks(xs + width * len(key_sigs) / 2)
    ax.set_xticklabels(scheds)
    ax.set_ylabel("AUROC (causally normalized)")
    ax.axhline(0.5, color="k", linewidth=0.8, linestyle=":")
    ax.legend(fontsize=8, ncol=4)
    ax.set_ylim(0.3, 1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"signal_auroc.{ext}", dpi=150)
    plt.close(fig)

    print(f"rows: {len(rows)} -> {TABLES}/signal_quality*.csv, {FIGS}/signal_*.png")


if __name__ == "__main__":
    main()
