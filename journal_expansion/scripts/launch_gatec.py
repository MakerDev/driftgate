"""Gate-C fairness pilot (Phase 5) — spatial equal_spread (the worst-cell
battleground: cell0 rho=0 ... cell4 rho=0.8), CIFAR-10, 100R.

Pre-registered matrix (seeds 0/1/2 unless noted):
  sc_delta_b10                — self-calibrating controller v3b, no fairness (new baseline)
  sc_delta_b10 + fair_deploy  — F1/F2 deployable donor selection
  sc_delta_b10 + fair_oracle  — oracle donors (analysis upper bound)
  sc_delta_b10 + fair_random  — random donors (seed 0 only; sanity control)
  fixed lam=0.2 / lam=0.4     — the v4 hand-tuned worst-cell references, now 3 seeds
Gate-C pass (pre-registered): deployable fairness lifts worst-cell >= +2 pp over
sc_delta_b10 with avg-acc loss <= 0.5 pp, and approaches fixed lam=0.2's worst cell.
"""
import json
from pathlib import Path

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = JOURNAL_ROOT.parent
OUT = JOURNAL_ROOT / "runs/gatec_fairness"
SEEDS = [(0, 100), (1, 101), (2, 102)]
SLOTS = ["cuda:0", "cuda:0", "cuda:1", "cuda:1"]  # [SERVER-GPU]

BASE = ("python -u {jr}/scripts/run_v2.py --schedule static --spatial equal_spread "
        "--rounds 100 --probe_n 64 --seed {ps} --model_seed {ms} "
        "--run_name {name} --output_dir {out} --device {{DEV}} ")


def build_jobs():
    jobs = []
    for ps, ms in SEEDS:
        common = dict(jr=JOURNAL_ROOT, ps=ps, ms=ms, out=OUT)
        jobs.append((f"sc_delta_b10_sp_s{ps}",
                     BASE.format(name=f"sc_delta_b10_sp_s{ps}", **common)
                     + "--mode selfcal --signal delta_hard --burn_in 10"))
        jobs.append((f"fair_deploy_sp_s{ps}",
                     BASE.format(name=f"fair_deploy_sp_s{ps}", **common)
                     + "--mode selfcal --signal delta_hard --burn_in 10 --fairness deployable"))
        jobs.append((f"fair_oracle_sp_s{ps}",
                     BASE.format(name=f"fair_oracle_sp_s{ps}", **common)
                     + "--mode selfcal --signal delta_hard --burn_in 10 --fairness oracle"))
        jobs.append((f"fixed02_sp_s{ps}",
                     BASE.format(name=f"fixed02_sp_s{ps}", **common)
                     + "--mode fixed --lambda_val 0.2"))
        jobs.append((f"fixed04_sp_s{ps}",
                     BASE.format(name=f"fixed04_sp_s{ps}", **common)
                     + "--mode fixed --lambda_val 0.4"))
    jobs.append((f"fair_random_sp_s0",
                 BASE.format(name="fair_random_sp_s0", jr=JOURNAL_ROOT, ps=0,
                             ms=100, out=OUT)
                 + "--mode selfcal --signal delta_hard --burn_in 10 --fairness random"))
    return [(n, c) for n, c in jobs if not (OUT / f"{n}.json").exists()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    logdir = OUT / "logs"
    logdir.mkdir(exist_ok=True)
    jobs = build_jobs()
    slot_jobs = {i: [] for i in range(len(SLOTS))}
    for j, job in enumerate(jobs):
        slot_jobs[j % len(SLOTS)].append(job)
    for i, dev in enumerate(SLOTS):
        lines = ["#!/bin/bash", f"cd {PROJECT_ROOT}"]
        for name, cmd in slot_jobs[i]:
            lines.append(f"echo '=== START {name}' $(date)")
            lines.append(f"{cmd.replace('{DEV}', dev)} >> {logdir}/{name}.log 2>&1")
            lines.append(f"echo '=== END {name}' $(date) exit=$?")
        script = JOURNAL_ROOT / f"scripts/_cslot{i}.sh"
        script.write_text("\n".join(lines) + "\n")
        script.chmod(0o755)
        print(f"cslot{i} [{dev}]: {len(slot_jobs[i])} jobs")
    with open(OUT / "manifest.json", "w") as f:
        json.dump({f"slot{i}": [n for n, _ in slot_jobs[i]] for i in slot_jobs},
                  f, indent=1)
    print(f"total: {len(jobs)} jobs. launch: for i in 0 1 2 3; do setsid nohup bash "
          f"{JOURNAL_ROOT}/scripts/_cslot$i.sh > {logdir}/cslot$i.log 2>&1 & done")


if __name__ == "__main__":
    main()
