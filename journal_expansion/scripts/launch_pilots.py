"""Gate-B pilot launcher: builds the job manifest, writes per-slot worker
scripts (sequential within a slot), and prints the setsid launch commands.

Pilot matrix (pre-registered):
  passive recording runs: 4 schedules x 3 seeds, fixed lambda=0.4 backbone,
  100 rounds, probe_n=64 -> 12 runs.
Slots: static round-robin across GPU slots. Jobs already having a result
JSON are skipped (new run IDs are only for genuinely re-executed runs).
"""
import json
import sys
from pathlib import Path

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = JOURNAL_ROOT.parent
sys.path.insert(0, str(JOURNAL_ROOT))

from src.schedules import PILOT_SCHEDULES

SEEDS = [(0, 100), (1, 101), (2, 102)]
ROUNDS = 100
OUT = JOURNAL_ROOT / "runs/signal_benchmark"
SLOTS = ["cuda:0", "cuda:1", "cuda:1"]  # slot0 shares GPU0 with repro; GPU1 x2  # [SERVER-GPU]


def build_jobs():
    jobs = []
    for sched in PILOT_SCHEDULES:
        for ps, ms in SEEDS:
            name = f"rec_{sched}_s{ps}"
            if (OUT / f"{name}.json").exists():
                continue
            jobs.append(
                f"python -u {JOURNAL_ROOT}/scripts/run_v2.py --mode passive "
                f"--schedule {sched} --seed {ps} --model_seed {ms} "
                f"--rounds {ROUNDS} --probe_n 64 --run_name {name} "
                f"--output_dir {OUT} --device {{DEV}}")
    return jobs


def main():
    jobs = build_jobs()
    OUT.mkdir(parents=True, exist_ok=True)
    logdir = JOURNAL_ROOT / "runs/signal_benchmark/logs"
    logdir.mkdir(parents=True, exist_ok=True)

    slot_jobs = {i: [] for i in range(len(SLOTS))}
    for j, job in enumerate(jobs):
        slot_jobs[j % len(SLOTS)].append(job)

    manifest = {"jobs": jobs, "slots": {}}
    for i, dev in enumerate(SLOTS):
        lines = ["#!/bin/bash", f"cd {PROJECT_ROOT}"]
        for job in slot_jobs[i]:
            cmd = job.replace("{DEV}", dev)
            tag = cmd.split("--run_name ")[1].split()[0]
            lines.append(f"echo '=== START {tag}' $(date)")
            lines.append(f"{cmd} >> {logdir}/{tag}.log 2>&1")
            lines.append(f"echo '=== END {tag}' $(date) exit=$?")
        script = JOURNAL_ROOT / f"scripts/_slot{i}.sh"
        script.write_text("\n".join(lines) + "\n")
        script.chmod(0o755)
        manifest["slots"][f"slot{i}"] = {"device": dev, "jobs": slot_jobs[i],
                                         "script": str(script)}
        print(f"slot{i} [{dev}]: {len(slot_jobs[i])} jobs -> {script}")

    with open(JOURNAL_ROOT / "runs/signal_benchmark/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nTotal jobs: {len(jobs)}")
    print("Launch each slot with:")
    for i in range(len(SLOTS)):
        print(f"  setsid nohup bash {JOURNAL_ROOT}/scripts/_slot{i}.sh "
              f"> {logdir}/slot{i}.log 2>&1 &")


if __name__ == "__main__":
    main()
