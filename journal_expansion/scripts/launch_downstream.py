"""Gate-B downstream pilot: controller-in-the-loop runs.

Pre-registered matrix (fixed BEFORE passive-run analysis):
  controllers x schedules {A (legacy's calibration home), abrupt (unseen)} x seeds {0,1,2}
    - selfcal + delta_hard   (main candidate)
    - selfcal + ent_client   (normalized-entropy baseline, B8)
    - legacy  + delta_hard   (hand-calibrated mu=0.31 — leaky reference)
  fixed-lambda baseline = the passive runs themselves (same seeds/schedules, lam=0.4).
100 rounds, probe_n=64. 18 runs. Skips existing result JSONs.
"""
import json
import sys
from pathlib import Path

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = JOURNAL_ROOT.parent

SEEDS = [(0, 100), (1, 101), (2, 102)]
SCHEDS = ["A", "abrupt"]
CONTROLLERS = [
    ("selfcal", "delta_hard", "sc_delta"),
    ("selfcal", "ent_client", "sc_ent"),
    ("legacy", "delta_hard", "legacy_delta"),
]
ROUNDS = 100
OUT = JOURNAL_ROOT / "runs/downstream_pilot"
SLOTS = ["cuda:0", "cuda:0", "cuda:1", "cuda:1"]  # [SERVER-GPU]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    logdir = OUT / "logs"
    logdir.mkdir(exist_ok=True)
    jobs = []
    for sched in SCHEDS:
        for mode, sig, tag in CONTROLLERS:
            for ps, ms in SEEDS:
                name = f"{tag}_{sched}_s{ps}"
                if (OUT / f"{name}.json").exists():
                    continue
                jobs.append((name,
                             f"python -u {JOURNAL_ROOT}/scripts/run_v2.py --mode {mode} "
                             f"--signal {sig} --schedule {sched} --seed {ps} "
                             f"--model_seed {ms} --rounds {ROUNDS} --probe_n 64 "
                             f"--run_name {name} --output_dir {OUT} --device {{DEV}}"))

    slot_jobs = {i: [] for i in range(len(SLOTS))}
    for j, job in enumerate(jobs):
        slot_jobs[j % len(SLOTS)].append(job)

    for i, dev in enumerate(SLOTS):
        lines = ["#!/bin/bash", f"cd {PROJECT_ROOT}"]
        for name, cmd in slot_jobs[i]:
            lines.append(f"echo '=== START {name}' $(date)")
            lines.append(f"{cmd.replace('{DEV}', dev)} >> {logdir}/{name}.log 2>&1")
            lines.append(f"echo '=== END {name}' $(date) exit=$?")
        script = JOURNAL_ROOT / f"scripts/_dslot{i}.sh"
        script.write_text("\n".join(lines) + "\n")
        script.chmod(0o755)
        print(f"dslot{i} [{dev}]: {len(slot_jobs[i])} jobs -> {script}")

    with open(OUT / "manifest.json", "w") as f:
        json.dump({f"slot{i}": [n for n, _ in slot_jobs[i]] for i in slot_jobs}, f, indent=1)
    print(f"total jobs: {len(jobs)}")
    print("launch: for i in 0 1 2 3; do setsid nohup bash "
          f"{JOURNAL_ROOT}/scripts/_dslot$i.sh > {logdir}/dslot$i.log 2>&1 & done")


if __name__ == "__main__":
    main()
