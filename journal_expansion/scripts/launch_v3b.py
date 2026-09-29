"""v3b downstream runs: selfcal-delta with burn_in=10 (the warm-up-transient fix
found in Gate-B signal analysis). 6 runs: {A, abrupt} x seeds {0,1,2}.
Run AFTER the v1 downstream pilot completes (GPU capacity)."""
import json
from pathlib import Path

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = JOURNAL_ROOT.parent
OUT = JOURNAL_ROOT / "runs/downstream_pilot"
SEEDS = [(0, 100), (1, 101), (2, 102)]
SLOTS = ["cuda:0", "cuda:0", "cuda:1", "cuda:1"]  # [SERVER-GPU]


def main():
    logdir = OUT / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for sched in ("A", "abrupt"):
        for ps, ms in SEEDS:
            name = f"sc_delta_b10_{sched}_s{ps}"
            if (OUT / f"{name}.json").exists():
                continue
            jobs.append((name,
                         f"python -u {JOURNAL_ROOT}/scripts/run_v2.py --mode selfcal "
                         f"--signal delta_hard --burn_in 10 --schedule {sched} "
                         f"--seed {ps} --model_seed {ms} --rounds 100 --probe_n 64 "
                         f"--run_name {name} --output_dir {OUT} --device {{DEV}}"))
    slot_jobs = {i: [] for i in range(len(SLOTS))}
    for j, job in enumerate(jobs):
        slot_jobs[j % len(SLOTS)].append(job)
    for i, dev in enumerate(SLOTS):
        if not slot_jobs[i]:
            continue
        lines = ["#!/bin/bash", f"cd {PROJECT_ROOT}"]
        for name, cmd in slot_jobs[i]:
            lines.append(f"echo '=== START {name}' $(date)")
            lines.append(f"{cmd.replace('{DEV}', dev)} >> {logdir}/{name}.log 2>&1")
            lines.append(f"echo '=== END {name}' $(date) exit=$?")
        script = JOURNAL_ROOT / f"scripts/_bslot{i}.sh"
        script.write_text("\n".join(lines) + "\n")
        script.chmod(0o755)
        print(f"bslot{i} [{dev}]: {[n for n, _ in slot_jobs[i]]}")
    print("launch: for i in 0 1 2 3; do setsid nohup bash "
          f"{JOURNAL_ROOT}/scripts/_bslot$i.sh > {logdir}/bslot$i.log 2>&1 & done")


if __name__ == "__main__":
    main()
