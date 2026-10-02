"""Round 7 gate: launch the 18 runs (6 arms x seeds 5-7, S1) at once on GPUs 0-3 (round-robin).

  F    fixed lambda 0.4, Lambda 0.5
  C    DriftGate, cell lambda (--no_neighbor_avg), Lambda adapted
  CL   DriftGate, cell lambda, Lambda fixed 0.5                      (--fixed_Lambda 0.5)
  DTV  device lambda from x_TV, Lambda 0.5                           (--device_lambda --device_signal tv)
  DSR  device lambda from x_SR, Lambda 0.5                           (--device_lambda --device_signal sr)
  O    oracle: lambda_k = 0.70 at home, 0.15 away, Lambda 0.5        (--oracle_home_away)
Every arm records x_TV and x_SR for every client (--record_device_signals).
Each run: setsid nohup, CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=<gpu>; log runs/phaseT7_gate/logs/<run>.log.
An existing result JSON is never overwritten (run_r6.py skips it).

  python journal_expansion/scripts/launch_r7_gate.py --dry-run
  python journal_expansion/scripts/launch_r7_gate.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/honeynaps/data/driftgate")  # [SERVER-PATH:REPO_ROOT]
OUT = ROOT / "journal_expansion/runs/phaseT7_gate"
GPUS = [0, 1, 2, 3]  # [SERVER-GPU]
DG = "--mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --spatial_norm --no_neighbor_avg"
ARMS = [
    ("F", "--mode fixed --lambda_val 0.4 --big_lambda_val 0.5"),
    ("C", DG),
    ("CL", f"{DG} --fixed_Lambda 0.5"),
    ("DTV", f"{DG} --device_lambda --device_signal tv --fixed_Lambda 0.5"),
    ("DSR", f"{DG} --device_lambda --device_signal sr --fixed_Lambda 0.5"),
    ("O", "--mode fixed --oracle_home_away --big_lambda_val 0.5 --fixed_Lambda 0.5"),
]
SEEDS = [5, 6, 7]


def jobs():
    out = []
    for s in SEEDS:
        for arm, flags in ARMS:
            rn = f"r7_{arm}_s{s}"
            cmd = (f"python -u {ROOT}/journal_expansion/scripts/run_r6.py --scenario S1 --env S1 --arm {arm} {flags} "
                   f"--record_device_signals --disjoint_pools --seed {s} --model_seed {100 + s} "
                   f"--run_name {rn} --output_dir {OUT} --device cuda:0")
            out.append((rn, cmd))
    assert len(out) == 18
    return out


def main():
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    with open(OUT / "launched_commands.txt", "w") as f:
        for rn, cmd in jobs():
            f.write(cmd + "\n")
    for i, (rn, cmd) in enumerate(jobs()):
        gpu = GPUS[i % len(GPUS)]
        print(f"GPU{gpu} {rn}")
        if "--dry-run" in sys.argv:
            continue
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu), JX_THREADS="2",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        log = open(OUT / "logs" / f"{rn}.log", "a")
        subprocess.Popen(cmd, shell=True, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)


if __name__ == "__main__":
    main()
