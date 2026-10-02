"""Round 8 check: launch the 6 runs (2 arms x seeds 5-7, S1) at once on GPUs 0-3 (round-robin).

  T40  fixed lambda_t 0.40, Lambda 0.5 (the same training as Round 7 F)
  T15  fixed lambda_t 0.15, Lambda 0.5
Both arms: --record_device_signals, --eval_infer_lambdas 0.15,0.3,0.4,0.55,0.7, --eval_mainaware_route.
Each run: own session (setsid), CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=<gpu>; log runs/phaseT8_check/logs/<run>.log.
An existing result JSON is never overwritten (run_r6.py skips it).

  python journal_expansion/scripts/launch_r8_check.py --dry-run
  python journal_expansion/scripts/launch_r8_check.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/honeynaps/data/driftgate")  # [SERVER-PATH:REPO_ROOT]
OUT = ROOT / "journal_expansion/runs/phaseT8_check"
GPUS = [0, 1, 2, 3]  # [SERVER-GPU]
R8 = "--record_device_signals --eval_infer_lambdas 0.15,0.3,0.4,0.55,0.7 --eval_mainaware_route"
ARMS = [
    ("T40", "--mode fixed --lambda_val 0.4 --big_lambda_val 0.5"),
    ("T15", "--mode fixed --lambda_val 0.15 --big_lambda_val 0.5"),
]
SEEDS = [5, 6, 7]


def jobs():
    out = []
    for s in SEEDS:
        for arm, flags in ARMS:
            rn = f"r8_{arm}_s{s}"
            cmd = (f"python -u {ROOT}/journal_expansion/scripts/run_r6.py --scenario S1 --env S1 --arm {arm} {flags} "
                   f"{R8} --disjoint_pools --seed {s} --model_seed {100 + s} "
                   f"--run_name {rn} --output_dir {OUT} --device cuda:0")
            out.append((rn, cmd))
    assert len(out) == 6
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
