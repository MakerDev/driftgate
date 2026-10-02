"""Round 10: launch the 3 runs (T40 = fixed lambda 0.4, Lambda 0.5, S1, seeds 5-7) on GPUs 0-2.

Same settings as runs/phaseT8_check/r8v2_T40_s{5,6,7} (their launched command), except: no lambda_inf block evaluation
(--eval_infer_lambdas and --record_eval_requests removed) and the Round 10 records on
(--record_eval_probs --record_train_label_hist). Output runs/phaseT10_prior/r10_T40_s<seed>.

  python journal_expansion/scripts/launch_r10_prior.py --dry-run
  python journal_expansion/scripts/launch_r10_prior.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/honeynaps/data/driftgate")  # [SERVER-PATH:REPO_ROOT]
OUT = ROOT / "journal_expansion/runs/phaseT10_prior"
GPUS = [0, 1, 2]  # [SERVER-GPU]
SEEDS = [5, 6, 7]


def jobs():
    src = (ROOT / "journal_expansion/runs/phaseT8_check/launched_commands_v2.txt").read_text().splitlines()
    out = []
    for s in SEEDS:
        base = [l for l in src if f"--run_name r8v2_T40_s{s} " in l + " "]
        assert len(base) == 1
        toks = base[0].split()
        i = toks.index("--eval_infer_lambdas")
        del toks[i:i + 2]
        toks.remove("--record_eval_requests")
        toks[toks.index("--run_name") + 1] = f"r10_T40_s{s}"
        toks[toks.index("--output_dir") + 1] = str(OUT)
        toks[toks.index("--device") - 0:toks.index("--device")] = ["--record_eval_probs", "--record_train_label_hist"]
        out.append((f"r10_T40_s{s}", " ".join(toks)))
    return out


def main():
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    with open(OUT / "launched_commands.txt", "w") as f:
        for rn, cmd in jobs():
            f.write(cmd + "\n")
    for i, (rn, cmd) in enumerate(jobs()):
        gpu = GPUS[i % len(GPUS)]
        print(f"GPU{gpu} {rn}: {cmd}")
        if "--dry-run" in sys.argv:
            continue
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu), JX_THREADS="2",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        log = open(OUT / "logs" / f"{rn}.log", "a")
        subprocess.Popen(cmd, shell=True, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)


if __name__ == "__main__":
    main()
