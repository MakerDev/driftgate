"""Round 19 section 5: one day of the multi-day frozen replay (protocol: R19_multiday_protocol.md).

Runs src/runner_r6.run_r6 unchanged in replay mode (--replay_from the Round 15 end-of-day checkpoint of the seed, the same
flags as the Round 15 replays) on runs/phaseT19_multiday/env/{S1,S2}_s{seed}_day{d}.npz. Model parameters stay frozen; the
controller and entropy history are carried across days in the analysis (r19_multiday.py), not in the runner.
Usage: python r19_multiday_replay.py S1 0 1 cuda:0   -> runs/phaseT19_multiday/r19_s1_s0_day1{.json,_trace.npz,_evalprobs.npz}
"""
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
ROOT = JR.parent
for p in (str(ROOT), str(JR)):
    if p not in sys.path:
        sys.path.insert(0, p)
import src.runner_r6 as R6  # noqa: E402


def main():
    sc, seed, day, device = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    out = JR / "runs" / "phaseT19_multiday"
    name = f"r19_{sc.lower()}_s{seed}_day{day}"
    if (out / f"{name}.json").exists():
        print(f"{name} exists; not re-running")
        return
    cfg = yaml.safe_load(open(ROOT / "configs/base_v3.yaml"))
    cfg.update(device=device, partition_seed=seed, model_seed=100 + seed, global_rounds=150)
    ckpt = JR / "runs" / "phaseT15_replay" / f"r15_{sc.lower()}_fixed040_s{seed}_ckpt.pt"
    R6.run_r6(cfg, out / "env" / f"{sc}_s{seed}_day{day}.npz", "fixed", signal="tv_dist", lambda_val=0.4, big_lambda_val=0.5,
              apfl_eta=None, signal_delay=0, probe_n=64, controller_kwargs={}, run_name=name, output_dir=str(out), eval_every=5,
              save_models=False, scenario=sc, arm="fixed040", record_device_signals=True, record_eval_probs=True,
              record_train_label_hist=True, record_eval_logprobs=True, replay_from=str(ckpt))


if __name__ == "__main__":
    main()
