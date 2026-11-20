"""Round 18 (own-class detector): job lists for the dispatcher (runs/phaseT18oc/jobs.json).

  python .../r18oc_jobs.py dev                      # stage 2, S1 development seeds 8-10, three model configurations
  python .../r18oc_jobs.py eval <cfg>=<variant> ... # stage 2 evaluation seeds of a configuration with its chosen variant
  python .../r18oc_jobs.py list

Every training run: 300 rounds (the same day twice, --days 2), fixed lambda 0.4 / Lambda 0.5 (v32 training), 10% held-out
samples, detectors, per-request records and the round-150 checkpoint; its frozen replay (150 rounds, --replay_from the
round-150 checkpoint) starts when that checkpoint exists. Jobs are only appended; an existing job name is never changed.
"""
import json
import sys
from pathlib import Path

JR = Path(__file__).resolve().parents[3]
ROOT = JR.parent
OUT = JR / "runs" / "phaseT18oc"
JOBS = OUT / "jobs.json"
TAG = {"cnn-scratch": "cnn", "pretrained-finetuned-block": "ft", "pretrained-frozen-block": "fz"}
COST = {"cnn-scratch": 3.0, "pretrained-finetuned-block": 4.0, "pretrained-frozen-block": 3.5}   # GB on the GPU
SETTING = {"s1": dict(scenario="S1", dataset=None), "s2": dict(scenario="S2", dataset=None),
           "c100": dict(scenario="S1", dataset="cifar100")}
DEV_ENV = JR / "runs" / "phaseT18oc_env"


def name(cfg, st, seed, replay=False):
    return f"r18oc_{TAG[cfg]}_{st}{'_replay' if replay else ''}_s{seed}"


def base(cfg, st, seed, main, holdout=0.1):
    a = ["python", "-u", str(JR / "scripts" / "run_r6.py"), "--scenario", SETTING[st]["scenario"], "--arm", "r18oc",
         "--mode", "fixed", "--lambda_val", "0.4", "--big_lambda_val", "0.5", "--disjoint_pools", "--seed", str(seed),
         "--model_seed", "100", "--output_dir", str(OUT), "--device", "cuda:0", "--record_eval_probs",
         "--record_train_label_hist", "--model_config", cfg, "--holdout_frac", str(holdout), "--detector"]
    if SETTING[st]["dataset"]:
        a += ["--dataset", SETTING[st]["dataset"]]
    if main:
        a += ["--detector_main", f"'{main}'"]
    if st == "s1" and seed >= 8:
        a += ["--env_dir", str(DEV_ENV)]
    return a


def pair(cfg, st, seed, main=None):
    n, nr = name(cfg, st, seed), name(cfg, st, seed, True)
    ck = OUT / f"{n}_ckpt_r150.pt"
    day = dict(name=n, kind="train", cfg=cfg, setting=st, seed=seed, main=main, cost=COST[cfg], requires=None, after=None,
               cmd=" ".join(base(cfg, st, seed, main) + ["--run_name", n, "--rounds", "300", "--days", "2", "--checkpoint_at", "150"]))
    rep = dict(name=nr, kind="replay", cfg=cfg, setting=st, seed=seed, main=main, cost=COST[cfg] - 1.0, requires=str(ck),
               after=None, cmd=" ".join(base(cfg, st, seed, main) + ["--run_name", nr, "--rounds", "150", "--replay_from", str(ck)]))
    return [day, rep]


def add(new):
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.load(open(JOBS)) if JOBS.exists() else []
    have = {j["name"] for j in old}
    add_ = [j for j in new if j["name"] not in have]
    json.dump(old + add_, open(JOBS, "w"), indent=1)
    print(f"{len(add_)} jobs added, {len(old) + len(add_)} in total")


def main():
    if sys.argv[1] == "dev":
        add([j for cfg in TAG for s in (8, 9, 10) for j in pair(cfg, "s1", s)])
    elif sys.argv[1] == "eval":
        new = []
        for arg in sys.argv[2:]:
            cfg, var = arg.split("=")
            for st, seeds in (("s1", (0, 1, 2)), ("s2", (0, 1, 2))) + ((("c100", (0, 1, 2)),) if cfg != "cnn-scratch" else ()):
                for s in seeds:
                    new += pair(cfg, st, s, var)
        add(new)
    elif sys.argv[1] == "list":
        for j in json.load(open(JOBS)):
            print(j["name"], "|", j["cmd"])


if __name__ == "__main__":
    main()
