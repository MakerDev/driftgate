"""Round 18 (own-class detector): job lists for the dispatcher (runs/phaseT18oc/jobs.json).

  python .../r18oc_jobs.py dev                      # stage 2, S1 development seeds 8-10, three model configurations
  python .../r18oc_jobs.py eval <cfg>=<variant> ... # stage 2 evaluation seeds of a configuration with its chosen variant
  python .../r18oc_jobs.py attempt <n> <cfg>=<variant> ...       # improvement attempt n (7.4) on the S1 development seeds 8-10
  python .../r18oc_jobs.py attempt_eval <n> <cfg>=<variant> ...  # the same attempt on the evaluation seeds (S1, S2 seeds 0-2)
  python .../r18oc_jobs.py stage3 <main cfg>=<variant> <other pretrained cfg>=<variant> <cnn-scratch>=<variant> [--part P]
  python .../r18oc_jobs.py list

stage3 (directive 6.3, 6.4) adds, in this order (P selects parts; default all):
  main     main configuration: S1 and S2 seeds 3-4, S1-fast, partial participation (S4_part05), stepwise change, random
           mobility, iNaturalist with the S1 and with the S2 movement (seeds 0-2); CIFAR-100 seeds 0-2 exist from stage 2
  cnn      cnn-scratch: S1 and S2 seeds 3-4, CIFAR-100 seeds 0-2
  k200     main configuration, 200 devices (S3_K200, seeds 0-2)
  pair     the other pretrained configuration on iNaturalist with the S1 movement (finetuned-vs-frozen ablation; S1, S2 and
           CIFAR-100 exist from stage 2)
  holdout  main configuration, held-out fraction 0.05 and 0.20: S1, S2, CIFAR-100, iNaturalist (S1 movement), seeds 0-2
  k500     main configuration, 500 devices (S3_K500, seeds 0-2)

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
           "c100": dict(scenario="S1", dataset="cifar100"),
           # stage 3
           "s1fast": dict(scenario="S1fast", dataset=None), "part": dict(scenario="S4_part05", dataset=None),
           "stepwise": dict(scenario="stepwise", dataset=None, r18_env=True),
           "randmob": dict(scenario="randmob", dataset=None, r18_env=True),
           "k200": dict(scenario="S3_K200", dataset=None), "k500": dict(scenario="S3_K500", dataset=None),
           "inat1": dict(scenario="S1", dataset="inat"), "inat2": dict(scenario="S2", dataset="inat")}
DEV_ENV = JR / "runs" / "phaseT18oc_env"
SCALE = {"k200": 2.2, "k500": 4.0}   # GPU memory factor: 5-round tests peaked at 8.6 GB (200 devices); 500 devices ran out of memory beside other runs at 5.3 GB


ATTEMPT_FLAGS = {1: ["--other_class"], 2: ["--device_split", "stride16"]}   # 7.4: exit-with-other-class, deeper-split


def name(cfg, st, seed, replay=False, attempt=None, holdout=None):
    ho = f"_h{round(100 * holdout):02d}" if holdout is not None else ""
    return f"r18oc_{TAG[cfg]}{f'a{attempt}' if attempt else ''}_{st}{ho}{'_replay' if replay else ''}_s{seed}"


def base(cfg, st, seed, main, holdout=0.1, extra=()):
    a = ["python", "-u", str(JR / "scripts" / "run_r6.py"), "--scenario", SETTING[st]["scenario"], "--arm", "r18oc",
         "--mode", "fixed", "--lambda_val", "0.4", "--big_lambda_val", "0.5", "--disjoint_pools", "--seed", str(seed),
         "--model_seed", "100", "--output_dir", str(OUT), "--device", "cuda:0", "--record_eval_probs",
         "--record_train_label_hist", "--model_config", cfg, "--holdout_frac", str(holdout), "--detector"]
    if SETTING[st]["dataset"]:
        a += ["--dataset", SETTING[st]["dataset"]]
    if main:
        a += ["--detector_main", f"'{main}'"]
    if (st == "s1" and seed >= 8) or (st == "s2" and seed >= 3) or SETTING[st].get("r18_env"):
        a += ["--env_dir", str(DEV_ENV)]
    return a + list(extra)


def pair(cfg, st, seed, main=None, attempt=None, holdout=None):
    n, nr = name(cfg, st, seed, attempt=attempt, holdout=holdout), name(cfg, st, seed, True, attempt=attempt, holdout=holdout)
    ck = OUT / f"{n}_ckpt_r150.pt"
    ex = ATTEMPT_FLAGS[attempt] if attempt else []
    ho = 0.1 if holdout is None else holdout
    c = COST[cfg] * SCALE.get(st, 1.0)
    day = dict(name=n, kind="train", cfg=cfg, setting=st, seed=seed, main=main, cost=c, requires=None, after=None,
               attempt=attempt, **({} if holdout is None else dict(holdout=holdout)),
               cmd=" ".join(base(cfg, st, seed, main, ho, extra=ex) + ["--run_name", n, "--rounds", "300", "--days", "2", "--checkpoint_at", "150"]))
    rep = dict(name=nr, kind="replay", cfg=cfg, setting=st, seed=seed, main=main, cost=c - 1.0, requires=str(ck),
               after=None, attempt=attempt, **({} if holdout is None else dict(holdout=holdout)),
               cmd=" ".join(base(cfg, st, seed, main, ho, extra=ex) + ["--run_name", nr, "--rounds", "150", "--replay_from", str(ck)]))
    return [day, rep]


def stage3(main_arg, other_arg, cnn_arg, parts):
    (cm, vm), (co, vo), (cc, vc) = (a.split("=") for a in (main_arg, other_arg, cnn_arg))
    assert cm != "cnn-scratch" and co != "cnn-scratch" and cm != co and cc == "cnn-scratch"
    new = []
    for part in parts:
        if part == "main":
            for st, ss in (("s1", (3, 4)), ("s2", (3, 4)), ("s1fast", (0, 1, 2)), ("part", (0, 1, 2)), ("stepwise", (0, 1, 2)),
                           ("randmob", (0, 1, 2)), ("inat1", (0, 1, 2)), ("inat2", (0, 1, 2))):
                new += [j for s in ss for j in pair(cm, st, s, vm)]
        elif part == "cnn":
            for st, ss in (("s1", (3, 4)), ("s2", (3, 4)), ("c100", (0, 1, 2))):
                new += [j for s in ss for j in pair(cc, st, s, vc)]
        elif part in ("k200", "k500"):
            new += [j for s in (0, 1, 2) for j in pair(cm, part, s, vm)]
        elif part == "pair":
            new += [j for s in (0, 1, 2) for j in pair(co, "inat1", s, vo)]
        elif part == "holdout":
            for ho in (0.05, 0.2):
                for st in ("s1", "s2", "c100", "inat1"):
                    new += [j for s in (0, 1, 2) for j in pair(cm, st, s, vm, holdout=ho)]
        else:
            raise SystemExit(f"unknown part {part}")
    add(new)


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
    elif sys.argv[1] in ("attempt", "attempt_eval"):
        n = int(sys.argv[2])
        assert n in ATTEMPT_FLAGS, "attempt 3 is defined in decisions.md before it is added here"
        new = []
        for arg in sys.argv[3:]:
            cfg, var = arg.split("=")
            assert cfg != "cnn-scratch" or n == 1, "deeper-split is a MobileNetV2 option"
            seeds = (("s1", (8, 9, 10)),) if sys.argv[1] == "attempt" else (("s1", (0, 1, 2)), ("s2", (0, 1, 2)))
            for st, ss in seeds:
                for s in ss:
                    new += pair(cfg, st, s, var, attempt=n)
        add(new)
    elif sys.argv[1] == "stage3":
        args = sys.argv[2:]
        parts = ["main", "cnn", "k200", "pair", "holdout", "k500"]
        if "--part" in args:
            i = args.index("--part")
            parts, args = args[i + 1].split(","), args[:i] + args[i + 2:]
        stage3(*args, parts)
    elif sys.argv[1] == "list":
        for j in json.load(open(JOBS)):
            print(j["name"], "|", j["cmd"])


if __name__ == "__main__":
    main()
