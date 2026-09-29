"""Gate-D full-evaluation matrix (pre-registered; queued in priority order).

Arms (pilot-selected):
  main    = selfcal v3c + ST  (delta_hard, burn_in 10, z_guard 0.5, spatial_norm)
  entswap = SAME controller, signal ent_client       (C1 signal swap at horizon)
  legacy  = hand-calibrated mu=0.31 (leaky reference, transfer test on unseen data)
  fixed02 / fixed04 = fixed grid (hindsight reference, marked non-deployable)
  oracle  = greedy labeled causal oracle (B11 upper bound)
  st_lm05 = main with lam_max=0.5 (bounds ablation, D3)

Blocks (priority order):
  D2: CIFAR-10 Schedule A @150R — entropy-death confirmation (main/entswap x5 + passive x3)
  D1: unseen schedules {gradual_sigmoid, asym_return, piecewise_random} @150R x 5 seeds
  D3: spatial equal_spread @150R + lambda-bounds ablation
  D4: CIFAR-100 (gradual_sigmoid + spatial) @150R x 3 seeds
  D5: Tiny-ImageNet gradual_sigmoid @100R x 3 seeds (scale point; added after smoke)
"""
import sys
from pathlib import Path

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
QDIR = JOURNAL_ROOT / "runs/queue"
RV2 = f"python -u {JOURNAL_ROOT}/scripts/run_v2.py"
S5 = [(0, 100), (1, 101), (2, 102), (3, 103), (4, 104)]
S3 = S5[:3]

MAIN = "--mode selfcal --signal delta_hard --burn_in 10 --z_guard 0.5 --spatial_norm"
ENT = "--mode selfcal --signal ent_client --burn_in 10 --z_guard 0.5 --spatial_norm"
LEG = "--mode legacy --signal delta_hard"
ORC = "--mode oracle_greedy"


def block_d2():
    out = JOURNAL_ROOT / "runs/gated/d2_horizon"
    for arm, flags, seeds in [("main", MAIN, S5), ("entswap", ENT, S5),
                              ("rec", "--mode passive", S3)]:
        for ps, ms in seeds:
            name = f"d2_{arm}_A_s{ps}"
            yield out, name, (f"{RV2} {flags} --schedule A --rounds 150 --probe_n 64 "
                              f"--seed {ps} --model_seed {ms} --run_name {name} "
                              f"--output_dir {out} --device {{DEV}}")


def block_d1():
    out = JOURNAL_ROOT / "runs/gated/d1_unseen"
    arms = [("main", MAIN, S5), ("legacy", LEG, S5),
            ("fixed02", "--mode fixed --lambda_val 0.2", S5),
            ("fixed04", "--mode fixed --lambda_val 0.4", S5),
            ("entswap", ENT, S3), ("oracle", ORC, S3)]
    for sched in ("gradual_sigmoid", "asym_return", "piecewise_random"):
        for arm, flags, seeds in arms:
            for ps, ms in seeds:
                name = f"d1_{arm}_{sched}_s{ps}"
                yield out, name, (f"{RV2} {flags} --schedule {sched} --rounds 150 "
                                  f"--probe_n 64 --seed {ps} --model_seed {ms} "
                                  f"--run_name {name} --output_dir {out} --device {{DEV}}")


def block_d3():
    out = JOURNAL_ROOT / "runs/gated/d3_spatial"
    arms = [("main", MAIN, S5), ("st_lm05", MAIN + " --lam_max 0.5", S5),
            ("fixed02", "--mode fixed --lambda_val 0.2", S5)]
    for arm, flags, seeds in arms:
        for ps, ms in seeds:
            name = f"d3_{arm}_sp_s{ps}"
            yield out, name, (f"{RV2} {flags} --schedule static --spatial equal_spread "
                              f"--rounds 150 --probe_n 64 --seed {ps} --model_seed {ms} "
                              f"--run_name {name} --output_dir {out} --device {{DEV}}")


def block_d4():
    out = JOURNAL_ROOT / "runs/gated/d4_cifar100"
    arms = [("main", MAIN), ("legacy", LEG),
            ("fixed04", "--mode fixed --lambda_val 0.4")]
    for arm, flags in arms:
        for ps, ms in S3:
            name = f"d4_{arm}_gsig_s{ps}"
            yield out, name, (f"{RV2} {flags} --dataset cifar100 --schedule gradual_sigmoid "
                              f"--rounds 150 --probe_n 64 --seed {ps} --model_seed {ms} "
                              f"--run_name {name} --output_dir {out} --device {{DEV}}")
    for arm, flags in [("main", MAIN), ("fixed04", "--mode fixed --lambda_val 0.4")]:
        for ps, ms in S3:
            name = f"d4_{arm}_sp_s{ps}"
            yield out, name, (f"{RV2} {flags} --dataset cifar100 --schedule static "
                              f"--spatial equal_spread --rounds 150 --probe_n 64 "
                              f"--seed {ps} --model_seed {ms} --run_name {name} "
                              f"--output_dir {out} --device {{DEV}}")


def block_d5():
    out = JOURNAL_ROOT / "runs/gated/d5_tinyimagenet"
    for arm, flags in [("main", MAIN), ("fixed04", "--mode fixed --lambda_val 0.4")]:
        for ps, ms in S3:
            name = f"d5_{arm}_gsig_s{ps}"
            yield out, name, (f"{RV2} {flags} --dataset tinyimagenet "
                              f"--schedule gradual_sigmoid --rounds 100 --probe_n 64 "
                              f"--seed {ps} --model_seed {ms} --run_name {name} "
                              f"--output_dir {out} --device {{DEV}}")


def main():
    blocks = {"d2": block_d2, "d1": block_d1, "d3": block_d3, "d4": block_d4,
              "d5": block_d5}
    wanted = sys.argv[1:] or ["d2", "d1", "d3", "d4"]
    QDIR.mkdir(parents=True, exist_ok=True)
    qfile = QDIR / "queue.txt"
    existing = set(qfile.read_text().strip().split("\n")) if qfile.exists() else set()
    added = 0
    with open(qfile, "a") as f:
        for b in wanted:
            for out, name, cmd in blocks[b]():
                out.mkdir(parents=True, exist_ok=True)
                if (out / f"{name}.json").exists() or cmd in existing:
                    continue
                f.write(cmd + "\n")
                added += 1
    print(f"enqueued {added} Gate-D jobs ({wanted}) -> {qfile}")


if __name__ == "__main__":
    main()
