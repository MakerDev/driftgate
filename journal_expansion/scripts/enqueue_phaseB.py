"""Phase-B job generator: adaptation-value decomposition + full fixed grid.

Blocks (priority order; pass block names as argv):
  g1core : global/per-cell mean-matched fixed + time-shuffled replay
           (3 envs x 3 seeds x 3 arms = 27 runs) — the Gate-1 verdict set
  g1ext  : shift10, shift20, reverse, cluster_shuffle, lowpass, identity replays
  grid   : full fixed lambda grid {0.0..0.8} on all 8 environments (existing
           0.2/0.4 points reused; Tiny-IN partial grid {0.1,0.3,0.6})
  Lpilot : Lambda pilot {0.4,0.6,0.7} x lambda {0.2,0.4} on Schedule A

Source DV runs (frozen, provenance-tracked) provide the trajectories/means:
  A       runs/gated/d2_horizon/d2_dual_A_s{s}.json
  gsig    runs/gated/d1_unseen/d1_dual_gradual_sigmoid_s{s}.json
  spatial runs/gated/d3_spatial/d3_dual_sp_s{s}.json
"""
import sys
from pathlib import Path

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JOURNAL_ROOT))
from src.replay import trajectory_means

QDIR = JOURNAL_ROOT / "runs/queue"
RV2 = f"python -u {JOURNAL_ROOT}/scripts/run_v2.py"
OUT = JOURNAL_ROOT / "runs/phaseB_decomp"
GRID_OUT = JOURNAL_ROOT / "runs/phaseB_grid"
S3 = [(0, 100), (1, 101), (2, 102)]

ENVS = {  # env -> (dv_source_pattern, schedule flags, rounds)
    "A": ("runs/gated/d2_horizon/d2_dual_A_s{s}.json", "--schedule A", 150),
    "gsig": ("runs/gated/d1_unseen/d1_dual_gradual_sigmoid_s{s}.json",
             "--schedule gradual_sigmoid", 150),
    "sp": ("runs/gated/d3_spatial/d3_dual_sp_s{s}.json",
           "--schedule static --spatial equal_spread", 150),
}

GRID_ENVS = {
    "A": ("--schedule A", 150, "", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    "gsig": ("--schedule gradual_sigmoid", 150, "", [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8]),
    "asym": ("--schedule asym_return", 150, "", [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8]),
    "pwr": ("--schedule piecewise_random", 150, "", [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8]),
    "sp": ("--schedule static --spatial equal_spread", 150, "",
           [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    "c100gsig": ("--schedule gradual_sigmoid", 150, "--dataset cifar100",
                 [0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8]),
    "c100sp": ("--schedule static --spatial equal_spread", 150, "--dataset cifar100",
               [0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8]),
    "tin": ("--schedule gradual_sigmoid", 100, "--dataset tinyimagenet",
            [0.1, 0.3, 0.6]),
}


def jobs_g1core():
    for env, (src, sched, rounds) in ENVS.items():
        for ps, ms in S3:
            path = JOURNAL_ROOT / src.format(s=ps)
            m = trajectory_means(path)
            base = (f"{RV2} {sched} --rounds {rounds} --probe_n 64 --seed {ps} "
                    f"--model_seed {ms} --output_dir {OUT} --device {{DEV}} ")
            yield OUT, f"gmm_{env}_s{ps}", base + (
                f"--mode fixed --lambda_val {m['lam_global']:.4f} "
                f"--big_lambda_val {m['Lam_global']:.4f} --run_name gmm_{env}_s{ps}")
            lpc = ",".join(f"{m['lam_per_cell'][z]:.4f}" for z in sorted(m['lam_per_cell']))
            Lpc = ",".join(f"{m['Lam_per_cell'][z]:.4f}" for z in sorted(m['Lam_per_cell']))
            yield OUT, f"pcm_{env}_s{ps}", base + (
                f"--mode fixed --lambda_per_cell {lpc} --Lambda_per_cell {Lpc} "
                f"--run_name pcm_{env}_s{ps}")
            yield OUT, f"shuf_{env}_s{ps}", base + (
                f"--mode replay --replay_from {path} --replay_transform shuffle "
                f"--run_name shuf_{env}_s{ps}")


def jobs_g1ext():
    for env, (src, sched, rounds) in ENVS.items():
        for ps, ms in S3:
            path = JOURNAL_ROOT / src.format(s=ps)
            base = (f"{RV2} {sched} --rounds {rounds} --probe_n 64 --seed {ps} "
                    f"--model_seed {ms} --output_dir {OUT} --device {{DEV}} ")
            for tr in ("identity", "shift10", "shift20", "reverse",
                       "cluster_shuffle", "lowpass"):
                yield OUT, f"{tr}_{env}_s{ps}", base + (
                    f"--mode replay --replay_from {path} --replay_transform {tr} "
                    f"--run_name {tr}_{env}_s{ps}")


def jobs_grid():
    for env, (sched, rounds, ds, lams) in GRID_ENVS.items():
        for lam in lams:
            for ps, ms in S3:
                name = f"fx{int(round(lam*100)):02d}_{env}_s{ps}"
                yield GRID_OUT, name, (
                    f"{RV2} {sched} {ds} --rounds {rounds} --probe_n 64 "
                    f"--mode fixed --lambda_val {lam} --seed {ps} --model_seed {ms} "
                    f"--run_name {name} --output_dir {GRID_OUT} --device {{DEV}}")


def jobs_Lpilot():
    for Lam in (0.4, 0.6, 0.7):
        for lam in (0.2, 0.4):
            for ps, ms in S3:
                name = f"fx{int(lam*100):02d}_L{int(Lam*100):02d}_A_s{ps}"
                yield GRID_OUT, name, (
                    f"{RV2} --schedule A --rounds 150 --probe_n 64 --mode fixed "
                    f"--lambda_val {lam} --big_lambda_val {Lam} --seed {ps} "
                    f"--model_seed {ms} --run_name {name} --output_dir {GRID_OUT} "
                    f"--device {{DEV}}")


def main():
    blocks = {"g1core": jobs_g1core, "g1ext": jobs_g1ext, "grid": jobs_grid,
              "Lpilot": jobs_Lpilot}
    wanted = sys.argv[1:] or ["g1core"]
    QDIR.mkdir(parents=True, exist_ok=True)
    qfile = QDIR / "queue.txt"
    existing = set(qfile.read_text().strip().split("\n")) if qfile.exists() else set()
    n = 0
    with open(qfile, "a") as f:
        for b in wanted:
            for out, name, cmd in blocks[b]():
                out.mkdir(parents=True, exist_ok=True)
                if (out / f"{name}.json").exists() or cmd in existing:
                    continue
                f.write(cmd + "\n")
                n += 1
    print(f"enqueued {n} jobs ({wanted})")


if __name__ == "__main__":
    main()
