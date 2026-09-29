"""Gate-2 / Phase-C: dual-exit signal family inside the frozen DV controller.

Arms (identical DV pipeline; only --signal changes; abs view uses the signal's
MATHEMATICAL range — no dataset constants):
  tv_dist        on 5 envs (A, abrupt, spatial, C100-gsig, TinyIN-gsig)
  dv delta_hard  on abrupt (the one env with no existing DV-delta run)
  js_div, kl_sym, cos_logit_dist on A + abrupt only (pilot; expected weaker)
3 seeds each. Existing DV-delta runs are reused for A/spatial/C100/TinyIN.
"""
import sys
from pathlib import Path

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
QDIR = JOURNAL_ROOT / "runs/queue"
RV2 = f"python -u {JOURNAL_ROOT}/scripts/run_v2.py"
OUT = JOURNAL_ROOT / "runs/phaseC_signals"
S3 = [(0, 100), (1, 101), (2, 102)]
DV = "--mode selfcal --burn_in 10 --z_guard 0.5 --spatial_norm --abs_cap"

ENVS = {
    "A": ("--schedule A", 150, ""),
    "abrupt": ("--schedule abrupt", 150, ""),
    "sp": ("--schedule static --spatial equal_spread", 150, ""),
    "c100gsig": ("--schedule gradual_sigmoid", 150, "--dataset cifar100"),
    "tin": ("--schedule gradual_sigmoid", 100, "--dataset tinyimagenet"),
}


def jobs():
    for env, (sched, rounds, ds) in ENVS.items():
        for sig, tag in [("tv_dist", "tv")]:
            for ps, ms in S3:
                name = f"dvsig_{tag}_{env}_s{ps}"
                yield name, (f"{RV2} {DV} --signal {sig} {sched} {ds} "
                             f"--rounds {rounds} --probe_n 64 --seed {ps} "
                             f"--model_seed {ms} --run_name {name} "
                             f"--output_dir {OUT} --device {{DEV}}")
    # DV-delta on abrupt (missing env for the delta arm)
    for ps, ms in S3:
        name = f"dvsig_delta_abrupt_s{ps}"
        yield name, (f"{RV2} {DV} --signal delta_hard --schedule abrupt "
                     f"--rounds 150 --probe_n 64 --seed {ps} --model_seed {ms} "
                     f"--run_name {name} --output_dir {OUT} --device {{DEV}}")
    for sig, tag in [("js_div", "js"), ("kl_sym", "kl"), ("cos_logit_dist", "cos")]:
        for env in ("A", "abrupt"):
            sched, rounds, ds = ENVS[env]
            for ps, ms in S3:
                name = f"dvsig_{tag}_{env}_s{ps}"
                yield name, (f"{RV2} {DV} --signal {sig} {sched} {ds} "
                             f"--rounds {rounds} --probe_n 64 --seed {ps} "
                             f"--model_seed {ms} --run_name {name} "
                             f"--output_dir {OUT} --device {{DEV}}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qfile = QDIR / "queue.txt"
    existing = set(qfile.read_text().strip().split("\n")) if qfile.exists() else set()
    n = 0
    with open(qfile, "a") as f:
        for name, cmd in jobs():
            if (OUT / f"{name}.json").exists() or cmd in existing:
                continue
            f.write(cmd + "\n")
            n += 1
    print(f"enqueued {n} Gate-2 signal jobs")


if __name__ == "__main__":
    main()
