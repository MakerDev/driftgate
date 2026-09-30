"""Round 6 — enqueue all 151 runs (conditions fixed BEFORE any result; no gate runs).

Queue: runs/queue_r6/queue.txt, one line per run: "<size class> <command>", served in file
order by scripts/r6_worker.py. Order = directive §4 priority table: all of P1 first, then P2
in table order; inside a block, seed-major (all arms of seed 0, then seed 1, ...).

  P1  S1 main arms (8 x seeds 0-4)            DriftGate, fixed 0.15/0.2/0.3/0.4/0.5/0.6, entropy
  P1  S1 APFL (2 x seeds 0-2)                 eta 0.01, 0.1
  P1  S2 GeoLife (10 x seeds 0-2)             DriftGate, 6 fixed, entropy, 2 APFL
  P1  S3 K=200 (4 x seeds 0-2)                DriftGate, fixed 0.2/0.4/0.6
  P2  S3 K=500 (4 x seeds 0-2)
  P2  S4 loss 0.1/0.3, delay 1/3, low request rate (DriftGate x seeds 0-2)
  P2  S4 participation 0.7/0.5 (DriftGate, fixed 0.2/0.4/0.6 x seeds 0-2)
  P2  S1-fast (DriftGate, fixed 0.2/0.4/0.6 x seeds 0-2)
DriftGate = --mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --spatial_norm (no --abs_cap).
All runs: --disjoint_pools, probe 64, model_seed = 100 + seed, env_seed = seed, 150 rounds,
evaluation rounds {1, 5, ..., 150}. S1 DriftGate seed 0 also saves its final model (§6.3).

  python journal_expansion/scripts/enqueue_r6.py --snapshot-only   # write the 151-line snapshot only
  python journal_expansion/scripts/enqueue_r6.py                   # append runs without a JSON
"""
import sys
from pathlib import Path

JR = Path("/home/honeynaps/data/driftgate/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
RUN = f"python -u {JR}/scripts/run_r6.py"
QD = JR / "runs/queue_r6"
QD.mkdir(parents=True, exist_ok=True)
(QD / "logs").mkdir(exist_ok=True)
DG = "--mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --spatial_norm"
ENT = "--mode selfcal --signal ent_client --burn_in 10 --z_guard 0.5 --spatial_norm"


def FIX(lam):
    return f"--mode fixed --lambda_val {lam} --big_lambda_val 0.5"


def APFL(eta):
    return f"--mode apfl --apfl_eta {eta} --big_lambda_val 0.5"


FIXED_ALL = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
FIXED3 = [0.2, 0.4, 0.6]


def tag(lam):
    return f"{int(round(lam * 100)):03d}"


def arm_list(kind):
    arms = []
    if kind in ("main", "s2"):
        arms.append(("driftgate", DG))
        arms += [(f"fixed{tag(l)}", FIX(l)) for l in FIXED_ALL]
        arms.append(("entropy", ENT))
    if kind in ("apfl", "s2"):
        arms += [("apfl001", APFL(0.01)), ("apfl010", APFL(0.1))]
    if kind == "dg3fixed":
        arms = [("driftgate", DG)] + [(f"fixed{tag(l)}", FIX(l)) for l in FIXED3]
    if kind == "dg":
        arms = [("driftgate", DG)]
    return arms


# (block name, scenario label, env prefix, extra flags, arms, seeds, size class, out dir, run prefix)
BLOCKS = [
    ("P1 S1 main", "S1", "S1", "", "main", range(5), "K50", "phaseT6_S1", "s1"),
    ("P1 S1 APFL", "S1", "S1", "", "apfl", range(3), "K50", "phaseT6_S1", "s1"),
    ("P1 S2", "S2", "S2", "", "s2", range(3), "K50", "phaseT6_S2", "s2"),
    ("P1 S3 K200", "S3_K200", "S3_K200", "", "dg3fixed", range(3), "K200", "phaseT6_S3", "s3k200"),
    ("P2 S3 K500", "S3_K500", "S3_K500", "", "dg3fixed", range(3), "K500", "phaseT6_S3", "s3k500"),
    ("P2 S4 loss0.1", "S4_loss01", "S4_loss01", "", "dg", range(3), "K50", "phaseT6_S4", "s4loss01"),
    ("P2 S4 loss0.3", "S4_loss03", "S4_loss03", "", "dg", range(3), "K50", "phaseT6_S4", "s4loss03"),
    ("P2 S4 delay1", "S4_delay1", "S1", "--signal_delay 1", "dg", range(3), "K50", "phaseT6_S4", "s4delay1"),
    ("P2 S4 delay3", "S4_delay3", "S1", "--signal_delay 3", "dg", range(3), "K50", "phaseT6_S4", "s4delay3"),
    ("P2 S4 lowreq", "S4_lowreq", "S4_lowreq", "", "dg", range(3), "K50", "phaseT6_S4", "s4lowreq"),
    ("P2 S4 part0.7", "S4_part07", "S4_part07", "", "dg3fixed", range(3), "K50", "phaseT6_S4", "s4part07"),
    ("P2 S4 part0.5", "S4_part05", "S4_part05", "", "dg3fixed", range(3), "K50", "phaseT6_S4", "s4part05"),
    ("P2 S1-fast", "S1fast", "S1fast", "", "dg3fixed", range(3), "K50", "phaseT6_S1fast", "s1fast"),
]


def all_jobs():
    jobs = []
    for block, scen, env, extra, kind, seeds, cls, out, pre in BLOCKS:
        for s in seeds:
            for arm, flags in arm_list(kind):
                rn = f"{pre}_{arm}_s{s}"
                save = " --save_models" if (scen == "S1" and arm == "driftgate" and s == 0) else ""
                cmd = (f"{RUN} --scenario {scen} --env {env} --arm {arm} {flags} --disjoint_pools {extra} "
                       f"--seed {s} --model_seed {100 + s} --run_name {rn} --output_dir {JR / 'runs' / out} "
                       f"--device {{DEV}}{save}").replace("  ", " ")
                jobs.append((block, cls, JR / "runs" / out, rn, cmd))
    return jobs


def main():
    jobs = all_jobs()
    assert len(jobs) == 151, len(jobs)
    assert len({j[3] for j in jobs}) == 151
    if "--snapshot-only" in sys.argv:
        with open(QD / "enqueued_snapshot.txt", "w") as f:
            for block, cls, out, rn, cmd in jobs:
                f.write(f"{cls} {cmd}\n")
        print("snapshot written: 151 runs")
        return
    q = QD / "queue.txt"
    existing = q.read_text().splitlines() if q.exists() else []
    n = 0
    with open(q, "a") as f:
        for block, cls, out, rn, cmd in jobs:
            line = f"{cls} {cmd}"
            if (out / f"{rn}.json").exists() or line in existing:
                continue
            out.mkdir(parents=True, exist_ok=True)
            f.write(line + "\n")
            n += 1
    print(f"enqueued {n} of 151")


if __name__ == "__main__":
    main()
