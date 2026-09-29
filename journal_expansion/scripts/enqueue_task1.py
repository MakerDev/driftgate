"""Task 1 — disjoint controller/eval pool re-verification.
`gate` = 1-seed sanity per family; `full` = remaining seeds.
Method frozen; only the probe/eval pools become disjoint (--disjoint_pools).
Fixed grid uses the same-pool-established peak region {fx30,fx40,fx50}
(argmax was fx40) to bound compute; documented in the report.
"""
import sys
from pathlib import Path

JR = Path("/disk2/Yujin/adaptive_splitomc_tmc/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
RV2 = f"python -u {JR}/scripts/run_v2.py"
DV = "--mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --spatial_norm --abs_cap"
ENT = "--mode selfcal --signal ent_client --burn_in 10 --z_guard 0.5 --spatial_norm"
A1 = DV + " --fixed_Lambda 0.5"
OUT = JR / "runs/phaseT1_disjoint"
Q = JR / "runs/queue/queue.txt"
FIXED_LAMS = [0.3, 0.4, 0.5]

def sched_A(seeds):
    jobs = []
    for s in seeds:
        base = f"--disjoint_pools --schedule A --rounds 150 --probe_n 64 --seed {s} --model_seed {100+s} --output_dir {OUT} --device {{DEV}}"
        jobs.append((f"t1_A_dg_s{s}",   f"{RV2} {DV} {base} --run_name t1_A_dg_s{s}"))
        jobs.append((f"t1_A_ent_s{s}",  f"{RV2} {ENT} {base} --run_name t1_A_ent_s{s}"))
        jobs.append((f"t1_A_a1_s{s}",   f"{RV2} {A1} {base} --run_name t1_A_a1_s{s}"))
        for lam in FIXED_LAMS:
            jobs.append((f"t1_A_fx{int(lam*100):02d}_s{s}",
                         f"{RV2} --mode fixed --lambda_val {lam} --big_lambda_val 0.5 {base} "
                         f"--run_name t1_A_fx{int(lam*100):02d}_s{s}"))
    return jobs

def mob(seeds):
    jobs = []
    for s in seeds:
        base = f"--disjoint_pools --mobility --mobility_speed_level med --schedule abrupt --rounds 120 --probe_n 64 --seed {s} --model_seed {100+s} --output_dir {OUT} --device {{DEV}}"
        jobs.append((f"t1_mob_dg_s{s}",  f"{RV2} {DV} {base} --run_name t1_mob_dg_s{s}"))
        jobs.append((f"t1_mob_ent_s{s}", f"{RV2} {ENT} {base} --run_name t1_mob_ent_s{s}"))
        jobs.append((f"t1_mob_a1_s{s}",  f"{RV2} {A1} {base} --run_name t1_mob_a1_s{s}"))
        for lam in FIXED_LAMS:
            jobs.append((f"t1_mob_fx{int(lam*100):02d}_s{s}",
                         f"{RV2} --mode fixed --lambda_val {lam} --big_lambda_val 0.5 {base} "
                         f"--run_name t1_mob_fx{int(lam*100):02d}_s{s}"))
    return jobs

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "gate"
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == "gate":
        jobs = sched_A([0]) + mob([0])
        jobs = [j for j in jobs if j[0] in
                ("t1_A_dg_s0", "t1_A_ent_s0", "t1_A_a1_s0", "t1_A_fx40_s0",
                 "t1_mob_dg_s0", "t1_mob_fx40_s0")]
    elif mode == "full":
        jobs = sched_A([1, 2, 3, 4]) + mob([1, 2]) + \
               [j for j in sched_A([0]) + mob([0])
                if not (OUT / f"{j[0]}.json").exists()]
    else:
        raise SystemExit("mode: gate|full")
    existing = set(Q.read_text().strip().split("\n")) if Q.exists() else set()
    n = 0
    with open(Q, "a") as f:
        for name, cmd in jobs:
            if (OUT / f"{name}.json").exists() or cmd in existing:
                continue
            f.write(cmd + "\n")
            n += 1
    print(f"enqueued {n} Task-1 jobs ({mode})")

if __name__ == "__main__":
    main()
