"""Enqueue TMC supplementary experiments (Task A/B). Gate-before-scale:
`python enqueue_supp.py gate`  -> 1-seed sanity runs for each new family
`python enqueue_supp.py full`  -> remaining seeds (run AFTER gate sanity passes)
Task C needs no training. All method params frozen; A1 only freezes Lambda (diagnostic).
"""
import sys
from pathlib import Path

JR = Path("/disk2/Yujin/adaptive_splitomc_tmc/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
RV2 = f"python -u {JR}/scripts/run_v2.py"
DV = "--mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --spatial_norm --abs_cap"
OUT = JR / "runs/phaseS_supp"
Q = JR / "runs/queue/queue.txt"

# A1: adaptive lambda + fixed Lambda=0.5
def a1(seeds_A, seeds_mob):
    jobs = []
    for s in seeds_A:
        jobs.append((f"a1_A_s{s}", f"{RV2} {DV} --fixed_Lambda 0.5 --schedule A --rounds 150 "
                     f"--probe_n 64 --seed {s} --model_seed {100+s} --run_name a1_A_s{s} "
                     f"--output_dir {OUT} --device {{DEV}}"))
    for s in seeds_mob:
        jobs.append((f"a1_mobmed_s{s}", f"{RV2} {DV} --fixed_Lambda 0.5 --mobility "
                     f"--mobility_speed_level med --schedule abrupt --rounds 120 --probe_n 64 "
                     f"--seed {s} --model_seed {100+s} --run_name a1_mobmed_s{s} "
                     f"--output_dir {OUT} --device {{DEV}}"))
    return jobs

# A2: fixed (lambda, Lambda) 2D grid; lambda in {best-1,best,best+1}, Lambda in {0.4,0.6,0.7}
A2_LAMS = [0.3, 0.4, 0.5]      # Schedule A best fixed = 0.4; mobility best = 0.4
A2_BIGLAMS = [0.4, 0.6, 0.7]   # Lambda=0.5 reused from existing grid
def a2(seeds):
    jobs = []
    for lam in A2_LAMS:
        for L in A2_BIGLAMS:
            tag = f"{int(lam*100):02d}L{int(L*100):02d}"
            for s in seeds:
                jobs.append((f"a2_A_{tag}_s{s}",
                             f"{RV2} --mode fixed --lambda_val {lam} --big_lambda_val {L} "
                             f"--schedule A --rounds 150 --probe_n 64 --seed {s} "
                             f"--model_seed {100+s} --run_name a2_A_{tag}_s{s} "
                             f"--output_dir {OUT} --device {{DEV}}"))
                jobs.append((f"a2_mobmed_{tag}_s{s}",
                             f"{RV2} --mode fixed --lambda_val {lam} --big_lambda_val {L} "
                             f"--mobility --mobility_speed_level med --schedule abrupt --rounds 120 "
                             f"--probe_n 64 --seed {s} --model_seed {100+s} "
                             f"--run_name a2_mobmed_{tag}_s{s} --output_dir {OUT} --device {{DEV}}"))
    return jobs

# B1: bring Schedule A adaptive + best-fixed to 5 paired seeds.
#  adaptive A (dvsig_tv_A) has s0,s1,s2 -> add s3,s4 (run as DriftGate on A)
#  fixed candidates near the peak: fx40 (has s1,s2 -> add s0,s3,s4), fx50 (has s0,s1,s2 -> add s3,s4)
def b1():
    jobs = []
    for s in (3, 4):
        jobs.append((f"b1_adaptive_A_s{s}", f"{RV2} {DV} --schedule A --rounds 150 --probe_n 64 "
                     f"--seed {s} --model_seed {100+s} --run_name b1_adaptive_A_s{s} "
                     f"--output_dir {OUT} --device {{DEV}}"))
    for lam, need in ((0.4, (0, 3, 4)), (0.5, (3, 4))):
        for s in need:
            tag = f"{int(lam*100):02d}"
            jobs.append((f"b1_fx{tag}_A_s{s}",
                         f"{RV2} --mode fixed --lambda_val {lam} --big_lambda_val 0.5 "
                         f"--schedule A --rounds 150 --probe_n 64 --seed {s} --model_seed {100+s} "
                         f"--run_name b1_fx{tag}_A_s{s} --output_dir {OUT} --device {{DEV}}"))
    return jobs


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "gate"
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == "gate":
        jobs = a1([0], [0]) + [j for j in a2([0]) if "_4040_" in j[0] or "_40L60_" in j[0]] \
               + [("b1_adaptive_A_s3", None)]
        # gate: one A2 point per env (lam0.4,Lam0.6) + a1 gate + b1 one seed
        jobs = a1([0], [0])
        jobs += [j for j in a2([0]) if j[0] in ("a2_A_40L60_s0", "a2_mobmed_40L60_s0")]
        jobs += [j for j in b1() if j[0] == "b1_adaptive_A_s3"]
    elif mode == "full":
        jobs = a1([1, 2, 3, 4], [1, 2]) + a2([0, 1, 2]) + b1()
    else:
        raise SystemExit("mode: gate|full")
    existing = set(Q.read_text().strip().split("\n")) if Q.exists() else set()
    n = 0
    with open(Q, "a") as f:
        for name, cmd in jobs:
            if cmd is None or (OUT / f"{name}.json").exists() or cmd in existing:
                continue
            f.write(cmd + "\n")
            n += 1
    print(f"enqueued {n} jobs ({mode})")


if __name__ == "__main__":
    main()
