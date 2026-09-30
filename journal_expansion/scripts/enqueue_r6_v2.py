"""Round 6, queue v2 (2026-10-01, user decision): DriftGate WITHOUT the one-step neighbour score
average (--no_neighbor_avg) for every DriftGate / entropy / absonly arm. Fixed-lambda and APFL arms are
unaffected (no controller) and keep their v1 run names and results.

Order (served in file order by r6_worker.py):
  P0 (67)  Round-5 controller runs re-run with --no_neighbor_avg, exact Round-5 command lines otherwise
           (run_v2.py, same seeds / rounds / disjoint pools), output runs/phaseT6_p0/, name p0_<original>.
           DriftGate 25, E3 constants 18, entropy 13, absonly 11 (absonly also averaged its raw signal over
           neighbours: self_calibrating.py:170-171). Longest first inside P0.
  R6 P1    fixed / APFL runs not yet started, then DriftGate / entropy with the new definition
           (arms driftgate_own / entropy_own).
  R6 P2    table order (K=500, S4 loss/delay/low rate, S4 participation, S1-fast), DriftGate arms new definition.
The v1 queue's DriftGate / entropy lines (neighbour average) are retired: runs/queue_r6/held_old_definition.txt.

  python journal_expansion/scripts/enqueue_r6_v2.py --snapshot-only   # enqueued_snapshot_v2.txt (218 lines)
  python journal_expansion/scripts/enqueue_r6_v2.py                   # rebuild queue.txt under the queue lock
"""
import fcntl
import os
import re
import sys
from pathlib import Path

JR = Path("/home/honeynaps/data/driftgate/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
sys.path.insert(0, str(JR / "scripts"))
import enqueue_r6 as v1  # noqa: E402

QD = JR / "runs" / "queue_r6"
RV2 = f"python -u {JR}/scripts/run_v2.py"
P0_OUT = JR / "runs" / "phaseT6_p0"
DG_NEW = v1.DG + " --no_neighbor_avg"
ENT_NEW = v1.ENT + " --no_neighbor_avg"

ENVF = {"A": "--schedule A --rounds 150",
        "mob": "--mobility --mobility_speed_level med --schedule abrupt --rounds 120",
        "svhn": "--dataset svhn --schedule gradual_sigmoid --rounds 150",
        "c10gsig": "--dataset cifar10 --schedule gradual_sigmoid --rounds 150",
        "c100gsig": "--dataset cifar100 --schedule gradual_sigmoid --rounds 150",
        "tiny": "--dataset tinyimagenet --schedule gradual_sigmoid --rounds 100",
        "resA": "--model_family resnet18 --split_point middle --num_clients 16 --schedule A --rounds 150"}
RELONLY = "--mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --spatial_norm"
ENTROPY = "--mode selfcal --signal ent_client --burn_in 10 --z_guard 0.5 --spatial_norm"
ABSONLY = "--mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --abs_only"   # R4 B1 command (no --spatial_norm)


def p0_jobs():
    """(orig run name, controller flags, env key, seed); flags/env exactly as the original Round-4/5 runs
    (provenance command lines / runs/queue_r5 snapshot)."""
    J = []
    for s in range(3):
        J.append((f"e1_relonly_tiny_s{s}", RELONLY, "tiny", s))
    for s in range(3):
        J.append((f"e2_res_relonly_A_s{s}", RELONLY, "resA", s))
    for s in range(5):
        J.append((f"e1_relonly_svhn_s{s}", RELONLY, "svhn", s))
    for s in range(5):
        J.append((f"t2_svhn_ent_s{s}", ENTROPY, "svhn", s))
    for s in range(3):
        J.append((f"e1_relonly_c10gsig_s{s}", RELONLY, "c10gsig", s))
    for s in range(3):
        J.append((f"b1_relonly_c100gsig_s{s}", RELONLY, "c100gsig", s))
    for s in range(3):
        J.append((f"b1_absonly_c100gsig_s{s}", ABSONLY, "c100gsig", s))
    for s in range(5):
        J.append((f"b1_relonly_A_s{s}", RELONLY, "A", s))
    for s in range(5):
        J.append((f"t1_A_ent_s{s}", ENTROPY, "A", s))
    for s in range(5):
        J.append((f"b1_absonly_A_s{s}", ABSONLY, "A", s))
    for g, tag in ((0.25, "guard025"), (1.0, "guard100")):
        for s in range(3):
            J.append((f"e3_{tag}_A_s{s}", f"--mode selfcal --signal tv_dist --burn_in 10 --z_guard {g} --spatial_norm", "A", s))
    for lm, tag in ((0.60, "lmax060"), (0.80, "lmax080")):
        for s in range(3):
            J.append((f"e3_{tag}_A_s{s}", f"{RELONLY} --lam_max {lm}", "A", s))
    for s in range(3):
        J.append((f"b1_relonly_mob_s{s}", RELONLY, "mob", s))
    for s in range(3):
        J.append((f"t1_mob_ent_s{s}", ENTROPY, "mob", s))
    for s in range(3):
        J.append((f"b1_absonly_mob_s{s}", ABSONLY, "mob", s))
    for lm, tag in ((0.60, "lmax060"), (0.80, "lmax080")):
        for s in range(3):
            J.append((f"e3_{tag}_mob_s{s}", f"{RELONLY} --lam_max {lm}", "mob", s))
    out = []
    for orig, flags, env, s in J:
        rn = f"p0_{orig}"
        cmd = (f"{RV2} {flags} --no_neighbor_avg --disjoint_pools {ENVF[env]} --probe_n 64 --seed {s} "
               f"--model_seed {100 + s} --run_name {rn} --output_dir {P0_OUT} --device {{DEV}}")
        out.append(("P0", "P0", P0_OUT, rn, cmd))
    assert len(out) == 67
    return out


def r6_jobs():
    """151 Round-6 runs; DriftGate / entropy arms renamed *_own and given --no_neighbor_avg."""
    jobs = []
    for block, cls, out, rn, cmd in v1.all_jobs():
        if re.search(r"--arm driftgate ", cmd):
            cmd = cmd.replace("--arm driftgate ", "--arm driftgate_own ").replace(v1.DG, DG_NEW)
            rn = rn.replace("_driftgate_", "_driftgate_own_")
        elif re.search(r"--arm entropy ", cmd):
            cmd = cmd.replace("--arm entropy ", "--arm entropy_own ").replace(v1.ENT, ENT_NEW)
            rn = rn.replace("_entropy_", "_entropy_own_")
        cmd = re.sub(r"--run_name \S+", f"--run_name {rn}", cmd)
        jobs.append((block, cls, out, rn, cmd))
    return jobs


def ordered():
    r6 = r6_jobs()
    is_ctrl = lambda j: "--mode selfcal" in j[4]
    p1 = [j for j in r6 if j[0].startswith("P1")]
    p2 = [j for j in r6 if j[0].startswith("P2")]
    return p0_jobs() + [j for j in p1 if not is_ctrl(j)] + [j for j in p1 if is_ctrl(j)] + p2


def main():
    jobs = ordered()
    assert len(jobs) == 218 and len({j[3] for j in jobs}) == 218
    if "--snapshot-only" in sys.argv:
        with open(QD / "enqueued_snapshot_v2.txt", "w") as f:
            for block, cls, out, rn, cmd in jobs:
                f.write(f"{cls} {cmd}\n")
        print("snapshot v2 written: 218 runs (P0 67 + Round 6 151)")
        return
    running = {f.rsplit(".", 1)[0] for d in QD.glob("running_gpu*") for f in os.listdir(d)}
    with open(QD / "queue.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        lines = []
        for block, cls, out, rn, cmd in jobs:
            if (Path(out) / f"{rn}.json").exists() or rn in running:
                continue
            Path(out).mkdir(parents=True, exist_ok=True)
            lines.append(f"{cls} {cmd}")
        old = (QD / "queue.txt").read_text() if (QD / "queue.txt").exists() else ""
        (QD / "queue_v1_before_v2.txt").write_text(old)
        (QD / "queue.txt").write_text("".join(l + "\n" for l in lines))
        fcntl.flock(lk, fcntl.LOCK_UN)
    print(f"queue rebuilt: {len(lines)} lines (skipped {len(jobs) - len(lines)} done or running)")


if __name__ == "__main__":
    main()
