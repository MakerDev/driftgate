"""Round 6, queue v3 (2026-10-01, user decision; replaces v2 before any v2 line ran).

DriftGate = the final controller WITHOUT the one-step neighbour score average (--no_neighbor_avg):
q_z = max of the EMA-smoothed temporal and spatial scores of cluster z. d-bar sharing for the spatial
score is kept. The entropy controller uses the same controller. Fixed-lambda and APFL arms do not use
the controller and keep their v1 names and results.

Order (served in file order by r6_worker.py; the order is execution order only):
  1. S3 K=500 (12)                      first, spread over the 4 GPUs (R6_MAX_K500 per GPU)
  2. R0 (67)                            Round-4/5 controller runs re-run with --no_neighbor_avg; the exact
                                        original command lines otherwise (checked against provenance /
                                        the Round-5 snapshot), output runs/phaseT6_R0/, name r0_<original>
                                        DriftGate 25, entropy 13, E3 constants 18, absonly 11
  3. S1, S2, S3 K=200                   (Round-6 P1)
  4. S4 (loss, delay, low rate, participation), S1-fast
Round-6 DriftGate / entropy arms are named driftgate_own / entropy_own (the v1 names driftgate / entropy
hold the superseded neighbour-average runs, kept as records).

  python journal_expansion/scripts/enqueue_r6_v3.py --snapshot-only   # enqueued_snapshot_v3.txt (218 lines)
  python journal_expansion/scripts/enqueue_r6_v3.py                   # rebuild queue.txt under the queue lock
"""
import fcntl
import os
import sys
from pathlib import Path

JR = Path("/home/honeynaps/data/driftgate/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
sys.path.insert(0, str(JR / "scripts"))
import enqueue_r6_v2 as v2  # noqa: E402

QD = JR / "runs" / "queue_r6"
R0_OUT = JR / "runs" / "phaseT6_R0"
HEAVY_R0 = ("tinyimagenet", "resnet18")


def r0_jobs():
    out = []
    for _, _, _, rn, cmd in v2.p0_jobs():
        new = "r0_" + rn[len("p0_"):]
        cmd = cmd.replace(f"--run_name {rn} ", f"--run_name {new} ").replace(str(v2.P0_OUT), str(R0_OUT))
        cls = "R0H" if any(h in cmd for h in HEAVY_R0) else "R0"
        out.append(("R0", cls, R0_OUT, new, cmd))
    assert len(out) == 67
    return out


def ordered():
    r6 = v2.r6_jobs()
    blk = lambda name: [j for j in r6 if j[0] == name]
    k500 = blk("P2 S3 K500")
    p1 = blk("P1 S1 main") + blk("P1 S1 APFL") + blk("P1 S2") + blk("P1 S3 K200")
    p2 = [j for j in r6 if j[0].startswith("P2") and j[0] != "P2 S3 K500"]
    return k500 + r0_jobs() + p1 + p2


def main():
    jobs = ordered()
    assert len(jobs) == 218 and len({j[3] for j in jobs}) == 218
    if "--snapshot-only" in sys.argv:
        with open(QD / "enqueued_snapshot_v3.txt", "w") as f:
            for block, cls, out, rn, cmd in jobs:
                f.write(f"{cls} {cmd}\n")
        print("snapshot v3 written: 218 runs (R0 67 + Round 6 151)")
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
        cur = (QD / "queue.txt").read_text() if (QD / "queue.txt").exists() else ""
        assert not cur.strip(), "queue.txt is not empty; pause it first"
        (QD / "queue.txt").write_text("".join(l + "\n" for l in lines))
        fcntl.flock(lk, fcntl.LOCK_UN)
    print(f"queue rebuilt: {len(lines)} lines (skipped {len(jobs) - len(lines)} done or running)")


if __name__ == "__main__":
    main()
