"""Round-6 progress check. Never modifies the queue (failed runs are reported, not re-queued).
Exit codes: 0 = all 218 runs (v3 snapshot) finished | 1 = still running | 2 = a run failed, or runs are missing
while the queue is empty and no worker is alive.
"""
import glob
import os
import re
import subprocess
import sys
import datetime

JR = "/home/honeynaps/data/driftgate/journal_expansion"  # [SERVER-PATH:REPO_ROOT]
QD = f"{JR}/runs/queue_r6"

expected = []
for line in open(f"{QD}/enqueued_snapshot_v3.txt"):   # v3: R0 67 + Round 6 151 (2026-10-01)
    rn = re.search(r"--run_name (\S+)", line).group(1)
    od = re.search(r"--output_dir (\S+)", line).group(1)
    expected.append((rn, od, line.split(" ", 1)[0]))
assert len(expected) == 218
N = len(expected)
done = [rn for rn, od, _ in expected if os.path.exists(f"{od}/{rn}.json")]
fails = {}
for wl in glob.glob(f"{QD}/logs/worker*.log"):
    for line in open(wl):
        m = re.search(r"END (\S+) exit=(\d+)", line)
        if m and m.group(2) != "0":
            fails[m.group(1)] = fails.get(m.group(1), 0) + 1
names = {rn for rn, _, _ in expected}
queued = set(re.findall(r"--run_name (\S+)", open(f"{QD}/queue.txt").read())) if os.path.exists(f"{QD}/queue.txt") else set()
running_now = {os.path.basename(f).rsplit('.', 1)[0] for f in glob.glob(f"{QD}/running_gpu*/*")}
# retired v1 runs and runs that were re-queued (or are running again) are not failures
failed = sorted(n for n in fails if n in names and n not in done and n not in queued and n not in running_now)
nq = sum(1 for l in open(f"{QD}/queue.txt") if l.strip()) if os.path.exists(f"{QD}/queue.txt") else 0
running = sorted(os.path.basename(f) for f in glob.glob(f"{QD}/running_gpu*/*"))
workers = subprocess.run("ps -eo args | grep -c '[r]6_worker\\.py'", shell=True, capture_output=True,
                         text=True).stdout.strip()
gpu = subprocess.run("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader", shell=True,
                     capture_output=True, text=True).stdout.strip().replace("\n", " | ")
by_cls = {}
for rn, od, cls in expected:
    d = by_cls.setdefault(cls, [0, 0])
    d[1] += 1
    d[0] += rn in done
print(f"{datetime.datetime.now():%m-%d %H:%M} done {len(done)}/{N} "
      f"({', '.join(f'{c} {a}/{b}' for c, (a, b) in sorted(by_cls.items()))}) | queue {nq} | running {len(running)} "
      f"| workers {workers} | GPU {gpu}")
if running:
    print("running:", " ".join(running))
if len(done) == N:
    sys.exit(0)
if failed:
    print("FAILED (no JSON):", failed)
    sys.exit(2)
if nq == 0 and not running and int(workers or 0) == 0:
    print("queue empty, no worker, runs missing")
    sys.exit(2)
sys.exit(1)
