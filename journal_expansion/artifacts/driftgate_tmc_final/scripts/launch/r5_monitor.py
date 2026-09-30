"""Round-5 progress check (called periodically).
Exit codes: 0 = all 64 runs finished | 1 = still running, nothing to report |
            2 = a run failed twice (needs attention) or a queue is empty while runs are missing.
A run whose worker logged END exit!=0 without a JSON is re-queued ONCE at the head of its queue.
"""
import fcntl, glob, json, os, re, subprocess, sys, datetime
JR = "/disk2/Yujin/adaptive_splitomc_tmc/journal_expansion"
QD = f"{JR}/runs/queue_r5"
EXPECTED = {}  # run_name -> (out_dir, queue)
def add(q, out, names):
    for n in names: EXPECTED[n] = (f"{JR}/{out}", q)
add("heavy", "runs/phaseT4_B3_fixedgrid", [f"b3_tiny_fx15_s{s}" for s in range(3)])
add("heavy", "runs/phaseT5_E1_relonly_transfer", [f"e1_relonly_tiny_s{s}" for s in range(3)])
add("heavy", "runs/phaseT5_E2_resnet", [f"e2_res_{a}_A_s{s}" for a in ("relonly", "fx20", "fx40") for s in range(3)])
add("light", "runs/phaseT4_B3_fixedgrid", [f"b3_svhn_fx15_s{s}" for s in (2, 3, 4)] + [f"b3_c10gsig_fx15_s{s}" for s in (0, 1)]
    + [f"b3_c100gsig_fx15_s{s}" for s in (1, 2)])
add("light", "runs/phaseT5_E1_relonly_transfer", [f"e1_relonly_svhn_s{s}" for s in range(5)] + [f"e1_relonly_c10gsig_s{s}" for s in range(3)])
add("light", "runs/phaseT4_B4_apfl", [f"b4_apfl_{t}_A_s{s}" for t in ("eta001", "eta010") for s in range(5)]
    + [f"b4_apfl_{t}_mob_s{s}" for t in ("eta001", "eta010") for s in range(3)])
add("light", "runs/phaseT5_E3_const", [f"e3_{t}_A_s{s}" for t in ("guard025", "guard100", "lmax060", "lmax080") for s in range(3)]
    + [f"e3_{t}_mob_s{s}" for t in ("lmax060", "lmax080") for s in range(3)])
assert len(EXPECTED) == 64

done = [n for n, (d, _) in EXPECTED.items() if os.path.exists(f"{d}/{n}.json")]
# failures from worker logs
fails = {}
for wl in glob.glob(f"{QD}/logs/worker*.log"):
    for line in open(wl):
        m = re.search(r"END (\S+) exit=(\d+)", line)
        if m and m.group(2) != "0":
            fails[m.group(1)] = fails.get(m.group(1), 0) + 1
reqf = f"{QD}/requeued.txt"
requeued = set(open(reqf).read().split()) if os.path.exists(reqf) else set()
need_attention = []
for n, cnt in fails.items():
    if n not in EXPECTED or n in done:
        continue
    if n in requeued:
        if cnt >= 2: need_attention.append(n)
        continue
    # re-queue once at the head of its queue (same command line as originally enqueued)
    q = EXPECTED[n][1]
    line = None
    for src in (f"{QD}/{q}.txt", *sorted(glob.glob(f"{QD}/enqueued_{q}_snapshot.txt"))):
        if os.path.exists(src):
            for l in open(src):
                if f"--run_name {n} " in l: line = l.rstrip("\n"); break
        if line: break
    if line is None:
        need_attention.append(n + "(no cmd found)"); continue
    with open(f"{QD}/queue.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        cur = open(f"{QD}/{q}.txt").read() if os.path.exists(f"{QD}/{q}.txt") else ""
        if f"--run_name {n} " not in cur:
            with open(f"{QD}/{q}.txt", "w") as f: f.write(line + "\n" + cur)
        fcntl.flock(lk, fcntl.LOCK_UN)
    with open(reqf, "a") as f: f.write(n + "\n")
    print(f"re-queued {n} (failed once)")

nh = sum(1 for l in open(f"{QD}/heavy.txt") if l.strip()) if os.path.exists(f"{QD}/heavy.txt") else 0
nl = sum(1 for l in open(f"{QD}/light.txt") if l.strip()) if os.path.exists(f"{QD}/light.txt") else 0
running = subprocess.run("pgrep -f 'run_v2\\.py --mode' | wc -l", shell=True, capture_output=True, text=True).stdout.strip()
workers = subprocess.run("ps -eo args | grep -c '[r]5_worker\\.py'", shell=True, capture_output=True, text=True).stdout.strip()
gpu = subprocess.run("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader", shell=True,
                     capture_output=True, text=True).stdout.strip().replace("\n", " | ")
print(f"{datetime.datetime.now():%m-%d %H:%M} done {len(done)}/64 | queue heavy {nh} light {nl} | run_v2 procs {running} | workers {workers} | GPU {gpu}")
if len(done) == 64: sys.exit(0)
if need_attention: print("NEEDS ATTENTION:", need_attention); sys.exit(2)
if nh + nl == 0 and int(workers or 0) == 0: print("queues empty and no workers but runs missing"); sys.exit(2)
sys.exit(1)
