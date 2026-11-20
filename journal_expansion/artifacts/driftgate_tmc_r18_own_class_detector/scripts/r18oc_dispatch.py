"""Round 18 (own-class detector) dispatcher on GPUs 0-3 (scheduling as in scripts/r15_dispatch.py).

Reads runs/phaseT18oc/jobs.json in every loop (jobs can be appended with r18oc_jobs.py while it runs). A job starts when its
`requires` file exists; a GPU is chosen by free memory minus the cost of jobs launched there in the last 4 minutes (margin
2.5 GB, at most MAX_PER_GPU jobs per GPU) and only while the machine has at least MIN_RAM_GB of available memory. A failed
job is retried after 10 minutes, at most 3 attempts. STOP file -> no new launches (running jobs continue). The dispatcher
exits when every job has its result JSON (ALL_DONE) or when nothing can run any more.

  setsid nohup python -u .../r18oc_dispatch.py < /dev/null > runs/phaseT18oc/dispatch.out 2>&1 &
"""
import json
import os
import subprocess
import time
from pathlib import Path

JR = Path(__file__).resolve().parents[3]
ROOT = JR.parent
OUT = JR / "runs" / "phaseT18oc"
JOBS = OUT / "jobs.json"
GPUS = [0, 1, 2, 3]  # [SERVER-GPU] the user's allowed GPUs
MAX_PER_GPU = 5
MARGIN_GB = 2.5
MIN_RAM_GB = 14.0


def alive_names():
    names = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            args = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        args = [a.decode(errors="ignore") for a in args]
        if "--run_name" in args and any(a.endswith("run_r6.py") for a in args):
            names.add(args[args.index("--run_name") + 1])
    return names


def free_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return {int(a): float(b) / 1024 for a, b in (l.split(",") for l in out)}


def ram_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "dispatch.log", "a") as f:
        f.write(line + "\n")


def main():
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    state_p = OUT / "dispatch_state.json"
    state = json.load(open(state_p)) if state_p.exists() else {}
    log(f"dispatcher start (pid {os.getpid()})")
    idle = 0
    while True:
        J = json.load(open(JOBS))
        alive = alive_names()
        done = {j["name"] for j in J if (OUT / f"{j['name']}.json").exists()}
        now = time.time()
        pending, waiting = [], []
        for j in J:
            n = j["name"]
            st = state.setdefault(n, {"attempts": 0, "last": 0, "gpu": None})
            if n in done or n in alive or st["attempts"] >= 3:
                continue
            if st["attempts"] > 0 and now - st["last"] < 600:
                waiting.append(j)
                continue
            if j["requires"] and not Path(j["requires"]).exists():
                waiting.append(j)
                continue
            pending.append(j)
        ours_alive = alive & {j["name"] for j in J}
        if len(done) == len(J):
            log(f"all {len(J)} jobs done")
            (OUT / "ALL_DONE").write_text(time.strftime("%F %T") + f" {len(J)} jobs\n")
            idle += 1
            if idle >= 3:   # leave after three quiet loops (new jobs restart a fresh dispatcher)
                break
            time.sleep(60)
            continue
        idle = 0
        if not pending and not ours_alive and not waiting:
            log("no job can run (failed jobs reached 3 attempts); stopping")
            break
        if (OUT / "STOP").exists():
            log("STOP file found; not launching (running jobs continue)")
            time.sleep(60)
            continue
        free = free_gb()
        ours = {g: 0 for g in GPUS}
        recent = {g: 0.0 for g in GPUS}
        for j in J:
            st = state[j["name"]]
            if j["name"] in alive and st["gpu"] is not None:
                ours[st["gpu"]] += 1
                if now - st["last"] < 240:
                    recent[st["gpu"]] += j["cost"]
        for j in pending:
            if ram_gb() < MIN_RAM_GB:
                log(f"available RAM {ram_gb():.1f} GB < {MIN_RAM_GB}; waiting")
                break
            eff = {g: free.get(g, 0) - recent[g] for g in GPUS if ours[g] < MAX_PER_GPU}
            if not eff:
                break
            g = max(eff, key=eff.get)
            if eff[g] - j["cost"] < MARGIN_GB:
                continue
            st = state[j["name"]]
            st["attempts"] += 1
            st["last"] = now
            st["gpu"] = g
            env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(g), JX_THREADS="2",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            env.pop("JX_PROVENANCE_DIR", None)
            lf = open(OUT / "logs" / f"{j['name']}.log", "a")
            lf.write(f"\n=== attempt {st['attempts']} on GPU {g} at {time.strftime('%F %T')}\n")
            lf.flush()
            subprocess.Popen(j["cmd"], shell=True, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
            log(f"launch {j['name']} on GPU {g} (attempt {st['attempts']}, free {eff[g]:.1f} GB, cost {j['cost']})")
            ours[g] += 1
            recent[g] += j["cost"]
            time.sleep(20)   # stagger the start-up (data loading, RAM)
        with open(state_p, "w") as f:
            json.dump(state, f, indent=1)
        time.sleep(45)


if __name__ == "__main__":
    main()
