#!/usr/bin/env python3
"""Round-5 file-queue worker — GPU 0 ONLY.

Every job runs with CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=0, so the
job's "cuda:0" is nvidia-smi index 0 and GPU 1 is invisible to it.

Two queues under runs/queue_r5/:
  heavy.txt : Tiny-ImageNet (~3.8 GB) and ResNet-18 jobs
  light.txt : CIFAR/SVHN CNN jobs (~1.5 GB)
A "heavy" worker pops only heavy jobs. A "light" worker pops light jobs and, once the
light queue is empty, heavy jobs. The number of heavy jobs running at once is capped
(R5_HEAVY_MAX, default 5) so GPU-0 memory stays within budget. Pops and the heavy
counter are updated under one flock, so two workers cannot exceed the cap together.
Stops when runs/queue_r5/STOP exists.
Usage: r5_worker.py <heavy|light> <worker_id>

Migration note (2026-09-29): the GPU is now read from R5_GPU (default "0", i.e. the old
behaviour) and the heavy counter is kept per GPU (running_heavy_gpu<N>), so the same worker
can serve any GPU of a multi-GPU server. R5_HEAVY_MAX is therefore a per-GPU cap.
"""
import datetime, fcntl, os, re, subprocess, sys, time

KIND, WID = sys.argv[1], sys.argv[2]
QDIR = "/disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/queue_r5"  # [SERVER-PATH:REPO_ROOT]
GPU = os.environ.get("R5_GPU", "0")  # physical GPU index (nvidia-smi, PCI order)  # [SERVER-GPU]
LOCK, LOGDIR, RUNH = f"{QDIR}/queue.lock", f"{QDIR}/logs", f"{QDIR}/running_heavy_gpu{GPU}"
HEAVY_MAX = int(os.environ.get("R5_HEAVY_MAX", "5"))
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=GPU)
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(RUNH, exist_ok=True)
os.chdir("/disk2/Yujin/adaptive_splitomc_tmc")  # [SERVER-PATH:REPO_ROOT]


def log(msg):
    with open(f"{LOGDIR}/worker{WID}.log", "a") as f:
        f.write(f"[{KIND}{WID} cuda:0/GPU{GPU}] {msg} {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")


def pop(name):
    """Atomically pop the first line of <name>.txt. Heavy pops respect HEAVY_MAX."""
    with open(LOCK, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            if name == "heavy" and len(os.listdir(RUNH)) >= HEAVY_MAX:
                return None, None
            path = f"{QDIR}/{name}.txt"
            if not os.path.exists(path):
                return None, None
            lines = [l for l in open(path).read().splitlines() if l.strip()]
            if not lines:
                return None, None
            job = lines[0]
            with open(path, "w") as f:
                f.write("".join(l + "\n" for l in lines[1:]))
            m = re.search(r"--run_name (\S+)", job)
            tag = m.group(1) if m else f"job_{int(time.time())}"
            if name == "heavy":
                open(f"{RUNH}/{tag}", "w").close()
            return job, tag
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


while not os.path.exists(f"{QDIR}/STOP"):
    src = "heavy"
    if KIND == "heavy":
        job, tag = pop("heavy")
    else:
        job, tag = pop("light")
        src = "light"
        if job is None:
            job, tag = pop("heavy")
            src = "heavy"
    if job is None:
        time.sleep(60)
        continue
    cmd = job.replace("{DEV}", "cuda:0")
    log(f"START {tag} ({src})")
    with open(f"{LOGDIR}/{tag}.log", "a") as out:
        rc = subprocess.call(cmd, shell=True, stdout=out, stderr=subprocess.STDOUT, env=ENV)
    log(f"END {tag} exit={rc}")
    if src == "heavy":
        try:
            os.remove(f"{RUNH}/{tag}")
        except FileNotFoundError:
            pass
