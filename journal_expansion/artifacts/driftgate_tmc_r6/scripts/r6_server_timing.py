"""§6.1 server-side timing (run on an otherwise idle server, after all Round-6 runs).

  A          feature bytes of one request: client block output [1,128,8,8] fp32 (checked from the model)
  t_s edge   server block + server exit on the edge GPU, batch 1 and 64 (20 warm-up, 200 timed runs)
  t_c host   client block + client exit on the server CPU, batch 1 and 64  -> PLACEHOLDER for t_c(device)
  t_s host   server block on the server CPU, batch 64                       -> PLACEHOLDER for t_s(device)
  t_train    one local training round on the server CPU (908 samples, 3 epochs, batch 32) -> PLACEHOLDER
Writes tables/T9a_server_timing.csv and tables/T9a_host_placeholder_device.csv (§6.2 device CSV
format, source = "host placeholder"). Weights: S1 DriftGate seed 0 final model (device/models/).

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python .../r6_server_timing.py
"""
import csv
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "device"))
from r6_device_models import FEATURE_BYTES, FEATURE_SHAPE, load  # noqa: E402
from bench_device import train_round  # noqa: E402

TAB = HERE / "tables"
W = HERE / "device" / "models" / "driftgate_s1_s0_weights.pth"


def timeit(fn, sync, warmup=20, reps=200):
    for _ in range(warmup):
        fn()
    sync()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        out.append((time.perf_counter() - t0) * 1000.0)
    out.sort()
    return statistics.median(out), out[int(math.ceil(0.95 * len(out))) - 1]


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main():
    load_avg = os.getloadavg()
    gpu_busy = subprocess.run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader",
                              shell=True, capture_output=True, text=True).stdout.strip().replace("\n", " | ")
    rows = []
    cg, sg = load(W, "cuda")
    gname = torch.cuda.get_device_name(0)
    with torch.no_grad():
        _, rep = cg(torch.zeros(1, 3, 32, 32, device="cuda"))
        a_bytes = rep.numel() * rep.element_size()
        assert a_bytes == FEATURE_BYTES == 32768, a_bytes
        rows.append(dict(quantity="A_feature_bytes", batch=1, median_ms="", p95_ms="", value=a_bytes,
                         hardware=f"shape {tuple(rep.shape)} {rep.dtype}", role="measured"))
        for b in (1, 64):
            f = torch.randn(b, *FEATURE_SHAPE, device="cuda")
            med, p95 = timeit(lambda: sg(f), torch.cuda.synchronize)
            rows.append(dict(quantity=f"t_s_edge_gpu_b{b}", batch=b, median_ms=f"{med:.4f}", p95_ms=f"{p95:.4f}",
                             value="", hardware=f"{gname} (CUDA {torch.version.cuda})", role="measured (edge)"))
    cc, sc = load(W, "cpu")
    threads = torch.get_num_threads()
    host = {}
    with torch.no_grad():
        for b in (1, 64):
            x = torch.randn(b, 3, 32, 32)
            f = torch.randn(b, *FEATURE_SHAPE)
            host[f"t_c_b{b}"] = timeit(lambda: cc(x), lambda: None)
            host[f"t_s_b{b}"] = timeit(lambda: sc(f), lambda: None)
    for k, (med, p95) in host.items():
        rows.append(dict(quantity=f"{k}_host_cpu", batch=int(k.split("b")[-1]), median_ms=f"{med:.4f}",
                         p95_ms=f"{p95:.4f}", value="", hardware=f"{cpu_name()}, {threads} threads",
                         role="host placeholder for the device"))
    train_round(cc, sc, torch.device("cpu"), 908, 32, 1, 0.01, 1e-4, lambda: None)
    t_train = statistics.median(train_round(cc, sc, torch.device("cpu"), 908, 32, 3, 0.01, 1e-4, lambda: None)
                                for _ in range(3))
    rows.append(dict(quantity="t_train_round_host_cpu", batch=32, median_ms=f"{t_train * 1000:.1f}", p95_ms="",
                     value="908 samples x 3 epochs", hardware=f"{cpu_name()}, {threads} threads",
                     role="host placeholder for the device"))
    TAB.mkdir(parents=True, exist_ok=True)
    with open(TAB / "T9a_server_timing.csv", "w", newline="") as fh:
        fh.write(f"# measured {time.strftime('%F %T %Z')} on {platform.node()}; load average before {load_avg}; "
                 f"GPUs before: {gpu_busy}\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(TAB / "T9a_host_placeholder_device.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["device", "backend", "t_c_b1_ms", "t_c_b64_ms", "t_s_b1_ms", "t_s_b64_ms", "t_train_round_s", "source"])
        w.writerow([f"host {cpu_name()}", "cpu", f"{host['t_c_b1'][0]:.4f}", f"{host['t_c_b64'][0]:.4f}",
                    f"{host['t_s_b1'][0]:.4f}", f"{host['t_s_b64'][0]:.4f}", f"{t_train:.4f}", "host placeholder"])
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
