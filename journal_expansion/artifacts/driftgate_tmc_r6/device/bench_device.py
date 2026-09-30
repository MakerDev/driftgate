"""§6.3 device benchmark (Jetson, PyTorch). Measures, per backend (cpu / cuda):
  t_c  client block + client exit, batch 1 and 64          (inference, eval mode)
  t_s  server block + server exit replica, batch 1 and 64  (inference, eval mode)
  t_train_round  one local training round: local_epochs x ceil(n/batch) SGD steps on the
                 multi-exit loss (gamma 0.5) with one server replica, n = samples per client
Each inference timing: `--warmup` runs, then `--reps` runs; median and p95 in ms.
Writes the §6.2 device CSV (columns device, backend, t_c_b1_ms, t_c_b64_ms, t_s_b1_ms,
t_s_b64_ms, t_train_round_s, then extra columns).

  python3 bench_device.py --weights models/driftgate_s1_s0_weights.pth --device-name "Jetson Orin Nano 8GB" \
      --backends cpu cuda --out device_timing_jetson.csv
Needs only PyTorch and r6_device_models.py (same folder). Training uses random images of the right
shape (timing does not depend on pixel values).
"""
import argparse
import csv
import math
import os
import platform
import statistics
import time

import torch

from r6_device_models import FEATURE_SHAPE, load, multi_exit_loss


def timeit(fn, sync, warmup, reps):
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
    return statistics.median(out), out[min(len(out) - 1, int(math.ceil(0.95 * len(out))) - 1)]


def train_round(client, server, dev, n, batch, epochs, lr, wd, sync):
    client.train(), server.train()
    params = list(client.parameters()) + list(server.parameters())
    opt = torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=wd)
    xs = torch.randn(n, 3, 32, 32, device=dev)
    ys = torch.randint(0, 2, (n,), device=dev)
    sync()
    t0 = time.perf_counter()
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            cl, rep = client(xs[idx])
            loss = multi_exit_loss(cl, server(rep), ys[idx])
            loss.backward()
            opt.step()
    sync()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device-name", required=True)
    ap.add_argument("--backends", nargs="+", default=["cpu", "cuda"])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--train-samples", type=int, default=908, help="samples per client (R6 median 908)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--train-reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: PyTorch default)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    rows = []
    for backend in a.backends:
        if backend == "cuda" and not torch.cuda.is_available():
            print("cuda not available, skipped")
            continue
        dev = torch.device(backend)
        sync = (lambda: torch.cuda.synchronize()) if backend == "cuda" else (lambda: None)
        client, server = load(a.weights, dev)
        res = {}
        with torch.no_grad():
            for b in (1, 64):
                x = torch.randn(b, 3, 32, 32, device=dev)
                f = torch.randn(b, *FEATURE_SHAPE, device=dev)
                res[f"t_c_b{b}"] = timeit(lambda: client(x), sync, a.warmup, a.reps)
                res[f"t_s_b{b}"] = timeit(lambda: server(f), sync, a.warmup, a.reps)
        train_round(client, server, dev, a.train_samples, a.batch, 1, 0.01, 1e-4, sync)     # warm-up
        tr = sorted(train_round(client, server, dev, a.train_samples, a.batch, a.local_epochs, 0.01, 1e-4, sync)
                    for _ in range(a.train_reps))
        rows.append({"device": a.device_name, "backend": backend,
                     "t_c_b1_ms": f"{res['t_c_b1'][0]:.4f}", "t_c_b64_ms": f"{res['t_c_b64'][0]:.4f}",
                     "t_s_b1_ms": f"{res['t_s_b1'][0]:.4f}", "t_s_b64_ms": f"{res['t_s_b64'][0]:.4f}",
                     "t_train_round_s": f"{statistics.median(tr):.4f}",
                     "t_c_b1_p95_ms": f"{res['t_c_b1'][1]:.4f}", "t_c_b64_p95_ms": f"{res['t_c_b64'][1]:.4f}",
                     "t_s_b1_p95_ms": f"{res['t_s_b1'][1]:.4f}", "t_s_b64_p95_ms": f"{res['t_s_b64'][1]:.4f}",
                     "train_samples": a.train_samples, "local_epochs": a.local_epochs, "batch": a.batch,
                     "warmup": a.warmup, "reps": a.reps, "torch": torch.__version__,
                     "threads": torch.get_num_threads(), "platform": platform.platform(),
                     "gpu": torch.cuda.get_device_name(0) if backend == "cuda" else "", "source": "measured"})
        print(rows[-1])
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
