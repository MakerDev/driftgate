"""Phase J — edge-server-side benchmark (server block under batched load).

Measures server-block forward latency/throughput at several batch sizes and the
per-round aggregation cost. Same honesty rule: only measured values, run on the
machine the user chooses.

  python benchmark_edge_server.py --export-dir exported/cnn_cifar10_middle \
      --format torchscript --device cuda:0 --batch-sizes 1 8 32 128 \
      --output results/edge_server_latency.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
JOURNAL_ROOT = HERE.parent
for p in (str(JOURNAL_ROOT), str(JOURNAL_ROOT.parent)):
    sys.path.insert(0, p)


def timed(fn, warmup, iters, sync=None):
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    xs = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        xs.append((time.perf_counter() - t0) * 1000)
    xs.sort()
    return {"mean_ms": statistics.fmean(xs), "median_ms": xs[len(xs) // 2],
            "p95_ms": xs[int(0.95 * len(xs)) - 1], "std_ms": statistics.pstdev(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--format", default="torchscript", choices=["torchscript"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    dev = torch.device(args.device)
    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    exp = Path(args.export_dir)
    manifest = json.load(open(exp / "export_manifest.json"))
    server = torch.jit.load(str(exp / "server_block.pt"), map_location=dev).eval()
    rep_shape = manifest["stats"]["rep_shape"]

    out = {"metadata": {"device": args.device, "runtime": f"torch {torch.__version__}",
                        "export": str(exp), "rep_shape": rep_shape},
           "server_forward": {}}
    with torch.no_grad():
        for b in args.batch_sizes:
            rep = torch.randn(b, *rep_shape, device=dev)
            r = timed(lambda: server(rep), args.warmup, args.iterations, sync)
            r["throughput_samples_per_s"] = b / (r["mean_ms"] / 1000)
            out["server_forward"][str(b)] = r
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["server_forward"], indent=1))


if __name__ == "__main__":
    main()
