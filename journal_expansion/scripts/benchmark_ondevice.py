"""Phase J — on-device latency benchmark (run BY THE USER on the target device).

Measures ONLY what it runs — never estimates, never fills defaults. If a stage
cannot run, it is recorded as null with the error string.

Example:
  python benchmark_ondevice.py \
      --export-dir exported/cnn_cifar10_middle --format torchscript \
      --batch-size 1 --probe-size 64 --warmup 50 --iterations 500 \
      --device cpu --threads 4 \
      --output results/device_latency.json

Stages measured:
  client_forward       one inference batch through the client block
  client_exit          softmax+argmax on client logits
  probe_path           full dual-exit probe (client + server tail) on probe batch
  signal_computation   16-signal library on probe outputs
  controller_update    one DV controller step (5 cells)
  e2e_local_probe      probe_path + signal + controller
Reported per stage: mean/median/p95/p99/std (ms), plus peak RSS. `energy_j` is
an OPTIONAL user-supplied field (external meter) — never auto-filled.
"""
import argparse
import json
import platform
import resource
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
            "p95_ms": xs[int(0.95 * len(xs)) - 1], "p99_ms": xs[int(0.99 * len(xs)) - 1],
            "std_ms": statistics.pstdev(xs), "n": len(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--format", default="torchscript", choices=["torchscript", "eager"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--probe-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    sync = (torch.cuda.synchronize if dev.type == "cuda" else None)

    exp = Path(args.export_dir)
    manifest = json.load(open(exp / "export_manifest.json"))
    size = manifest["input_size"]

    if args.format == "torchscript":
        client = torch.jit.load(str(exp / "client_block.pt"), map_location=dev).eval()
        probe_path = torch.jit.load(str(exp / "probe_path.pt"), map_location=dev).eval()
    else:
        from src.models_ext import FlexModelFactory
        from src.datasets_ext import META
        ncls, in_spatial, _ = META.get(manifest["dataset"], (10, 8, 2))
        f = FlexModelFactory(family=manifest["family"], num_classes=ncls,
                             in_spatial=in_spatial, split=manifest["split"])
        sd = torch.load(exp / "eager_state.pt", map_location="cpu")
        client = f.make_client(); client.load_state_dict(sd["client"]); client.to(dev).eval()
        server = f.make_server(); server.load_state_dict(sd["server"]); server.to(dev).eval()

        class PP(torch.nn.Module):
            def __init__(s):
                super().__init__(); s.c, s.s = client, server
            def forward(s, x):
                cl, rep = s.c(x); sl, _ = s.s(rep); return cl, sl, rep
        probe_path = PP().eval()

    x1 = torch.randn(args.batch_size, 3, size, size, device=dev)
    xp = torch.randn(args.probe_size, 3, size, size, device=dev)

    from src.signals.library import raw_signals
    from src.controllers.self_calibrating import SelfCalController
    import torch.nn.functional as F

    results, errors = {}, {}
    with torch.no_grad():
        try:
            results["client_forward"] = timed(lambda: client(x1), args.warmup,
                                              args.iterations, sync)
        except Exception as e:
            errors["client_forward"] = str(e)
        try:
            cl, _ = client(x1)
            results["client_exit"] = timed(lambda: F.softmax(cl, 1).argmax(1),
                                           args.warmup, args.iterations, sync)
        except Exception as e:
            errors["client_exit"] = str(e)
        try:
            results["probe_path"] = timed(lambda: probe_path(xp), args.warmup,
                                          args.iterations, sync)
        except Exception as e:
            errors["probe_path"] = str(e)
        try:
            clp, slp, rep = probe_path(xp)
            rp = F.adaptive_avg_pool2d(rep, 1).flatten(1)
            results["signal_computation"] = timed(lambda: raw_signals(clp, slp, rp),
                                                  args.warmup, args.iterations, sync)
        except Exception as e:
            errors["signal_computation"] = str(e)
        try:
            ctrl = SelfCalController(neighbors={i: [] for i in range(5)}, abs_cap=True)
            for _ in range(30):
                ctrl.step({i: 0.3 for i in range(5)})
            results["controller_update"] = timed(
                lambda: ctrl.step({i: 0.31 for i in range(5)}),
                args.warmup, args.iterations, None)
        except Exception as e:
            errors["controller_update"] = str(e)
        try:
            def e2e():
                clp, slp, rep = probe_path(xp)
                rp = F.adaptive_avg_pool2d(rep, 1).flatten(1)
                s = raw_signals(clp, slp, rp)
                ctrl.step({i: s["delta_hard"] for i in range(5)})
            results["e2e_local_probe"] = timed(e2e, args.warmup, args.iterations, sync)
        except Exception as e:
            errors["e2e_local_probe"] = str(e)

    out = {
        "metadata": {
            "device_model": None, "chipset": None, "cpu": platform.processor() or None,
            "gpu_npu": None, "ram_gb": None, "os": platform.platform(),
            "runtime": f"torch {torch.__version__}", "thread_count": torch.get_num_threads(),
            "precision": "fp32", "power_mode": None, "thermal_state": None,
            "model_format": args.format, "batch_size": args.batch_size,
            "probe_size": args.probe_size, "warmup": args.warmup,
            "iterations": args.iterations, "export": str(exp),
            "note": "null metadata fields MUST be filled in by the person who ran this",
        },
        "stages": results,
        "errors": errors,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "energy_j": None,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: round(v["mean_ms"], 3) for k, v in results.items()}, indent=1))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
