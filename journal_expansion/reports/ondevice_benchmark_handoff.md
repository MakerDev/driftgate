# §10 — On-Device Benchmark Handoff (user runs; no numbers fabricated)

**Date**: 2026-08-02 · Verified end-to-end on this host (export + refuse-on-empty tooling).
The user runs the benchmark on real devices; no latency values are estimated here.

## Verified capabilities
- Export produces (checked in `journal_expansion/exported/cnn_cifar10_middle/`):
  client_block, server_block, probe_path (dual-exit) in **TorchScript (.pt)** and **ONNX**,
  plus eager `state.pt` and `export_manifest.json`.
- `benchmark_ondevice.py` measures: client forward, client exit, probe path, TV signal
  computation, controller update, end-to-end local probe path — mean/median/p95/p99/std,
  peak RSS, optional energy field.
- `build_ondevice_table.py` / `plot_ondevice_overhead.py` **refuse to run without valid,
  metadata-complete measurements** (verified: errors out on missing input, no placeholder).

## Exact commands for the user

**1. Export (any machine):**
```
python journal_expansion/scripts/export_benchmark_models.py \
    --family cnn --dataset cifar10 --split middle \
    --formats torchscript onnx --outdir journal_expansion/exported
```

**2. Copy to the device** `journal_expansion/exported/cnn_cifar10_middle/`,
`scripts/benchmark_ondevice.py`, `src/signals/library.py`,
`src/controllers/{self_calibrating,normalizers}.py` (or clone the repo).

**3. Run (repeat 3×; fix power mode, cool device, pin threads):**
```
# CPU (e.g. Raspberry Pi / mobile CPU)
python benchmark_ondevice.py --export-dir exported/cnn_cifar10_middle \
    --format torchscript --batch-size 1 --probe-size 64 \
    --warmup 50 --iterations 500 --device cpu --threads 4 \
    --output results/<device>_run1.json
# Jetson / CUDA device: --device cuda:0
# ONNX runtime path: convert with the device's ORT and time client_block.onnx/server_block.onnx
```

**4. Edge-server side (optional):**
```
python benchmark_edge_server.py --export-dir exported/cnn_cifar10_middle \
    --format torchscript --device <cpu|cuda:0> --batch-sizes 1 8 32 128 \
    --output results/edge_<device>.json
```

**5. Fill ALL null metadata by hand** (device_model, chipset, RAM, power_mode, thermal_state;
energy only if a meter was used) per `templates/ondevice_measurement_protocol.md`.

**6. Build the paper table + figure:**
```
python journal_expansion/scripts/build_ondevice_table.py results/*.json
python journal_expansion/scripts/plot_ondevice_overhead.py results/*.json
```

## Required metadata (validation enforced)
device_model, chipset, cpu, gpu_npu, ram_gb, os, runtime, thread_count, precision,
power_mode, thermal_state, model_format, batch_size, probe_size, warmup, iterations.

## Expected output columns for the final paper table
device_model · chipset · precision · model_format · batch_size · probe_size ·
client_forward_ms · probe_path_ms · signal_computation_ms · controller_update_ms ·
e2e_local_probe_ms · p95_e2e_ms · peak_rss_mb · energy_j.

## Numbers pending from the user (leave blank until measured)
- device name(s) / chipset / RAM / OS / runtime
- per-stage latency mean/median/p95/p99
- peak memory
- energy (optional)

Host-RTX reference only (NOT a device number, for context): probe path ≈ 1.5 ms/client/round.
