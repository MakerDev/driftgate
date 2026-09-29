# On-Device Measurement Protocol (Phase J)

1. Export models on any machine:
   `python journal_expansion/scripts/export_benchmark_models.py --family cnn --dataset cifar10 --split middle --formats torchscript onnx`
2. Copy `journal_expansion/exported/<name>/` + `scripts/benchmark_ondevice.py` +
   `src/signals/library.py` + `src/controllers/{self_calibrating,normalizers}.py`
   to the device (or install the repo).
3. Fix the device state BEFORE measuring and record it in metadata:
   - plug in / fixed power mode, screen state, governor
   - cool device (note thermal_state), close background apps
   - pin thread count (`--threads`)
4. Run (repeat 3x, report all runs):
   `python benchmark_ondevice.py --export-dir exported/cnn_cifar10_middle --format torchscript --batch-size 1 --probe-size 64 --warmup 50 --iterations 500 --device cpu --threads 4 --output results/device_latency_run1.json`
5. Fill ALL null metadata fields by hand (device model, chipset, RAM, power mode).
   Energy: only if an external meter was used; otherwise leave null.
6. Build the paper table/figure:
   `python journal_expansion/scripts/build_ondevice_table.py results/*.json`
   `python journal_expansion/scripts/plot_ondevice_overhead.py results/*.json`
   Both scripts REFUSE to run without at least one valid result file — no
   placeholder or estimated numbers are ever produced.
