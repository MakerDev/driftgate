# journal_expansion — Adaptive-SplitOMC → TMC/ToN

Journal-grade expansion of the ICTC Adaptive-SplitOMC work. The parent directory holds the
untouched v4 training stack (`data/ models/ train/ eval/ controller/ scripts/`); everything
new lives here. Nothing in the parent is overwritten; legacy results in `../results/` are
frozen references.

## Environment

```bash
# Python 3.10+ (tested 3.12.7), torch 2.x + CUDA, numpy, matplotlib, pyyaml, pytest
pip install -r ../requirements.txt
```

Datasets: CIFAR-10 auto-downloads to `../data_cache` on first run.

## Verify correctness (Gate A)

```bash
cd /disk2/Yujin/adaptive_splitomc_tmc
python -m pytest tests/test_all.py journal_expansion/tests/test_journal.py -q   # 29 tests
```

## Reproduce the ICTC main run

```bash
python scripts/run_e2_temporal.py --schedule A --device cuda:0 \
    --methods adaptive_splitomc \
    --output_dir ./journal_expansion/runs/reproduction/e2_temporal
# compare against ../results/e2_temporal/schedule_A/adaptive_splitomc.json
# report: journal_expansion/reports/phase0_reproduction_and_audit.md
```

## Gate-B pilot (signal benchmark)

```bash
python journal_expansion/scripts/launch_pilots.py     # writes slot scripts + manifest
setsid nohup bash journal_expansion/scripts/_slot0.sh > .../slot0.log 2>&1 &   # etc.
# when done:
python journal_expansion/scripts/analyze_signals.py
# -> tables/signal_quality*.csv, figures/signal_*.png
```

Single runs (any mode):

```bash
python journal_expansion/scripts/run_v2.py --mode passive|legacy|selfcal \
    --schedule A|abrupt|recurring|burst|gradual_sigmoid|asym_return|piecewise_random|staggered \
    --signal delta_hard|ent_client|kl_sym|... --seed N --rounds 100 \
    --run_name NAME --output_dir journal_expansion/runs/... --device cuda:N
```

## Layout

```
src/
  provenance.py            run records (ID, git, config SHA, split/model hashes, env, metrics)
  schedules.py             drift schedules (fraction-based; scalar or per-cell)
  signals/library.py       16 raw signals: S0 entropy, S1 hard δ, S2 soft divergences,
                           S3 representation drift, confidences (S5 gate inputs)
  controllers/normalizers.py       S4 causal normalizers (guarded-z main; anchored/ewma/rollq/cusum)
  controllers/self_calibrating.py  self-calibrating z-space controller + source-calibrated
  evaluation/signal_metrics.py     AUROC/corr/delay/retention/recovery/stability
  runner.py                extended experiment loop (recording + pluggable controllers)
scripts/   run_v2.py, launch_pilots.py, analyze_signals.py
tests/     test_journal.py (17 audit tests)
runs/      per-experiment outputs (JSON + per-client signal npz)
provenance/ all_runs.jsonl + per-run records
reports/   master_experiment_plan.md, phase reports, progress_log.md
tables/ figures/  paper-ready outputs
```

## Provenance rules

- every run → `provenance/<run_id>.json` + line in `provenance/all_runs.jsonl`
- same config re-run ⇒ NEW run id; result JSONs are only skipped, never overwritten
- controllers never see labels or ρ (enforced by API + tests); calibration protocol in
  `reports/phase1_calibration.md` is binding.
