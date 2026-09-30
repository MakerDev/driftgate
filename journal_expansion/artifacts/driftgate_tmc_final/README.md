# DriftGate TMC final package (Round 5)

All numbers come from raw run JSON files under `journal_expansion/runs/`. Nothing in `tables/`, `paper_numbers.csv`
or the report is typed by hand.

## Contents

| Path | What it holds |
|---|---|
| `DriftGate_final_report_ko.md` | Final report (Korean): claim verdicts, pre-checks, results, captions, discrepancy log, run manifest |
| `paper_numbers.csv` | Every number meant for the paper: item, value, 95% CI, seed count, source runs |
| `tables/T1..T10*.csv` | Tables 1 to 10 (T2a/b/c, T4a/b/c/d and T6a/b/c are sub-tables) |
| `tables/lambda_trajectories.csv` | Per-round λ (cluster mean, then seed mean) behind Figure 3 |
| `tables/run_manifest.csv` | The 64 Round-5 runs: run_id, seed, arm, setting, completion, probe/eval overlap |
| `figures/` | Five figures as vector PDF and 300 dpi PNG, plus `figure_captions.md` |
| `precheck/` | Pre-check script and output (controller path identity), `code_identity.txt` (sha256 of the source files) |
| `scripts/` | Regeneration scripts |
| `scripts/launch/` | The queue, worker, monitor and supervisor scripts exactly as they ran on the old server (ubuntu20, old paths), and the exact enqueued command lines. Versions adapted to a new server live in `journal_expansion/scripts/` |

## Regenerate

Run from `journal_expansion/` in this order. The scripts read the raw run JSON files listed in `tables/run_manifest.csv` and the earlier runs named in the `source_runs` columns, so those files must be present.

```bash
python3 artifacts/driftgate_tmc_final/precheck/precheck.py
python3 artifacts/driftgate_tmc_final/scripts/r5_tables.py        # tables/*.csv and paper_numbers.csv
python3 artifacts/driftgate_tmc_final/scripts/r5_config_comm.py   # T8 and T9
python3 artifacts/driftgate_tmc_final/scripts/r5_manifest.py      # run_manifest.csv
python3 artifacts/driftgate_tmc_final/scripts/r5_figures.py       # figures/*.pdf and *.png
python3 artifacts/driftgate_tmc_final/scripts/r5_report.py        # DriftGate_final_report_ko.md
```

## Re-running the experiments

`scripts/launch/enqueue_r5.py` writes the 64 commands to `runs/queue_r5/{heavy,light}.txt`.
`scripts/launch/supervisor.sh` (called by cron every 10 minutes) starts `r5_worker.py` processes pinned to physical
GPU 0 (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`). Creating `runs/queue_r5/STOP` stops the workers
and makes the supervisor exit. The file exists now, so the cron entry does nothing.
