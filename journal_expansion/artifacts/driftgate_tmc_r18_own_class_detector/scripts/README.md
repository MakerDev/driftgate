# Round 18 (own-class detector) scripts

Run from the repository root after `source ~/venvs/driftgate/bin/activate`.

| Script | What it does |
|---|---|
| `r18oc_make_env.py` | S1 environment files for the development seeds 8-10 in `runs/phaseT18oc_env/` (generator checked on S1 seed 5 first). |
| `r18oc_jobs.py` | Appends training and replay jobs to `runs/phaseT18oc/jobs.json` (`dev`, `eval <cfg>=<variant> ...`, `list`). |
| `r18oc_dispatch.py` | Launches the jobs on GPUs 0-3 (memory-based placement, RAM floor, retries, STOP file). |

Code used by the runs: `src/r18oc_models.py` (MobileNetV2 split, vectors), `src/r18oc_detector.py` (detectors, validation
statistics, request scores) and the Round 18 options of `src/runner_r6.py` / `scripts/run_r6.py` (`--model_config`,
`--holdout_frac`, `--days`, `--detector`, `--detector_main`, `--checkpoint_at`, `--env_dir`; all off by default).
| `r18oc_analysis.py` | Per-run, per-condition analysis (`run <name> [conditions]` -> `cache/<run>__<condition>.pkl`) and the stage-1 checks (`check <smoke run path>` -> `tables/R18oc_stage1_checks.json`). |
| `r18oc_progress.sh` | Progress of the jobs (last round and ETA only; no accuracy values). |
