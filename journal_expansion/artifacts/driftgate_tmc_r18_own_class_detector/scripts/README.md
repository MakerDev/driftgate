# Round 18 (own-class detector) scripts

Run from the repository root after `source ~/venvs/driftgate/bin/activate`.

| Script | What it does |
|---|---|
| `r18oc_make_env.py` | S1 environment files for the development seeds 8-10 in `runs/phaseT18oc_env/` (generator checked on S1 seed 5 first). |
| `r18oc_jobs.py` | Appends training and replay jobs to `runs/phaseT18oc/jobs.json` (`dev`, `eval`, `attempt`, `attempt_eval`, `stage3`, `list`). |
| `r18oc_dispatch.py` | Launches the jobs on GPUs 0-3 (memory-based placement, RAM floor, retries, STOP file). |

Code used by the runs: `src/r18oc_models.py` (MobileNetV2 split, vectors), `src/r18oc_detector.py` (detectors, validation
statistics, request scores) and the Round 18 options of `src/runner_r6.py` / `scripts/run_r6.py` (`--model_config`,
`--holdout_frac`, `--days`, `--detector`, `--detector_main`, `--checkpoint_at`, `--env_dir`; all off by default).
| `r18oc_analysis.py` | Per-run, per-condition analysis (`run <name> [conditions]` -> `cache/<run>__<condition>.pkl`) and the stage-1 checks (`check <smoke run path>` -> `tables/R18oc_stage1_checks.json`). |
| `r18oc_progress.sh` | Progress of the jobs (last round and ETA only; no accuracy values). |
| `r18oc_tables.py` | Design selection on the development seeds (`select <cfg>`), stage-2 cells and judgements (`stage2`), held-out effect (`holdout`), attempt gates and cells (`attempt_dev <n> <cfg>`, `attempt_cells <n> <cfg>`). |
| `r18oc_autoanalyse.sh` | Background loop: per-run analysis as soon as a training run and its replay have finished. |
| `r18oc_storage.sh` | Links `runs/phaseT18oc` and `cache` to the 8T disk folder (`/home/honeynaps/8T/2_Data/3_DataPreprocessed/driftgate_r18_own_class_261120`, decisions.md D9). |
| `r18oc_inat_prepare.py` | iNaturalist 2021 subset with real observation locations (five North American regions, 100 species; decisions.md D7). |
| `r18oc_diagnose.py` | Stage-2 failure diagnosis (`stage2` -> `tables/R18oc_stage2_diagnosis_{rules,kind,r}.csv`). |
| `r18oc_figures.py` | Accuracy vs offloading-ratio figures from the curve CSVs (`curves <csv> <prefix> [cfg ...]`). |
| `r18oc_report.py` | Markdown tables for the reports from `tables/` (`stage2`, `stage2_summary`, `stage3`, `times`, `times_summary`). |
| `r18oc_attempt_context.py` | Device only, edge only and envelope B(0.5) of the runs behind an attempt gate (`<n> <cfg>` -> `tables/R18oc_attempt<n>_<tag>_dev_context.csv`). |
| `r18oc_attempt3.py` | Attempt 3 `average-corrected-by-detector` (decisions.md D18): `run`, `check`, `dev`, `cells`. |
| `r18oc_attempt3_batch.sh` | Background batch of `r18oc_attempt3.py run` over the development and evaluation runs (one analysis worker at a time). |
| `r18oc_stage3.py` | Stage-3 tables and `decision_stage3.json` (`tables <main cfg> <other cfg>`, `check`). |

Storage: run outputs, analysis caches, the iNaturalist data and the exported mobile models are kept on the 8T disk
(user instruction 2026-11-20); the repository paths used by the scripts are links (`r18oc_storage.sh`).
