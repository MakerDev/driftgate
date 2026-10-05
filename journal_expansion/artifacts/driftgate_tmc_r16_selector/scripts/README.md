# Round 16 stage 1 scripts

Run from the repository root (`/home/honeynaps/data/driftgate`) with `source ~/venvs/driftgate/bin/activate`.
Rules: `../decision_rule.md`; settings: `../r16_config.json` (read by the script; part of the cache key).

1. Facts about the labeled pools for `calibration_route.md` (no accuracy):
   `python journal_expansion/artifacts/driftgate_tmc_r16_selector/scripts/r16_calibration_pool_check.py` -> `tables/R16_C_pools.csv`
2. Unit checks of the new helpers (synthetic data):
   `OMP_NUM_THREADS=4 python .../scripts/r16_unit_checks.py`
3. Evaluation rows (evaluation rounds only): `python .../scripts/r16_stage1.py manifest` -> `evaluation_manifest.csv`
4. Phase A, reference answers and start check against the Round 15 cache (GPU for Logit-entropy weighting and Label-shift EM):
   `R16_WORKERS=3 python .../scripts/r16_stage1.py check` -> `tables/R16_T0_start_check.csv`, `cache/START_CHECK.json`
5. Phase B, grid, simulated calibration, selectors, oracles, online controller, and all tables (only after a passed check):
   `R16_WORKERS=3 python .../scripts/r16_stage1.py run` -> `tables/R16_*.csv`, `tables/R16_stage1_summary.json`

Records larger than 1 GB (K = 500, CIFAR-100) go to a separate single worker. `cache/` holds the per-run reference answers
(phase A) and per-run summaries (phase B) and is not committed.
