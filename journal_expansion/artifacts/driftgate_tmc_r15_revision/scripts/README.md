# Round 15 scripts

Run from the repository root (`/home/honeynaps/data/driftgate`) with `source ~/venvs/driftgate/bin/activate`.

1. A2 runs (retrained first day with end-of-day checkpoints, then frozen-model replays), GPUs 0-3:
   `setsid nohup python -u journal_expansion/scripts/r15_dispatch.py < /dev/null > journal_expansion/runs/phaseT15_replay/dispatch.out 2>&1 &`
   (runner flags: `--save_checkpoint`, `--record_eval_logprobs`, `--replay_from <ckpt>` in `journal_expansion/scripts/run_r6.py`)
2. Analysis (all settings; per-run summaries are cached in `cache/`, about 75 min with K = 500):
   `R15_WORKERS=3 python journal_expansion/artifacts/driftgate_tmc_r15_revision/scripts/r15_analysis.py`
   -> `tables/R15_*.csv`, `figures/R15_fig_replay_time_of_day.*`
3. Derived table (from the cache): `python .../scripts/r15_derived_tables.py` -> `tables/R15_E_scale_reach.csv`
4. Paper figures (from the cache): `python .../scripts/r15_paper_figures.py` -> `figures/paper/*`
5. Mobile cost figure from an input JSON (schema: `mobile/mobile_cost_schema.json`):
   `python .../scripts/r15_mobile_cost.py <input.json> <out_prefix>`; demo: `mobile/demo/mobile_cost_demo.json` -> `figures/demo/`
6. Manuscript: `cd manuscript/v28 && ~/.local/opt/tectonic/tectonic -X compile main.tex`
