# Round 20 scripts

Run from the repository root with `source ~/venvs/driftgate/bin/activate`. No training and no new inference.

| Script | What it does |
|---|---|
| `r20_recompute.py run SETTING SEED` | Recomputes the inference rules from the stored request outputs of one run (beta 0.25/0.5/0.75/1 with the Round 18 controller, beta 0 = device only; confidence-based offloading at 0.8 nats; Round 16 refs for the learned and EM rules at beta 1; Task 3 correction ablation for S1, S2 and their frozen replays). Stores correct counts per evaluation round and device, costs and reproduction checks against Rounds 15, 16, 18 and 19 in `cache/<run>.pkl`. |
| `r20_recompute.py a0` | Median a_k over present device-rounds of the S1 development seeds 5-7 -> `cache/a0.json` (run after the three development runs, before the Task 3 settings). |
| `r20_tables.py` | All Round 20 tables in `tables/` (checks, comparator sets, reconciliation values, Task 1-3). |
| `r20_report_tables.py` | Markdown tables for the final report (`tables/R20_report_tables.md`), read only from `tables/R20_*.csv`. |
| `r20_sources.py` | Source manifest `tables/R20_sources.csv` (size, sha256, last commit or git status of every file read). |

Queues used: `cache/queue_small.sh` (development seeds first, then `a0`, then the other settings) and `cache/queue_big.sh` (K=200, K=500), each started with `setsid nohup ... < /dev/null &`.

Settings and runs are those of Round 18 (`r18_analysis.SETTINGS`) plus `S1 development` (`runs/phaseT10_prior/r10_T40_s{5,6,7}`) and the Round 19 three-day replays (`runs/phaseT19_multiday`, combined per seed with `r19_multiday.combine`).
