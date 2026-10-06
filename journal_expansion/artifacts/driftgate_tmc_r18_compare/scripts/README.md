# Round 18 scripts

Run from the repository root (`/home/honeynaps/data/driftgate`) with `source ~/venvs/driftgate/bin/activate`.
Definitions and decision rules: `../R18_plan.md` (committed before any result, 45d250d).

Inputs (no new training): Round 12 / 13b first-day records, Round 15 frozen replays, Round 16 phase A cache of the
reference answers (`driftgate_tmc_r16_selector/cache/*__refs-v1.npz`), Round 15 cache (curves, A3), Round 17 replay
records with features (`runs/phaseT17_featgate`), Round 15 end-of-day checkpoints (BTFL training-set entropies).

1. Per-run analysis (CPU; K = 500 about 7 minutes and 13 GB):
   `python journal_expansion/artifacts/driftgate_tmc_r18_compare/scripts/r18_analysis.py run "S1" 0`
   (settings: S1, S2, S1-fast, partial participation, stepwise change, random mobility, CIFAR-100, ResNet-18, K=200, K=500,
   S1 replay, S2 replay) -> `cache/<run>.pkl`
2. Detector diagnostics on the replays (GPU for k-NN): `python .../r18_analysis.py features "S1 replay" 0` -> `cache/<run>__feat.pkl`
3. Tables: `python .../r18_analysis.py tables` -> `tables/R18_*.csv`, `tables/R18_conditional_decision.json`
4. Summary of the same-budget table: `python .../r18_summary.py` -> `tables/R18_online_same_budget_summary.csv`
5. Audit of the Round 17 comparison: `python .../r18_audit.py` -> `tables/R18_R17_corrected_margins.csv`, `tables/R18_R17_seed_sets.csv`
6. BTFL inference adaptation (GPU forward pass of the frozen checkpoints): `python .../r18_btfl.py S1 0` ... then
   `python .../r18_btfl.py table` -> `tables/R18_BTFL_adaptation.csv`
The queue scripts used here are in `cache/` (not committed): three CPU queues for step 1, one GPU queue for step 2.
