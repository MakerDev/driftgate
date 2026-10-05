# Round 17 stage 0 scripts

Run from the repository root (`/home/honeynaps/data/driftgate`) with `source ~/venvs/driftgate/bin/activate`.
Rules: `../decision_rule.md`; settings: `../r17_config.json`.

1. Replay of a Round 15 checkpoint with feature records (GPU; about 2 minutes per run):
   `python journal_expansion/artifacts/driftgate_tmc_r17_featgate/scripts/r17_extract.py --scenario S1 --seed 0 --device cuda:0`
   -> `journal_expansion/runs/phaseT17_featgate/r17_s1_replay_s0{.json,_trace.npz,_evalprobs.npz,_features.npz,_own.npz,_extract.json}`
2. Per-run analysis (GPU for Logit-entropy weighting, Label-shift EM and k-NN; about 5 minutes per run):
   `python .../scripts/r17_stage0.py run r17_s1_replay_s0` -> `cache/r17_s1_replay_s0.pkl`
3. Tables and decision: `python .../scripts/r17_stage0.py tables` -> `tables/R17_*.csv`, `decision_stage0.json`
