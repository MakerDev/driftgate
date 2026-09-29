#!/bin/bash
cd /disk2/Yujin/adaptive_splitomc_tmc  # [SERVER-PATH:REPO_ROOT]
echo '=== START sc_delta_A_s2' $(date)
python -u /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/scripts/run_v2.py --mode selfcal --signal delta_hard --schedule A --seed 2 --model_seed 102 --rounds 100 --probe_n 64 --run_name sc_delta_A_s2 --output_dir /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/downstream_pilot --device cuda:1 >> /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/downstream_pilot/logs/sc_delta_A_s2.log 2>&1  # [SERVER-PATH:REPO_ROOT]
echo '=== END sc_delta_A_s2' $(date) exit=$?
echo '=== START legacy_delta_A_s0' $(date)
python -u /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/scripts/run_v2.py --mode legacy --signal delta_hard --schedule A --seed 0 --model_seed 100 --rounds 100 --probe_n 64 --run_name legacy_delta_A_s0 --output_dir /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/downstream_pilot --device cuda:1 >> /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/downstream_pilot/logs/legacy_delta_A_s0.log 2>&1  # [SERVER-PATH:REPO_ROOT]
echo '=== END legacy_delta_A_s0' $(date) exit=$?
echo '=== START sc_delta_abrupt_s1' $(date)
python -u /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/scripts/run_v2.py --mode selfcal --signal delta_hard --schedule abrupt --seed 1 --model_seed 101 --rounds 100 --probe_n 64 --run_name sc_delta_abrupt_s1 --output_dir /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/downstream_pilot --device cuda:1 >> /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/downstream_pilot/logs/sc_delta_abrupt_s1.log 2>&1  # [SERVER-PATH:REPO_ROOT]
echo '=== END sc_delta_abrupt_s1' $(date) exit=$?
echo '=== START sc_ent_abrupt_s2' $(date)
python -u /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/scripts/run_v2.py --mode selfcal --signal ent_client --schedule abrupt --seed 2 --model_seed 102 --rounds 100 --probe_n 64 --run_name sc_ent_abrupt_s2 --output_dir /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/downstream_pilot --device cuda:1 >> /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/downstream_pilot/logs/sc_ent_abrupt_s2.log 2>&1  # [SERVER-PATH:REPO_ROOT]
echo '=== END sc_ent_abrupt_s2' $(date) exit=$?
