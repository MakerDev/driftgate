#!/bin/bash
cd /disk2/Yujin/adaptive_splitomc_tmc  # [SERVER-PATH:REPO_ROOT]
echo '=== START rec_A_s0' $(date)
python -u /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/scripts/run_v2.py --mode passive --schedule A --seed 0 --model_seed 100 --rounds 100 --probe_n 64 --run_name rec_A_s0 --output_dir /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/signal_benchmark --device cuda:0 >> /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/signal_benchmark/logs/rec_A_s0.log 2>&1  # [SERVER-PATH:REPO_ROOT]
echo '=== END rec_A_s0' $(date) exit=$?
echo '=== START rec_abrupt_s0' $(date)
python -u /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/scripts/run_v2.py --mode passive --schedule abrupt --seed 0 --model_seed 100 --rounds 100 --probe_n 64 --run_name rec_abrupt_s0 --output_dir /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/signal_benchmark --device cuda:0 >> /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/signal_benchmark/logs/rec_abrupt_s0.log 2>&1  # [SERVER-PATH:REPO_ROOT]
echo '=== END rec_abrupt_s0' $(date) exit=$?
echo '=== START rec_recurring_s0' $(date)
python -u /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/scripts/run_v2.py --mode passive --schedule recurring --seed 0 --model_seed 100 --rounds 100 --probe_n 64 --run_name rec_recurring_s0 --output_dir /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/signal_benchmark --device cuda:0 >> /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/signal_benchmark/logs/rec_recurring_s0.log 2>&1  # [SERVER-PATH:REPO_ROOT]
echo '=== END rec_recurring_s0' $(date) exit=$?
echo '=== START rec_burst_s0' $(date)
python -u /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/scripts/run_v2.py --mode passive --schedule burst --seed 0 --model_seed 100 --rounds 100 --probe_n 64 --run_name rec_burst_s0 --output_dir /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/signal_benchmark --device cuda:0 >> /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/signal_benchmark/logs/rec_burst_s0.log 2>&1  # [SERVER-PATH:REPO_ROOT]
echo '=== END rec_burst_s0' $(date) exit=$?
