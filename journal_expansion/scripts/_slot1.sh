#!/bin/bash
cd /home/honeynaps/data/driftgate  # [SERVER-PATH:REPO_ROOT]
echo '=== START rec_A_s1' $(date)
python -u /home/honeynaps/data/driftgate/journal_expansion/scripts/run_v2.py --mode passive --schedule A --seed 1 --model_seed 101 --rounds 100 --probe_n 64 --run_name rec_A_s1 --output_dir /home/honeynaps/data/driftgate/journal_expansion/runs/signal_benchmark --device cuda:1 >> /home/honeynaps/data/driftgate/journal_expansion/runs/signal_benchmark/logs/rec_A_s1.log 2>&1  # [SERVER-PATH:REPO_ROOT]
echo '=== END rec_A_s1' $(date) exit=$?
echo '=== START rec_abrupt_s1' $(date)
python -u /home/honeynaps/data/driftgate/journal_expansion/scripts/run_v2.py --mode passive --schedule abrupt --seed 1 --model_seed 101 --rounds 100 --probe_n 64 --run_name rec_abrupt_s1 --output_dir /home/honeynaps/data/driftgate/journal_expansion/runs/signal_benchmark --device cuda:1 >> /home/honeynaps/data/driftgate/journal_expansion/runs/signal_benchmark/logs/rec_abrupt_s1.log 2>&1  # [SERVER-PATH:REPO_ROOT]
echo '=== END rec_abrupt_s1' $(date) exit=$?
echo '=== START rec_recurring_s1' $(date)
python -u /home/honeynaps/data/driftgate/journal_expansion/scripts/run_v2.py --mode passive --schedule recurring --seed 1 --model_seed 101 --rounds 100 --probe_n 64 --run_name rec_recurring_s1 --output_dir /home/honeynaps/data/driftgate/journal_expansion/runs/signal_benchmark --device cuda:1 >> /home/honeynaps/data/driftgate/journal_expansion/runs/signal_benchmark/logs/rec_recurring_s1.log 2>&1  # [SERVER-PATH:REPO_ROOT]
echo '=== END rec_recurring_s1' $(date) exit=$?
echo '=== START rec_burst_s1' $(date)
python -u /home/honeynaps/data/driftgate/journal_expansion/scripts/run_v2.py --mode passive --schedule burst --seed 1 --model_seed 101 --rounds 100 --probe_n 64 --run_name rec_burst_s1 --output_dir /home/honeynaps/data/driftgate/journal_expansion/runs/signal_benchmark --device cuda:1 >> /home/honeynaps/data/driftgate/journal_expansion/runs/signal_benchmark/logs/rec_burst_s1.log 2>&1  # [SERVER-PATH:REPO_ROOT]
echo '=== END rec_burst_s1' $(date) exit=$?
