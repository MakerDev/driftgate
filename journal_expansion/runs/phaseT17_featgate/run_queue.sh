#!/bin/bash
# usage: run_queue.sh GPU "S1:1 S2:2 ..."  (replay with features, then the stage-0 analysis, one run after another)
cd /home/honeynaps/data/driftgate
source ~/venvs/driftgate/bin/activate
export OMP_NUM_THREADS=6
S=journal_expansion/artifacts/driftgate_tmc_r17_featgate/scripts
for job in $2; do
  sc=${job%%:*}; seed=${job##*:}; name=r17_$(echo $sc | tr A-Z a-z)_replay_s$seed
  echo "$(date '+%F %T') start $name on GPU $1"
  python -u $S/r17_extract.py --scenario $sc --seed $seed --device cuda:$1 > journal_expansion/runs/phaseT17_featgate/logs/$name.log 2>&1 || { echo "$(date '+%F %T') extract FAILED $name"; continue; }
  CUDA_VISIBLE_DEVICES=$1 python -u $S/r17_stage0.py run $name > journal_expansion/runs/phaseT17_featgate/logs/${name}_stage0.log 2>&1 || { echo "$(date '+%F %T') analysis FAILED $name"; continue; }
  echo "$(date '+%F %T') done $name"
done
