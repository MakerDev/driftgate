#!/bin/bash
# usage: queue.sh GPU "S1:0:1 S1:0:2 ..."
cd /home/honeynaps/data/driftgate
source ~/venvs/driftgate/bin/activate
export OMP_NUM_THREADS=4
for j in $2; do
  IFS=':' read sc sd dy <<< "$j"
  python -u journal_expansion/artifacts/driftgate_tmc_r19_weights/scripts/r19_multiday_replay.py $sc $sd $dy cuda:$1 > journal_expansion/runs/phaseT19_multiday/logs/r19_${sc}_s${sd}_day${dy}.log 2>&1 && echo "done $j" || echo "FAILED $j"
done
echo "queue finished"
