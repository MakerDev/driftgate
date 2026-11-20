#!/bin/bash
# Round 18: run the per-run analysis (one run at a time) for every training job whose training and replay results exist.
# Exits when the dispatcher wrote ALL_DONE and every training job has its three condition caches.
cd /home/honeynaps/data/driftgate
source ~/venvs/driftgate/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
A=journal_expansion/artifacts/driftgate_tmc_r18_own_class_detector
OUT=journal_expansion/runs/phaseT18oc
while true; do
  pending=0
  for n in $(python -c "import json; print(' '.join(j['name'] for j in json.load(open('$OUT/jobs.json')) if j['kind']=='train'))"); do
    r=$(echo $n | sed -E 's/_s([0-9]+)$/_replay_s\1/')
    miss=0
    for c in first-day second-day frozen-replay; do [ -f $A/cache/${n}__$c.pkl ] || miss=1; done
    [ $miss = 0 ] && continue
    pending=1
    if [ -f $OUT/$n.json ] && [ -f $OUT/$r.json ]; then
      echo "$(date '+%F %T') analyse $n" >> $A/cache/autoanalyse.log
      python $A/scripts/r18oc_analysis.py run $n >> $A/cache/autoanalyse.log 2>&1
    fi
  done
  if [ $pending = 0 ] && [ -f $OUT/ALL_DONE ]; then echo "$(date '+%F %T') all analysed" >> $A/cache/autoanalyse.log; break; fi
  sleep 120
done
