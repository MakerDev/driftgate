#!/bin/bash
# Round 18: attempt-3 caches (r18oc_attempt3.py run; prints only the reproduction check, no attempt-3 accuracy) for the
# S1 development runs (frozen-replay) and the stage-2 evaluation runs (S1, S2 seeds 0-2; second-day, frozen-replay) of the
# two pretrained configurations. One analysis worker at a time: waits while the per-run analysis (autoanalyse) runs.
cd /home/honeynaps/data/driftgate
source ~/venvs/driftgate/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
A=journal_expansion/artifacts/driftgate_tmc_r18_own_class_detector
LOG=$A/cache/attempt3_batch.log
wait_free() { while pgrep -f "r18oc_analysis.py run" > /dev/null; do sleep 60; done; }
jobs=()
for t in ft fz; do for s in 8 9 10; do jobs+=("r18oc_${t}_s1_s$s frozen-replay"); done; done
for t in ft fz; do for st in s1 s2; do for s in 0 1 2; do jobs+=("r18oc_${t}_${st}_s$s second-day frozen-replay"); done; done; done
for j in "${jobs[@]}"; do
  wait_free
  echo "$(date '+%F %T') attempt3 run $j" >> $LOG
  python $A/scripts/r18oc_attempt3.py run $j >> $LOG 2>&1 || echo "$(date '+%F %T') FAILED $j" >> $LOG
done
echo "$(date '+%F %T') attempt3 batch done" >> $LOG
