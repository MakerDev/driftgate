#!/bin/bash
# Round 18 progress: per job the last round reached and the ETA printed by the runner (no accuracy values).
OUT=/home/honeynaps/data/driftgate/journal_expansion/runs/phaseT18oc
for f in $OUT/logs/*.log; do
  n=$(basename $f .log)
  if [ -f $OUT/$n.json ]; then echo "$n done"; continue; fi
  l=$(grep -o "R *[0-9]* .*elapsed=[0-9.]*m ETA=[0-9]*m" $f | tail -n 1 | sed -E 's/ loss=.*active=[0-9]+//')
  err=$(grep -c "Traceback" $f)
  echo "$n ${l:-starting} errors=$err"
done
tail -n 2 $OUT/dispatch.log
