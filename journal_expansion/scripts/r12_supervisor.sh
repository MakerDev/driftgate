#!/bin/bash
# Round 12: restart the dispatcher (journal_expansion/scripts/r12_dispatch.py) if it is not running.
# cron: */10 * * * * /home/honeynaps/data/driftgate/journal_expansion/scripts/r12_supervisor.sh
ROOT=/home/honeynaps/data/driftgate  # [SERVER-PATH:REPO_ROOT]
OUT=$ROOT/journal_expansion/runs/phaseT12_fusion
[ -f "$OUT/STOP" ] && exit 0
[ -f "$OUT/ALL_DONE" ] && exit 0
if ! pgrep -f "^python -u $ROOT/journal_expansion/scripts/r12_dispatch.py" > /dev/null; then
  cd $ROOT
  source /home/honeynaps/venvs/driftgate/bin/activate
  echo "$(date '+%F %T') supervisor: starting dispatcher" >> $OUT/supervisor.log
  setsid nohup python -u $ROOT/journal_expansion/scripts/r12_dispatch.py < /dev/null >> $OUT/dispatch.out 2>&1 &
fi
