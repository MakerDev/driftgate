#!/bin/bash
# Usage: timed_worker.sh <device> <id> <deadline_epoch>
# Pulls jobs until deadline_epoch, then exits (in-flight job finishes naturally).
# Re-queues a job on non-zero exit so an end-of-window kill loses nothing.
DEV="$1"; WID="$2"; DEADLINE="$3"
QDIR="/home/honeynaps/data/driftgate/journal_expansion/runs/queue"  # [SERVER-PATH:REPO_ROOT]
QFILE="$QDIR/queue.txt"; LOCK="$QDIR/queue.lock"; LOGDIR="$QDIR/logs"
mkdir -p "$LOGDIR"; cd /home/honeynaps/data/driftgate  # [SERVER-PATH:REPO_ROOT]
export JX_THREADS=4
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  JOB=$(flock "$LOCK" bash -c "head -n 1 '$QFILE' 2>/dev/null; sed -i '1d' '$QFILE' 2>/dev/null")
  [ -z "$JOB" ] && sleep 30 && continue
  TAG=$(echo "$JOB" | grep -o 'run_name [^ ]*' | cut -d' ' -f2)
  CMD=${JOB//\{DEV\}/$DEV}
  echo "[tw$WID $DEV] START $TAG $(date)" >> "$LOGDIR/tw$WID.log"
  eval "$CMD" >> "$LOGDIR/$TAG.log" 2>&1
  RC=$?
  echo "[tw$WID $DEV] END $TAG exit=$RC $(date)" >> "$LOGDIR/tw$WID.log"
  if [ $RC -ne 0 ]; then flock "$LOCK" bash -c "echo '$JOB' >> '$QFILE'"; fi
done
echo "[tw$WID $DEV] DEADLINE reached, stopping new pulls $(date)" >> "$LOGDIR/tw$WID.log"
