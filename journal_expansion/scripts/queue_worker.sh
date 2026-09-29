#!/bin/bash
# File-queue worker: pops one job line at a time from queue.txt (flock-atomic),
# substitutes {DEV} with this worker's device, runs it, repeats. Sleeps when
# the queue is empty; exits only when runs/queue/STOP exists.
# Usage: queue_worker.sh <device> <worker_id>
DEV="$1"
WID="$2"
QDIR="/disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/queue"  # [SERVER-PATH:REPO_ROOT]
QFILE="$QDIR/queue.txt"
LOCK="$QDIR/queue.lock"
LOGDIR="$QDIR/logs"
mkdir -p "$LOGDIR"
cd /disk2/Yujin/adaptive_splitomc_tmc  # [SERVER-PATH:REPO_ROOT]

while [ ! -f "$QDIR/STOP" ]; do
  JOB=$(flock "$LOCK" bash -c "head -n 1 '$QFILE' 2>/dev/null; sed -i '1d' '$QFILE' 2>/dev/null")
  if [ -z "$JOB" ]; then
    sleep 60
    continue
  fi
  TAG=$(echo "$JOB" | grep -o 'run_name [^ ]*' | cut -d' ' -f2)
  [ -z "$TAG" ] && TAG="job_$(date +%s)"
  CMD=${JOB//\{DEV\}/$DEV}
  echo "[worker$WID $DEV] START $TAG $(date)" >> "$LOGDIR/worker$WID.log"
  eval "$CMD" >> "$LOGDIR/$TAG.log" 2>&1
  echo "[worker$WID $DEV] END $TAG exit=$? $(date)" >> "$LOGDIR/worker$WID.log"
done
