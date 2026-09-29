#!/bin/bash
# Dedicated ResNet worker: expandable segments + RE-QUEUE on failure (transient
# OOM from fluctuating shared-machine load must not drop a job). Backs off when
# the job fails so it doesn't hot-loop against a full GPU.
DEV="$1"; WID="$2"
QDIR="/home/honeynaps/data/driftgate/journal_expansion/runs/resnet_queue"  # [SERVER-PATH:REPO_ROOT]
QFILE="$QDIR/queue.txt"; LOCK="$QDIR/queue.lock"; LOGDIR="$QDIR/logs"
mkdir -p "$LOGDIR"; cd /home/honeynaps/data/driftgate  # [SERVER-PATH:REPO_ROOT]
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ ! -f "$QDIR/STOP" ]; do
  JOB=$(flock "$LOCK" bash -c "head -n 1 '$QFILE' 2>/dev/null; sed -i '1d' '$QFILE' 2>/dev/null")
  [ -z "$JOB" ] && sleep 60 && continue
  TAG=$(echo "$JOB" | grep -o 'run_name [^ ]*' | cut -d' ' -f2)
  CMD=${JOB//\{DEV\}/$DEV}
  echo "[rw$WID $DEV] START $TAG $(date)" >> "$LOGDIR/rw$WID.log"
  eval "$CMD" >> "$LOGDIR/$TAG.log" 2>&1
  RC=$?
  echo "[rw$WID $DEV] END $TAG exit=$RC $(date)" >> "$LOGDIR/rw$WID.log"
  if [ $RC -ne 0 ] && [ ! -f "$QDIR/STOP" ]; then
    # re-queue at the tail and back off (likely transient OOM)
    flock "$LOCK" bash -c "echo '$JOB' >> '$QFILE'"
    echo "[rw$WID $DEV] REQUEUED $TAG (backoff 300s)" >> "$LOGDIR/rw$WID.log"
    sleep 300
  fi
done
