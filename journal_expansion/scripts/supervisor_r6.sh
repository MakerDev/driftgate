#!/bin/bash
# Round 6 supervisor. Restarts the Round-6 worker pool if it has died and there is queued work.
# Run by cron every 10 minutes (cron does not pass your shell environment, so set values here).
# New server (honeynaps, 4x RTX 4090 24 GB): the user allowed GPUs 0-3.
cd /home/honeynaps/data/driftgate  # [SERVER-PATH:REPO_ROOT]
GPUS="${R6_GPUS:-0 1 2 3}"            # physical GPU indices allowed for R6  # [SERVER-GPU]
NPER="${R6_WORKERS_PER_GPU:-4}"       # worker slots per GPU  # [SERVER-GPU]
Q=journal_expansion/runs/queue_r6
[ -f "$Q/STOP" ] && exit 0
nq=$(grep -c . "$Q/queue.txt" 2>/dev/null || echo 0)
nw=$(ps -eo args | grep -c '[r]6_worker\.py')
[ "$nq" -eq 0 ] && exit 0
[ "$nw" -gt 0 ] && exit 0
# running markers are NOT cleared here: workers count only markers whose job is alive (r6_worker.py)
for g in $GPUS; do
  for i in $(seq 1 "$NPER"); do
    setsid nohup env R6_GPU="$g" python3 journal_expansion/scripts/r6_worker.py "g${g}w${i}" >/dev/null 2>&1 < /dev/null &
  done
done
echo "$(date) supervisor(R6, GPUs: $GPUS, $NPER per GPU) restarted workers (queue=$nq)" >> "$Q/logs/supervisor.log"
