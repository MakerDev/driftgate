#!/bin/bash
# Round 5 supervisor. Restarts the Round-5 worker pool if it has died and there is queued work.
# Run by cron every 10 minutes (see MIGRATION/START_HERE.md, step 7).
# Old server (ubuntu20): GPU 0 ONLY, 3 heavy + 5 light workers (GPU 1 was reserved for the user).
# Migration (2026-09-29): GPUs and worker counts are variables. The defaults reproduce the old
# server. On the new server set them here (cron does not pass your shell environment).
# New server (honeynaps, 4x RTX 4090 24 GB): the user allowed GPUs 0-3. Under plan A (old server
# finishes Round 5) no R5 worker runs here; the values below only apply if the plan changes.
# The pre-R5 pool (queue_worker.sh / resnet_worker.sh) is intentionally NOT launched any more.
cd /home/honeynaps/data/driftgate  # [SERVER-PATH:REPO_ROOT]
GPUS="${R5_GPUS:-0 1 2 3}"           # space-separated physical GPU indices allowed for R5  # [SERVER-GPU]
NHEAVY="${R5_HEAVY_PER_GPU:-2}"      # heavy workers per GPU (Tiny-ImageNet ~3.8 GB, ResNet-18 ~2.5 GB)  # [SERVER-GPU]
NLIGHT="${R5_LIGHT_PER_GPU:-4}"      # light workers per GPU (CIFAR/SVHN CNN ~1.5 GB)  # [SERVER-GPU]
Q=journal_expansion/runs/queue_r5
[ -f "$Q/STOP" ] && exit 0
nq=$(( $(cat "$Q/heavy.txt" 2>/dev/null | grep -c .) + $(cat "$Q/light.txt" 2>/dev/null | grep -c .) ))
nw=$(ps -eo args | grep -c '[r]5_worker\.py')
[ "$nq" -eq 0 ] && exit 0
[ "$nw" -gt 0 ] && exit 0
launch(){ g=$1; shift; setsid nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID R5_GPU="$g" python3 "$@" >/dev/null 2>&1 < /dev/null & }
for g in $GPUS; do
  for i in $(seq 1 "$NHEAVY"); do launch "$g" journal_expansion/scripts/r5_worker.py heavy "g${g}h${i}"; done
  for i in $(seq 1 "$NLIGHT"); do launch "$g" journal_expansion/scripts/r5_worker.py light "g${g}l${i}"; done
done
echo "$(date) supervisor(R5, GPUs: $GPUS, ${NHEAVY}h+${NLIGHT}l per GPU) restarted workers (queue=$nq)" >> "$Q/logs/supervisor.log"
