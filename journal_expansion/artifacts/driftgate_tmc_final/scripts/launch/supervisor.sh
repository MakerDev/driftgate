#!/bin/bash
# Round 5 supervisor — GPU 0 ONLY (physical nvidia-smi index 0, PCI order).
# Restarts the Round-5 worker pool if it has died and there is queued work.
# The pre-R5 pool (queue_worker.sh / resnet_worker.sh, which used cuda:1) is
# intentionally NOT launched any more. Original kept as supervisor.sh.bak_pre_r5_*.
cd /disk2/Yujin/adaptive_splitomc_tmc
Q=journal_expansion/runs/queue_r5
[ -f "$Q/STOP" ] && exit 0
nq=$(( $(cat "$Q/heavy.txt" 2>/dev/null | grep -c .) + $(cat "$Q/light.txt" 2>/dev/null | grep -c .) ))
nw=$(ps -eo args | grep -c '[r]5_worker\.py')
[ "$nq" -eq 0 ] && exit 0
[ "$nw" -gt 0 ] && exit 0
launch(){ setsid nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 "$@" >/dev/null 2>&1 < /dev/null & }
for i in 0 1 2; do launch journal_expansion/scripts/r5_worker.py heavy $i; done
for i in 10 11 12 13 14; do launch journal_expansion/scripts/r5_worker.py light $i; done
echo "$(date) supervisor(R5, GPU0 only) restarted workers (queue=$nq)" >> "$Q/logs/supervisor.log"
