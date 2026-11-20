#!/bin/bash
# Round 18 (own-class detector): run outputs, analysis caches, data and exported models live on the 8T disk (user
# instruction 2026-11-20, decisions.md D9); the repository keeps links at the paths the scripts use. Idempotent.
S=/home/honeynaps/8T/2_Data/3_DataPreprocessed/driftgate_r18_own_class_261120   # [SERVER-PATH:R18OC_STORE]
cd /home/honeynaps/data/driftgate
mkdir -p $S/runs/phaseT18oc $S/analysis_cache $S/datasets/inat2021_r18 $S/mobile_export
link() {
  if [ -L "$2" ]; then echo "ok   $2 -> $(readlink $2)"; return; fi
  if [ -e "$2" ]; then echo "STOP $2 exists and is not a link; copy its contents to $1 first"; return 1; fi
  ln -s "$1" "$2" && echo "made $2 -> $1"
}
link $S/runs/phaseT18oc journal_expansion/runs/phaseT18oc
link $S/analysis_cache journal_expansion/artifacts/driftgate_tmc_r18_own_class_detector/cache
