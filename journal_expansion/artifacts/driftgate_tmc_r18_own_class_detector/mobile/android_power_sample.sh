#!/bin/bash
# Round 18, stage 4: battery voltage and current samples on an Android phone while a TFLite model runs (adb, USB).
# Usage: bash android_power_sample.sh MODEL_TFLITE SECONDS OUT_PREFIX [THREADS]
# 1. idle: samples for SECONDS with nothing running; 2. load: benchmark_model runs MODEL for SECONDS while sampling.
# The battery files differ between phones; check that both paths exist with `adb shell ls /sys/class/power_supply/battery/`.
# Units are device-specific (often microvolts and microamperes; Samsung phones may report mA). State them in the JSON.
# USB charging distorts the current: disable charging (e.g. `adb shell dumpsys battery unplug`) or use a power monitor.
set -e
MODEL=$1; SECS=$2; OUT=$3; THR=${4:-4}
V=/sys/class/power_supply/battery/voltage_now
I=/sys/class/power_supply/battery/current_now
sample() {  # $1 seconds, $2 file
  adb shell "end=\$((\$(date +%s)+$1)); while [ \$(date +%s) -lt \$end ]; do echo \$(date +%s%N) \$(cat $V) \$(cat $I); sleep 0.1; done" > $2
}
sample $SECS ${OUT}_idle.txt
adb shell "/data/local/tmp/benchmark_model --graph=/data/local/tmp/$(basename $MODEL) --num_threads=$THR --warmup_runs=50 --num_runs=1000000 --min_secs=$SECS --max_secs=$SECS" > ${OUT}_bench.log 2>&1 &
BP=$!
sample $SECS ${OUT}_load.txt
wait $BP || true
echo "samples: ${OUT}_idle.txt ${OUT}_load.txt; runs and timing: ${OUT}_bench.log"
echo "energy per request = (mean V*I under load - mean V*I idle) x SECS / number of runs (see README_mobile_ko.md)"
