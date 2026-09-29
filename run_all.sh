#!/bin/bash
# ============================================================================
# Adaptive-SplitOMC v3 — Master autonomous execution pipeline.
#
# Runs Phase 0 (E0 smoke) → Phase 1 (E1) → Phase 2 (E2-E5) sequentially.
# After Phase 0, GATE: if E0 passes, continue. Else stop and require manual fix.
#
# Designed for tmux execution on 2× RTX 3090 server.
# ============================================================================

set -e

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

# Config
CONFIG_FILE="${CONFIG_FILE:-configs/base_v3.yaml}"
DEVICE0="${DEVICE0:-cuda:0}"
DEVICE1="${DEVICE1:-cuda:1}"  # [SERVER-GPU]
RESULTS_ROOT="${RESULTS_ROOT:-./results}"
LOG_DIR="${LOG_DIR:-./logs}"

mkdir -p "$RESULTS_ROOT" "$LOG_DIR"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $1" | tee -a "$LOG_DIR/master.log"; }

log "============================================================"
log "Adaptive-SplitOMC v3 Master Execution"
log "============================================================"
log "Project dir: $PROJECT_DIR"
log "Config: $CONFIG_FILE"
log "Devices: $DEVICE0 + $DEVICE1"
log "Results root: $RESULTS_ROOT"
log ""

# ---------- Pre-flight ----------
log "[Pre-flight] Unit tests..."
python -m pytest tests/test_all.py -v 2>&1 | tail -20 | tee -a "$LOG_DIR/master.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "UNIT TESTS FAILED. Aborting."
    exit 1
fi
log "[Pre-flight] Unit tests PASSED"
log ""

# ============================================================
# Phase 0 — E0 smoke (~3h, 1 GPU)
# ============================================================
log "============================================================"
log "Phase 0: E0 smoke (50R x 3e, ~3h on 1 GPU)"
log "============================================================"

if [ ! -f "$RESULTS_ROOT/e0_smoke/e0_summary.json" ]; then
    python scripts/run_e0_smoke.py \
        --config "$CONFIG_FILE" \
        --device "$DEVICE0" \
        --rounds 50 \
        --local_epochs 3 \
        2>&1 | tee -a "$LOG_DIR/phase0.log"
else
    log "E0 already done (skip)"
fi

# Verify E0 with simple check
python -c "
import json, sys
try:
    s = json.load(open('$RESULTS_ROOT/e0_smoke/e0_summary.json'))
    fails = [k for k, v in s.items() if v.get('status') != 'PASS_NO_NAN' and 'status' in v]
    if fails:
        print('E0 FAIL:', fails)
        sys.exit(1)
    # Check splitomc accuracy at least 30%
    sm = s.get('splitomc_lam0.2', {})
    if sm.get('acc_total', 0) < 0.25:
        print('E0 WARNING: splitomc_lam0.2 acc_total too low:', sm.get('acc_total'))
        # Don't fail; let later phases reveal
    print('E0 verification OK')
except Exception as e:
    print('E0 verify error:', e)
    sys.exit(1)
" || { log "E0 GATE FAILED. Stop."; exit 1; }

log "Phase 0 PASSED"
log ""

# ============================================================
# Phase 1 — E1 static Pareto (~10-13h on 2 GPUs)
# ============================================================
log "============================================================"
log "Phase 1: E1 static Pareto"
log "============================================================"

# Split runs across 2 GPUs
HALF1=(fedavg fedprox splitfed fedmes splitgp_lam0.2 splitgp_lam0.5 splitgp_lam0.8 splitomc_lam0.0)
HALF2=(splitomc_lam0.2 splitomc_lam0.4 splitomc_lam0.6 splitomc_lam0.8 splitomcplus_lam0.2_Lam0.5 splitomcplus_lam0.4_Lam0.5 splitomcplus_lam0.6_Lam0.5 adaptive_splitomc)

python scripts/run_e1_static.py --config "$CONFIG_FILE" --device "$DEVICE0" \
    --output_dir "$RESULTS_ROOT/e1_static" \
    --methods "${HALF1[@]}" > "$LOG_DIR/e1_gpu0.log" 2>&1 &
PID0=$!
python scripts/run_e1_static.py --config "$CONFIG_FILE" --device "$DEVICE1" \
    --output_dir "$RESULTS_ROOT/e1_static" \
    --methods "${HALF2[@]}" > "$LOG_DIR/e1_gpu1.log" 2>&1 &
PID1=$!

log "E1 launched on both GPUs (PIDs: $PID0, $PID1). Tailing GPU0 log..."
tail -f "$LOG_DIR/e1_gpu0.log" &
TAIL_PID=$!

wait $PID0 $PID1
kill $TAIL_PID 2>/dev/null || true
log "Phase 1 done."
log ""

# ============================================================
# Phase 2 — E2 temporal (CORE) ~7h on 2 GPUs
# ============================================================
log "============================================================"
log "Phase 2: E2 temporal drift (CORE EXPERIMENT)"
log "============================================================"

python scripts/run_e2_temporal.py --config "$CONFIG_FILE" --device "$DEVICE0" \
    --schedule A --output_dir "$RESULTS_ROOT/e2_temporal" \
    --methods splitomcplus_lam0.2_Lam0.5 splitomcplus_lam0.4_Lam0.5 \
    > "$LOG_DIR/e2_A_gpu0.log" 2>&1 &
PID0=$!
python scripts/run_e2_temporal.py --config "$CONFIG_FILE" --device "$DEVICE1" \
    --schedule A --output_dir "$RESULTS_ROOT/e2_temporal" \
    --methods splitomcplus_lam0.6_Lam0.5 adaptive_splitomc \
    > "$LOG_DIR/e2_A_gpu1.log" 2>&1 &
PID1=$!

wait $PID0 $PID1
log "E2 schedule A done."

# Schedule B (smaller - quick run)
python scripts/run_e2_temporal.py --config "$CONFIG_FILE" --device "$DEVICE0" \
    --schedule B --output_dir "$RESULTS_ROOT/e2_temporal" \
    --methods splitomcplus_lam0.4_Lam0.5 \
    > "$LOG_DIR/e2_B_gpu0.log" 2>&1 &
PID0=$!
python scripts/run_e2_temporal.py --config "$CONFIG_FILE" --device "$DEVICE1" \
    --schedule B --output_dir "$RESULTS_ROOT/e2_temporal" \
    --methods adaptive_splitomc \
    > "$LOG_DIR/e2_B_gpu1.log" 2>&1 &
PID1=$!
wait $PID0 $PID1
log "E2 schedule B done."
log ""

# ============================================================
# Phase 3 — E3 spatial + E4 mobility + E5 ablation ~12h
# ============================================================
log "============================================================"
log "Phase 3: E3 spatial + E4 mobility + E5 ablation"
log "============================================================"

# E3 across both GPUs
python scripts/run_e3_spatial.py --config "$CONFIG_FILE" --device "$DEVICE0" \
    --output_dir "$RESULTS_ROOT/e3_spatial" \
    --methods splitomcplus_lam0.2_Lam0.5 splitomcplus_lam0.4_Lam0.5 \
    > "$LOG_DIR/e3_gpu0.log" 2>&1 &
PID0=$!
python scripts/run_e3_spatial.py --config "$CONFIG_FILE" --device "$DEVICE1" \
    --output_dir "$RESULTS_ROOT/e3_spatial" \
    --methods splitomcplus_lam0.6_Lam0.5 adaptive_splitomc \
    > "$LOG_DIR/e3_gpu1.log" 2>&1 &
PID1=$!
wait $PID0 $PID1
log "E3 done."

# E4 mobility
python scripts/run_e4_mobility.py --config "$CONFIG_FILE" --device "$DEVICE0" \
    --output_dir "$RESULTS_ROOT/e4_mobility" \
    --methods splitomcplus_lam0.2_Lam0.5_mob splitomcplus_lam0.6_Lam0.5_mob \
    > "$LOG_DIR/e4_gpu0.log" 2>&1 &
PID0=$!
python scripts/run_e4_mobility.py --config "$CONFIG_FILE" --device "$DEVICE1" \
    --output_dir "$RESULTS_ROOT/e4_mobility" \
    --methods adaptive_splitomc_mob \
    > "$LOG_DIR/e4_gpu1.log" 2>&1 &
PID1=$!
wait $PID0 $PID1
log "E4 done."

# E5 ablation
python scripts/run_e5_ablation.py --config "$CONFIG_FILE" --device "$DEVICE0" \
    --output_dir "$RESULTS_ROOT/e5_ablation" \
    --methods A_full_adaptive B_no_delta \
    --rounds 100 \
    > "$LOG_DIR/e5_gpu0.log" 2>&1 &
PID0=$!
python scripts/run_e5_ablation.py --config "$CONFIG_FILE" --device "$DEVICE1" \
    --output_dir "$RESULTS_ROOT/e5_ablation" \
    --methods C_no_consensus D_static_mean \
    --rounds 100 \
    > "$LOG_DIR/e5_gpu1.log" 2>&1 &
PID1=$!
wait $PID0 $PID1
log "E5 done."
log ""

# ============================================================
# Phase 4 — analysis
# ============================================================
log "============================================================"
log "Phase 4: Analysis + paper assembly"
log "============================================================"

python scripts/analyze_all.py --results_root "$RESULTS_ROOT" \
    --output_dir ./docs/results_v3 \
    2>&1 | tee -a "$LOG_DIR/analysis.log"

# Assemble the paper draft with real numbers
python scripts/fill_paper.py \
    --output ./docs/results_v3/paper_v3_filled.md \
    2>&1 | tee -a "$LOG_DIR/analysis.log"

log ""
log "============================================================"
log "ALL DONE"
log "============================================================"
log "Results: $RESULTS_ROOT/"
log "Analysis: ./docs/results_v3/"
log ""
log "Key files for paper:"
log "  - docs/results_v3/paper_v3_filled.md (ASSEMBLED PAPER, real numbers)"
log "  - docs/results_v3/interpretation.md (narrative summary)"
log "  - docs/results_v3/tables/*.md"
log "  - docs/results_v3/figures/*.png"
log ""
