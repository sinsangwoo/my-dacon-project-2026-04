#!/bin/bash

# ==============================================================================
# [PIPELINE: SINGLE-SOURCE-OF-TRUTH ARCHITECTURE]
#
# Philosophy: "Run once, wake up to submission.csv"
# Execution: ./pipeline.sh [--debug]
# ==============================================================================

set -euo pipefail

DEBUG_MODE=${1:-""}
if [ "$DEBUG_MODE" = "--debug" ]; then
    PYTHON_MODE="debug"
    echo "[DEBUG MODE] Activated"
else
    PYTHON_MODE="full"
    echo "[FULL MODE] Activated"
fi

SMOKE_FLAG=""
if [ "${2:-""}" = "--smoke-test" ] || [ "${1:-""}" = "--smoke-test" ]; then
    SMOKE_FLAG="--smoke-test"
    echo "[SMOKE TEST MODE] Activated"
fi

run_pipeline() {
    export RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
    LOG_DIR="logs/$RUN_ID"
    PIPELINE_LOG="$LOG_DIR/pipeline.log"
    mkdir -p "$LOG_DIR"
    
    # [DETERMINISM_FIX] Safe sync wrapper
    safe_sync() {
        if command -v sync >/dev/null 2>&1; then
            sync && echo "[SYNC] Disk buffers flushed." || echo "[SYNC] Sync failed but continuing..."
        else
            echo "[SYNC] command not found, skipping."
        fi
    }

    echo "[PRE-CLEAN] Starting..." | tee -a "$PIPELINE_LOG"
    rm -rf logs/latest .done/latest 2>/dev/null || true
    mkdir -p ".done/$RUN_ID"
    echo "[PRE-CLEAN] Complete." | tee -a "$PIPELINE_LOG"

    echo "============================================================" | tee -a "$PIPELINE_LOG"
    echo "  DACON SUBMISSION PIPELINE - [RUN_ID: $RUN_ID]" | tee -a "$PIPELINE_LOG"
    echo "  Start Time: $(date)" | tee -a "$PIPELINE_LOG"
    echo "============================================================" | tee -a "$PIPELINE_LOG"

    # [PHASE 2: PIPELINE AUTO-SYNC]
    echo "[PIPELINE_SYNC] Fetching phase list from main.py..." | tee -a "$PIPELINE_LOG"
    
    PHASES_STR=$(python -c "from main import VALID_PHASES; print(' '.join(VALID_PHASES))")
    read -a PHASES <<< "$PHASES_STR"

    if [ ${#PHASES[@]} -eq 0 ]; then
        echo "[FATAL] Failed to sync phases from main.py or list is empty." | tee -a "$PIPELINE_LOG"
        exit 1
    fi

    echo "[PIPELINE_PHASE_LIST] ${PHASES[*]}" | tee -a "$PIPELINE_LOG"

    for phase in "${PHASES[@]}"; do
        DONE_MARKER=".done/$RUN_ID/$phase"
        if [ -f "$DONE_MARKER" ]; then
            echo "[SKIPPING] Phase $phase (Already Completed)" | tee -a "$PIPELINE_LOG"
            continue
        fi

        echo "------------------------------------------------------------" | tee -a "$PIPELINE_LOG"
        echo "[PHASE_EXECUTION_START] $phase" | tee -a "$PIPELINE_LOG"
        
        # [MEMORY_GUARD] Clear caches before major phases
        if [ "$phase" == "5_train_leakage_free" ]; then
            echo "[MEMORY_CLEAN] Aggressive sync for Phase 5..." | tee -a "$PIPELINE_LOG"
            safe_sync
        fi

        start_ts=$(date +%s)
        phase_log="$LOG_DIR/$phase.log"

        if python main.py --phase "$phase" --mode "$PYTHON_MODE" $SMOKE_FLAG >> "$phase_log" 2>&1; then
            end_ts=$(date +%s)
            duration=$((end_ts - start_ts))
            echo "[PHASE_EXECUTION_SUCCESS] $phase (${duration}s)" | tee -a "$PIPELINE_LOG"
            touch "$DONE_MARKER"
        else
            echo "[PHASE_EXECUTION_FAILED] $phase" | tee -a "$PIPELINE_LOG"
            echo "------------------------------------------------------------" | tee -a "$PIPELINE_LOG"
            echo "LAST 20 LINES OF $phase.log:" | tee -a "$PIPELINE_LOG"
            tail -n 20 "$phase_log" | tee -a "$PIPELINE_LOG"
            echo "------------------------------------------------------------" | tee -a "$PIPELINE_LOG"
            exit 1
        fi
    done

    # [PHASE 4: LATEST SYMLINK]
    rm -rf logs/latest 2>/dev/null || true
    ln -s "$RUN_ID" logs/latest 2>/dev/null || true
    echo "[PIPELINE] Symlink logs/latest -> $RUN_ID" | tee -a "$PIPELINE_LOG"

    SUBMISSION_PATH="./outputs/$RUN_ID/submission.csv"
    INTELLIGENCE_PATH="logs/$RUN_ID/validation_report.json"

    BEST_SCORE="N/A"
    if [ -f "$INTELLIGENCE_PATH" ]; then
        # Use Python for portable and robust JSON parsing
        BEST_SCORE=$(python -c "import json; print(json.load(open('$INTELLIGENCE_PATH'))['metrics']['mae'])" 2>/dev/null || echo "N/A")
    fi

    echo "============================================================" | tee -a "$PIPELINE_LOG"
    echo "  [PIPELINE SUCCESS]" | tee -a "$PIPELINE_LOG"
    echo "  RUN_ID: $RUN_ID" | tee -a "$PIPELINE_LOG"
    echo "  BEST SCORE (MAE): $BEST_SCORE" | tee -a "$PIPELINE_LOG"
    echo "  SUBMISSION: $SUBMISSION_PATH" | tee -a "$PIPELINE_LOG"
    echo "  Finish Time: $(date)" | tee -a "$PIPELINE_LOG"
    echo "============================================================" | tee -a "$PIPELINE_LOG"
}

run_pipeline