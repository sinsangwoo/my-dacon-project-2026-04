#!/bin/bash

# Configuration
MODE="full"
if [ "$1" == "debug" ]; then
    MODE="debug"
fi

HALT_ON_REGRESSION=false
MAX_RETRIES=2
LOG_DIR="logs"
DONE_DIR=".done"
PIPELINE_LOG="$LOG_DIR/pipeline.log"
ERROR_LOG="$LOG_DIR/error.log"
BEST_MAE_FILE="$LOG_DIR/best_mae.txt"

# Clean Mode
if [ "$1" == "clean" ]; then
    echo "[CLEAN] Resetting pipeline artifacts..."
    rm -rf "$LOG_DIR" "$DONE_DIR" "outputs/"
    echo "Done."
    exit 0
fi

# Initialization
mkdir -p "$LOG_DIR" "$DONE_DIR" "outputs/processed" "outputs/models" "outputs/predictions"

# Log Header
echo "============================================" | tee -a "$PIPELINE_LOG"
echo " PIPELINE EXECUTION START: $(date)" | tee -a "$PIPELINE_LOG"
echo " MODE: $MODE" | tee -a "$PIPELINE_LOG"
echo "============================================" | tee -a "$PIPELINE_LOG"

# Initial Best MAE (high value)
if [ ! -f "$BEST_MAE_FILE" ]; then
    echo "999.999" > "$BEST_MAE_FILE"
fi

# Trap for critical errors
trap 'echo "[CRITICAL] Pipeline failed unexpectedly. See $ERROR_LOG." | tee -a "$PIPELINE_LOG"' ERR

check_performance() {
    local phase=$1
    local current_mae_file="$LOG_DIR/current_mae.txt"
    
    if [ ! -f "$current_mae_file" ]; then
        return 0
    fi
    
    current_mae=$(cat "$current_mae_file")
    best_mae=$(cat "$BEST_MAE_FILE")
    
    # Use awk for robust float comparison
    is_better=$(awk -v c="$current_mae" -v b="$best_mae" 'BEGIN {print (c < b)}')
    
    if [ "$is_better" -eq 1 ]; then
        echo "[METRIC] Phase $phase Improved MAE: $best_mae -> $current_mae" | tee -a "$PIPELINE_LOG"
        echo "$current_mae" > "$BEST_MAE_FILE"
    else
        echo "[WARNING] Performance Regression in $phase! Current: $current_mae, Best: $best_mae" | tee -a "$PIPELINE_LOG"
        if [ "$HALT_ON_REGRESSION" = "true" ]; then
            echo "[ERROR] Halting due to regression policy." | tee -a "$PIPELINE_LOG"
            return 1
        fi
    fi
    return 0
}

run_phase() {
    local phase=$1
    local done_marker="$DONE_DIR/$phase.done"
    local phase_log="$LOG_DIR/$phase.log"
    local attempt=1
    local delays=(0 10 30)

    if [ -f "$done_marker" ]; then
        echo "[SKIP] Phase $phase already completed." | tee -a "$PIPELINE_LOG"
        return 0
    fi

    echo "[START] Phase: $phase" | tee -a "$PIPELINE_LOG"
    start_time=$(date +%s)

    while [ $attempt -le $((MAX_RETRIES + 1)) ]; do
        echo "Attempt $attempt starting at $(date)..." >> "$phase_log"
        if python main.py --phase "$phase" --mode "$MODE" >> "$phase_log" 2>&1; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            echo "[END] Phase $phase | Duration: ${duration}s" | tee -a "$PIPELINE_LOG"
            touch "$done_marker"
            
            # Performance Fail-safe
            if ! check_performance "$phase"; then
                return 1
            fi
            return 0
        else
            echo "[RETRY] Phase $phase failed (Attempt $attempt). Waiting ${delays[$attempt]}s..." | tee -a "$PIPELINE_LOG"
            if [ $attempt -le $MAX_RETRIES ]; then
                sleep ${delays[$attempt]}
            fi
            ((attempt++))
        fi
    done

    echo "[FAIL] Phase $phase failed after $((attempt-1)) retries." | tee -a "$PIPELINE_LOG"
    echo "--- Last 20 lines of $phase_log ---" >> "$ERROR_LOG"
    tail -n 20 "$phase_log" >> "$ERROR_LOG"
    echo "-----------------------------------" >> "$ERROR_LOG"
    return 1
}

# Main Phase Loop
PHASES=(
    "1_data_check"
    "2_preprocess"
    "3_train_base"
    "4_stacking"
    "5_pseudo_labeling"
    "6_retrain"
    "7_inference"
    "8_submission"
)

for phase in "${PHASES[@]}"; do
    if ! run_phase "$phase"; then
        echo "[CRIT] Pipeline stopped at phase $phase" | tee -a "$PIPELINE_LOG"
        exit 1
    fi
done

# Final Summary
echo "============================================" | tee -a "$PIPELINE_LOG"
echo " PIPELINE COMPLETE: $(date)" | tee -a "$PIPELINE_LOG"
echo " FINAL BEST MAE: $(cat "$BEST_MAE_FILE")" | tee -a "$PIPELINE_LOG"
echo " SUBMISSION: outputs/submission.csv" | tee -a "$PIPELINE_LOG"
echo "============================================" | tee -a "$PIPELINE_LOG"
