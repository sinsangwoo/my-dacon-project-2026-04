#!/bin/bash

# ==============================================================================
# [PIPELINE: POWER-DAMPED ATTACK ZONE RECOVERY SYSTEM]
#
# Philosophy: "Risk-Controlled Execution for Tail MAE Dominance"
# Execution: ./pipeline.sh [--debug] [--smoke-test]
# ==============================================================================

set -euo pipefail

PYTHON_MODE="full"
SMOKE_FLAG=""
START_PHASE=""
ONLY_PHASE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug) PYTHON_MODE="debug"; shift ;;
        --smoke-test) SMOKE_FLAG="--smoke-test"; shift ;;
        --start-phase) START_PHASE="$2"; shift 2 ;;
        --only-phase) ONLY_PHASE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [ "$PYTHON_MODE" = "debug" ]; then echo "[DEBUG MODE] Activated"; else echo "[FULL MODE] Activated"; fi
if [ -n "$SMOKE_FLAG" ]; then echo "[SMOKE TEST MODE] Activated"; fi

run_pipeline() {
    # [SSOT_FIX] Only generate RUN_ID if it's not already provided via env
    export RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
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

    # [PHASE 0: HEALTH CHECK]
    verify_environment() {
        echo "[HEALTH_CHECK] Verifying environment..." | tee -a "$PIPELINE_LOG"
        
        # Check Python & Required Modules for Attack Zone System
        local missing_modules=()
        for module in lightgbm pandas numpy sklearn scipy; do
            if ! python -c "import $module" >/dev/null 2>&1; then
                missing_modules+=("$module")
            fi
        done
        
        if [ ${#missing_modules[@]} -gt 0 ]; then
            echo "[FATAL] Missing required Python modules: ${missing_modules[*]}" | tee -a "$PIPELINE_LOG"
            exit 1
        fi
        echo "[HEALTH_CHECK] Environment OK." | tee -a "$PIPELINE_LOG"
    }

    echo "[PRE-CLEAN] Starting..." | tee -a "$PIPELINE_LOG"
    rm -rf logs/latest .done/latest 2>/dev/null || true
    mkdir -p ".done/$RUN_ID"
    echo "[PRE-CLEAN] Complete." | tee -a "$PIPELINE_LOG"

    echo "============================================================" | tee -a "$PIPELINE_LOG"
    echo "  ATTACK ZONE RECOVERY PIPELINE - [RUN_ID: $RUN_ID]" | tee -a "$PIPELINE_LOG"
    echo "  System: Power-Damped Asymmetric Execution (P2)" | tee -a "$PIPELINE_LOG"
    echo "  Start Time: $(date)" | tee -a "$PIPELINE_LOG"
    echo "============================================================" | tee -a "$PIPELINE_LOG"

    verify_environment

    # [PHASE 2: PIPELINE AUTO-SYNC]
    PHASES_STR=$(python -c "from main import VALID_PHASES; print(' '.join(VALID_PHASES))")
    read -a PHASES <<< "$PHASES_STR"

    # [PHASE 3: SMART EXECUTION LOOP]
    START_PHASE=${START_PHASE:-""}
    ONLY_PHASE=${ONLY_PHASE:-""}
    found_start=false
    if [ -z "$START_PHASE" ]; then found_start=true; fi

    for phase in "${PHASES[@]}"; do
        if [ "$found_start" = false ]; then
            if [ "$phase" = "$START_PHASE" ]; then found_start=true; else continue; fi
        fi
        if [ -n "$ONLY_PHASE" ] && [ "$phase" != "$ONLY_PHASE" ]; then continue; fi

        echo "------------------------------------------------------------" | tee -a "$PIPELINE_LOG"
        echo "[PHASE_START] $phase" | tee -a "$PIPELINE_LOG"
        
        if [ "$phase" == "5_train_leakage_free" ]; then safe_sync; fi

        start_ts=$(date +%s)
        phase_log="$LOG_DIR/$phase.log"

        verify_artifacts() {
            local phase=$1
            echo "[VERIFY] Checking artifacts for $phase..." | tee -a "$PIPELINE_LOG"
            
            case $phase in
                2_build_base)
                    files=("./outputs/$RUN_ID/processed/train_base.pkl" "./outputs/$RUN_ID/processed/test_base.pkl" "./outputs/$RUN_ID/processed/y_train.npy")
                    ;;
                2.5_drift_audit)
                    files=("./outputs/$RUN_ID/processed/stability_manifest.json")
                    ;;
                3_train_raw)
                    files=("./outputs/$RUN_ID/processed/oof_raw.npy")
                    ;;
                5_train_leakage_free)
                    files=("./outputs/$RUN_ID/predictions/oof_raw.npy" "./outputs/$RUN_ID/models/lgbm/model_fold_0.pkl" "./outputs/$RUN_ID/models/reconstructors/global_means.pkl")
                    ;;
                6_calibrate)
                    files=("./outputs/$RUN_ID/predictions/oof_stable.npy" "./outputs/$RUN_ID/predictions/test_stable.npy")
                    ;;
                7_inference)
                    files=("./outputs/$RUN_ID/predictions/final_submission.npy")
                    ;;
                8_submission)
                    files=("./outputs/$RUN_ID/submission.csv")
                    ;;
                *)
                    echo "[VERIFY] No specific artifact check defined for $phase. Skipping." | tee -a "$PIPELINE_LOG"
                    return 0
                    ;;
            esac
            
            for file in "${files[@]}"; do
                if [ ! -f "$file" ]; then
                    echo "[FATAL] Artifact MISSING: $file" | tee -a "$PIPELINE_LOG"
                    exit 1
                fi
                if [ ! -s "$file" ]; then
                    echo "[FATAL] Artifact EMPTY: $file" | tee -a "$PIPELINE_LOG"
                    exit 1
                fi
            done
            echo "[VERIFY] $phase artifacts OK." | tee -a "$PIPELINE_LOG"
        }

        if python main.py --phase "$phase" --mode "$PYTHON_MODE" --run-id "$RUN_ID" $SMOKE_FLAG >> "$phase_log" 2>&1; then
            end_ts=$(date +%s)
            duration=$((end_ts - start_ts))
            echo "[PHASE_SUCCESS] $phase (${duration}s)" | tee -a "$PIPELINE_LOG"
            
            # [MISSION_CRITICAL] Verify artifacts immediately
            verify_artifacts "$phase"
            
            touch ".done/$RUN_ID/$phase"
        else
            echo "[PHASE_FAILED] $phase. Check $phase_log" | tee -a "$PIPELINE_LOG"
            tail -n 20 "$phase_log" | tee -a "$PIPELINE_LOG"
            exit 1
        fi
    done

    # [PHASE 4: RESULTS HARVESTING]
    rm -rf logs/latest 2>/dev/null || true
    ln -s "$RUN_ID" logs/latest 2>/dev/null || true

    INTELLIGENCE_PATH="logs/$RUN_ID/validation_report.json"
    
    BEST_SCORE="N/A"
    TAIL_MAE="N/A"
    FP_COST="N/A"
    GAIN_CAPTURE="N/A"
    
    if [ -f "$INTELLIGENCE_PATH" ]; then
        BEST_SCORE=$(python -c "import json; m=json.load(open('$INTELLIGENCE_PATH'))['metrics']; print(f\"{m.get('mae', 0.0):.4f}\")" 2>/dev/null || echo "N/A")
        TAIL_MAE=$(python -c "import json; m=json.load(open('$INTELLIGENCE_PATH'))['metrics']; print(f\"{m.get('Q99_100_mae', 0.0):.4f}\")" 2>/dev/null || echo "N/A")
        FP_COST=$(python -c "import json; m=json.load(open('$INTELLIGENCE_PATH'))['metrics']; print(f\"{m.get('fp_cost', 0.0):.4f}\")" 2>/dev/null || echo "N/A")
        GAIN_CAPTURE=$(python -c "import json; m=json.load(open('$INTELLIGENCE_PATH'))['metrics']; print(f\"{m.get('gain_capture', 0.0):.4f}\")" 2>/dev/null || echo "N/A")
    fi

    echo "============================================================" | tee -a "$PIPELINE_LOG"
    echo "  [PIPELINE SUCCESS] - ATTACK ZONE ACTIVATED" | tee -a "$PIPELINE_LOG"
    echo "  RUN_ID:      $RUN_ID" | tee -a "$PIPELINE_LOG"
    echo "  GLOBAL MAE:  $BEST_SCORE" | tee -a "$PIPELINE_LOG"
    echo "  TAIL MAE:    $TAIL_MAE (Q99-100 Target)" | tee -a "$PIPELINE_LOG"
    echo "  ----------------------------------------------------------" | tee -a "$PIPELINE_LOG"
    echo "  FP COST:     $FP_COST (Risk Exposure)" | tee -a "$PIPELINE_LOG"
    echo "  GAIN CAPTURE: $GAIN_CAPTURE (Signal Recovery)" | tee -a "$PIPELINE_LOG"
    echo "  ----------------------------------------------------------" | tee -a "$PIPELINE_LOG"
    echo "  SUBMISSION:  ./outputs/$RUN_ID/submission.csv" | tee -a "$PIPELINE_LOG"
    echo "  Finish Time: $(date)" | tee -a "$PIPELINE_LOG"
    echo "============================================================" | tee -a "$PIPELINE_LOG"
}

run_pipeline