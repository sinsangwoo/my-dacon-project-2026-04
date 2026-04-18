#!/bin/bash

# ==============================================================================
# [DEVELOPMENT ENGINE - Internal Use Only]
# 
# Usage: ./dev.sh [mode] [phase]
# Modes: debug, trace, smoke-test, dry-run
# ==============================================================================

MODE=$1
PHASE=$2

if [ -z "$MODE" ] || [ -z "$PHASE" ]; then
    echo "Usage: ./dev.sh [mode] [phase]"
    echo "Modes: debug, trace, smoke-test, dry-run"
    exit 1
fi

export RUN_ID="dev_$(date +%Y%m%d_%H%M%S)"
echo "[DEV] Starting experiment with RUN_ID: $RUN_ID"
echo "[DEV] Mode: $MODE | Phase: $PHASE"

case "$MODE" in
    "debug")
        python main.py --phase "$PHASE" --mode debug
        ;;
    "trace")
        python main.py --phase "$PHASE" --mode trace
        ;;
    "smoke-test")
        python main.py --phase "$PHASE" --mode debug --smoke-test
        ;;
    "dry-run")
        python main.py --phase "$PHASE" --mode full --dry-run
        ;;
    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac
