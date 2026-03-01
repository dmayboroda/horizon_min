#!/bin/bash
# =============================================================================
# Run all training experiments (optimizer × forward_mod grid)
# =============================================================================
# Usage:
#   ./run_all.sh                    # Run everything
#   ./run_all.sh --dry-run          # Print commands without executing
#   ./run_all.sh --optimizer muon   # Run only Muon optimizer experiments
#   ./run_all.sh --forward-mod action_full  # Run only full action mode
#   ./run_all.sh --max-steps 100    # Override max steps for quick test
#
# Each run layers: base_config + ministral_config + optimizer + forward_mod

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
DRY_RUN=false
FILTER_OPT=""
FILTER_MOD=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --optimizer)
            FILTER_OPT="$2"
            shift 2
            ;;
        --forward-mod)
            FILTER_MOD="$2"
            shift 2
            ;;
        --max-steps)
            EXTRA_ARGS="$EXTRA_ARGS --override training.max_steps=$2"
            shift 2
            ;;
        --no-wandb)
            EXTRA_ARGS="$EXTRA_ARGS --override logging.report_to=none"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Experiment grid
OPTIMIZERS=(
    "adamw"
    "adamw_8bit"
    "muon"
    "soap"
    "shampoo"
    "prodigy"
)

FORWARD_MODS=(
    "text_baseline"
    "temporal_only"
    "cross_frame_only"
    "action_spatial_only"
    "action_full"
)

echo "============================================="
echo "  Duck Hunt GRPO — Experiment Grid"
echo "============================================="
echo ""
echo "Optimizers:    ${OPTIMIZERS[*]}"
echo "Forward mods:  ${FORWARD_MODS[*]}"
echo "Extra args:    ${EXTRA_ARGS:-none}"
echo ""

RUN_COUNT=0

for opt in "${OPTIMIZERS[@]}"; do
    # Filter
    if [[ -n "$FILTER_OPT" && "$opt" != "$FILTER_OPT" ]]; then
        continue
    fi

    for mod in "${FORWARD_MODS[@]}"; do
        # Filter
        if [[ -n "$FILTER_MOD" && "$mod" != "$FILTER_MOD" ]]; then
            continue
        fi

        RUN_NAME="${opt}-${mod}"
        OUTPUT_DIR="./outputs/${RUN_NAME}"

        CMD="python train.py --custom"
        CMD="$CMD --config configs/base_config.yaml"
        CMD="$CMD --config configs/ministral_config.yaml"
        CMD="$CMD --config configs/optimizers/${opt}.yaml"
        CMD="$CMD --config configs/forward_mods/${mod}.yaml"
        CMD="$CMD --override training.output_dir=${OUTPUT_DIR}"
        CMD="$CMD --override logging.wandb_run_name=${RUN_NAME}"
        CMD="$CMD $EXTRA_ARGS"

        RUN_COUNT=$((RUN_COUNT + 1))

        echo "---------------------------------------------"
        echo "  Run ${RUN_COUNT}: ${RUN_NAME}"
        echo "  Output: ${OUTPUT_DIR}"
        echo "---------------------------------------------"

        if $DRY_RUN; then
            echo "  [DRY RUN] $CMD"
            echo ""
        else
            echo "  Running: $CMD"
            eval "$CMD"
            echo ""
            echo "  Completed: ${RUN_NAME}"
            echo ""
        fi
    done
done

echo "============================================="
echo "  All ${RUN_COUNT} experiments complete!"
echo "============================================="
