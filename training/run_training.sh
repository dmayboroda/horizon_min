#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
#  Duck Hunt GRPO Training — Launch Script
# ===================================================================
#
#  Usage:
#    ./run_training.sh                    # TRL mode (default)
#    ./run_training.sh --custom           # Custom GRPO loop
#    ./run_training.sh --custom --resume outputs/ministral_duckhunt_grpo/checkpoint-500
#    ./run_training.sh --push-to-hub --hub-model-id user/duckhunt-grpo
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -------------------------------------------------------------------
#  Configuration
# -------------------------------------------------------------------
CONFIG="configs/ministral_config.yaml"
NUM_SAMPLES=1000
MODE="trl"                   # "trl" or "custom"
RESUME_FROM=""
WANDB_PROJECT="duckhunt-grpo"
PUSH_TO_HUB=false
HUB_MODEL_ID=""
EXTRA_OVERRIDES=()

# -------------------------------------------------------------------
#  Parse arguments
# -------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --custom)
            MODE="custom"
            shift
            ;;
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --override)
            EXTRA_OVERRIDES+=("--override" "$2")
            shift 2
            ;;
        --no-wandb)
            EXTRA_OVERRIDES+=("--override" "logging.report_to=none")
            shift
            ;;
        --push-to-hub)
            PUSH_TO_HUB=true
            shift
            ;;
        --hub-model-id)
            HUB_MODEL_ID="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --custom              Use custom GRPO loop instead of TRL"
            echo "  --resume PATH         Resume from checkpoint (custom mode only)"
            echo "  --config PATH         Config YAML (default: configs/ministral_config.yaml)"
            echo "  --num-samples N       Training samples to generate (default: 1000, TRL only)"
            echo "  --wandb-project NAME  W&B project name (default: duckhunt-grpo)"
            echo "  --override KEY=VAL    Override config value (repeatable)"
            echo "  --no-wandb            Disable W&B logging"
            echo "  --push-to-hub         Push final model to Hugging Face Hub"
            echo "  --hub-model-id ID     HF repo id (e.g. user/duckhunt-grpo)"
            echo "  -h, --help            Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# -------------------------------------------------------------------
#  Environment setup
# -------------------------------------------------------------------

# Create virtualenv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# W&B
export WANDB_PROJECT="$WANDB_PROJECT"

# -------------------------------------------------------------------
#  Create output directory
# -------------------------------------------------------------------
mkdir -p outputs

# -------------------------------------------------------------------
#  Launch training
# -------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Duck Hunt GRPO Training"
echo "============================================================"
echo "  Mode:        $MODE"
echo "  Config:      $CONFIG"
if [ "$MODE" = "trl" ]; then
    echo "  Samples:     $NUM_SAMPLES"
fi
if [ -n "$RESUME_FROM" ]; then
    echo "  Resume from: $RESUME_FROM"
fi
if [ "$PUSH_TO_HUB" = true ]; then
    echo "  Push to Hub: $HUB_MODEL_ID"
fi
echo "============================================================"
echo ""

CMD=(python train.py --config "$CONFIG")

if [ "$MODE" = "custom" ]; then
    CMD+=(--custom)
    if [ -n "$RESUME_FROM" ]; then
        CMD+=(--resume "$RESUME_FROM")
    fi
else
    CMD+=(--num-samples "$NUM_SAMPLES")
fi

# HF Hub
if [ "$PUSH_TO_HUB" = true ]; then
    CMD+=(--push-to-hub)
    if [ -n "$HUB_MODEL_ID" ]; then
        CMD+=(--hub-model-id "$HUB_MODEL_ID")
    fi
fi

# Append any extra overrides
if [ ${#EXTRA_OVERRIDES[@]} -gt 0 ]; then
    CMD+=("${EXTRA_OVERRIDES[@]}")
fi

echo "Running: ${CMD[*]}"
echo ""

exec "${CMD[@]}"
