#!/bin/bash
set -e

# SegmentAnyTree Training Wrapper
# Usage: bash scripts/run_training.sh [job_name] [extra hydra overrides...]
#
# Example:
#   bash scripts/run_training.sh my_experiment
#   bash scripts/run_training.sh my_run training.epochs=200 training.batch_size=4

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAT_ROOT="${SAT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PYTHONPATH="$SAT_ROOT:${PYTHONPATH:-}"

JOB_NAME="${1:-treeins_run}"
shift 2>/dev/null || true

cd "$SAT_ROOT"

echo "Starting training: $JOB_NAME"
echo "Extra overrides: $@"

python3 train.py \
    task=panoptic \
    data=panoptic/treeins \
    models=panoptic/area4_ablation_3heads \
    model_name=PointGroup-PAPER \
    training=treeins \
    job_name="$JOB_NAME" \
    "$@"
