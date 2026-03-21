#!/bin/bash
set -e

# Run SegmentAnyTree Docker container
# Usage:
#   bash scripts/run_docker.sh jupyter <input_dir> <output_dir>   # JupyterLab (default)
#   bash scripts/run_docker.sh infer <input_dir> <output_dir>     # Batch inference
#   bash scripts/run_docker.sh shell <input_dir> <output_dir>     # Interactive shell

IMAGE="${SAT_IMAGE:-segmentanytree:latest}"
MODE="${1:-jupyter}"
INPUT_DIR="${2:-$HOME/segmentanytree/input}"
OUTPUT_DIR="${3:-$HOME/segmentanytree/output}"

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

COMMON_ARGS=(
    --gpus all
    --rm
    -v "$INPUT_DIR:/data/input"
    -v "$OUTPUT_DIR:/data/output"
)

case "$MODE" in
    jupyter)
        echo "Starting JupyterLab at http://localhost:8888"
        docker run -it -p 8888:8888 "${COMMON_ARGS[@]}" "$IMAGE"
        ;;
    infer)
        echo "Running batch inference..."
        docker run "${COMMON_ARGS[@]}" "$IMAGE" \
            bash scripts/run_inference.sh /data/input /data/output true
        echo "Results in: $OUTPUT_DIR"
        ;;
    shell)
        echo "Starting interactive shell..."
        docker run -it "${COMMON_ARGS[@]}" "$IMAGE" bash
        ;;
    *)
        echo "Usage: run_docker.sh <jupyter|infer|shell> [input_dir] [output_dir]"
        exit 1
        ;;
esac
