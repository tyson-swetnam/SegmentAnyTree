#!/bin/bash
set -e

# Multi-GPU Parallel Inference
# Usage: bash scripts/run_inference_parallel.sh <input_dir> <output_dir> [num_gpus]
#
# Distributes input files across N GPUs, running one inference pipeline per GPU.
# Default: uses all available GPUs.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAT_ROOT="${SAT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

SOURCE_DIR="${1:?Usage: $0 <input_dir> <output_dir> [num_gpus]}"
DEST_DIR="${2:?Usage: $0 <input_dir> <output_dir> [num_gpus]}"
NUM_GPUS="${3:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

[[ "$SOURCE_DIR" != /* ]] && SOURCE_DIR="$(pwd)/$SOURCE_DIR"
[[ "$DEST_DIR" != /* ]] && DEST_DIR="$(pwd)/$DEST_DIR"

# Discover input files
shopt -s nullglob
FILES=("$SOURCE_DIR"/*.{las,laz,ply,LAS,LAZ,PLY})
shopt -u nullglob

NUM_FILES=${#FILES[@]}
if [ "$NUM_FILES" -eq 0 ]; then
    echo "No .las/.laz/.ply files found in $SOURCE_DIR"
    exit 1
fi

echo "================================================"
echo "SegmentAnyTree Parallel Inference"
echo "================================================"
echo "Input:       $SOURCE_DIR ($NUM_FILES files)"
echo "Output:      $DEST_DIR"
echo "GPUs:        $NUM_GPUS"
echo "================================================"

# If only 1 GPU or 1 file, fall back to sequential
if [ "$NUM_GPUS" -le 1 ] || [ "$NUM_FILES" -le 1 ]; then
    echo "Using single-GPU mode"
    exec bash "$SCRIPT_DIR/run_inference.sh" "$SOURCE_DIR" "$DEST_DIR" true
fi

# Split files into per-GPU directories
mkdir -p "$DEST_DIR"
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    mkdir -p "$DEST_DIR/gpu_${gpu_id}/input"
done

# Round-robin assignment of files to GPUs
for i in "${!FILES[@]}"; do
    gpu_id=$((i % NUM_GPUS))
    cp "${FILES[$i]}" "$DEST_DIR/gpu_${gpu_id}/input/"
done

# Report assignment
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    n=$(ls "$DEST_DIR/gpu_${gpu_id}/input/" 2>/dev/null | wc -l)
    echo "GPU $gpu_id: $n files"
done

# Launch parallel inference (one per GPU)
PIDS=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    input_dir="$DEST_DIR/gpu_${gpu_id}/input"
    output_dir="$DEST_DIR/gpu_${gpu_id}/output"
    # Skip if no files assigned
    [ "$(ls "$input_dir" 2>/dev/null | wc -l)" -eq 0 ] && continue

    echo "Starting GPU $gpu_id..."
    SAT_GPU="$gpu_id" bash "$SCRIPT_DIR/run_inference.sh" \
        "$input_dir" "$output_dir" true \
        > "$DEST_DIR/gpu_${gpu_id}/log.txt" 2>&1 &
    PIDS+=($!)
done

# Wait for all and collect exit codes
FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "GPU $i FAILED (see $DEST_DIR/gpu_${i}/log.txt)"
        FAILED=1
    else
        echo "GPU $i done"
    fi
done

# Merge results from all GPUs into final directory
FINAL_DIR="$DEST_DIR/final_results"
mkdir -p "$FINAL_DIR"
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_final="$DEST_DIR/gpu_${gpu_id}/output/final_results"
    if [ -d "$gpu_final" ]; then
        cp "$gpu_final"/* "$FINAL_DIR/" 2>/dev/null || true
    fi
done

num_results=$(find "$FINAL_DIR" -maxdepth 1 -type f | wc -l)
echo "================================================"
echo "Done! $num_results result files in: $FINAL_DIR"
if [ "$FAILED" -ne 0 ]; then
    echo "WARNING: Some GPUs failed. Check gpu_*/log.txt"
fi
echo "================================================"
exit $FAILED
