#!/bin/bash
set -e

# COPC Octant-Parallel Inference
#
# Splits a single large COPC file into spatial tiles, processes each tile
# on a separate GPU, and merges the results.
#
# Usage:
#   bash scripts/run_inference_copc.sh <input.copc.laz> <output_dir> [num_gpus] [overlap_m]
#
# Requires: COPC-format input file (.copc.laz)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAT_ROOT="${SAT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PYTHONPATH="$SAT_ROOT:${PYTHONPATH:-}"
export SPARSE_BACKEND="${SPARSE_BACKEND:-spconv}"
export CUDA_VISIBLE_DEVICES="${SAT_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

COPC_FILE="${1:?Usage: $0 <input.copc.laz> <output_dir> [num_gpus] [overlap_m]}"
OUTPUT_DIR="${2:?Usage: $0 <input.copc.laz> <output_dir> [num_gpus] [overlap_m]}"
NUM_GPUS="${3:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
OVERLAP="${4:-2.0}"

[[ "$COPC_FILE" != /* ]] && COPC_FILE="$(pwd)/$COPC_FILE"
[[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$(pwd)/$OUTPUT_DIR"

echo "================================================"
echo "SegmentAnyTree COPC Parallel Inference"
echo "================================================"
echo "Input:   $COPC_FILE"
echo "Output:  $OUTPUT_DIR"
echo "GPUs:    $NUM_GPUS"
echo "Overlap: ${OVERLAP}m"
echo "================================================"

# Step 1: Split COPC into tiles
TILE_DIR="$OUTPUT_DIR/tiles"
echo "Splitting COPC into $NUM_GPUS tiles..."
python3 -m sat.pipeline.copc_parallel "$COPC_FILE" -o "$TILE_DIR" -n "$NUM_GPUS" --overlap "$OVERLAP"

# Count tiles (may be fewer than NUM_GPUS if some tiles are empty)
NUM_TILES=$(find "$TILE_DIR" -name "tile_*.laz" | wc -l)
echo "Created $NUM_TILES tiles"

if [ "$NUM_TILES" -eq 0 ]; then
    echo "No tiles created — input may be empty"
    exit 1
fi

# Step 2: Create per-tile input directories (run_inference.sh expects a directory)
for tile_file in "$TILE_DIR"/tile_*.laz; do
    tile_name=$(basename "$tile_file" .laz)
    tile_input="$TILE_DIR/${tile_name}_input"
    mkdir -p "$tile_input"
    cp "$tile_file" "$tile_input/"
done

# Step 3: Run inference on each tile (one per GPU)
PIDS=()
gpu_id=0
for tile_file in "$TILE_DIR"/tile_*.laz; do
    tile_name=$(basename "$tile_file" .laz)
    tile_input="$TILE_DIR/${tile_name}_input"
    tile_output="$TILE_DIR/${tile_name}_output"

    # Wrap around GPUs if more tiles than GPUs
    assigned_gpu=$((gpu_id % NUM_GPUS))

    echo "Starting $tile_name on GPU $assigned_gpu..."
    SAT_GPU="$assigned_gpu" bash "$SCRIPT_DIR/run_inference.sh" \
        "$tile_input" "$tile_output" false \
        > "$TILE_DIR/${tile_name}_log.txt" 2>&1 &
    PIDS+=($!)
    gpu_id=$((gpu_id + 1))

    # If we've filled all GPUs, wait for the batch to finish before starting more
    if [ "${#PIDS[@]}" -ge "$NUM_GPUS" ]; then
        for pid in "${PIDS[@]}"; do
            wait "$pid" || echo "WARNING: A tile inference failed"
        done
        PIDS=()
    fi
done

# Wait for remaining
for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "WARNING: A tile inference failed"
done

echo "All tiles processed."

# Step 4: Merge tile results
# TODO: Implement merge step using sat.pipeline.copc_parallel.merge_tile_results
# For now, collect results into final directory
FINAL_DIR="$OUTPUT_DIR/final_results"
mkdir -p "$FINAL_DIR"
for tile_dir in "$TILE_DIR"/tile_*_output; do
    if [ -d "$tile_dir/final_results" ]; then
        cp "$tile_dir/final_results"/* "$FINAL_DIR/" 2>/dev/null || true
    fi
done

num_results=$(find "$FINAL_DIR" -maxdepth 1 -type f | wc -l)
echo "================================================"
echo "Done! $num_results tile result files in: $FINAL_DIR"
echo "================================================"
