#!/bin/bash
set -e

# SegmentAnyTree Batch Inference — processes files in chunks of N
# Usage: bash scripts/run_batch_inference.sh <input_dir> <output_dir> [batch_size]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAT_ROOT="${SAT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

INPUT_DIR="${1:?Usage: run_batch_inference.sh <input_dir> <output_dir> [batch_size]}"
OUTPUT_DIR="${2:?Usage: run_batch_inference.sh <input_dir> <output_dir> [batch_size]}"
BATCH_SIZE="${3:-10}"

[[ "$INPUT_DIR" != /* ]] && INPUT_DIR="$(pwd)/$INPUT_DIR"
[[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$(pwd)/$OUTPUT_DIR"

TEMP_IN="$OUTPUT_DIR/.batch_temp_in"
TEMP_OUT="$OUTPUT_DIR/.batch_temp_out"
FINAL_DIR="$OUTPUT_DIR/final_results"

rm -rf "$TEMP_IN" "$TEMP_OUT"
mkdir -p "$TEMP_IN" "$TEMP_OUT" "$FINAL_DIR"

# Collect input files
mapfile -t files < <(find "$INPUT_DIR" -maxdepth 1 -type f)
total=${#files[@]}

echo "Batch inference: $total files in batches of $BATCH_SIZE"

for ((i=0; i<total; i+=BATCH_SIZE)); do
    batch_num=$(( i / BATCH_SIZE + 1 ))
    batch_end=$(( i + BATCH_SIZE ))
    [ $batch_end -gt $total ] && batch_end=$total

    echo ""
    echo "=== Batch $batch_num: files $((i+1))-$batch_end of $total ==="

    rm -rf "$TEMP_IN"/*
    cp "${files[@]:i:BATCH_SIZE}" "$TEMP_IN/"

    bash "$SCRIPT_DIR/run_inference.sh" "$TEMP_IN" "$TEMP_OUT" true

    if [ -d "$TEMP_OUT/final_results" ]; then
        cp "$TEMP_OUT/final_results"/* "$FINAL_DIR/" 2>/dev/null || true
    fi
done

# Cleanup
rm -rf "$TEMP_IN" "$TEMP_OUT"

num_files=$(find "$FINAL_DIR" -maxdepth 1 -type f | wc -l)
echo ""
echo "Batch processing complete. $num_files result files in: $FINAL_DIR"
