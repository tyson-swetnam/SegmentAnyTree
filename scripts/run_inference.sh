#!/bin/bash
set -e

# SegmentAnyTree Inference Pipeline
# Usage: bash scripts/run_inference.sh <input_dir> <output_dir> [clean:true|false]
#
# All paths are portable. SAT_ROOT is auto-detected from the script location.

# Auto-detect SAT_ROOT (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAT_ROOT="${SAT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PYTHONPATH="$SAT_ROOT:${PYTHONPATH:-}"

# Parse arguments with portable defaults
SOURCE_DIR="${1:-}"
DEST_DIR="${2:-}"
CLEAN_OUTPUT_DIR="${3:-true}"

if [ -z "$SOURCE_DIR" ] || [ -z "$DEST_DIR" ]; then
    echo "Usage: bash scripts/run_inference.sh <input_dir> <output_dir> [clean:true|false]"
    echo ""
    echo "Arguments:"
    echo "  input_dir      Directory containing .las/.laz/.ply files to segment"
    echo "  output_dir     Directory for output results"
    echo "  clean          Clean output directory before running (default: true)"
    echo ""
    echo "Environment variables:"
    echo "  SAT_ROOT       Project root (default: auto-detected)"
    echo "  SAT_MODEL      Model checkpoint dir (default: \$SAT_ROOT/model_file)"
    exit 1
fi

# Resolve to absolute paths
[[ "$SOURCE_DIR" != /* ]] && SOURCE_DIR="$(pwd)/$SOURCE_DIR"
[[ "$DEST_DIR" != /* ]] && DEST_DIR="$(pwd)/$DEST_DIR"

echo "================================================"
echo "SegmentAnyTree Inference Pipeline"
echo "================================================"
echo "SAT_ROOT:    $SAT_ROOT"
echo "Input:       $SOURCE_DIR"
echo "Output:      $DEST_DIR"
echo "Clean first: $CLEAN_OUTPUT_DIR"
echo "================================================"

# Clean output directory if requested
if [ "$CLEAN_OUTPUT_DIR" = "true" ]; then
    rm -rf "$DEST_DIR"/*
fi

# Step 1: Copy and sanitize input files
mkdir -p "$DEST_DIR/input_data"
cp -r "$SOURCE_DIR/"* "$DEST_DIR/input_data/"
python3 -m sat.pipeline.file_preparation "$DEST_DIR/input_data"

# Step 2: UTM to local coordinate transform
python3 -m sat.pipeline.coordinate_transform -i "$DEST_DIR/input_data" -o "$DEST_DIR/utm2local"

# Step 3: Update eval.yaml with file paths
cp "$SAT_ROOT/conf/eval.yaml" "$DEST_DIR/"
python3 -m sat.pipeline.config_update "$DEST_DIR/eval.yaml" "$DEST_DIR/utm2local" "$DEST_DIR"

# Step 4: Clear cache
python3 -m sat.pipeline.cache --eval_yaml "$DEST_DIR/eval.yaml"

# Step 5: Run model inference
# Hydra needs --config-path (directory) and --config-name (filename without .yaml)
cd "$SAT_ROOT"
python3 eval.py --config-path "$DEST_DIR" --config-name eval
echo "Inference complete."

# Step 6: Rename output files
python3 -m sat.pipeline.result_rename instance "$DEST_DIR/eval.yaml" "$DEST_DIR"
python3 -m sat.pipeline.result_rename semantic "$DEST_DIR/eval.yaml" "$DEST_DIR"

# Step 7: Merge results with original point clouds
FINAL_DIR="$DEST_DIR/final_results"
python3 -m sat.pipeline.result_merge \
    -i "$DEST_DIR/utm2local" \
    -s "$DEST_DIR" \
    -o "$FINAL_DIR" \
    -v

# Step 8: Clean up numbered prefixes in output filenames
for file in "$FINAL_DIR"/*; do
    filename=$(basename "$file")
    new_name=$(echo "$filename" | sed 's/^[0-9]*_//')
    [ "$filename" != "$new_name" ] && mv -n "$file" "$FINAL_DIR/$new_name"
done

num_files=$(find "$FINAL_DIR" -maxdepth 1 -type f | wc -l)
echo "================================================"
echo "Done! $num_files result files in: $FINAL_DIR"
echo "================================================"
