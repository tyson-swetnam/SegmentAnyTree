# Inference Guide

## Pipeline Overview

The inference pipeline transforms input point clouds through several stages:

```
Input (.las/.laz/.ply)
  → File preparation (sanitize filenames)
  → UTM → local coordinates (subtract min x/y/z, save offsets)
  → Model inference (PointGroup-PAPER, panoptic segmentation)
  → Result renaming (index → descriptive names)
  → Result merging (combine predictions with original point cloud, restore UTM)
  → Output (.las with PredSemantic + PredInstance fields)
```

## Running Inference

### Single run

```bash
bash scripts/run_inference.sh /path/to/input /path/to/output true
```

Arguments:
- `input_dir`: Directory containing point cloud files
- `output_dir`: Directory for results (created if needed)
- `clean`: `true` to clear output directory first, `false` to keep existing files

### Batch mode

For large collections, batch mode processes files in configurable chunks:

```bash
bash scripts/run_batch_inference.sh /path/to/input /path/to/output 10
```

The third argument is the batch size (default: 10 files per batch).

### Python API

You can also run individual pipeline steps from Python:

```python
from sat.pipeline.coordinate_transform import utm_to_local_folder
from sat.pipeline.file_preparation import sanitize_filenames
from sat.pipeline.config_update import modify_eval_yaml
from sat.pipeline.result_merge import FolderMerger

# Step 1: Prepare files
sanitize_filenames("input_data/")

# Step 2: Coordinate transform
utm_to_local_folder("input_data/", "utm2local/")

# Step 3: Configure and run eval.py
modify_eval_yaml("eval.yaml", "utm2local/", "output/")

# Step 4: Merge results
FolderMerger("utm2local/", "output/", "final_results/", verbose=True).run()
```

## Coordinate Systems

The model expects points in a **local coordinate system** near the origin. The pipeline handles this automatically:

1. **UTM → Local**: Subtracts the minimum x, y, z values from all points. The offsets are saved to `*_min_values.json` files.
2. **Local → UTM**: After inference, the merge step adds the offsets back to restore original UTM coordinates.

You never need to manually transform coordinates.

## Input Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| LAS | `.las` | Standard LiDAR format |
| LAZ | `.laz` | Compressed LAS (requires lazrs) |
| PLY | `.ply` | Stanford polygon format |

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| PredSemantic | uint8 | 0=unclassified, 1=non-tree, 2=tree |
| PredInstance | uint16 | Unique tree ID (0=unassigned, 1+ = tree instances) |

All original point attributes are preserved in the output.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SAT_ROOT` | Auto-detected | Project root directory |
| `SAT_MODEL` | `$SAT_ROOT/model_file` | Checkpoint directory |
| `SAT_DATA` | `$SAT_ROOT/data` | Data directory |
| `SAT_CACHE` | `/tmp/sat_cache` | Temporary files |

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for common issues.
