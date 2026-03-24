# Inference Guide

## Scientific Context

SegmentAnyTree performs **panoptic segmentation** on 3D LiDAR point clouds, combining two complementary tasks:

- **Semantic segmentation**: classifies each point as tree, non-tree, or unclassified
- **Instance segmentation**: assigns a unique ID to each individual tree

This enables downstream forestry analysis including per-tree height estimation, crown diameter measurement, stem position mapping, and biomass estimation. The model works across sensor types (ALS, UAS, TLS, MLS) without retraining.

See the [Scientific Workflow](workflow.md) guide for the full end-to-end process.

## Pipeline Overview

The inference pipeline transforms input point clouds through several stages:

```mermaid
graph LR
    A[Input .las/.laz/.ply] --> B[Sanitize filenames]
    B --> C[UTM → local coords]
    C --> D[Model inference]
    D --> E[Rename & merge results]
    E --> F[Restore UTM coords]
    F --> G[Convert to COPC]
    G --> H[Output .copc.laz]
```

1. **File preparation** — sanitize filenames (replace spaces/dashes)
2. **UTM → local coordinates** — subtract min x/y/z, save offsets to JSON
3. **Model inference** — PointGroup-PAPER panoptic segmentation
4. **Result renaming** — map generic indices to descriptive filenames
5. **Result merging** — combine predictions with original point cloud, restore UTM
6. **COPC conversion** — convert LAZ to Cloud-Optimized Point Cloud format via PDAL

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

## COPC Output

The pipeline automatically converts output files to **COPC** (Cloud-Optimized Point Cloud) format using [PDAL](https://pdal.io/). COPC is a LAZ 1.4 file with an embedded spatial octree index that enables:

- Efficient HTTP range-request streaming for web viewers
- Fast partial reads — load only the spatial region you need
- Native support in QGIS 3.26+, CloudCompare, Potree, and [copc.io](https://viewer.copc.io/)

Output files have the `.copc.laz` extension. If PDAL is not available, the pipeline falls back to standard `.laz` output.

### Standalone COPC conversion

To convert existing LAZ files to COPC format outside the inference pipeline:

```python
from sat.io.las_io import laz_to_copc

# Single file
laz_to_copc("segmented.laz", "segmented.copc.laz", verbose=True)
```

```bash
# Or via PDAL directly
pdal translate input.laz output.copc.laz --writer copc
```

See the [COPC Conversion notebook](notebooks.md) for batch conversion examples.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SAT_ROOT` | Auto-detected | Project root directory |
| `SAT_MODEL` | `$SAT_ROOT/model_file` | Checkpoint directory |
| `SAT_DATA` | `$SAT_ROOT/data` | Data directory |
| `SAT_CACHE` | `/tmp/sat_cache` | Temporary files |
| `SAT_GPU` | `0` | GPU device index for single-GPU inference |
| `NUM_GPUS` | Auto-detect all | Number of GPUs for parallel inference |

## Multi-GPU Inference

Distribute files across all available GPUs for parallel processing:

```bash
# Auto-detect all GPUs
bash scripts/run_inference_parallel.sh /data/input /data/output

# Specify GPU count
bash scripts/run_inference_parallel.sh /data/input /data/output 4
```

Files are assigned round-robin across GPUs. Each GPU runs an independent inference pipeline. Results are merged into `output/final_results/`.

For Docker:

=== "Harbor Registry"

    ```bash
    docker run --gpus all \
      -v $HOME/data/input:/data/input \
      -v $HOME/data/output:/data/output \
      harbor.cyverse.org/vice/segmentanytree:cuda12 \
      bash scripts/run_inference_parallel.sh /data/input /data/output
    ```

=== "Local Build"

    ```bash
    docker run --gpus all \
      -v $HOME/data/input:/data/input \
      -v $HOME/data/output:/data/output \
      segmentanytree:cuda12 \
      bash scripts/run_inference_parallel.sh /data/input /data/output
    ```

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for common issues.
