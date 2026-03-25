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

## COPC Octant-Parallel Inference

For a single large COPC file, split it into spatial tiles and process each tile on a separate GPU:

```bash
# Split a COPC file into 4 tiles (one per GPU) with 2m overlap buffer
bash scripts/run_inference_copc.sh input.copc.laz /path/to/output 4 2.0
```

This uses the COPC octree index to extract spatial tiles efficiently, processes each independently, and collects results. The overlap buffer ensures trees at tile boundaries are segmented correctly.

You can also use the Python API directly:

```python
from sat.pipeline.copc_parallel import split_copc_to_tiles, merge_tile_results

# Split into tiles
tiles = split_copc_to_tiles("input.copc.laz", "workdir/", num_tiles=4, overlap=2.0)

# Process each tile (e.g., on different GPUs)
# ... run inference on each tile.file_path ...

# Merge results (deduplicates overlap zones)
merge_tile_results(tiles, "workdir/", "output.laz", overlap=2.0)
```

## Weight Migration (CUDA 12 only)

When using the CUDA 12.4 image with pre-trained weights from the original MinkowskiEngine model, the weights must be converted to SpConv format:

```bash
python scripts/migrate_weights.py \
  --input model_file/PointGroup-PAPER.pt \
  --output model_file/PointGroup-PAPER-spconv.pt
```

This converts 118 convolution kernels from ME's `(K³, C_in, C_out)` layout to SpConv's `(C_out, kD, kH, kW, C_in)` format. All other parameters (BatchNorm, MLP heads) transfer unchanged. The CUDA 11.8 image uses the original weights directly.

## Customizing Parameters

The inference pipeline uses [Hydra](https://hydra.cc/) for configuration. Parameters can be customized by editing `conf/eval.yaml` or passing overrides on the command line.

### Key parameters

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `batch_size` | `1` | `conf/eval.yaml` | Batch size for inference. Increase for faster processing if GPU memory allows |
| `data.first_subsampling` | `0.2` | `conf/data/panoptic/treeins_rad8.yaml` | Grid voxel size in meters. Smaller = more detail but more memory |
| `data.radius` | `8` | `conf/data/panoptic/treeins_rad8.yaml` | Cylinder sampling radius in meters |
| `tracker_options.min_score` | `0.0` | `conf/eval.yaml` | Minimum cluster score for instance acceptance (0.0–1.0). Higher = fewer but more confident trees |
| `checkpoint_dir` | `model_file` | `conf/eval.yaml` | Path to model checkpoint directory |
| `cuda` | `0` | `conf/eval.yaml` | GPU device index |

### Using Hydra overrides

When running `eval.py` directly, pass overrides on the command line:

```bash
# Increase batch size for faster inference
python eval.py batch_size=2

# Use stricter cluster filtering (fewer, more confident tree detections)
python eval.py tracker_options.min_score=0.5

# Change grid sampling size (smaller = finer detail, more memory)
python eval.py data.first_subsampling=0.1

# Use a different model checkpoint
python eval.py checkpoint_dir=/path/to/my_trained_model

# Combine multiple overrides
python eval.py batch_size=2 tracker_options.min_score=0.5 data.first_subsampling=0.15
```

### Customizing via Docker

Pass overrides through the inference script by running `eval.py` directly:

```bash
# Run with custom parameters in Docker
docker run --gpus all \
  -v $HOME/data/input:/data/input \
  -v $HOME/data/output:/data/output \
  harbor.cyverse.org/vice/segmentanytree:cuda12 \
  bash -c "cd /opt/segmentanytree && \
    python eval.py batch_size=2 tracker_options.min_score=0.5"
```

Or mount a modified `eval.yaml`:

```bash
# Edit eval.yaml locally, then mount it
docker run --gpus all \
  -v $HOME/data/input:/data/input \
  -v $HOME/data/output:/data/output \
  -v $HOME/my_eval.yaml:/opt/segmentanytree/conf/eval.yaml \
  harbor.cyverse.org/vice/segmentanytree:cuda12 \
  bash scripts/run_inference.sh /data/input /data/output true
```

### Tuning guidance

| Scenario | Parameter to adjust | Recommendation |
|----------|-------------------|----------------|
| GPU out of memory | `data.first_subsampling` | Increase to 0.3 or 0.4 (coarser voxels) |
| GPU out of memory | `batch_size` | Keep at 1 (already minimal) |
| Too many small/spurious trees | `tracker_options.min_score` | Increase to 0.3–0.7 |
| Missing small trees | `tracker_options.min_score` | Decrease toward 0.0 |
| Very dense point cloud (UAS/TLS) | `data.first_subsampling` | Increase to 0.3+ or pre-sparsify |
| Very sparse point cloud (ALS) | `data.first_subsampling` | Keep at 0.2 or decrease to 0.15 |
| Custom-trained model | `checkpoint_dir` | Point to your training output directory |

### Configuration files

All configuration lives in `conf/`:

```
conf/
├── eval.yaml                          # Main inference config
├── config.yaml                        # Main training config
├── data/panoptic/treeins_rad8.yaml    # Data loading & transforms
├── models/panoptic/                   # Model architecture configs
├── training/                          # Training hyperparameters
└── lr_scheduler/                      # Learning rate schedules
```

See [Architecture](architecture.md) for details on the Hydra configuration system.

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for common issues.
