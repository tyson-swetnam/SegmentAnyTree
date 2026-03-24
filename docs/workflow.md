# Scientific Workflow

End-to-end guide: from raw LiDAR data to per-tree segmentation results.

## Overview

```mermaid
graph LR
    A[Acquire LiDAR Data] --> B[Prepare Input Files]
    B --> C[Run Inference]
    C --> D[COPC Output]
    D --> E[Visualize & Analyze]
```

## 1. Acquiring LiDAR Data

SegmentAnyTree works with point cloud data from any LiDAR sensor platform:

| Platform | Abbreviation | Typical Use |
|----------|-------------|-------------|
| Airborne Laser Scanning | ALS | Landscape-scale forest inventory |
| Uncrewed Aerial System | UAS | Plot-level high-density scanning |
| Terrestrial Laser Scanning | TLS | Single-tree detail (stem, branches) |
| Mobile Laser Scanning | MLS | Road corridor and urban trees |

### Supported file formats

| Format | Extension | Notes |
|--------|-----------|-------|
| LAS | `.las` | Standard ASPRS LiDAR format |
| LAZ | `.laz` | Compressed LAS (recommended for storage) |
| PLY | `.ply` | Stanford polygon format |

### Public data sources

- **CyVerse Data Store** — Example ALS, UAS, and MLS clips with pre-segmented outputs. See [Example Data](example-data.md).
- **[NEON AOP](https://www.neonscience.org/)** — Free airborne LiDAR across US ecological sites
- **[OpenTopography](https://opentopography.org/)** — Community LiDAR datasets searchable by location

!!! tip "Start with example data"
    Download the ALS clip (15 MB) from the [Example Data](example-data.md) page to verify your setup before processing your own data.

## 2. Preparing Input Files

### Directory setup

Create a directory with your input point cloud files:

```bash
mkdir -p $HOME/segmentanytree/input
mkdir -p $HOME/segmentanytree/output

# Copy your files
cp /path/to/your/*.laz $HOME/segmentanytree/input/
```

### File naming

The pipeline automatically sanitizes filenames (replaces spaces and dashes with underscores). No manual renaming is needed.

### Large files

For point clouds with more than 500 million points, split into spatial tiles for best performance and to avoid GPU memory limits. Use PDAL or LAStools to tile:

```bash
# Split into 100m x 100m tiles with PDAL
pdal split --capacity 0 --length 100 input.laz output_dir/
```

### Reducing point density

For very dense point clouds, reduce density before processing:

```bash
python -m sat.preprocessing.sparsify -i input/ -o sparse_input/ -d 100
```

This randomly samples to ~100 points/m². See [Troubleshooting](troubleshooting.md) for GPU memory management tips.

### Coordinate reference system

Any projected CRS works (UTM recommended). The pipeline automatically converts to a local coordinate system for model inference and restores the original CRS in the output.

## 3. Running Inference

=== "Docker (Harbor)"

    ```bash
    docker pull harbor.cyverse.org/vice/segmentanytree:cuda12
    docker run --gpus all \
      -v $HOME/segmentanytree/input:/data/input \
      -v $HOME/segmentanytree/output:/data/output \
      harbor.cyverse.org/vice/segmentanytree:cuda12 \
      bash scripts/run_inference.sh /data/input /data/output true
    ```

=== "Docker (Local Build)"

    ```bash
    docker run --gpus all \
      -v $HOME/segmentanytree/input:/data/input \
      -v $HOME/segmentanytree/output:/data/output \
      segmentanytree:cuda12 \
      bash scripts/run_inference.sh /data/input /data/output true
    ```

=== "Singularity (HPC)"

    ```bash
    singularity exec --nv \
      --bind $HOME/segmentanytree/input:/data/input \
      --bind $HOME/segmentanytree/output:/data/output \
      segmentanytree-cuda12.sif \
      bash /opt/segmentanytree/scripts/run_inference.sh /data/input /data/output true
    ```

=== "Local Install"

    ```bash
    conda activate sat
    export SAT_ROOT=/path/to/SegmentAnyTree
    export SPARSE_BACKEND=spconv
    bash scripts/run_inference.sh $HOME/segmentanytree/input $HOME/segmentanytree/output true
    ```

### Batch processing

For many files, process in configurable batches to manage memory:

```bash
bash scripts/run_batch_inference.sh /path/to/input /path/to/output 10
```

### Multi-GPU

Distribute files across all available GPUs:

```bash
bash scripts/run_inference_parallel.sh /data/input /data/output
```

See the [Inference](inference.md) guide for full pipeline details and the Python API.

## 4. Understanding Output

### Output directory structure

```
output/
├── input_data/          # Sanitized copies of input files
├── utm2local/           # Coordinate-transformed PLY files
├── ...                  # Intermediate model outputs
└── final_results/       # Final segmented point clouds
    ├── file1.copc.laz
    ├── file2.copc.laz
    └── ...
```

### Output fields

Each output file contains all original point attributes plus two new dimensions:

| Field | Type | Values |
|-------|------|--------|
| **PredSemantic** | uint8 | `0` = unclassified, `1` = non-tree, `2` = tree |
| **PredInstance** | uint16 | `0` = unassigned, `1`+ = unique tree ID |

### COPC format

Output files are in **COPC** (Cloud-Optimized Point Cloud) format (`.copc.laz`):

- Standard LAZ 1.4 with an embedded spatial octree index
- Enables efficient HTTP range-request streaming for web viewers
- Supported by QGIS 3.26+, CloudCompare, Potree, and [copc.io viewer](https://viewer.copc.io/)
- Compatible with all standard LAS/LAZ tools (PDAL, LAStools, laspy)

If PDAL is not available, the pipeline outputs standard `.laz` files instead.

## 5. Visualization

### CloudCompare

1. Open your `.copc.laz` file in [CloudCompare](https://www.danielgm.net/cc/)
2. In the Properties panel, select **PredInstance** as the active scalar field
3. Set the color ramp to **Random** for distinct tree colors
4. Each color represents an individual segmented tree

To view semantic classes instead, select **PredSemantic** as the scalar field:
- Blue (0) = unclassified
- Green (1) = non-tree (ground, low vegetation)
- Red (2) = tree

### QGIS

1. Open [QGIS](https://qgis.org/) 3.26 or later
2. Drag your `.copc.laz` file into the map canvas (or Layer → Add Layer → Add Point Cloud Layer)
3. In Layer Styling, change the renderer to **Classification** and select the **PredInstance** attribute
4. COPC files load efficiently — QGIS reads only the spatial tiles needed for the current view

### Potree / Web Viewers

COPC files can be served directly over HTTP for browser-based 3D visualization:

- **[copc.io viewer](https://viewer.copc.io/)** — paste a public URL to any COPC file
- **[Potree](https://potree.github.io/)** — self-hosted web viewer for large point clouds

To serve locally:

```bash
# Simple HTTP server
cd $HOME/segmentanytree/output/final_results
python -m http.server 8080
# Then open the copc.io viewer and point it to http://localhost:8080/file.copc.laz
```

## 6. Downstream Analysis

Extract per-tree metrics from the segmented point cloud using Python:

```python
import laspy
import numpy as np
import pandas as pd

# Load segmented point cloud
las = laspy.read("output/final_results/my_forest.copc.laz")

# Get unique tree IDs (skip 0 = unassigned)
tree_ids = np.unique(las.PredInstance)
tree_ids = tree_ids[tree_ids > 0]

# Extract per-tree metrics
trees = []
for tid in tree_ids:
    mask = las.PredInstance == tid
    x, y, z = las.x[mask], las.y[mask], las.z[mask]
    trees.append({
        "tree_id": int(tid),
        "n_points": int(mask.sum()),
        "height_m": float(z.max() - z.min()),
        "crown_x_m": float(x.max() - x.min()),
        "crown_y_m": float(y.max() - y.min()),
        "centroid_x": float(x.mean()),
        "centroid_y": float(y.mean()),
        "z_max": float(z.max()),
    })

df = pd.DataFrame(trees)
print(f"Detected {len(df)} trees")
print(df.describe())

# Export to CSV
df.to_csv("tree_metrics.csv", index=False)
```

This gives you a table of tree-level metrics suitable for forest inventory analysis, allometric modeling, or GIS integration.

### Export to GeoJSON

```python
import json

features = []
for _, row in df.iterrows():
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [row.centroid_x, row.centroid_y]},
        "properties": {
            "tree_id": row.tree_id,
            "height_m": round(row.height_m, 2),
            "crown_x_m": round(row.crown_x_m, 2),
            "crown_y_m": round(row.crown_y_m, 2),
            "n_points": row.n_points,
        }
    })

geojson = {"type": "FeatureCollection", "features": features}
with open("tree_locations.geojson", "w") as f:
    json.dump(geojson, f, indent=2)
```

Load `tree_locations.geojson` in QGIS or any GIS tool to overlay tree positions on maps.

## 7. Tips & Best Practices

- **Start small**: Process the ALS example file (15 MB) first to verify your setup
- **One file first**: Run a single file before batching to catch issues early
- **Monitor GPU memory**: Use `nvidia-smi` during inference. If you get OOM errors, reduce point density with `sat.preprocessing.sparsify` or split into smaller tiles
- **Use COPC for sharing**: COPC files are self-contained and streamable — ideal for collaboration and web visualization
- **Keep originals**: The pipeline never modifies input files. Output is always written to a separate directory
- **Check the notebooks**: The [Jupyter notebooks](notebooks.md) provide interactive examples for each stage of the workflow
