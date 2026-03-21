# Quick Start

Get tree segmentation results in 5 minutes using the pre-built Docker image.

## Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- An NVIDIA GPU (Volta or newer: V100, A100, RTX 20xx/30xx/40xx)
- Point cloud files in `.las`, `.laz`, or `.ply` format

## Steps

### 1. Prepare directories

```bash
mkdir -p $HOME/segmentanytree/input
mkdir -p $HOME/segmentanytree/output
```

### 2. Add your point cloud files

Copy your `.las`, `.laz`, or `.ply` files into `$HOME/segmentanytree/input/`.

### 3. Run inference

```bash
docker run --gpus all \
  -v $HOME/segmentanytree/input:/data/input \
  -v $HOME/segmentanytree/output:/data/output \
  segmentanytree:latest \
  bash scripts/run_inference.sh /data/input /data/output true
```

### 4. Check results

Segmented files appear in `$HOME/segmentanytree/output/final_results/`:
- `*_instance_segmentation.las` — each tree gets a unique instance ID
- `*_semantic_segmentation.las` — per-point tree/non-tree classification

### 5. Interactive mode (JupyterLab)

```bash
docker run --gpus all -p 8888:8888 \
  -v $HOME/segmentanytree/input:/data/input \
  -v $HOME/segmentanytree/output:/data/output \
  segmentanytree:latest
```

Open http://localhost:8888 and use the notebooks in `notebooks/`.

## Output format

The output LAS files contain two extra dimensions:
- **PredSemantic**: `0` = unclassified, `1` = non-tree, `2` = tree
- **PredInstance**: unique integer ID per detected tree (0 = unassigned)

These files preserve all original point attributes (coordinates, intensity, return number, etc.) with UTM coordinates restored.
