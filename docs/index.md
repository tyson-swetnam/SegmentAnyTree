# SegmentAnyTree

Deep learning framework for **tree instance segmentation from 3D LiDAR point clouds**.

Based on [Wielgosz et al. (2024) "SegmentAnyTree: A sensor and platform agnostic deep learning model for tree segmentation using laser scanning data"](https://www.sciencedirect.com/science/article/pii/S0034425724003936), *Remote Sensing of Environment*.

Built on the [torch-points3d](https://github.com/torch-points3d/torch-points3d) framework using PointGroup panoptic segmentation architecture.

---

## What it does

SegmentAnyTree takes 3D point cloud data (LAS, LAZ, or PLY) and segments individual trees using panoptic segmentation — combined semantic classification (tree vs. non-tree) and instance segmentation (each tree gets a unique ID).

```
Input Point Cloud → SegmentAnyTree → Per-tree Instance Labels
```

The output preserves all original point attributes and adds:

- **PredSemantic**: `0` = unclassified, `1` = non-tree, `2` = tree
- **PredInstance**: unique integer ID per detected tree

## Quick start

```bash
# Create directories
mkdir -p $HOME/segmentanytree/input $HOME/segmentanytree/output

# Copy your .las/.laz/.ply files into input/

# Run inference
docker run --gpus all \
  -v $HOME/segmentanytree/input:/data/input \
  -v $HOME/segmentanytree/output:/data/output \
  segmentanytree:latest \
  bash scripts/run_inference.sh /data/input /data/output true

# Results in $HOME/segmentanytree/output/final_results/
```

See the [Quick Start guide](quickstart.md) for detailed setup instructions.

## Documentation

| Guide | Description |
|-------|-------------|
| [Quick Start](quickstart.md) | Get running in 5 minutes |
| [Docker Guide](docker.md) | Docker build, run, and configuration |
| [Inference](inference.md) | Detailed inference pipeline |
| [Training](training.md) | Training and data preparation |
| [Example Data](example-data.md) | Download and test with example LiDAR data |
| [Architecture](architecture.md) | Model architecture and design |
| [HPC / Singularity](singularity.md) | SLURM and Singularity/Apptainer usage |
| [Troubleshooting](troubleshooting.md) | Common issues and solutions |

## Docker image stack

| Component | Version |
|-----------|---------|
| Ubuntu | 22.04 |
| CUDA | 11.8.0 + cuDNN 8 |
| Python | 3.10 |
| PyTorch | 2.1.2 |
| JupyterLab | 4.x |

!!! note "Why CUDA 11.8?"
    MinkowskiEngine (a core dependency for sparse 3D convolutions) is incompatible with CUDA 12.x due to unresolved `libcu++` template conflicts. Your NVIDIA driver (525+) supports running CUDA 11.8 containers.

## Citation

```bibtex
@article{WIELGOSZ2024114367,
  title = {SegmentAnyTree: A sensor and platform agnostic deep learning model
           for tree segmentation using laser scanning data},
  journal = {Remote Sensing of Environment},
  volume = {313},
  pages = {114367},
  year = {2024},
  doi = {https://doi.org/10.1016/j.rse.2024.114367},
  author = {Maciej Wielgosz and Stefano Puliti and Binbin Xiang
            and Konrad Schindler and Rasmus Astrup},
}
```
