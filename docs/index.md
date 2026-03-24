# SegmentAnyTree

Deep learning framework for **tree instance segmentation from 3D LiDAR point clouds**.

Based on [Wielgosz et al. (2024) "SegmentAnyTree: A sensor and platform agnostic deep learning model for tree segmentation using laser scanning data"](https://www.sciencedirect.com/science/article/pii/S0034425724003936), *Remote Sensing of Environment*.

Built on the [torch-points3d](https://github.com/torch-points3d/torch-points3d) framework using PointGroup panoptic segmentation architecture.

---

## What it does

SegmentAnyTree takes 3D point cloud data (LAS, LAZ, or PLY) and segments individual trees using panoptic segmentation — combined semantic classification (tree vs. non-tree) and instance segmentation (each tree gets a unique ID).

```
Input Point Cloud (.las/.laz/.ply)
  → SegmentAnyTree
  → Per-tree Instance Labels (.copc.laz)
```

The output preserves all original point attributes and adds:

- **PredSemantic**: `0` = unclassified, `1` = non-tree, `2` = tree
- **PredInstance**: unique integer ID per detected tree

Output files are in [COPC](https://copc.io/) (Cloud-Optimized Point Cloud) format for efficient streaming and visualization.

## Quick start

=== "Harbor Registry"

    ```bash
    # Create directories
    mkdir -p $HOME/segmentanytree/input $HOME/segmentanytree/output

    # Copy your .las/.laz/.ply files into input/

    # Pull and run
    docker pull harbor.cyverse.org/vice/segmentanytree:cuda12
    docker run --gpus all \
      -v $HOME/segmentanytree/input:/data/input \
      -v $HOME/segmentanytree/output:/data/output \
      harbor.cyverse.org/vice/segmentanytree:cuda12 \
      bash scripts/run_inference.sh /data/input /data/output true

    # Results in $HOME/segmentanytree/output/final_results/
    ```

=== "Local Build"

    ```bash
    # Create directories
    mkdir -p $HOME/segmentanytree/input $HOME/segmentanytree/output

    # Copy your .las/.laz/.ply files into input/

    # Build and run
    git clone https://github.com/tyson-swetnam/SegmentAnyTree.git
    cd SegmentAnyTree
    docker build -f docker/Dockerfile.cuda12 -t segmentanytree:cuda12 .
    docker run --gpus all \
      -v $HOME/segmentanytree/input:/data/input \
      -v $HOME/segmentanytree/output:/data/output \
      segmentanytree:cuda12 \
      bash scripts/run_inference.sh /data/input /data/output true

    # Results in $HOME/segmentanytree/output/final_results/
    ```

See the [Quick Start guide](quickstart.md) for detailed setup instructions.

## Documentation

| Guide | Description |
|-------|-------------|
| [Quick Start](quickstart.md) | Get running in 5 minutes |
| [Docker Guide](docker.md) | Docker build, run, and configuration |
| [Local Build](local-build.md) | Install locally on Linux with conda |
| [Scientific Workflow](workflow.md) | End-to-end: LAZ input to COPC output |
| [Inference](inference.md) | Detailed inference pipeline |
| [Training](training.md) | Training and data preparation |
| [Notebooks](notebooks.md) | Interactive Jupyter notebook examples |
| [Example Data](example-data.md) | Download and test with example LiDAR data |
| [Architecture](architecture.md) | Model architecture and design |
| [HPC / Singularity](singularity.md) | SLURM and Singularity/Apptainer usage |
| [Troubleshooting](troubleshooting.md) | Common issues and solutions |

## Docker image variants

| Component | CUDA 12.4 (recommended) | CUDA 11.8 (legacy) |
|-----------|------------------------|---------------------|
| **Registry** | `harbor.cyverse.org/vice/segmentanytree:cuda12` | `harbor.cyverse.org/vice/segmentanytree:cuda11` |
| **Base** | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` |
| **Python** | 3.10 | 3.10 |
| **PyTorch** | 2.5.1 | 2.1.2 |
| **Sparse backend** | SpConv v2.x | MinkowskiEngine |
| **JupyterLab** | 4.x | 4.x |
| **Build time** | ~15 min | ~40 min |
| **GPU support** | Volta through Blackwell (sm_70 - sm_100) | Volta through Hopper (sm_70 - sm_90) |

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
