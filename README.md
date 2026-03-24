
![SegmentAnyTree_logo](https://github.com/user-attachments/assets/8849a4b2-3bb3-4c6d-b1f1-13f91efc0936)

# SegmentAnyTree (Fork)

> **This is a fork of [SmartForest-no/SegmentAnyTree](https://github.com/SmartForest-no/SegmentAnyTree)** focused on updating the Docker build environment and finding additional improvements in computation speed for inference and training workflows.

Deep learning framework for **tree instance segmentation from 3D LiDAR point clouds**.

Based on [Wielgosz et al. (2024) "SegmentAnyTree: A sensor and platform agnostic deep learning model for tree segmentation using laser scanning data"](https://www.sciencedirect.com/science/article/pii/S0034425724003936), Remote Sensing of Environment.

Built on the [torch-points3d](https://github.com/torch-points3d/torch-points3d) framework using PointGroup panoptic segmentation architecture.

## Fork Goals

- **Docker build modernization** — Update base images, dependencies, and build process for reliability and reproducibility
- **Computation speed improvements** — Profile and optimize inference/training pipelines, clustering, and I/O bottlenecks
- **Dependency updates** — Evaluate newer CUDA, PyTorch, and sparse convolution library versions for compatibility and performance

## Quick Start

```bash
# 1. Create directories
mkdir -p $HOME/segmentanytree/input $HOME/segmentanytree/output

# 2. Copy your .las/.laz/.ply files into the input directory

# 3. Run inference
docker run --gpus all \
  -v $HOME/segmentanytree/input:/data/input \
  -v $HOME/segmentanytree/output:/data/output \
  segmentanytree:latest \
  bash scripts/run_inference.sh /data/input /data/output true

# 4. Results in $HOME/segmentanytree/output/final_results/
```

# 5. Multi-GPU inference (if multiple GPUs available)
docker run --gpus all \
  -v $HOME/segmentanytree/input:/data/input \
  -v $HOME/segmentanytree/output:/data/output \
  segmentanytree:latest \
  bash scripts/run_inference_parallel.sh /data/input /data/output

### Interactive Mode (JupyterLab)

```bash
docker run --gpus all -p 8888:8888 \
  -v $HOME/segmentanytree/input:/data/input \
  -v $HOME/segmentanytree/output:/data/output \
  segmentanytree:latest
```

Open http://localhost:8888 and use the starter notebooks.

## Build from Source

```bash
git clone https://github.com/tyson-swetnam/SegmentAnyTree.git
cd SegmentAnyTree

# CUDA 12.4 (recommended — faster build, modern GPU support)
make build-cuda12

# CUDA 11.8 (legacy — for older drivers or MinkowskiEngine compatibility)
make build-cuda11
```

**Requirements**: Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), NVIDIA GPU (Volta+).

See [docker/README.md](docker/README.md) for details on both image variants.

## Training

```bash
# Prepare data: convert LAS to PLY with semantic labels
python -m sat.io.conversion --las_dir /path/to/data --output_dir /path/to/ply

# Train
python train.py task=panoptic data=panoptic/treeins \
  models=panoptic/area4_ablation_3heads \
  model_name=PointGroup-PAPER \
  training=treeins \
  job_name=my_experiment
```

## Documentation

| Guide | Description |
|-------|-------------|
| [docs/quickstart.md](docs/quickstart.md) | Get running in 5 minutes |
| [docs/inference.md](docs/inference.md) | Detailed inference pipeline |
| [docs/training.md](docs/training.md) | Training and data preparation |
| [docs/docker.md](docs/docker.md) | Docker build, run, and configuration |
| [docs/architecture.md](docs/architecture.md) | Model architecture and design |
| [docs/singularity.md](docs/singularity.md) | HPC / SLURM / Singularity usage |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common issues and solutions |

## Project Structure

```
SegmentAnyTree/
├── sat/                    # Python package (pipeline, I/O, metrics, preprocessing)
├── torch_points3d/         # Core ML framework (PointGroup model, training, datasets)
├── conf/                   # Hydra configuration (model, data, training configs)
├── model_file/             # Pre-trained PointGroup-PAPER checkpoint
├── scripts/                # Shell scripts (inference, batch, docker, training)
├── notebooks/              # JupyterLab starter notebooks
├── docker/                 # Dockerfiles for CUDA 11 and CUDA 12
├── docs/                   # Documentation
├── tests/                  # Automated tests
├── train.py                # Training entry point
├── eval.py                 # Evaluation/inference entry point
└── Dockerfile              # Default (CUDA 12.4 + SpConv v2.x)
```

## Docker Image Stack

| | CUDA 12.4 (default) | CUDA 11.8 (legacy) |
|---|---|---|
| **Dockerfile** | `docker/Dockerfile.cuda12` | `docker/Dockerfile.cuda11` |
| **PyTorch** | 2.5.1 | 2.1.2 |
| **Sparse Backend** | SpConv v2.x (pip) | MinkowskiEngine (source) |
| **Build Time** | ~15 min | ~40 min |
| **GPU Support** | Volta → Blackwell | Volta → Hopper |

## Issues

If you encounter problems, please [open an issue](https://github.com/tyson-swetnam/SegmentAnyTree/issues). For issues with the core model or algorithm, consider reporting to the [upstream repository](https://github.com/SmartForest-no/SegmentAnyTree/issues).

## Citation

```bibtex
@article{WIELGOSZ2024114367,
  title = {SegmentAnyTree: A sensor and platform agnostic deep learning model for tree segmentation using laser scanning data},
  journal = {Remote Sensing of Environment},
  volume = {313},
  pages = {114367},
  year = {2024},
  doi = {https://doi.org/10.1016/j.rse.2024.114367},
  author = {Maciej Wielgosz and Stefano Puliti and Binbin Xiang and Konrad Schindler and Rasmus Astrup},
}
```
