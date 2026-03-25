
![SegmentAnyTree_logo](https://github.com/user-attachments/assets/8849a4b2-3bb3-4c6d-b1f1-13f91efc0936)

# SegmentAnyTree (Fork)

> **This is a fork of [SmartForest-no/SegmentAnyTree](https://github.com/SmartForest-no/SegmentAnyTree)** focused on modernizing the Docker build environment, adding CUDA 12 support, and improving computation speed for inference and training.

Deep learning framework for **tree instance segmentation from 3D LiDAR point clouds**.

Based on [Wielgosz et al. (2024) "SegmentAnyTree: A sensor and platform agnostic deep learning model for tree segmentation using laser scanning data"](https://www.sciencedirect.com/science/article/pii/S0034425724003936), Remote Sensing of Environment.

Built on the [torch-points3d](https://github.com/torch-points3d/torch-points3d) framework using PointGroup panoptic segmentation architecture.

## Fork Highlights

- **CUDA 12.4 + SpConv v2.x** — Replaced MinkowskiEngine (unmaintained) with pip-installable SpConv. No source compilation needed.
- **Multi-GPU parallel inference** — Process N files across N GPUs simultaneously
- **COPC octant-parallel processing** — Split a single large COPC file across GPUs using octree spatial indexing
- **GPU memory auto-tuning** — Auto-scales `cluster_nsample`, `num_workers`, and `batch_size` based on detected GPU memory (e.g., A100 80GB → paper-quality settings)
- **DDP training** — Multi-GPU training via `torchrun`
- **Numba-accelerated clustering** — 50-100x speedup on union-find region growing
- **Pure PyTorch dependencies** — Removed all unmaintained C++/CUDA libraries (MinkowskiEngine, torchsparse, torch-points-kernels)
- **COPC output** — Cloud-Optimized Point Cloud format for efficient streaming and web visualization

## Quick Start

```bash
# 1. Create directories
mkdir -p $HOME/segmentanytree/input $HOME/segmentanytree/output

# 2. Copy your .las/.laz/.ply files into the input directory

# 3. Run inference (single GPU)
docker run --gpus all \
  -v $HOME/segmentanytree/input:/data/input \
  -v $HOME/segmentanytree/output:/data/output \
  segmentanytree:cuda12 \
  bash scripts/run_inference.sh /data/input /data/output true

# 4. Multi-GPU inference (distributes files across all GPUs)
docker run --gpus all \
  -v $HOME/segmentanytree/input:/data/input \
  -v $HOME/segmentanytree/output:/data/output \
  segmentanytree:cuda12 \
  bash scripts/run_inference_parallel.sh /data/input /data/output

# 5. Results in $HOME/segmentanytree/output/final_results/
```

### Interactive Mode (JupyterLab)

```bash
docker run --gpus all -p 8888:8888 \
  -v $HOME/segmentanytree/input:/data/input \
  -v $HOME/segmentanytree/output:/data/output \
  segmentanytree:cuda12
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

### Local Development (no Docker)

```bash
# Using conda/mamba
mamba env create -f environment.yml
conda activate sat

# Or using pip/venv — see docs/local-build.md
```

See [docker/README.md](docker/README.md) for details on both image variants.

## Weight Migration

If using the CUDA 12.4 image with pre-trained weights from the original MinkowskiEngine-based model:

```bash
python scripts/migrate_weights.py \
  --input model_file/PointGroup-PAPER.pt \
  --output model_file/PointGroup-PAPER-spconv.pt
```

This reshapes kernel tensors from ME format `(K³, C_in, C_out)` to SpConv format `(C_out, kD, kH, kW, C_in)` and renames state dict keys. The CUDA 11.8 image uses the original weights directly.

## Training

```bash
# Single GPU
python train.py task=panoptic data=panoptic/treeins \
  models=panoptic/area4_ablation_3heads \
  model_name=PointGroup-PAPER \
  training=treeins \
  job_name=my_experiment

# Multi-GPU DDP (4 GPUs)
make train-ddp GPUS=4
```

## Documentation

| Guide | Description |
|-------|-------------|
| [docs/quickstart.md](docs/quickstart.md) | Get running in 5 minutes |
| [docs/inference.md](docs/inference.md) | Inference pipeline, multi-GPU, parameter tuning |
| [docs/training.md](docs/training.md) | Training, data preparation, DDP |
| [docs/docker.md](docs/docker.md) | Docker build, run, CUDA 11 vs 12 comparison |
| [docs/local-build.md](docs/local-build.md) | Local development setup (conda or venv) |
| [docs/architecture.md](docs/architecture.md) | Model architecture and design |
| [docs/singularity.md](docs/singularity.md) | HPC / SLURM / Singularity usage |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common issues and solutions |

## Project Structure

```
SegmentAnyTree/
├── sat/                    # Python package (pipeline, I/O, metrics, clustering, compat shims)
├── torch_points3d/         # Core ML framework (PointGroup model, training, datasets)
├── conf/                   # Hydra configuration (model, data, training, GPU profiles)
├── model_file/             # Pre-trained PointGroup-PAPER checkpoint
├── scripts/                # Inference, parallel, COPC, batch, migration scripts
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
| **Clustering** | Numba-accelerated PyTorch | torch-points-kernels (C++) |
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
