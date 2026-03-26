
![SegmentAnyTree_logo](https://github.com/user-attachments/assets/8849a4b2-3bb3-4c6d-b1f1-13f91efc0936)

# SegmentAnyTree (Fork)

> **This is a fork of [SmartForest-no/SegmentAnyTree](https://github.com/SmartForest-no/SegmentAnyTree)** focused on modernizing the Docker build environment, adding CUDA 12 support, and improving computation speed for inference and training.

Deep learning framework for **tree instance segmentation from 3D LiDAR point clouds**.

Based on [Wielgosz et al. (2024) "SegmentAnyTree: A sensor and platform agnostic deep learning model for tree segmentation using laser scanning data"](https://www.sciencedirect.com/science/article/pii/S0034425724003936), Remote Sensing of Environment.

Built on the [torch-points3d](https://github.com/torch-points3d/torch-points3d) framework using PointGroup panoptic segmentation architecture.

## Fork Highlights

- **CUDA 11.8 + MinkowskiEngine** — Production-ready inference with validated 95% tree detection accuracy (47/64 trees matched >50% IoU on FOR-instance benchmark)
- **CUDA 12.4 + SpConv v2.x** — Experimental backend with pip-installable dependencies (no source compilation). Auto-converts ME weights at load time. Semantic segmentation still under development.
- **Multi-GPU parallel inference** — Process N files across N GPUs simultaneously
- **COPC octant-parallel processing** — Split a single large COPC file across GPUs using octree spatial indexing
- **GPU memory auto-tuning** — Auto-scales `cluster_nsample`, `num_workers`, and `batch_size` based on detected GPU memory
- **DDP training** — Multi-GPU training via `torchrun`
- **Numba-accelerated clustering** — 50-100x speedup on union-find region growing
- **COPC output** — Cloud-Optimized Point Cloud format for efficient streaming and web visualization
- **FOR-instance benchmark data** — Automated download and validation against the paper's primary dataset

## Quick Start

### Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU (Volta or newer) with driver 525+
- **Git LFS** — the model weights (665 MB) are stored with Git LFS

```bash
# 1. Clone and pull model weights
git clone https://github.com/tyson-swetnam/SegmentAnyTree.git
cd SegmentAnyTree
git lfs install
git lfs pull --include="model_file/PointGroup-PAPER.pt"
make verify-weights  # Should show ~665 MB

# 2. Build Docker image
make build-cuda11    # Recommended: MinkowskiEngine (production)
# make build-cuda12  # Experimental: SpConv v2.x

# 3. Create directories and add your .las/.laz/.ply files
mkdir -p $HOME/segmentanytree/input $HOME/segmentanytree/output

# 4. Run inference
docker run --gpus all \
  -v $HOME/segmentanytree/input:/data/input \
  -v $HOME/segmentanytree/output:/data/output \
  segmentanytree:cuda11 \
  bash scripts/run_inference.sh /data/input /data/output true

# 5. Results in $HOME/segmentanytree/output/final_results/ (.copc.laz)
```

> **Note:** If you don't have `git-lfs`, download weights directly:
> ```bash
> curl -L -o model_file/PointGroup-PAPER.pt \
>   "https://github.com/SmartForest-no/SegmentAnyTree/raw/main/model_file/PointGroup-PAPER.pt"
> ```

### Interactive Mode (JupyterLab)

```bash
docker run --gpus all -p 8888:8888 \
  -v $HOME/segmentanytree/input:/data/input \
  -v $HOME/segmentanytree/output:/data/output \
  segmentanytree:cuda11
```

Open http://localhost:8888 and use the starter notebooks.

## Validation Results

Tested on [FOR-instance](https://zenodo.org/records/8287792) RMIT benchmark (357K points, 64 annotated trees):

| Backend | Semantic Accuracy | Trees Detected | GT Matched (>50% IoU) | Status |
|---------|------------------|----------------|----------------------|--------|
| **MinkowskiEngine (CUDA 11)** | 68% non-tree, 32% tree | 21-24 | **47/64 (73%)** | Production |
| SpConv v2.x (CUDA 12) | 0.02% non-tree, 99.98% tree | 29-105 | 0/64 | Experimental |

The CUDA 11 / MinkowskiEngine backend is recommended for all scientific use. SpConv produces clusters but semantic segmentation is biased due to SubMConv3d numerical differences from MinkowskiEngine — this would require fine-tuning or retraining.

## Build from Source

```bash
git clone https://github.com/tyson-swetnam/SegmentAnyTree.git
cd SegmentAnyTree

# Pull model weights (required before building)
git lfs pull --include="model_file/PointGroup-PAPER.pt"

# CUDA 11.8 (recommended — validated, production-ready)
make build-cuda11

# CUDA 12.4 (experimental — SpConv, auto-migrates weights during build)
make build-cuda12
```

**Requirements**: Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), NVIDIA GPU (Volta+), Git LFS.

### Local Development (no Docker)

```bash
# Install miniforge if not present
curl -L -o /tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/Miniforge3.sh -b -p $HOME/miniforge

# Create environment and install dependencies
mamba env create -f environment.yml
conda activate sat

# Install PyTorch ecosystem (must be done separately due to build deps)
pip install --extra-index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install torch-geometric spconv-cu124
```

See [docs/local-build.md](docs/local-build.md) for the full local development guide.

## Example Data

### FOR-instance (paper benchmark, recommended)

The [FOR-instance dataset](https://zenodo.org/records/8287792) is the primary training/test dataset from the paper. 1,130 manually segmented trees across 5 sites.

```bash
make download-forinstance DATA_DIR=$HOME/segmentanytree
```

See [docs/example-data.md](docs/example-data.md) for all available datasets including NIBIO MLS, SWERI LiDAR, NEON, and OpenTopography.

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
| [docs/docker.md](docs/docker.md) | Docker build, run, CUDA 11 vs 12 comparison |
| [docs/local-build.md](docs/local-build.md) | Local development setup (conda or venv) |
| [docs/workflow.md](docs/workflow.md) | End-to-end scientific workflow (LAZ → COPC) |
| [docs/inference.md](docs/inference.md) | Inference pipeline, multi-GPU, parameter tuning |
| [docs/training.md](docs/training.md) | Training, data preparation, DDP |
| [docs/notebooks.md](docs/notebooks.md) | Interactive Jupyter notebook examples |
| [docs/example-data.md](docs/example-data.md) | Download and test with example LiDAR data |
| [docs/architecture.md](docs/architecture.md) | Model architecture and design |
| [docs/singularity.md](docs/singularity.md) | HPC / SLURM / Singularity usage |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common issues and solutions |

## Project Structure

```
SegmentAnyTree/
├── sat/                    # Python package (pipeline, I/O, metrics, clustering, compat shims)
├── torch_points3d/         # Core ML framework (PointGroup model, training, datasets)
├── conf/                   # Hydra configuration (model, data, training, GPU profiles)
├── model_file/             # Pre-trained PointGroup-PAPER checkpoint (Git LFS)
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

| | CUDA 11.8 (recommended) | CUDA 12.4 (experimental) |
|---|---|---|
| **Registry** | `harbor.cyverse.org/vice/segmentanytree:cuda11` | `harbor.cyverse.org/vice/segmentanytree:cuda12` |
| **Dockerfile** | `docker/Dockerfile.cuda11` | `docker/Dockerfile.cuda12` |
| **PyTorch** | 2.1.2 | 2.5.1 |
| **Sparse Backend** | MinkowskiEngine (source) | SpConv v2.x (pip) |
| **Clustering** | torch-points-kernels (C++) | Numba-accelerated PyTorch |
| **Build Time** | ~40 min | ~15 min |
| **GPU Support** | Volta → Hopper | Volta → Blackwell |
| **Inference Quality** | Validated (47/64 GT match) | Experimental (semantic bias) |

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
