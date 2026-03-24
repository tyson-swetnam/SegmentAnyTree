# Docker Guide

## Pulling Pre-built Images

Pre-built images are available on the CyVerse Harbor registry:

=== "CUDA 12.4 (recommended)"

    ```bash
    docker pull harbor.cyverse.org/vice/segmentanytree:cuda12
    ```

=== "CUDA 11.8 (legacy)"

    ```bash
    docker pull harbor.cyverse.org/vice/segmentanytree:cuda11
    ```

## Image Variants

| | CUDA 12.4 (recommended) | CUDA 11.8 (legacy) |
|---|---|---|
| **Registry** | `harbor.cyverse.org/vice/segmentanytree:cuda12` | `harbor.cyverse.org/vice/segmentanytree:cuda11` |
| **Dockerfile** | `docker/Dockerfile.cuda12` | `docker/Dockerfile.cuda11` |
| **PyTorch** | 2.5.1 | 2.1.2 |
| **Sparse Backend** | SpConv v2.x (pip install) | MinkowskiEngine + torchsparse (source build) |
| **Clustering** | Pure PyTorch (`sat.clustering`) | torch-points-kernels (C++/CUDA) |
| **Build Time** | ~15 min | ~40 min |
| **GPU Support** | Volta through Blackwell (sm_70 - sm_100) | Volta through Hopper (sm_70 - sm_90) |
| **Driver** | 525+ | 525+ |

See [docker/README.md](../docker/README.md) for additional comparison details.

## Building Locally

```bash
git clone https://github.com/tyson-swetnam/SegmentAnyTree.git
cd SegmentAnyTree

# CUDA 12.4 (recommended)
make build-cuda12

# CUDA 11.8 (legacy)
make build-cuda11

# Or directly:
docker build -f docker/Dockerfile.cuda12 -t segmentanytree:cuda12 .
docker build -f docker/Dockerfile.cuda11 -t segmentanytree:cuda11 .
```

### Build requirements

- Docker 20.10+ with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA driver 525+ (supports both CUDA 11.8 and 12.4 containers)
- ~15 GB disk space for the final image

### Build without cache

```bash
docker build --no-cache -f docker/Dockerfile.cuda12 -t segmentanytree:cuda12 .
```

## Running

### JupyterLab (default)

=== "Harbor Registry"

    ```bash
    docker run --gpus all -p 8888:8888 \
      -v $HOME/data/input:/data/input \
      -v $HOME/data/output:/data/output \
      harbor.cyverse.org/vice/segmentanytree:cuda12
    ```

=== "Local Build"

    ```bash
    docker run --gpus all -p 8888:8888 \
      -v $HOME/data/input:/data/input \
      -v $HOME/data/output:/data/output \
      segmentanytree:cuda12
    ```

Open http://localhost:8888 in your browser.

### Batch inference

=== "Harbor Registry"

    ```bash
    docker run --gpus all \
      -v $HOME/data/input:/data/input \
      -v $HOME/data/output:/data/output \
      harbor.cyverse.org/vice/segmentanytree:cuda12 \
      bash scripts/run_inference.sh /data/input /data/output true
    ```

=== "Local Build"

    ```bash
    docker run --gpus all \
      -v $HOME/data/input:/data/input \
      -v $HOME/data/output:/data/output \
      segmentanytree:cuda12 \
      bash scripts/run_inference.sh /data/input /data/output true
    ```

### Interactive shell

=== "Harbor Registry"

    ```bash
    docker run --gpus all -it \
      -v $HOME/data/input:/data/input \
      -v $HOME/data/output:/data/output \
      harbor.cyverse.org/vice/segmentanytree:cuda12 bash
    ```

=== "Local Build"

    ```bash
    docker run --gpus all -it \
      -v $HOME/data/input:/data/input \
      -v $HOME/data/output:/data/output \
      segmentanytree:cuda12 bash
    ```

## Volume Mounts

| Container path | Purpose |
|---------------|---------|
| `/data/input` | Input point cloud files |
| `/data/output` | Output segmentation results |
| `/tmp/sat_cache` | Temporary processing files (optional mount) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SAT_ROOT` | `/opt/segmentanytree` | Project root directory |
| `SAT_DATA` | `/data` | Input/output data directory |
| `SAT_MODEL` | `/opt/segmentanytree/model_file` | Model checkpoint directory |
| `SAT_CACHE` | `/tmp/sat_cache` | Temporary files |
| `SPARSE_BACKEND` | `spconv` (CUDA 12) / not set (CUDA 11) | Sparse convolution backend |
| `SAT_GPU` | `0` | GPU device index for single-GPU inference |
| `NUM_GPUS` | auto-detect | Number of GPUs for parallel inference |

## Pre-trained Weights

The CUDA 12.4 image uses SpConv v2.x which has different weight format than MinkowskiEngine. To convert weights:

```bash
python scripts/migrate_weights.py \
  --input model_file/PointGroup-PAPER.pt \
  --output model_file/PointGroup-PAPER-spconv.pt
```

The CUDA 11.8 image uses the original weights directly.

## Dockerfile Architecture

Both images use a **multi-stage build**:

1. **Builder stage**: Installs PyTorch, GPU libraries, and Python dependencies
2. **Runtime stage**: Copies compiled libraries, adds JupyterLab, PDAL, and project code

### CUDA 12.4 build highlights
- SpConv v2.x installed via `pip install spconv-cu124` (no source compilation)
- Clustering uses pure PyTorch (no C++/CUDA compilation)
- ~20-30 min faster builds than CUDA 11.8

### CUDA 11.8 build highlights
- MinkowskiEngine, torchsparse, and torch-points-kernels compiled from source
- Requires `libsparsehash-dev` and `libopenblas-dev` build dependencies

## Local Build (without Docker)

See [local-build.md](local-build.md) for the full local build guide.

## Singularity / Apptainer

See [singularity.md](singularity.md) for HPC usage.
