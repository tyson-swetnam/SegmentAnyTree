# Docker Guide

## Pulling Pre-built Images

Pre-built images are available on the CyVerse Harbor registry:

=== "CUDA 11.8 (recommended)"

    ```bash
    docker pull harbor.cyverse.org/vice/segmentanytree:cuda11
    ```

=== "CUDA 12.4 (experimental)"

    ```bash
    docker pull harbor.cyverse.org/vice/segmentanytree:cuda12
    ```

## Image Variants

| | CUDA 11.8 (recommended) | CUDA 12.4 (experimental) |
|---|---|---|
| **Registry** | `harbor.cyverse.org/vice/segmentanytree:cuda11` | `harbor.cyverse.org/vice/segmentanytree:cuda12` |
| **Dockerfile** | `docker/Dockerfile.cuda11` | `docker/Dockerfile.cuda12` |
| **PyTorch** | 2.1.2 | 2.5.1 |
| **Sparse Backend** | MinkowskiEngine + torchsparse (source build) | SpConv v2.x (pip install) |
| **Clustering** | torch-points-kernels (C++/CUDA) | Pure PyTorch (`sat.clustering`) |
| **Build Time** | ~40 min | ~15 min |
| **GPU Support** | Volta through Hopper (sm_70 - sm_90) | Volta through Blackwell (sm_70 - sm_100) |
| **Driver** | 525+ | 525+ |
| **Inference Quality** | Validated (47/64 GT match) | Experimental (semantic bias) |

!!! warning "CUDA 12 is experimental"
    The SpConv backend auto-converts MinkowskiEngine weights at load time, but produces biased semantic segmentation (~100% tree classification). Instance segmentation produces clusters but they don't match ground truth. Use CUDA 11 for production inference.

See [docker/README.md](../docker/README.md) for additional comparison details.

## Building Locally

```bash
git clone https://github.com/tyson-swetnam/SegmentAnyTree.git
cd SegmentAnyTree

# IMPORTANT: Pull model weights before building
git lfs install
git lfs pull --include="model_file/PointGroup-PAPER.pt"

# Verify weights are real (should be ~665 MB, not 134 bytes)
make verify-weights

# CUDA 12.4 (recommended)
make build-cuda12

# CUDA 11.8 (legacy)
make build-cuda11

# Or directly:
docker build -f docker/Dockerfile.cuda12 -t segmentanytree:cuda12 .
docker build -f docker/Dockerfile.cuda11 -t segmentanytree:cuda11 .
```

!!! warning "Git LFS required"
    The model weights file (`model_file/PointGroup-PAPER.pt`) is stored with Git LFS. The Dockerfiles now include a build-time check that **fails the build** if the weights are LFS pointers. If you don't have `git-lfs`, you can download the weights directly:
    ```bash
    curl -L -o model_file/PointGroup-PAPER.pt \
      "https://github.com/SmartForest-no/SegmentAnyTree/raw/main/model_file/PointGroup-PAPER.pt"
    ```
    The CUDA 12 Dockerfile automatically runs `migrate_weights.py` during the build to generate SpConv-format weights.

### Build requirements

- Docker 20.10+ with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA driver 525+ (supports both CUDA 11.8 and 12.4 containers)
- **Git LFS** (`git-lfs`) for pulling model weights
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

The CUDA 12.4 image automatically converts weights to SpConv format during the Docker build — no manual migration needed.

For local (non-Docker) usage with SpConv, convert manually:

```bash
python scripts/migrate_weights.py \
  --input model_file/PointGroup-PAPER.pt \
  --output model_file/PointGroup-PAPER-spconv.pt
```

The CUDA 11.8 image uses the original MinkowskiEngine weights directly.

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

## CyVerse VICE

Both images include `/bin/entry.sh` for use as an entrypoint in CyVerse VICE deployments. The script automatically:

- Configures iRODS for Data Store access (`~/.irods/irods_environment.json`)
- Copies the user's `.gitconfig` and `.ssh` keys from the Data Store if available
- Starts JupyterLab

To use it in a VICE app definition, set the entrypoint to `bash /bin/entry.sh`.

## Local Build (without Docker)

See [local-build.md](local-build.md) for the full local build guide.

## Singularity / Apptainer

See [singularity.md](singularity.md) for HPC usage.
