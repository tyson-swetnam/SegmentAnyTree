# Docker Guide

## Pre-built Image

```bash
docker pull segmentanytree:latest
```

## Building From Source

```bash
git clone https://github.com/tyson-swetnam/SegmentAnyTree.git
cd SegmentAnyTree
docker build -t segmentanytree:latest .
```

Build takes 30-60 minutes due to GPU library compilation (MinkowskiEngine, torchsparse, torch-points-kernels).

### Build requirements

- Docker 20.10+ with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA driver 525+ (supports CUDA 11.8 containers)
- ~15 GB disk space for the final image

### Build without cache

If you encounter stale layer issues:

```bash
docker build --no-cache -t segmentanytree:latest .
```

## Running

### JupyterLab (default)

```bash
docker run --gpus all -p 8888:8888 \
  -v $HOME/data/input:/data/input \
  -v $HOME/data/output:/data/output \
  segmentanytree:latest
```

Open http://localhost:8888 in your browser.

### Batch inference

```bash
docker run --gpus all \
  -v $HOME/data/input:/data/input \
  -v $HOME/data/output:/data/output \
  segmentanytree:latest \
  bash scripts/run_inference.sh /data/input /data/output true
```

### Interactive shell

```bash
docker run --gpus all -it \
  -v $HOME/data/input:/data/input \
  -v $HOME/data/output:/data/output \
  segmentanytree:latest bash
```

### Helper script

```bash
bash scripts/run_docker.sh jupyter ~/data/input ~/data/output  # JupyterLab
bash scripts/run_docker.sh infer ~/data/input ~/data/output    # Batch inference
bash scripts/run_docker.sh shell ~/data/input ~/data/output    # Shell
```

## Volume Mounts

| Container path | Purpose |
|---------------|---------|
| `/data/input` | Input point cloud files |
| `/data/output` | Output segmentation results |
| `/tmp/sat_cache` | Temporary processing files (optional mount) |

## Image Details

| Component | Version |
|-----------|---------|
| Base | Ubuntu 22.04 |
| CUDA | 11.8.0 + cuDNN 8 |
| Python | 3.10 |
| PyTorch | 2.1.2 |
| MinkowskiEngine | latest (source build) |
| torchsparse | v1.4.0 (source build) |
| JupyterLab | 4.x |
| PDAL | via Miniforge/Mamba |
| GPU architectures | Volta (7.0), Turing (7.5), Ampere (8.0/8.6), Hopper (9.0) |

!!! note "Why CUDA 11.8 instead of 12.x?"
    MinkowskiEngine is incompatible with CUDA 12.x due to unresolved `libcu++` template conflicts in the sparse convolution kernels. CUDA 11.8 with PyTorch 2.1.2 is the most modern compatible combination. Your NVIDIA driver (525+) supports running CUDA 11.8 containers via backward compatibility.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SAT_ROOT` | `/opt/segmentanytree` | Project root directory |
| `SAT_DATA` | `/data` | Input/output data directory |
| `SAT_MODEL` | `/opt/segmentanytree/model_file` | Model checkpoint directory |
| `SAT_CACHE` | `/tmp/sat_cache` | Temporary files |

## Dockerfile Architecture

The image uses a **multi-stage build**:

1. **Builder stage**: Compiles MinkowskiEngine, torchsparse, and torch-points-kernels from source with CUDA support
2. **Runtime stage**: Copies compiled libraries, adds JupyterLab, PDAL, and project code

This keeps the final image smaller by excluding build tools (cmake, ninja, etc.).

## Local Build (without Docker)

For development or systems where Docker isn't available, you can build locally:

### Prerequisites

- NVIDIA GPU with CUDA 11.8 toolkit installed
- Python 3.10
- OpenBLAS (`libopenblas-dev`)
- sparsehash (`libsparsehash-dev`)

### Steps

```bash
# Clone the repository
git clone https://github.com/tyson-swetnam/SegmentAnyTree.git
cd SegmentAnyTree

# Create a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric ecosystem
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.2+cu118.html

# Build MinkowskiEngine from source
git clone --depth 1 https://github.com/NVIDIA/MinkowskiEngine.git /tmp/MinkowskiEngine
cd /tmp/MinkowskiEngine
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0" python setup.py install --blas=openblas --force_cuda
cd -

# Build torchsparse v1.4.0
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0" FORCE_CUDA=1 \
    pip install --no-build-isolation git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

# Build torch-points-kernels
git clone --depth 1 https://github.com/torch-points3d/torch-points-kernels.git /tmp/torch-points-kernels
cd /tmp/torch-points-kernels
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0" FORCE_CUDA=1 python setup.py install
cd -

# Install remaining Python dependencies
pip install hydra-core==1.3.2 omegaconf==2.3.0 wandb tensorboard tqdm pandas \
    scikit-learn scikit-image matplotlib seaborn h5py plyfile "laspy[lazrs]" \
    open3d gdown numba joblib dask pykdtree jaklas pytorch-metric-learning \
    addict python-louvain hdbscan jupyterlab ipywidgets

# Set environment variables
export SAT_ROOT=$(pwd)
export SAT_DATA=$SAT_ROOT/data
export SAT_MODEL=$SAT_ROOT/model_file
export SAT_CACHE=/tmp/sat_cache
export PYTHONPATH="${SAT_ROOT}:${PYTHONPATH}"
mkdir -p $SAT_DATA/input $SAT_DATA/output $SAT_CACHE
```

### Verify installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "import MinkowskiEngine; print(f'MinkowskiEngine {MinkowskiEngine.__version__}')"
python -c "import torchsparse; print('torchsparse OK')"
python -c "import torch_points_kernels; print('torch-points-kernels OK')"
```

### Run inference locally

```bash
bash scripts/run_inference.sh $SAT_DATA/input $SAT_DATA/output true
```

## Singularity / Apptainer

See [singularity.md](singularity.md) for HPC usage.
