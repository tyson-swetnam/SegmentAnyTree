# Docker Guide

## Pre-built Image

```bash
docker pull segmentanytree:latest
```

## Building From Source

```bash
git clone https://github.com/<org>/SegmentAnyTree.git
cd SegmentAnyTree
docker build -t segmentanytree:latest .
```

Build takes 30-60 minutes due to GPU library compilation (MinkowskiEngine, torchsparse).

### Build requirements
- Docker 20.10+
- NVIDIA driver 525+ (for CUDA 12.4 container support)
- ~15 GB disk space for the final image

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

## Image Details

| Component | Version |
|-----------|---------|
| Base | Ubuntu 22.04 |
| CUDA | 12.4.1 + cuDNN |
| Python | 3.10 |
| PyTorch | 2.4.1 |
| JupyterLab | 4.x |
| GPU architectures | Volta (7.0), Turing (7.5), Ampere (8.0/8.6), Ada (8.9), Hopper (9.0) |

## Singularity / Apptainer

See [singularity.md](singularity.md) for HPC usage.
