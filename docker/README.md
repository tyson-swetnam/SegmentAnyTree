# Docker Images

Two Dockerfile variants are provided for different GPU/CUDA requirements.

## Variants

| File | CUDA | PyTorch | Sparse Backend | GPU Support |
|------|------|---------|----------------|-------------|
| `Dockerfile.cuda12` | 12.4 | 2.5.1 | SpConv v2.x (pip) | Ada, Hopper, Blackwell + older |
| `Dockerfile.cuda11` | 11.8 | 2.1.2 | MinkowskiEngine + torchsparse (source) | Volta, Turing, Ampere |

**Recommendation:** Use `Dockerfile.cuda12` unless you specifically need CUDA 11.8 for older hardware or driver compatibility.

## Build

```bash
# CUDA 12.4 (default, recommended)
docker build -f docker/Dockerfile.cuda12 -t segmentanytree:cuda12 .

# CUDA 11.8 (legacy)
docker build -f docker/Dockerfile.cuda11 -t segmentanytree:cuda11 .

# Or use Make targets
make build-cuda12    # default
make build-cuda11    # legacy
make build           # alias for build-cuda12
```

Note: Both Dockerfiles use the repo root as build context (`.`), so run from the repo root.

## Run

```bash
# JupyterLab (default CMD)
docker run --gpus all -p 8888:8888 \
  -v $HOME/data/input:/data/input \
  -v $HOME/data/output:/data/output \
  segmentanytree:cuda12

# Batch inference
docker run --gpus all \
  -v $HOME/data/input:/data/input \
  -v $HOME/data/output:/data/output \
  segmentanytree:cuda12 \
  bash scripts/run_inference.sh /data/input /data/output true
```

## Key Differences

### CUDA 12.4 (`Dockerfile.cuda12`)
- **SpConv v2.x** replaces MinkowskiEngine for sparse convolutions (pip-installable, no source build)
- **Pure PyTorch** `region_grow` replaces torch-points-kernels (no C++/CUDA compilation)
- **Faster builds**: ~20-30 min saved by eliminating 3 source builds
- GPU arch support: sm_70 through sm_100 (Volta → Blackwell)

### CUDA 11.8 (`Dockerfile.cuda11`)
- **MinkowskiEngine** + **torchsparse v1.4.0** (source-built from NVIDIA and MIT repos)
- **torch-points-kernels** (source-built C++/CUDA)
- Original pre-trained weights work directly (no migration needed)
- GPU arch support: sm_70 through sm_90 (Volta → Hopper)

## Pre-trained Weights

The CUDA 12.4 image requires migrated weights because the sparse convolution backend changed:

```bash
# Convert weights (run inside the container or with the CUDA 12 env)
python scripts/migrate_weights.py \
  --input model_file/PointGroup-PAPER.pt \
  --output model_file/PointGroup-PAPER-spconv.pt
```

The CUDA 11.8 image uses the original `PointGroup-PAPER.pt` directly.
