# Docker Images

Two Dockerfile variants are provided for different GPU/CUDA requirements.

## Variants

| File | CUDA | PyTorch | Sparse Backend | Inference Quality | GPU Support |
|------|------|---------|----------------|-------------------|-------------|
| `Dockerfile.cuda11` | 11.8 | 2.1.2 | MinkowskiEngine (source) | **Validated** (47/64 GT match) | Volta → Hopper |
| `Dockerfile.cuda12` | 12.4 | 2.5.1 | SpConv v2.x (pip) | Experimental (semantic bias) | Volta → Blackwell |

**Recommendation:** Use `Dockerfile.cuda11` for production inference. The CUDA 12 / SpConv backend is faster to build and supports newer GPUs, but its semantic segmentation is biased due to numerical differences from MinkowskiEngine. Use CUDA 12 only for development or if retraining with SpConv.

## Prerequisites

**Git LFS required.** The model weights (`model_file/PointGroup-PAPER.pt`, 665 MB) are stored with Git LFS. Both Dockerfiles include a build-time check that fails if the weights are LFS pointers.

```bash
git lfs install
git lfs pull --include="model_file/PointGroup-PAPER.pt"
make verify-weights  # Should show ~665 MB

# Or download directly without git-lfs:
curl -L -o model_file/PointGroup-PAPER.pt \
  "https://github.com/SmartForest-no/SegmentAnyTree/raw/main/model_file/PointGroup-PAPER.pt"
```

## Build

```bash
# CUDA 11.8 (recommended for production)
docker build -f docker/Dockerfile.cuda11 -t segmentanytree:cuda11 .

# CUDA 12.4 (experimental)
docker build -f docker/Dockerfile.cuda12 -t segmentanytree:cuda12 .

# Or use Make targets
make build-cuda11    # recommended
make build-cuda12    # experimental
make build           # alias for build-cuda12
```

Note: Both Dockerfiles use the repo root as build context (`.`), so run from the repo root.

## Run

```bash
# JupyterLab (default CMD)
docker run --gpus all -p 8888:8888 \
  -v $HOME/data/input:/data/input \
  -v $HOME/data/output:/data/output \
  segmentanytree:cuda11

# Batch inference
docker run --gpus all \
  -v $HOME/data/input:/data/input \
  -v $HOME/data/output:/data/output \
  segmentanytree:cuda11 \
  bash scripts/run_inference.sh /data/input /data/output true
```

## Key Differences

### CUDA 11.8 (`Dockerfile.cuda11`) — Recommended
- **MinkowskiEngine** + **torchsparse v1.4.0** (source-built)
- **torch-points-kernels** (source-built C++/CUDA)
- Original pre-trained weights work directly (no migration needed)
- Sets `SPARSE_BACKEND=minkowski` environment variable
- **Validated**: 47/64 ground truth trees matched on FOR-instance benchmark
- GPU arch support: sm_70 through sm_90 (Volta → Hopper)

### CUDA 12.4 (`Dockerfile.cuda12`) — Experimental
- **SpConv v2.x** replaces MinkowskiEngine (pip-installable, no source build)
- **Pure PyTorch** `region_grow` replaces torch-points-kernels
- Auto-converts ME weights to SpConv format during Docker build
- **Experimental**: Semantic segmentation biased (~100% tree) due to SubMConv3d differences
- ~20-30 min faster builds than CUDA 11.8
- GPU arch support: sm_70 through sm_100 (Volta → Blackwell)

## Pre-trained Weights

The CUDA 12.4 Dockerfile automatically runs `scripts/migrate_weights.py` during the build to convert MinkowskiEngine weights to SpConv format. No manual conversion needed.

The CUDA 11.8 image uses the original `PointGroup-PAPER.pt` directly.

Both Dockerfiles verify at build time that the model file is a real checkpoint (not a Git LFS pointer). The build will fail with a clear error message if weights are missing.

## Container Details

| Setting | Value |
|---------|-------|
| User | `sat` (UID 1000) |
| WORKDIR | `/opt/segmentanytree` |
| Port | 8888 (JupyterLab) |
| Data dirs | `/data/input`, `/data/output` (world-writable) |
| Cache | `/tmp/sat_cache` (world-writable) |
