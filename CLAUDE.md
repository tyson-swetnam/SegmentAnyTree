# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**This is a fork of [SmartForest-no/SegmentAnyTree](https://github.com/SmartForest-no/SegmentAnyTree)** focused on:
- Updating the Docker build environment for reliability and reproducibility
- Finding and implementing computation speed improvements for inference and training

SegmentAnyTree is a deep learning framework for **tree instance segmentation from 3D point cloud data** (LiDAR). It builds on the [torch-points3d](https://github.com/torch-points3d/torch-points3d) framework and implements panoptic segmentation (combined semantic + instance segmentation) using the PointGroup architecture with a 3-head variant. The primary model is `PointGroup-PAPER`.

Paper: Wielgosz et al. (2024), "SegmentAnyTree: A sensor and platform agnostic deep learning model for tree segmentation using laser scanning data", Remote Sensing of Environment.

### Fork-Specific Notes
- Upstream repository: `https://github.com/SmartForest-no/SegmentAnyTree`
- This fork's repository: `https://github.com/tyson-swetnam/SegmentAnyTree`
- Branch `2026-update` contains Docker build and performance work
- When making changes, consider whether they should be contributed back upstream via PR

## Build & Run Commands

### Docker (primary usage)
```bash
# Build
docker build -t segmentanytree:latest .

# Run JupyterLab (default)
docker run --gpus all -p 8888:8888 \
  -v $HOME/data/input:/data/input \
  -v $HOME/data/output:/data/output \
  segmentanytree:latest

# Run batch inference
docker run --gpus all \
  -v $HOME/data/input:/data/input \
  -v $HOME/data/output:/data/output \
  segmentanytree:latest \
  bash scripts/run_inference.sh /data/input /data/output true
```

### Training
```bash
python train.py task=panoptic data=panoptic/treeins \
  models=panoptic/area4_ablation_3heads \
  model_name=PointGroup-PAPER \
  training=treeins \
  job_name=my_run
```

### Testing
```bash
make test          # Run unit tests (no GPU needed)
make test-paths    # Verify no hardcoded paths remain
make test-gpu      # Run all tests including GPU smoke test
make lint          # flake8 + mypy
make build         # Docker build
make test-docker   # Verify Docker image
```

### Code Style
Black formatter with 120 char line length (`[tool.black]` in pyproject.toml).

## Architecture

### Package Layout
- **`sat/`** — Python package with pipeline, I/O, metrics, and preprocessing modules
  - `sat/pipeline/` — Inference pipeline steps (coordinate_transform, config_update, result_merge, etc.)
  - `sat/io/` — Point cloud file I/O (LAS, PLY, format conversion)
  - `sat/metrics/` — Instance segmentation evaluation metrics
  - `sat/preprocessing/` — Point cloud sparsification and filtering
  - `sat/utils/paths.py` — Central path resolution via `SAT_ROOT`/`SAT_DATA` env vars
- **`torch_points3d/`** — Core ML framework (PointGroup model, trainer, datasets, modules)
- **`conf/`** — Hydra configuration (model architectures, data configs, training params)
- **`scripts/`** — Shell scripts for inference, batch processing, Docker, training
- **`notebooks/`** — JupyterLab starter notebooks
- **`tests/`** — Automated tests

### Configuration System
Uses **Hydra 1.x** for hierarchical config composition. All configs live in `conf/`.

### Inference Pipeline (`scripts/run_inference.sh`)
1. Sanitize filenames → 2. UTM→local coords → 3. Model inference via `eval.py` → 4. Rename outputs → 5. Merge results with original point cloud → 6. Restore UTM coords

### Environment Variables
All paths are derived from environment variables (no hardcoded paths):
- `SAT_ROOT` — Project root (auto-detected from git or script location)
- `SAT_DATA` — Data directory (Docker: `/data`)
- `SAT_MODEL` — Model checkpoint directory
- `SAT_CACHE` — Temporary files

### Key Technical Details
- **Python 3.10**, **PyTorch 2.1.2**, **CUDA 11.8** (Docker base: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`)
- GPU libraries: MinkowskiEngine, torchsparse v1.4.0, torch-points-kernels
- Pre-trained model: `model_file/PointGroup-PAPER.pt`
- Input formats: .las, .laz, .ply; Output format: .las with PredSemantic + PredInstance fields
