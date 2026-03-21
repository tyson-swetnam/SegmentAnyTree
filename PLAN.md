# PLAN.md — SegmentAnyTree 2026 Update

This document describes the full modernization plan for the SegmentAnyTree repository. The goal is to make the project buildable, runnable, and usable by anyone — not just the original authors — while preserving the core deep learning pipeline.

### Build Server Hardware (confirmed 2026-03-20)

- **GPUs**: 4x NVIDIA A100 80GB PCIe (compute capability 8.0)
- **Driver**: 560.35.03 (supports up to CUDA 12.6)
- **Decision**: Targeting **CUDA 12.4 + PyTorch 2.4** (primary plan, not fallback)

---

## Table of Contents

1. [Current State & Problems](#1-current-state--problems)
2. [Goals](#2-goals)
3. [Phase 1: Dockerfile Modernization](#3-phase-1-dockerfile-modernization)
4. [Phase 2: Remove Hardcoded Paths](#4-phase-2-remove-hardcoded-paths)
5. [Phase 3: Script Reorganization](#5-phase-3-script-reorganization)
6. [Phase 4: JupyterLab Integration](#6-phase-4-jupyterlab-integration)
7. [Phase 5: Documentation](#7-phase-5-documentation)
8. [Phase 6: Testing](#8-phase-6-testing)
9. [Dependency Compatibility Matrix](#9-dependency-compatibility-matrix)
10. [File-by-File Change Manifest](#10-file-by-file-change-manifest)
11. [Risk Assessment](#11-risk-assessment)

---

## 1. Current State & Problems

### Why users cannot build or run this project

1. **Hardcoded paths everywhere**: 30+ files reference `/home/nibio/mutable-outside-world/` and `/home/datascience/` — paths that only exist on the original author's machine or their specific Oracle Cloud setup. These paths appear in shell scripts, Python files, YAML configs, and Dockerfiles.

2. **Ancient base image**: The Dockerfile uses `nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04` with Python 3.8, PyTorch 1.9, and CUDA 11.1 — all end-of-life. The alternative `Dockerfile_cuda:11.8.0` upgrades PyTorch but still uses Ubuntu 20.04 and Python 3.8.

3. **Fragile dependency pinning**: The Dockerfile pins ~150 packages to exact versions from 2021. Many of these packages are no longer available at those exact versions from PyPI, causing build failures.

4. **No interactive interface**: The container's entrypoint is `bash run_oracle_pipeline.sh` — an Oracle Cloud-specific batch pipeline. There is no way for a user to interactively explore, configure, or run the model.

5. **Disorganized scripts**: 18+ shell scripts and 15+ Python utilities are scattered across root, `nibio_inference/`, `bash_helpers/`, `big_table_creation/`, `scripts/`, and `forward_scripts/`. Many have overlapping functionality and unclear purposes.

6. **No tests**: No automated tests exist. No CI/CD pipeline. The `Makefile` runs flake8 and mypy but these aren't integrated into any workflow.

### Hardcoded paths found (exhaustive list)

**Critical (breaks functionality):**

| File | Hardcoded Path | Count |
|------|---------------|-------|
| `run_inference.sh` | `/home/nibio/mutable-outside-world` (defaults + SCRIPT_DIR) | 3 |
| `run_batch_inference.sh` | `/home/nibio/mutable-outside-world` (4 directory paths + script path) | 5 |
| `run_oracle_pipeline.sh` | `/home/nibio/mutable-outside-world` + `/home/datascience` | 4 |
| `run_docker_locally.sh` | `/home/nibio/mutable-outside-world/code/PanopticSegForLargeScalePointCloud_maciej/` | 2 |
| `conf/eval.yaml` | `/home/nibio/mutable-outside-world/model_file` + data fold path | 2 |
| `conf/data/panoptic/treeins_rad8.yaml` | `/home/datascience/data` as dataroot | 1 |
| `oracle_wrapper.py` | `/home/datascience` | 1 |

**Moderate (auxiliary scripts):**

| File | Hardcoded Path |
|------|---------------|
| `run_paper_test.sh` | `/home/nibio/mutable-outside-world`, `/home/nibio/data/timing_check` |
| `compute_capacity.sh` | `/home/nibio/data/timing_check` |
| `merge_all.sh` | `/home/nibio/mutable-outside-world/merge_all.sh` |
| `run_podman_with_gpu.sh` | `/home/nibio/mutable-outside-world/bucket_{in,out}_folder` |
| `run_bash_in_podman_with_gpu.sh` | `/home/nibio/mutable-outside-world/bucket_{in,out}_folder` |
| `bash_helpers/get_sample_labelled_data.sh` | `/home/nibio/mutable-outside-world/sample_test_data/` |
| `bash_helpers/get_sample_small_unlabelled.sh` | `/home/nibio/mutable-outside-world/sample_test_data/` |
| `bash_helpers/merge_multiple_results.sh` | `/home/nibio/mutable-outside-world/for_instance_no_outer_sparse_many_times/` |
| `big_table_creation/*.sh` | `/home/nibio/data/test_data_agnostic_*`, `/home/nibio/mutable-outside-world/tmp_*` |
| `nibio_inference/merge_inference_results_in_folders.py` | `/home/nibio/data/test_data_agnostic_instanceSeg/results_*/` |
| `visualization/viz.py` | `/home/nibio/mutable-outside-world/for_instance_no_outer_sparse_many_times/` |

**Low priority (comments/examples):**

| File | Note |
|------|------|
| `nibio_inference/split_point_cloud.py` | Commented example paths |
| `nibio_inference/remove_outer_points_for_instance.py` | Commented example path |
| `nibio_inference/ply_to_pandas.py` | Commented example path |
| `model_file/.hydra/overrides.yaml` | Checkpoint metadata (read-only) |

---

## 2. Goals

1. **Buildable Dockerfile** on Ubuntu 22.04 with CUDA 12.x, Python 3.10+, PyTorch 2.x
2. **Zero hardcoded paths** — all paths derived from environment variables or relative to `$SAT_ROOT`
3. **JupyterLab as default interface** — users can run inference interactively via notebooks
4. **Organized script layout** — clear separation of inference, training, utilities, and deployment
5. **Documentation** — `docs/` folder explaining setup, inference, training, and architecture
6. **Tests** — Docker build verification, inference smoke tests, path portability checks

---

## 3. Phase 1: Dockerfile Modernization

### Target stack

| Component | Current | Target | Rationale |
|-----------|---------|--------|-----------|
| Base image | `nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04` | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` | Ubuntu 20.04 EOL April 2025; CUDA 12.4 supports all modern GPUs |
| Python | 3.8 | 3.10 | 3.8 EOL October 2024; 3.10 has wide library support |
| PyTorch | 1.9.0 | 2.4.x | PyTorch 1.9 EOL; 2.4 has native CUDA 12.4 support |
| CUDA archs | `6.0;7.0;7.5;8.0;8.6` | `7.0;7.5;8.0;8.6;8.9;9.0` | Drop Pascal (6.0), add Ada/Hopper |
| MinkowskiEngine | source build | source build | Still requires compilation; may need patches for PyTorch 2.x |
| torchsparse | v1.4.0 | v2.1.0 or latest | v1.4.0 incompatible with PyTorch 2.x |
| JupyterLab | none | 4.x | Interactive interface requirement |

### Dockerfile structure (new)

```
Dockerfile
├── Stage 1: builder
│   ├── Ubuntu 22.04 + CUDA 12.4 + Python 3.10
│   ├── PyTorch 2.4.x + torch-geometric ecosystem
│   ├── MinkowskiEngine (source build)
│   ├── torchsparse (source build, compatible version)
│   ├── torch-points-kernels (may need patching)
│   └── All Python dependencies
│
├── Stage 2: runtime
│   ├── Copy compiled libraries from builder
│   ├── Install JupyterLab + extensions
│   ├── Install runtime dependencies (laspy, plyfile, dask, etc.)
│   ├── Copy project code
│   ├── Set up non-root user
│   └── Configure JupyterLab as default entrypoint
│
└── Environment variables
    ├── SAT_ROOT=/opt/segmentanytree (project code)
    ├── SAT_DATA=/data (mount point for input/output)
    ├── SAT_MODEL=/opt/segmentanytree/model_file (pre-trained model)
    └── PYTHONPATH includes SAT_ROOT
```

### Key decisions

- **Keep multi-stage build** to minimize final image size
- **Non-root user** (`sat`) for JupyterLab security
- **Entrypoint flexibility**: Default to JupyterLab, but support `docker run ... bash` for CLI access and `docker run ... bash run_inference.sh` for batch mode
- **No Oracle-specific code in entrypoint** — Oracle pipeline becomes an optional script

### GPU library compatibility risks

The biggest risk is **MinkowskiEngine** and **torchsparse** compatibility with PyTorch 2.x:

- **MinkowskiEngine**: The NVIDIA repo has community patches for PyTorch 2.x. We will try the latest `main` branch first. Fallback: use the [openmm fork](https://github.com/NVIDIA/MinkowskiEngine) with PyTorch 2.x patches.
- **torchsparse v1.4.0**: Incompatible with PyTorch 2.x. We need v2.1.0+. The model configs reference `torchsparse` via the `SPARSE` conv_type. We need to verify the API hasn't changed for the operations used (SparseTensor, sparse convolutions).
- **torch-points-kernels 0.7.0**: Uses custom CUDA kernels. May need recompilation or patching for CUDA 12.x. The critical function is `region_grow()` used in clustering.

**Fallback plan**: If PyTorch 2.x proves incompatible after testing, we fall back to:
- CUDA 11.8 + PyTorch 2.1.x (the most-tested "bridge" configuration)
- Ubuntu 22.04 is still viable with CUDA 11.8

---

## 4. Phase 2: Remove Hardcoded Paths

### Strategy

All paths will be derived from a single environment variable `SAT_ROOT` (the project root directory). Inside Docker, this is `/opt/segmentanytree`. Outside Docker, it's wherever the user cloned the repo.

**Standard directory layout:**

```
$SAT_ROOT/                          # Project code
├── model_file/                     # Pre-trained model checkpoint
├── conf/                           # Hydra configs
└── ...

$SAT_DATA/                          # User data (mount point in Docker)
├── input/                          # Input point clouds
├── output/                         # Output segmentation results
└── cache/                          # Temporary processing files
```

### Environment variables

| Variable | Default (Docker) | Default (local) | Purpose |
|----------|-----------------|-----------------|---------|
| `SAT_ROOT` | `/opt/segmentanytree` | `$(git rev-parse --show-toplevel)` | Project root |
| `SAT_DATA` | `/data` | `$SAT_ROOT/data` | Input/output data directory |
| `SAT_MODEL` | `$SAT_ROOT/model_file` | `$SAT_ROOT/model_file` | Model checkpoint directory |
| `SAT_CACHE` | `/tmp/sat_cache` | `/tmp/sat_cache` | Temporary files, cleared between runs |

### Changes per file

**Shell scripts** — replace hardcoded defaults with env var lookups:
```bash
# Before:
SCRIPT_DIR="/home/nibio/mutable-outside-world"
# After:
SCRIPT_DIR="${SAT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
```

**YAML configs** — use Hydra variable interpolation or environment variables:
```yaml
# Before:
checkpoint_dir: "/home/nibio/mutable-outside-world/model_file"
# After:
checkpoint_dir: ${oc.env:SAT_MODEL,model_file}
```

Note: Hydra 1.0.x supports `${oc.env:VAR}` for environment variable interpolation via OmegaConf.

**Python files** — use `os.environ.get()` with sensible defaults:
```python
# Before:
PATH_DATA = '/home/datascience'
# After:
PATH_DATA = os.environ.get('SAT_DATA', os.path.join(os.path.dirname(__file__), 'data'))
```

---

## 5. Phase 3: Script Reorganization

### Current layout (chaotic)

```
./                              # 10+ shell scripts, 4+ Python scripts at root
├── nibio_inference/            # 15+ utilities (coordinate transforms, merging, format conversion)
├── nibio_sparsify/             # 2 scripts (point cloud density reduction)
├── bash_helpers/               # 4 shell scripts (auxiliary)
├── big_table_creation/         # 3 scripts (paper-specific metrics)
├── forward_scripts/            # 1 script (forward pass)
├── metrics/                    # 5 scripts (evaluation metrics)
├── scripts/                    # Misc scripts (sanity checks, visualizations)
└── visualization/              # 1 script (viz.py)
```

### Proposed layout

```
./
├── sat/                        # Renamed from nibio_inference + nibio_sparsify + root utilities
│   ├── __init__.py
│   ├── inference.py            # Main inference entrypoint (replaces eval.py logic)
│   ├── training.py             # Main training entrypoint (replaces train.py logic)
│   ├── pipeline/               # Inference pipeline steps
│   │   ├── __init__.py
│   │   ├── coordinate_transform.py   # UTM ↔ local (from pipeline_utm2local*.py, pipeline_local2utm.py)
│   │   ├── file_preparation.py       # Naming fixes, file copying (from fix_naming_of_input_files.py)
│   │   ├── config_update.py          # eval.yaml modification (from modify_eval.py)
│   │   ├── cache.py                  # Cache clearing (from clear_cache.py)
│   │   ├── result_merge.py           # Point cloud merging (from merge_pt_ss_is*.py)
│   │   └── result_rename.py          # Output file renaming (from rename_result_files_*.py)
│   ├── io/                     # File format I/O
│   │   ├── __init__.py
│   │   ├── las_io.py           # LAS/LAZ read/write (from las_to_pandas.py, pandas_to_las.py)
│   │   ├── ply_io.py           # PLY read/write (from ply_to_pandas.py, pandas_to_ply.py)
│   │   └── conversion.py      # Format conversion (from sample_data_conversion.py)
│   ├── metrics/                # Evaluation metrics (from metrics/ top-level)
│   │   ├── __init__.py
│   │   ├── instance.py         # Instance segmentation metrics
│   │   ├── semantic.py         # Semantic segmentation metrics
│   │   └── aggregate.py        # Metrics aggregation
│   ├── preprocessing/          # Data preprocessing
│   │   ├── __init__.py
│   │   ├── sparsify.py         # Point cloud density reduction (from nibio_sparsify/)
│   │   ├── split.py            # Point cloud splitting (from split_point_cloud.py)
│   │   └── filter.py           # Point filtering (from remove_outer_points_for_instance.py, distance_filtering*)
│   └── utils/                  # Shared utilities
│       ├── __init__.py
│       └── paths.py            # Central path resolution using SAT_ROOT/SAT_DATA env vars
│
├── scripts/                    # User-facing scripts (thin wrappers around sat/ modules)
│   ├── run_inference.sh        # Main inference script (simplified, uses env vars)
│   ├── run_batch_inference.sh  # Batch processing
│   ├── run_training.sh         # Training wrapper
│   ├── run_docker.sh           # Docker run helper (replaces run_docker_locally.sh)
│   ├── convert_data.sh         # Data format conversion helper
│   └── oracle/                 # Oracle Cloud-specific (isolated from main pipeline)
│       ├── run_oracle_pipeline.sh
│       └── oracle_wrapper.py
│
├── notebooks/                  # JupyterLab notebooks
│   ├── 01_quickstart.ipynb     # Quick inference demo
│   ├── 02_inference.ipynb      # Full inference with visualization
│   ├── 03_training.ipynb       # Training walkthrough
│   └── 04_evaluation.ipynb     # Metrics and evaluation
│
├── tests/                      # Automated tests
│   ├── test_docker_build.sh    # Docker build smoke test
│   ├── test_paths.py           # Verify no hardcoded paths remain
│   ├── test_io.py              # File format I/O tests
│   ├── test_pipeline.py        # Pipeline step unit tests
│   └── test_inference.py       # End-to-end inference smoke test (requires GPU)
│
├── docs/                       # Documentation
│   ├── quickstart.md           # Getting started
│   ├── inference.md            # How to run inference
│   ├── training.md             # How to train
│   ├── docker.md               # Docker usage
│   ├── architecture.md         # Code architecture
│   ├── singularity.md          # Singularity/Apptainer HPC usage
│   └── troubleshooting.md      # Common issues and solutions
│
├── torch_points3d/             # Core ML framework (minimal changes)
│   └── ...                     # Only fix imports if sat/ replaces nibio_inference/
│
├── conf/                       # Hydra configs (path fixes only)
│   ├── config.yaml
│   ├── eval.yaml
│   └── ...
│
├── train.py                    # Keep at root for Hydra compatibility (thin wrapper)
├── eval.py                     # Keep at root for Hydra compatibility (thin wrapper)
├── Dockerfile                  # New modernized Dockerfile
├── Makefile                    # Updated with new targets
├── pyproject.toml              # Updated dependencies
└── PLAN.md                     # This file
```

### What gets deleted vs moved

| Current File | Action | Destination |
|-------------|--------|-------------|
| `run_inference.sh` | Rewrite | `scripts/run_inference.sh` |
| `run_batch_inference.sh` | Rewrite | `scripts/run_batch_inference.sh` |
| `run_oracle_pipeline.sh` | Move | `scripts/oracle/run_oracle_pipeline.sh` |
| `oracle_wrapper.py` | Move | `scripts/oracle/oracle_wrapper.py` |
| `run_docker_locally.sh` | Rewrite | `scripts/run_docker.sh` |
| `run_paper_test.sh` | Delete | Research-specific, not needed for users |
| `compute_capacity.sh` | Delete | Research-specific utility |
| `merge_all.sh` | Delete | Research-specific utility |
| `run_podman_with_gpu.sh` | Merge | Into `scripts/run_docker.sh` (add podman flag) |
| `run_bash_in_podman_with_gpu.sh` | Delete | Covered by `scripts/run_docker.sh` |
| `run_pipeline.sh` | Delete | Research-specific training wrapper |
| `build.sh` | Delete | `docker build` is self-explanatory |
| `evaluation_stats_FOR.py` | Move | `sat/metrics/` |
| `evaluation_stats_NPM3D.py` | Move | `sat/metrics/` |
| `sample_data_conversion.py` | Move | `sat/io/conversion.py` |
| `nibio_inference/*.py` | Reorganize | Into `sat/pipeline/`, `sat/io/` |
| `nibio_sparsify/*.py` | Move | `sat/preprocessing/sparsify.py` |
| `bash_helpers/*.sh` | Delete | Research-specific utilities |
| `big_table_creation/*.sh` | Delete | Paper-specific utilities |
| `forward_scripts/forward.py` | Move | `sat/inference.py` (merge) |
| `metrics/*.py` | Move | `sat/metrics/` |
| `scripts/` (existing) | Delete | Sanity checks, not needed |
| `visualization/viz.py` | Delete | Research-specific, hardcoded paths |
| `Dockerfile_cuda:11.8.0` | Delete | Replaced by new Dockerfile |

### Backward compatibility

- `train.py` and `eval.py` remain at root (Hydra requires `config_path` relative to the script)
- `torch_points3d/` package is NOT reorganized (too risky, deep interdependencies)
- `conf/` directory structure is preserved (Hydra config groups depend on it)
- `model_file/PointGroup-PAPER.pt` stays in place

---

## 6. Phase 4: JupyterLab Integration

### Container entrypoint

```dockerfile
# Default: JupyterLab on port 8888
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--NotebookApp.token=''", "--NotebookApp.password=''"]
```

### Docker run examples

```bash
# Interactive (JupyterLab)
docker run --gpus all -p 8888:8888 \
  -v $HOME/data/input:/data/input \
  -v $HOME/data/output:/data/output \
  segmentanytree:latest

# Batch inference (CLI)
docker run --gpus all \
  -v $HOME/data/input:/data/input \
  -v $HOME/data/output:/data/output \
  segmentanytree:latest \
  bash scripts/run_inference.sh /data/input /data/output

# Interactive shell
docker run --gpus all -it \
  -v $HOME/data/input:/data/input \
  -v $HOME/data/output:/data/output \
  segmentanytree:latest bash
```

### Notebooks

Four starter notebooks will be provided:

1. **01_quickstart.ipynb** — Drop a .las/.laz file in, get segmented output. 10 cells max.
2. **02_inference.ipynb** — Full pipeline with coordinate transforms, configuration, visualization of results using open3d/matplotlib.
3. **03_training.ipynb** — Data preparation, config setup, training loop, checkpoint management.
4. **04_evaluation.ipynb** — Load predictions vs ground truth, compute metrics, generate plots.

---

## 7. Phase 5: Documentation

### docs/ folder structure

| File | Contents |
|------|----------|
| `quickstart.md` | Prerequisites, pull image, run inference in 5 minutes |
| `inference.md` | Detailed inference: input formats, pipeline steps, output formats, batch mode, coordinate systems |
| `training.md` | Data preparation (LAS→PLY conversion, label mapping), Hydra config, training command, monitoring with wandb |
| `docker.md` | Building from source, GPU setup, volume mounts, environment variables, Singularity/Apptainer conversion |
| `architecture.md` | Model architecture (PointGroup 3-head), backbone (Minkowski U-Net), clustering, loss functions. Includes diagram. |
| `singularity.md` | HPC usage: converting Docker → SIF, SLURM scripts, bind mounts, checkpointing, storage management |
| `troubleshooting.md` | Common errors: CUDA OOM, storage full, empty instances, dtype mismatches, coordinate transform issues |

### Key content sources

- `README.md` — quick start, Docker usage, citation
- `CLAUDE.md` — architecture overview, technical details
- The SLURM script from the user's issue report — Singularity/HPC patterns
- Research paper — model architecture explanation
- Inline comments in `run_inference.sh` — pipeline step explanations

---

## 8. Phase 6: Testing

### Test categories

#### 1. Path portability test (`tests/test_paths.py`)
```python
# Scan all .py, .sh, .yaml files for hardcoded paths
# Assert no occurrences of:
#   /home/nibio/
#   /home/datascience
#   /home/nibio/mutable-outside-world
#   PanopticSegForLargeScalePointCloud
```

#### 2. Docker build test (`tests/test_docker_build.sh`)
```bash
# Build the image
# Verify key binaries exist: python3, jupyter, nvidia-smi (if GPU)
# Verify key Python imports: torch, MinkowskiEngine, torchsparse, torch_points3d
# Verify model file exists
# Verify environment variables are set
```

#### 3. I/O tests (`tests/test_io.py`)
```python
# Test LAS → pandas → LAS roundtrip
# Test PLY → pandas → PLY roundtrip
# Test coordinate transform (UTM → local → UTM roundtrip)
```

#### 4. Pipeline tests (`tests/test_pipeline.py`)
```python
# Test file naming fix (dashes → underscores)
# Test eval.yaml config modification
# Test result file renaming
# Test merge logic with synthetic data
```

#### 5. Inference smoke test (`tests/test_inference.py`)
```python
# Requires GPU
# Create small synthetic point cloud (100 points)
# Run through full pipeline
# Verify output files exist and have expected columns
```

### Makefile targets (updated)

```makefile
.PHONY: test lint build

lint:
    flake8 . --count --select=E9,F402,F6,F7,F5,F8,F9 --show-source --statistics
    mypy torch_points3d

test:
    python -m pytest tests/ -v --ignore=tests/test_inference.py

test-gpu:
    python -m pytest tests/ -v

test-paths:
    python -m pytest tests/test_paths.py -v

build:
    docker build -t segmentanytree:latest .

test-docker:
    bash tests/test_docker_build.sh
```

---

## 9. Dependency Compatibility Matrix

### PyTorch 2.4 + CUDA 12.4 target

| Library | Current Version | Target Version | Compatibility Notes |
|---------|----------------|----------------|-------------------|
| PyTorch | 1.9.0+cu111 | 2.4.x+cu124 | Major API changes; sparse tensor API restructured |
| torch-geometric | 1.7.2 | 2.5.x | Stable upgrade path, wheels available |
| torch-scatter | 2.0.8 | 2.1.2+ | Wheels for PyTorch 2.4 available |
| torch-sparse | 0.6.12 | 0.6.18+ | Wheels for PyTorch 2.4 available |
| torch-cluster | 1.5.9 | 1.6.3+ | Wheels for PyTorch 2.4 available |
| MinkowskiEngine | git main | git main | Must test: `SparseTensor` API may differ in PyTorch 2.x |
| torchsparse | v1.4.0 | v2.1.0+ | v1.4.0 WILL NOT WORK with PyTorch 2.x |
| torch-points-kernels | 0.7.0 | 0.7.0 (rebuild) | Custom CUDA kernels need recompile; may need source patches |
| hydra-core | 1.0.7 | 1.3.x | Config API mostly backward compatible |
| omegaconf | 2.0.6 | 2.3.x | `oc.env` resolver available in 2.1+ |
| numpy | 1.19.5 | 1.26.x | Drop `<1.20` constraint |
| numba | 0.50.1 | 0.59.x | Significant improvements, new LLVM backend |
| open3d | 0.12.0 | 0.18.x | Visualization API changes; used only for preprocessing |
| laspy | 2.0.3 | 2.5.x | Stable upgrade |
| wandb | 0.8.36 | 0.17.x | Major upgrades but backward compatible |
| JupyterLab | none | 4.x | New addition |

### Fallback: PyTorch 2.1 + CUDA 11.8

If CUDA 12.4 causes GPU library compilation failures:

| Component | Fallback Version |
|-----------|-----------------|
| Base image | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` |
| PyTorch | 2.1.2+cu118 |
| torchsparse | v2.1.0 |
| CUDA archs | `7.0;7.5;8.0;8.6;9.0` |

---

## 10. File-by-File Change Manifest

### New files to create

| File | Purpose |
|------|---------|
| `Dockerfile` | Complete rewrite with modern stack + JupyterLab |
| `sat/__init__.py` | Package init |
| `sat/utils/paths.py` | Central path resolution |
| `sat/pipeline/*.py` | Reorganized pipeline modules |
| `sat/io/*.py` | File I/O modules |
| `sat/metrics/*.py` | Metrics modules |
| `sat/preprocessing/*.py` | Preprocessing modules |
| `scripts/run_inference.sh` | Rewritten inference script |
| `scripts/run_batch_inference.sh` | Rewritten batch script |
| `scripts/run_docker.sh` | Docker/Podman run helper |
| `scripts/run_training.sh` | Training wrapper |
| `notebooks/01_quickstart.ipynb` | Quick start notebook |
| `notebooks/02_inference.ipynb` | Full inference notebook |
| `notebooks/03_training.ipynb` | Training notebook |
| `notebooks/04_evaluation.ipynb` | Evaluation notebook |
| `tests/test_paths.py` | Path portability tests |
| `tests/test_io.py` | I/O roundtrip tests |
| `tests/test_pipeline.py` | Pipeline unit tests |
| `tests/test_inference.py` | Inference smoke tests |
| `tests/test_docker_build.sh` | Docker build verification |
| `docs/quickstart.md` | Quick start guide |
| `docs/inference.md` | Inference documentation |
| `docs/training.md` | Training documentation |
| `docs/docker.md` | Docker documentation |
| `docs/architecture.md` | Architecture documentation |
| `docs/singularity.md` | HPC/Singularity documentation |
| `docs/troubleshooting.md` | Troubleshooting guide |

### Files to modify

| File | Changes |
|------|---------|
| `conf/eval.yaml` | Replace hardcoded paths with env var interpolation |
| `conf/data/panoptic/treeins_rad8.yaml` | Replace hardcoded dataroot |
| `train.py` | Minor: update imports if needed |
| `eval.py` | Minor: update imports if needed |
| `Makefile` | Add new targets (test, build, test-docker, test-paths) |
| `pyproject.toml` | Update dependency versions |
| `CLAUDE.md` | Update to reflect new structure |
| `README.md` | Major rewrite for new usage patterns |
| `.gitignore` | Add `data/`, `__pycache__/`, `.ipynb_checkpoints/` |

### Files to delete

| File | Reason |
|------|--------|
| `Dockerfile_cuda:11.8.0` | Replaced by new Dockerfile |
| `run_inference.sh` (root) | Replaced by `scripts/run_inference.sh` |
| `run_batch_inference.sh` (root) | Replaced by `scripts/run_batch_inference.sh` |
| `run_oracle_pipeline.sh` (root) | Moved to `scripts/oracle/` |
| `oracle_wrapper.py` (root) | Moved to `scripts/oracle/` |
| `run_docker_locally.sh` | Replaced by `scripts/run_docker.sh` |
| `run_paper_test.sh` | Research-specific |
| `compute_capacity.sh` | Research-specific |
| `merge_all.sh` | Research-specific |
| `run_podman_with_gpu.sh` | Merged into `scripts/run_docker.sh` |
| `run_bash_in_podman_with_gpu.sh` | Merged into `scripts/run_docker.sh` |
| `run_pipeline.sh` | Research-specific |
| `build.sh` | Trivial; documented in docs/ |
| `evaluation_stats_FOR.py` | Moved to `sat/metrics/` |
| `evaluation_stats_NPM3D.py` | Moved to `sat/metrics/` |
| `sample_data_conversion.py` | Moved to `sat/io/conversion.py` |
| `bash_helpers/` (entire dir) | Research-specific |
| `big_table_creation/` (entire dir) | Paper-specific |
| `forward_scripts/` (entire dir) | Merged into `sat/` |
| `visualization/` (entire dir) | Research-specific, hardcoded |
| `nibio_inference/` (entire dir) | Reorganized into `sat/` |
| `nibio_sparsify/` (entire dir) | Reorganized into `sat/` |
| `scripts/` (existing dir) | Sanity checks, not user-facing |
| `metrics/` (top-level dir) | Moved to `sat/metrics/` |

---

## 11. Risk Assessment

### High risk

| Risk | Mitigation |
|------|-----------|
| MinkowskiEngine won't compile with PyTorch 2.4/CUDA 12.4 | Fallback to CUDA 11.8 + PyTorch 2.1; test compilation first |
| torchsparse v2.x API breaks model code | Compare `SparseTensor` API between v1.4 and v2.x; wrap differences |
| torch-points-kernels `region_grow` fails on CUDA 12.x | Rebuild from source with new CUDA; patch kernel code if needed |
| Pre-trained model checkpoint incompatible with new PyTorch | Test `torch.load()` with `weights_only=False`; may need model re-export |

### Medium risk

| Risk | Mitigation |
|------|-----------|
| Hydra 1.0 → 1.3 config API changes | Test all config files; Hydra 1.3 has good backward compat |
| NumPy 1.19 → 1.26 dtype behavior changes | Test all I/O roundtrips; watch for float64→float32 silent casts |
| Script reorganization breaks `import` paths | Keep `torch_points3d/` unchanged; only `nibio_inference` imports need updating |

### Low risk

| Risk | Mitigation |
|------|-----------|
| JupyterLab version conflicts | JupyterLab 4.x has minimal dependencies |
| Documentation accuracy | Review against working pipeline |

---

## Execution Order

1. **Phase 2 first** (hardcoded paths) — independent of Dockerfile, can test locally
2. **Phase 3 next** (script reorganization) — depends on Phase 2 for path strategy
3. **Phase 1 then** (Dockerfile) — depends on Phase 3 for knowing what to COPY
4. **Phase 4** (JupyterLab) — depends on Phase 1 for working Docker image
5. **Phase 5** (documentation) — depends on all prior phases being stable
6. **Phase 6** (testing) — written incrementally during each phase, final integration at end

---

*Plan created: 2026-03-20*
*Branch: `2026-update`*
