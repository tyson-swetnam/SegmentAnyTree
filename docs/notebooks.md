# Notebooks

Interactive Jupyter notebooks for working with SegmentAnyTree. Available in the `notebooks/` directory, or run them inside the Docker container via JupyterLab (port 8888).

## Available Notebooks

| Notebook | Description |
|----------|-------------|
| **01_quickstart** | GPU check, run full inference pipeline, inspect results |
| **02_inference** | Step-by-step pipeline walkthrough with matplotlib visualization |
| **03_copc_conversion** | Convert LAS/LAZ files to COPC format, verify with PDAL, serve for web |
| **04_batch_processing** | Process multiple files in batches, monitor progress, aggregate tree counts |
| **05_visualization** | 2D segmentation maps, per-tree metrics extraction, height/crown histograms |

## Running Notebooks

=== "Docker (Harbor)"

    ```bash
    docker pull harbor.cyverse.org/vice/segmentanytree:cuda11
    docker run --gpus all -p 8888:8888 \
      -v $HOME/segmentanytree/input:/data/input \
      -v $HOME/segmentanytree/output:/data/output \
      harbor.cyverse.org/vice/segmentanytree:cuda11
    ```

    Open http://localhost:8888 and navigate to `notebooks/`.

=== "Docker (Local Build)"

    ```bash
    docker run --gpus all -p 8888:8888 \
      -v $HOME/segmentanytree/input:/data/input \
      -v $HOME/segmentanytree/output:/data/output \
      segmentanytree:cuda11
    ```

    Open http://localhost:8888 and navigate to `notebooks/`.

=== "CyVerse VICE"

    Launch the SegmentAnyTree app from the [CyVerse Discovery Environment](https://de.cyverse.org/). The notebooks are pre-loaded in the JupyterLab workspace with your Data Store files accessible at `/home/sat/data-store/`.

=== "Local Install"

    ```bash
    conda activate sat
    export SAT_ROOT=/path/to/SegmentAnyTree
    export SPARSE_BACKEND=spconv
    cd $SAT_ROOT
    jupyter lab
    ```

    Open the URL shown in the terminal and navigate to `notebooks/`.

    !!! note "Kernel registration"
        If the notebooks show "Kernel not found", register the kernel:
        ```bash
        conda activate sat
        python -m ipykernel install --user --name sat --display-name "Python 3 (sat)"
        ```

All notebooks auto-detect whether they are running in Docker or locally using `sat.utils.paths`. Input/output directories are resolved automatically — no path editing needed.

## Notebook Details

### 01 — Quick Start

The fastest path to results. Checks GPU availability, runs the full inference pipeline on files in `/data/input/`, and inspects the output with laspy. Start here to verify your setup works.

### 02 — Inference Pipeline

Walks through each pipeline step individually: file preparation, coordinate transform, model inference, result merging, and matplotlib visualization. Useful for understanding the pipeline internals or debugging issues.

### 03 — COPC Conversion

Converts LAS/LAZ output files to COPC (Cloud-Optimized Point Cloud) format using PDAL. Includes single-file conversion, batch conversion, PDAL verification, and instructions for serving COPC files over HTTP for web-based 3D viewers.

### 04 — Batch Processing

Processes many files with progress tracking. Surveys input files, runs batch inference with configurable chunk sizes, monitors completion, and aggregates per-file statistics (tree count, point count, tree percentage) into a summary table.

### 05 — Visualization & Analysis

Creates publication-ready plots from segmentation results: bird's-eye instance and semantic maps, points-per-tree histograms, per-tree metrics extraction (height, crown diameter, centroid), and tree height/crown distributions. Also covers 3D visualization with CloudCompare, QGIS, and web viewers.
