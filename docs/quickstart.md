# Quick Start

Get tree segmentation results in 5 minutes using the pre-built Docker image.

## Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- An NVIDIA GPU (Volta or newer: V100, A100, RTX 20xx/30xx/40xx/50xx) with driver 525+
- **Git LFS** installed (`git lfs install`) — the model weights are stored via Git LFS
- Point cloud files in `.las`, `.laz`, or `.ply` format

!!! warning "Git LFS required"
    The pre-trained model (`model_file/PointGroup-PAPER.pt`, 665 MB) is stored with Git LFS. Without `git-lfs` installed, you'll get a 134-byte pointer file instead of real weights, and inference will produce garbage results. See [Troubleshooting](troubleshooting.md#model-weights-are-a-git-lfs-pointer) for details.

!!! tip "No data yet?"
    The [FOR-instance dataset](example-data.md#for-instance-dataset-recommended-for-validation) from the paper is the best choice for validating your setup. Also see the [Example Data](example-data.md) page for other options.

## Steps

### 1. Prepare directories

```bash
mkdir -p $HOME/segmentanytree/input
mkdir -p $HOME/segmentanytree/output
```

### 2. Add your point cloud files

Copy your `.las`, `.laz`, or `.ply` files into `$HOME/segmentanytree/input/`.

### 3. Run inference

=== "Harbor Registry"

    ```bash
    docker pull harbor.cyverse.org/vice/segmentanytree:cuda11
    docker run --gpus all \
      -v $HOME/segmentanytree/input:/data/input \
      -v $HOME/segmentanytree/output:/data/output \
      harbor.cyverse.org/vice/segmentanytree:cuda11 \
      bash scripts/run_inference.sh /data/input /data/output true
    ```

=== "Local Build"

    ```bash
    docker run --gpus all \
      -v $HOME/segmentanytree/input:/data/input \
      -v $HOME/segmentanytree/output:/data/output \
      segmentanytree:cuda11 \
      bash scripts/run_inference.sh /data/input /data/output true
    ```

### 4. Check results

Segmented files appear in `$HOME/segmentanytree/output/final_results/` as `.copc.laz` files (Cloud-Optimized Point Clouds). Each file contains the original point cloud with two extra dimensions added by the model.

If COPC conversion is not available (PDAL not installed), output files will be `.laz` instead.

### 5. Interactive mode (JupyterLab)

=== "Harbor Registry"

    ```bash
    docker run --gpus all -p 8888:8888 \
      -v $HOME/segmentanytree/input:/data/input \
      -v $HOME/segmentanytree/output:/data/output \
      harbor.cyverse.org/vice/segmentanytree:cuda11
    ```

=== "Local Build"

    ```bash
    docker run --gpus all -p 8888:8888 \
      -v $HOME/segmentanytree/input:/data/input \
      -v $HOME/segmentanytree/output:/data/output \
      segmentanytree:cuda11
    ```

Open http://localhost:8888 and use the notebooks in `notebooks/`. See the [Notebooks](notebooks.md) page for descriptions of each notebook.

## Output format

The output files contain two extra dimensions:
- **PredSemantic**: `0` = unclassified, `1` = non-tree, `2` = tree
- **PredInstance**: unique integer ID per detected tree (0 = unassigned)

These files preserve all original point attributes (coordinates, intensity, return number, etc.) with UTM coordinates restored. Output is in [COPC](https://copc.io/) format (`.copc.laz`) for efficient streaming and visualization in tools like [CloudCompare](https://www.danielgm.net/cc/), [QGIS](https://qgis.org/), and [Potree](https://potree.github.io/).

See the [Scientific Workflow](workflow.md) guide for the full end-to-end process including visualization.
