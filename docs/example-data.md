# Example Data

Test SegmentAnyTree with publicly available LiDAR point cloud data.

## FOR-instance Dataset (recommended for validation)

The [FOR-instance](https://zenodo.org/records/8287792) dataset is the **primary training and test dataset** used in the SegmentAnyTree paper. It contains 1,130 manually segmented trees across 5 sites with full ground-truth annotations (`treeID`, classification labels). This is the best dataset for validating that your installation produces correct results.

| Site | Country | Sensor | Files | Size |
|------|---------|--------|-------|------|
| CULS | Czech Republic | UAS | 3 plots | 287 MB |
| NIBIO | Norway | UAS | 20 plots | 4.2 GB |
| RMIT | Australia | UAS | train + test | 60 MB |
| SCION | New Zealand | UAS | 5 plots | 485 MB |
| TUWIEN | Austria | PLS | train + test | 395 MB |

**Classification labels**: 0=Unclassified, 1=Low-vegetation, 2=Terrain, 3=Out-points, 4=Stem, 5=Live-branches, 6=Woody-branches

### Download FOR-instance

```bash
mkdir -p $HOME/segmentanytree/for-instance
cd $HOME/segmentanytree/for-instance

# Download (1.6 GB zip)
curl -L -o FORinstance_dataset.zip \
  "https://zenodo.org/records/8287792/files/FORinstance_dataset.zip?download=1"

# Extract
unzip FORinstance_dataset.zip
```

### Run inference on FOR-instance

!!! tip "Start with RMIT/test.las"
    The RMIT test set (60 MB, 357K points) is the smallest file — ideal for validating your setup quickly.

=== "CUDA 12 (SpConv)"

    ```bash
    # Copy a test file
    mkdir -p $HOME/segmentanytree/input
    cp $HOME/segmentanytree/for-instance/RMIT/test.las $HOME/segmentanytree/input/

    docker run --gpus all \
      -v $HOME/segmentanytree/input:/data/input \
      -v $HOME/segmentanytree/output:/data/output \
      segmentanytree:cuda12 \
      bash scripts/run_inference.sh /data/input /data/output true
    ```

=== "CUDA 11 (MinkowskiEngine)"

    ```bash
    mkdir -p $HOME/segmentanytree/input
    cp $HOME/segmentanytree/for-instance/RMIT/test.las $HOME/segmentanytree/input/

    docker run --gpus all \
      -v $HOME/segmentanytree/input:/data/input \
      -v $HOME/segmentanytree/output:/data/output \
      segmentanytree:cuda11 \
      bash scripts/run_inference.sh /data/input /data/output true
    ```

### Validate results against ground truth

The FOR-instance files contain `treeID` fields for ground-truth comparison. After inference, compare `PredInstance` IDs against `treeID` in [CloudCompare](https://www.danielgm.net/cc/) or programmatically:

```python
import laspy
import numpy as np

# Load ground truth
gt = laspy.read("for-instance/RMIT/test.las")
gt_tree_ids = np.array(gt.treeID)

# Load prediction
pred = laspy.read("output/final_results/test_out.copc.laz")
pred_instances = np.array(pred.PredInstance)
pred_semantic = np.array(pred.PredSemantic)

# Check: PredSemantic should have class 2 (tree) for most forest points
unique_sem, counts_sem = np.unique(pred_semantic, return_counts=True)
print(f"Semantic classes: {dict(zip(unique_sem, counts_sem))}")

# Check: PredInstance should have multiple unique tree IDs
unique_inst = np.unique(pred_instances[pred_instances > 0])
print(f"Detected {len(unique_inst)} tree instances")
print(f"Ground truth has {len(np.unique(gt_tree_ids[gt_tree_ids > 0]))} trees")
```

## NIBIO MLS Dataset

The [NIBIO MLS](https://zenodo.org/records/12754726) dataset was introduced in the SegmentAnyTree paper as a new benchmark. It contains 16 mobile laser scanning plots (~250 m² each) with train/val/test splits.

```bash
mkdir -p $HOME/segmentanytree/nibio-mls
cd $HOME/segmentanytree/nibio-mls

# Download (1.3 GB zip)
curl -L -o NIBIO_MLS.zip \
  "https://zenodo.org/api/records/12754726/files/NIBIO_MLS.zip?download=1"

unzip NIBIO_MLS.zip
```

## SWERI LiDAR Examples

Three example datasets from different sensor platforms are available on the CyVerse Data Store, along with their pre-segmented outputs:

| File | Sensor | Size | Description |
|------|--------|------|-------------|
| `2020-07-04_09-46-07_ALS_Clip.laz` | Airborne (ALS) | 15 MB | Airborne laser scanning clip |
| `2020-07-04_09-46-07_UAS_Clip.laz` | Drone (UAS) | 106 MB | Uncrewed aerial system clip |
| `2020-07-04_09-46-07_MLS_Clip.laz` | Mobile (MLS) | 682 MB | Mobile laser scanning clip |

Pre-segmented results (in `segmented/` subdirectory) are available as COPC LAZ files for comparison.

### Download via WebDAV (no login required)

```bash
mkdir -p $HOME/segmentanytree/input

# Download ALS example (smallest, good for quick testing)
curl -O -L https://data.cyverse.org/dav-anon/iplant/projects/sweri/lidar_examples/2020-07-04_09-46-07_ALS_Clip.laz \
  --output-dir $HOME/segmentanytree/input/

# Download UAS example
curl -O -L https://data.cyverse.org/dav-anon/iplant/projects/sweri/lidar_examples/2020-07-04_09-46-07_UAS_Clip.laz \
  --output-dir $HOME/segmentanytree/input/

# Download MLS example (largest — 682 MB)
curl -O -L https://data.cyverse.org/dav-anon/iplant/projects/sweri/lidar_examples/2020-07-04_09-46-07_MLS_Clip.laz \
  --output-dir $HOME/segmentanytree/input/
```

### Download pre-segmented results for comparison

```bash
mkdir -p $HOME/segmentanytree/expected

curl -O -L https://data.cyverse.org/dav-anon/iplant/projects/sweri/lidar_examples/segmented/2020-07-04_09-46-07_ALS_Clip_segmented.copc.laz \
  --output-dir $HOME/segmentanytree/expected/

curl -O -L https://data.cyverse.org/dav-anon/iplant/projects/sweri/lidar_examples/segmented/2020-07-04_09-46-07_UAS_Clip_segmented.copc.laz \
  --output-dir $HOME/segmentanytree/expected/

curl -O -L https://data.cyverse.org/dav-anon/iplant/projects/sweri/lidar_examples/segmented/2020-07-04_09-46-07_MLS_Clip_segmented.copc.laz \
  --output-dir $HOME/segmentanytree/expected/
```

### Download with GoCommands

```bash
# Install GoCommands (if not already installed)
cd /usr/local/bin && \
GOCMD_VER=$(curl -L -s https://raw.githubusercontent.com/cyverse/gocommands/main/VERSION.txt) && \
curl -L -s https://github.com/cyverse/gocommands/releases/download/${GOCMD_VER}/gocmd-${GOCMD_VER}-linux-amd64.tar.gz | tar zxvf -

# Initialize GoCommands with CyVerse credentials
gocmd init

# Download all example data
gocmd get --progress /iplant/projects/sweri/lidar_examples/ $HOME/segmentanytree/input/
```

### Download with iCommands

```bash
iinit
ils /iplant/projects/sweri/lidar_examples/
iget -r /iplant/projects/sweri/lidar_examples/ $HOME/segmentanytree/input/
```

## Running with example data

Once you have data downloaded:

=== "Docker (Harbor)"

    ```bash
    docker pull harbor.cyverse.org/vice/segmentanytree:cuda11
    docker run --gpus all \
      -v $HOME/segmentanytree/input:/data/input \
      -v $HOME/segmentanytree/output:/data/output \
      harbor.cyverse.org/vice/segmentanytree:cuda11 \
      bash scripts/run_inference.sh /data/input /data/output true
    ```

=== "Docker (Local Build)"

    ```bash
    docker run --gpus all \
      -v $HOME/segmentanytree/input:/data/input \
      -v $HOME/segmentanytree/output:/data/output \
      segmentanytree:cuda11 \
      bash scripts/run_inference.sh /data/input /data/output true
    ```

=== "Singularity"

    ```bash
    singularity exec --nv \
      --bind $HOME/segmentanytree/input:/data/input \
      --bind $HOME/segmentanytree/output:/data/output \
      segmentanytree-cuda12.sif \
      bash /opt/segmentanytree/scripts/run_inference.sh /data/input /data/output true
    ```

=== "Local"

    ```bash
    export SAT_ROOT=$(pwd)
    export SAT_DATA=$HOME/segmentanytree
    bash scripts/run_inference.sh $SAT_DATA/input $SAT_DATA/output true
    ```

!!! tip "Start with the ALS file"
    The ALS clip (15 MB) is the smallest and fastest to process — ideal for verifying your setup works before running the larger UAS or MLS files.

## Verifying results

After inference, check the output:

```bash
# List output files
ls -la $HOME/segmentanytree/output/final_results/

# Quick validation with PDAL
pdal info --summary $HOME/segmentanytree/output/final_results/*.las
```

### Compare with pre-segmented reference

If you downloaded the pre-segmented COPC files, you can compare your results visually in [CloudCompare](https://www.danielgm.net/cc/) by loading both your output and the reference file, coloring by `PredInstance`.

The output LAS files contain `PredSemantic` and `PredInstance` extra dimensions. Visualize results in:

- [CloudCompare](https://www.danielgm.net/cc/) — color by `PredInstance` to see individual trees
- [QGIS](https://qgis.org/) — with the point cloud renderer
- [Potree](https://potree.github.io/) — web-based 3D viewer

## Other public LiDAR sources

### NEON AOP Data

The [National Ecological Observatory Network (NEON)](https://www.neonscience.org/) provides free airborne LiDAR data across ecological sites in the US.

```bash
# Example: Download a single LAZ tile from NEON
# Replace SITE and YEAR with your site of interest
curl -O https://data.neonscience.org/api/v0/data/DP1.30003.001/SITE/YEAR
```

NEON data is classified but requires re-labeling for SegmentAnyTree's expected format. See the [Training guide](training.md) for classification label mapping.

### OpenTopography

[OpenTopography](https://opentopography.org/) provides access to LiDAR point cloud datasets. Search for datasets in your area of interest and download in LAZ format.

## Data requirements

| Requirement | Details |
|-------------|---------|
| Format | `.las`, `.laz`, or `.ply` |
| Coordinate system | Any projected CRS (UTM recommended) |
| Point density | Works with various densities; very sparse data may yield fewer detections |
| Classification | Not required for inference (only for training) |
| File size | Split large files (>500M points) into tiles for best performance |

!!! warning "CUDA memory"
    Large point clouds may exceed GPU memory. Use the sparsification tool to reduce density:
    ```bash
    python -m sat.preprocessing.sparsify -i input/ -o sparse_input/ -d 100
    ```
    See [Troubleshooting](troubleshooting.md) for more memory management tips.
