# Architecture

## Model: PointGroup-PAPER (3-head variant)

SegmentAnyTree performs **panoptic segmentation** — combined semantic classification and instance segmentation — on 3D LiDAR point clouds.

### Overview

```
Input Point Cloud (x, y, z, features)
       ↓
  Minkowski U-Net Backbone (7-layer encoder-decoder, sparse convolutions)
       ↓
  ┌────┴────┐────────┐
  ↓         ↓        ↓
Semantic  Offset  Embedding
 Head      Head     Head
  ↓         ↓        ↓
Class     3D vote  5D cluster
logits    offsets  embeddings
  ↓         ↓        ↓
  └────┬────┘────────┘
       ↓
  Clustering (region growing + embedding similarity)
       ↓
  Scoring Network (instance quality estimation)
       ↓
  Instance Segmentation Output
```

### Three Output Heads

1. **Semantic Head**: Per-point classification (non-tree / tree). Uses NLL loss.
2. **Offset Head**: Predicts 3D offset vectors that "vote" toward instance centers. Uses norm + directional loss.
3. **Embedding Head**: Produces 5D discriminative embeddings for clustering. Uses variance + distance + regularization loss.

### Backbone

The backbone is a **Minkowski U-Net** with sparse 3D convolutions:
- 7 encoder layers, 7 decoder layers with skip connections
- Feature dimension: 16 (configurable via `feat_size`)
- Convolution type: SPARSE (via SpConv v2.x for CUDA 12.4, or MinkowskiEngine for CUDA 11.8)
- Stride pattern: `[1, 2, 2, 2, 2, 2, 2]`

### Clustering

After the three heads produce their outputs, clustering combines offset-based voting with embedding similarity:
- `cluster_type: 5` (default): Combined offset + embedding clustering
- Region growing with `cluster_radius_search: 1.5 * grid_size`
- Only applied to "thing" classes (trees), not "stuff" (non-tree)

### Data Pipeline

- **Grid sampling**: 0.2m voxel grid (configurable)
- **Sampling radius**: 8m cylinders for training samples
- **Augmentation** (training only): Random noise (σ=0.01), rotation (180°), scale (0.9-1.1), symmetry
- **Features**: Relative XYZ position + absolute Z height (4D input features)

### Configuration

All model hyperparameters are in `conf/models/panoptic/area4_ablation_3heads_5.yaml`. The model architecture, loss weights, and clustering parameters can be modified there.

## Framework: torch-points3d

The codebase builds on the [torch-points3d](https://github.com/torch-points3d/torch-points3d) framework, which provides:

- **Trainer** (`torch_points3d/trainer.py`): Training/evaluation orchestration
- **Model factory**: Instantiates models from Hydra config
- **Dataset factory**: Loads and transforms point cloud data
- **Metrics tracking**: Panoptic/segmentation evaluation

## Configuration: Hydra

All configuration uses [Hydra 1.x](https://hydra.cc/) for hierarchical YAML composition:

```
conf/
├── config.yaml          # Training defaults
├── eval.yaml            # Inference config
├── models/panoptic/     # Model architectures
├── data/panoptic/       # Dataset configs
├── training/            # Training hyperparameters
└── lr_scheduler/        # Learning rate schedules
```

Override any parameter via command line: `python train.py training.epochs=200`.
