# Training Guide

## Data Preparation

Training data must be LAS files with these fields:
- X, Y, Z coordinates
- `classification`: Point class (see mapping below)
- `treeID`: Ground truth instance ID per tree
- `intensity`: LiDAR intensity

### Classification Label Mapping

| LAS classification | Semantic label | Meaning |
|-------------------|----------------|---------|
| 0 | 0 | Unclassified |
| 1 | 1 | Low vegetation → non-tree |
| 2 | 1 | Ground → non-tree |
| 3 | 0 | Outpoints → unclassified |
| 4 | 2 | Stem → tree |
| 5 | 2 | Live branches → tree |
| 6 | 2 | Branches → tree |

### Convert LAS to PLY

```bash
python -m sat.io.conversion \
    --las_dir /path/to/las/data \
    --output_dir /path/to/ply/output
```

This reads `data_split_metadata.csv` from the LAS directory to determine train/val/test splits. The CSV should have columns: `path, region, split` (where split is "train" or "test"; validation is automatically sampled from training at 25%).

## Training Command

```bash
python train.py \
    task=panoptic \
    data=panoptic/treeins \
    models=panoptic/area4_ablation_3heads \
    model_name=PointGroup-PAPER \
    training=treeins \
    job_name=my_experiment
```

Or use the wrapper script:

```bash
bash scripts/run_training.sh my_experiment
```

### Key Parameters

Override via command line (Hydra syntax):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.epochs` | 300 | Number of training epochs |
| `training.batch_size` | 4 | Batch size |
| `training.cuda` | 0 | GPU device index |
| `data.first_subsampling` | 0.2 | Grid sampling size (meters) |
| `data.radius` | 8 | Cylinder sampling radius (meters) |

## Monitoring

Training logs to:
- **wandb**: Set `WANDB_API_KEY` environment variable
- **tensorboard**: Logs in `outputs/<job_name>/`

```bash
tensorboard --logdir outputs/
```

## Checkpoints

Model checkpoints are saved in `outputs/<job_name>/`. The best model is selected based on validation metrics. Use the checkpoint directory path in `conf/eval.yaml` for inference.

## Fine-tuning

To fine-tune the pre-trained model on new data:

```bash
python train.py \
    task=panoptic \
    data=panoptic/treeins \
    models=panoptic/area4_ablation_3heads \
    model_name=PointGroup-PAPER \
    training=treeins \
    job_name=finetune_run \
    training.checkpoint_dir=model_file
```

## Multi-GPU Training (DDP)

Train across multiple GPUs using PyTorch DistributedDataParallel:

```bash
# 4-GPU training
torchrun --nproc_per_node=4 train.py task=panoptic data=panoptic/treeins \
  models=panoptic/area4_ablation_3heads model_name=PointGroup-PAPER \
  training=treeins job_name=ddp_4gpu

# Or use the Makefile
make train-ddp GPUS=4
```

DDP automatically shards the dataset across GPUs and synchronizes gradients. Only rank 0 saves checkpoints and logs to wandb. Single-GPU training (`python train.py ...`) is unchanged.
