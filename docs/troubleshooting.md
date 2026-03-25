# Troubleshooting

## CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce point cloud density with sparsification:
  ```bash
  python -m sat.preprocessing.sparsify -i input/ -o sparse_input/ -d 100
  ```
- Split large point clouds into smaller tiles before processing
- Use a GPU with more VRAM (the model works well with 8+ GB)

## Empty Instances / Zero-Size Array

**Symptom**: `ValueError: zero-size array to reduction operation maximum` or files with no detected trees

**Cause**: Some point clouds contain no tree points (e.g., ground-only areas). This is normal.

**Solution**: The pipeline handles this gracefully in most cases. For batch processing, check the skipped files log. Files with only ground points will not produce instance segmentation output.

## Storage Full During Batch Processing

**Symptom**: Disk full errors during processing of many files

**Cause**: Intermediate files (coordinate transforms, model outputs, merged results) accumulate.

**Solutions**:
- Use `scripts/run_batch_inference.sh` which cleans up between batches
- For Singularity/SLURM: use the checkpointed script in [singularity.md](singularity.md) which cleans temp files per file
- Monitor storage with `df -h` during processing

## uint16 Overflow in Output

**Symptom**: `OverflowError` or corrupted PredInstance values

**Cause**: Instance IDs or other fields contain values outside uint16 range (0-65535).

**Solution**: The `sat.io.las_io.pandas_to_las` function clips uint16 fields automatically. If you encounter this with custom code, ensure values are in range before writing.

## Docker Build Fails at MinkowskiEngine (CUDA 11.8 only)

**Symptom**: Compilation errors during `pip install MinkowskiEngine`

!!! note
    This only applies to the CUDA 11.8 image (`docker/Dockerfile.cuda11`). The CUDA 12.4 image uses SpConv v2.x which installs via pip with no compilation.

**Solutions**:
- Ensure NVIDIA driver is 525+
- Check that `TORCH_CUDA_ARCH_LIST` includes your GPU's compute capability
- Try building with `--no-cache` to avoid stale layers:
  ```bash
  docker build --no-cache -f docker/Dockerfile.cuda11 -t segmentanytree:cuda11 .
  ```

## Segmentation Fault (exit code 139)

**Symptom**: Process exits with code 139 (common in Singularity environments)

**Cause**: Usually a memory issue or library incompatibility.

**Solutions**:
- Increase memory allocation (`#SBATCH --mem=128G`)
- Check that output files were still created — segfaults sometimes occur during cleanup after results are written
- Ensure the Singularity image was built from the correct Docker image

## Model Weights Are a Git LFS Pointer

**Symptom**: Inference produces garbage results — PredSemantic has no class 2 (tree), PredInstance is empty or has only a handful of tiny clusters. Eval log shows all cluster scores ~0.43 and cluster sizes of 5-15 points.

**Cause**: `model_file/PointGroup-PAPER.pt` is tracked by Git LFS. If you cloned the repo without `git-lfs` installed, the file is a 134-byte text pointer instead of the actual 665 MB model checkpoint. Docker builds will also fail or produce broken images.

**Diagnosis**:
```bash
# Check file size — should be ~665 MB, not 134 bytes
ls -lh model_file/PointGroup-PAPER.pt

# Check if it's a pointer file
file model_file/PointGroup-PAPER.pt
# Bad:  "ASCII text" (LFS pointer)
# Good: "Zip archive data" (PyTorch checkpoint)
```

**Solution**:
```bash
# Option 1: Pull via git-lfs
git lfs install
git lfs pull --include="model_file/PointGroup-PAPER.pt"

# Option 2: Download directly from upstream repo
curl -L -o model_file/PointGroup-PAPER.pt \
  "https://github.com/SmartForest-no/SegmentAnyTree/raw/main/model_file/PointGroup-PAPER.pt"
```

After fixing, rebuild Docker images (`make build-cuda12` / `make build-cuda11`). The Dockerfiles now include a build-time check that will fail fast if the weights are LFS pointers.

For CUDA 12 (SpConv), you must also regenerate the migrated weights:
```bash
python scripts/migrate_weights.py \
  --input model_file/PointGroup-PAPER.pt \
  --output model_file/PointGroup-PAPER-spconv.pt
```

## Model File Not Found

**Symptom**: `FileNotFoundError: model_file/PointGroup-PAPER.pt`

**Solution**: The pre-trained model must be in `$SAT_ROOT/model_file/`. In Docker, this is bundled in the image. For local usage, ensure `model_file/PointGroup-PAPER.pt` exists in the repository root.
