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

## Docker Build Fails at MinkowskiEngine

**Symptom**: Compilation errors during `pip install MinkowskiEngine`

**Solutions**:
- Ensure NVIDIA driver is 525+ (for CUDA 12.4 support)
- Check that `TORCH_CUDA_ARCH_LIST` includes your GPU's compute capability
- Try building with `--no-cache` to avoid stale layers:
  ```bash
  docker build --no-cache -t segmentanytree:latest .
  ```

## Segmentation Fault (exit code 139)

**Symptom**: Process exits with code 139 (common in Singularity environments)

**Cause**: Usually a memory issue or library incompatibility.

**Solutions**:
- Increase memory allocation (`#SBATCH --mem=128G`)
- Check that output files were still created — segfaults sometimes occur during cleanup after results are written
- Ensure the Singularity image was built from the correct Docker image

## Model File Not Found

**Symptom**: `FileNotFoundError: model_file/PointGroup-PAPER.pt`

**Solution**: The pre-trained model must be in `$SAT_ROOT/model_file/`. In Docker, this is bundled in the image. For local usage, ensure `model_file/PointGroup-PAPER.pt` exists in the repository root.
