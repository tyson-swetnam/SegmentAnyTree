# Singularity / Apptainer (HPC Usage)

## Convert Docker Image to SIF

=== "CUDA 11.8 (recommended)"

    ```bash
    singularity pull segmentanytree-cuda11.sif \
      docker://harbor.cyverse.org/vice/segmentanytree:cuda11
    ```

=== "CUDA 12.4 (experimental)"

    ```bash
    singularity pull segmentanytree-cuda12.sif \
      docker://harbor.cyverse.org/vice/segmentanytree:cuda12
    ```

=== "From Local Build"

    ```bash
    singularity pull segmentanytree-cuda11.sif docker://segmentanytree:cuda11
    ```

## Basic Inference

```bash
singularity exec --nv \
    --bind /path/to/input:/data/input \
    --bind /path/to/output:/data/output \
    segmentanytree-cuda11.sif \
    bash /opt/segmentanytree/scripts/run_inference.sh /data/input /data/output true
```

## SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=segmentanytree
#SBATCH --output=sat_%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

module load singularity  # or apptainer

SIF="/path/to/segmentanytree-cuda11.sif"
INPUT="/scratch/$USER/input"
OUTPUT="/scratch/$USER/output"
TEMP="/scratch/$USER/sat_temp"

mkdir -p "$OUTPUT" "$TEMP"

singularity exec --nv \
    --bind "$INPUT:/data/input" \
    --bind "$OUTPUT:/data/output" \
    --bind "$TEMP:/tmp/sat_cache" \
    "$SIF" \
    bash /opt/segmentanytree/scripts/run_inference.sh /data/input /data/output true
```

## Checkpointed Processing (Large Datasets)

For processing many files with potential job time limits, process one file at a time with checkpointing:

```bash
#!/bin/bash
#SBATCH --job-name=sat_batch
#SBATCH --output=sat_batch_%j.log
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

SIF="/path/to/segmentanytree-cuda11.sif"
INPUT="/scratch/$USER/input"
OUTPUT="/scratch/$USER/output"
CHECKPOINT="/scratch/$USER/checkpoints"
TEMP="/scratch/$USER/sat_temp"

mkdir -p "$OUTPUT" "$CHECKPOINT" "$TEMP"

for laz in "$INPUT"/*.laz; do
    fname=$(basename "$laz")
    marker="$CHECKPOINT/$fname.done"

    [ -f "$marker" ] && echo "Skip (done): $fname" && continue

    echo "Processing: $fname"

    TEMP_IN="$TEMP/in_$$"
    TEMP_OUT="$TEMP/out_$$"
    mkdir -p "$TEMP_IN" "$TEMP_OUT"
    cp "$laz" "$TEMP_IN/"

    singularity exec --nv \
        --bind "$TEMP_IN:/data/input" \
        --bind "$TEMP_OUT:/data/output" \
        "$SIF" \
        bash /opt/segmentanytree/scripts/run_inference.sh /data/input /data/output true \
        && touch "$marker"

    # Copy results
    [ -d "$TEMP_OUT/final_results" ] && cp "$TEMP_OUT/final_results"/* "$OUTPUT/" 2>/dev/null

    # Clean temp to free storage
    rm -rf "$TEMP_IN" "$TEMP_OUT"
done

echo "Done: $(ls "$OUTPUT" | wc -l) output files"
```

## Storage Tips

- Use `/scratch` or `/tmp` for temporary processing files — they are large
- Each file generates ~3x its size in intermediate data (UTM transform + model output + merged result)
- Clean `$TEMP` between files to avoid filling `/scratch`
- The `--bind` flag is the Singularity equivalent of Docker's `-v` mount
