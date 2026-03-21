#!/bin/bash
set -e

# Docker build verification test
# Usage: bash tests/test_docker_build.sh [image_name]

IMAGE="${1:-segmentanytree:latest}"

echo "=== Docker Build Verification ==="
echo "Image: $IMAGE"
echo ""

FAIL=0

check() {
    local desc="$1"
    shift
    printf "  %-50s " "$desc"
    if docker run --rm "$IMAGE" "$@" > /dev/null 2>&1; then
        echo "OK"
    else
        echo "FAIL"
        FAIL=1
    fi
}

echo "Checking binaries..."
check "python3 exists" python3 --version
check "jupyter exists" jupyter --version
check "bash exists" bash --version

echo ""
echo "Checking Python imports..."
check "import torch" python3 -c "import torch; print(torch.__version__)"
check "import MinkowskiEngine" python3 -c "import MinkowskiEngine"
check "import torchsparse" python3 -c "import torchsparse"
check "import torch_points_kernels" python3 -c "import torch_points_kernels"
check "import torch_geometric" python3 -c "import torch_geometric"
check "import hydra" python3 -c "import hydra"
check "import laspy" python3 -c "import laspy"
check "import plyfile" python3 -c "import plyfile"
check "import sat" python3 -c "import sat"
check "import torch_points3d" python3 -c "import torch_points3d"

echo ""
echo "Checking environment..."
check "SAT_ROOT set" python3 -c "import os; assert os.environ.get('SAT_ROOT')"
check "model file exists" python3 -c "from pathlib import Path; import os; assert (Path(os.environ['SAT_ROOT']) / 'model_file' / 'PointGroup-PAPER.pt').exists()"
check "/data/input exists" test -d /data/input
check "/data/output exists" test -d /data/output

echo ""
echo "Checking CUDA (requires --gpus)..."
printf "  %-50s " "torch.cuda.is_available()"
if docker run --rm --gpus all "$IMAGE" python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "OK"
else
    echo "SKIP (no GPU or --gpus not available)"
fi

echo ""
if [ $FAIL -eq 0 ]; then
    echo "All checks passed!"
else
    echo "Some checks FAILED"
    exit 1
fi
