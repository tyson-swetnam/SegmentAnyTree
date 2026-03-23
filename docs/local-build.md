# Local Build (without Docker)

This guide sets up SegmentAnyTree for development and inference directly on a Linux machine with an NVIDIA GPU, without Docker.

## Prerequisites

- Linux (Ubuntu 20.04/22.04 recommended)
- NVIDIA GPU with CUDA support (Compute Capability 7.0+)
- NVIDIA driver installed (`nvidia-smi` works)
- ~20 GB disk for CUDA toolkit + conda env

## 1. Install Miniforge (conda + mamba)

```bash
curl -fsSL -o /tmp/miniforge.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge.sh -b -p $HOME/miniforge
rm /tmp/miniforge.sh
export PATH="$HOME/miniforge/bin:$PATH"
```

## 2. Install CUDA 11.8 Toolkit

The GPU libraries (MinkowskiEngine, torchsparse, torch-points-kernels) require CUDA 11.8 to match the PyTorch build. Install it to a user directory (no sudo needed):

```bash
wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run \
  -O /tmp/cuda_11.8.run
sh /tmp/cuda_11.8.run --toolkit --silent --override --installpath=$HOME/cuda-11.8 --no-man-page
rm /tmp/cuda_11.8.run
```

Verify: `$HOME/cuda-11.8/bin/nvcc --version` should show `release 11.8`.

## 3. Create Conda Environment

```bash
mamba create -n sat python=3.10 pdal cmake ninja git -y -c conda-forge
```

## 4. Install Python Dependencies

Activate the env and set CUDA paths:

```bash
export PATH="$HOME/miniforge/envs/sat/bin:$HOME/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/cuda-11.8/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME=$HOME/cuda-11.8
```

Install PyTorch 2.1.2 + CUDA 11.8:

```bash
pip install --no-cache-dir --upgrade pip "setuptools<70" wheel ninja
pip install --no-cache-dir "numpy<2" scipy cython
pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

Install PyTorch Geometric (pre-built wheels):

```bash
pip install --no-cache-dir \
  torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
  -f https://data.pyg.org/whl/torch-2.1.2+cu118.html
```

Install remaining Python packages:

```bash
pip install --no-cache-dir \
  hydra-core==1.3.2 omegaconf==2.3.0 wandb tensorboard \
  tqdm pandas scikit-learn scikit-image matplotlib seaborn \
  h5py plyfile "laspy[lazrs]" open3d gdown numba joblib dask \
  pykdtree jaklas pytorch-metric-learning addict python-louvain \
  hdbscan jupyterlab ipywidgets absl-py protobuf"<7" \
  pydantic gitpython sentry-sdk markdown
```

Install torchnet and patch the visdom import:

```bash
pip install --no-cache-dir --no-deps torchnet
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
cat > $SITE/torchnet/logger/__init__.py << 'PATCH'
try:
    from .visdomlogger import *
except ImportError:
    pass
PATCH
```

## 5. Install GPU Libraries

These must be compiled from source with matching CUDA 11.8 and GCC <= 11.

### Option A: Copy from Docker image (fastest)

If you have the Docker image built:

```bash
CONDA_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
DOCKER_SITE=/usr/local/lib/python3.10/dist-packages

docker create --name sat_extract segmentanytree:latest true
for pkg in MinkowskiEngine MinkowskiEngineBackend torchsparse torch_points_kernels torchnet; do
  docker cp sat_extract:$DOCKER_SITE/$pkg $CONDA_SITE/$pkg 2>/dev/null
done
docker rm sat_extract
```

### Option B: Build from source

Install GCC 11 (required for CUDA 11.8):

```bash
mamba install -n sat -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11
export CC=$(which x86_64-conda-linux-gnu-gcc)
export CXX=$(which x86_64-conda-linux-gnu-g++)
```

Build each library:

```bash
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
export FORCE_CUDA=1

# torch-points-kernels
git clone --depth 1 https://github.com/torch-points3d/torch-points-kernels.git /tmp/tpk
cd /tmp/tpk && python setup.py install

# torchsparse v1.4.0
pip install --no-cache-dir --no-build-isolation git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

# MinkowskiEngine
git clone --depth 1 https://github.com/NVIDIA/MinkowskiEngine.git /tmp/ME
cd /tmp/ME && python setup.py install --blas=openblas --force_cuda
```

## 6. Verify Installation

```bash
python -c "
import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
from torch_points_kernels import region_grow; print('torch-points-kernels: OK')
import MinkowskiEngine; print('MinkowskiEngine:', MinkowskiEngine.__version__)
import torchsparse; print('torchsparse: OK')
import laspy; print('laspy: OK')
import subprocess; subprocess.run(['pdal', '--version'])
"
```

## 7. Run Inference

```bash
export PATH="$HOME/miniforge/envs/sat/bin:$HOME/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/cuda-11.8/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/path/to/SegmentAnyTree:$PYTHONPATH"
export SAT_ROOT="/path/to/SegmentAnyTree"

cd $SAT_ROOT
bash scripts/run_inference.sh /path/to/input /path/to/output true
```

## Shell Setup (add to ~/.bashrc)

```bash
# SegmentAnyTree environment
alias sat-env='
  export PATH="$HOME/miniforge/envs/sat/bin:$HOME/cuda-11.8/bin:$PATH"
  export LD_LIBRARY_PATH="$HOME/cuda-11.8/lib64:$LD_LIBRARY_PATH"
  export PYTHONPATH="$HOME/github/SegmentAnyTree:$PYTHONPATH"
  export SAT_ROOT="$HOME/github/SegmentAnyTree"
  echo "SegmentAnyTree env activated"
'
```

Then just run `sat-env` before working with SegmentAnyTree.
