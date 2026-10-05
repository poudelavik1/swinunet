#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-swinunet}"
source "$CONDA_SH"
if ! conda run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
    conda env create -n "$ENV_NAME" -f "$ROOT/environment.yml"
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip
# Matched official CUDA 12.6 wheels; override the index for a different driver.
python -m pip install torch==2.9.1 torchvision==0.24.1 \
    --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
python -m pip install -r "$ROOT/requirements.txt"
python -m pip install --no-deps -e "$ROOT"
python -m pip check
python -c "import torch, torchvision, cv2, numpy, PIL, matplotlib; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda)"
echo "Environment ready. GPU access is verified inside the Slurm job."
