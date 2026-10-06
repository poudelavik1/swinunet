#!/bin/bash
#SBATCH --job-name=gpu_check
#SBATCH --partition=ASL-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Ten-minute check of the GPU node, driver, and Conda environment before a long run.
# Submit from the repository root after setup: sbatch slurm/gpu_check.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "Node: $(hostname) | Job: $SLURM_JOB_ID | $(date)"
# As in pgat-length: the login node's /opt/miniconda3 is not on the compute node,
# so put the environment on PATH instead of activating it.
ENV_DIR="${ENV_DIR:-$HOME/.conda/envs/${ENV_NAME:-swinunet}}"
test -x "$ENV_DIR/bin/python" || { echo "Conda environment not found: $ENV_DIR" >&2; exit 1; }
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi
python - <<'PY'
from pathlib import Path
import torch, torchvision
print('PyTorch:', torch.__version__, '| torchvision:', torchvision.__version__, '| CUDA build:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA unavailable: check the GPU driver and PyTorch wheel compatibility'
memory = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
print(f'GPU: {torch.cuda.get_device_name(0)} | {memory:.0f} GB')
x = torch.randn(4096, 4096, device='cuda')
print('CUDA matmul check:', tuple((x @ x).shape))
weights = Path(torch.hub.get_dir(), 'checkpoints', Path(torchvision.models.Swin_T_Weights.DEFAULT.url).name)
print('Pretrained swin_t weights:', weights if weights.is_file() else
      'MISSING - run bash scripts/setup_conda.sh on the login node')
PY
echo "GPU check passed."
