#!/bin/bash
#SBATCH --job-name=swin_unet
#SBATCH --partition=ASL-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${DATASET:?Submit using scripts/submit.sh with the dataset path}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set by the submission script}"
# As in pgat-length: the login node's /opt/miniconda3 is not on the compute node,
# so put the environment on PATH instead of activating it.
ENV_DIR="${ENV_DIR:-$HOME/.conda/envs/${ENV_NAME:-swinunet}}"
test -x "$ENV_DIR/bin/python" || { echo "Conda environment not found: $ENV_DIR" >&2; exit 1; }
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR"
nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable: check the GPU driver and PyTorch wheel compatibility'
print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
# Verify actual CUDA execution, not just device discovery.
print('CUDA tensor check:', (torch.ones(1, device='cuda') + 1).item())
PY
git rev-parse HEAD > "$OUTPUT_DIR/git_commit.txt"
python -m pip freeze > "$OUTPUT_DIR/packages.txt"
args=("$DATASET" --output "$OUTPUT_DIR")
[[ -z "${EPOCHS:-}" ]] || args+=(--epochs "$EPOCHS")
[[ -z "${BATCH_SIZE:-}" ]] || args+=(--batch-size "$BATCH_SIZE")
[[ -z "${WORKERS:-}" ]] || args+=(--workers "$WORKERS")
if [[ "${RESUME:-0}" == 1 ]]; then
    test -f "$OUTPUT_DIR/last.pt" || { echo "Resume checkpoint missing" >&2; exit 1; }
    args+=(--resume)
fi
# Run in the batch step, as in the ASL GPU example. Since Slurm 22.05 an srun
# step ignores --cpus-per-task and would leave the data workers a single core.
python -u scripts/train.py "${args[@]}"
