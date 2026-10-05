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
source "${CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}"
conda activate "${ENV_NAME:-swinunet}"
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
args=("$DATASET" --output "$OUTPUT_DIR" --epochs "${EPOCHS:-80}"
      --batch-size "${BATCH_SIZE:-4}" --workers "${WORKERS:-4}")
if [[ "${RESUME:-0}" == 1 ]]; then
    test -f "$OUTPUT_DIR/last.pt" || { echo "Resume checkpoint missing" >&2; exit 1; }
    args+=(--resume)
fi
srun python -u train_swin_unet_hairline.py "${args[@]}"
