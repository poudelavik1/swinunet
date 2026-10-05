# Swin U-Net hairline crack segmentation

Training for thin and faint concrete cracks with native-resolution crops,
Swin encoder/decoder, BCE + Tversky + clDice, and tolerant crack metrics.

```text
configs/             Training defaults
src/swinunet/        Data, model, loss, evaluation, plotting, training modules
scripts/             Conda setup, training CLI, submission, epoch monitoring
slurm/               ASL GPU check and training jobs
inference/           Existing legacy CAD/inference work
tests/               Workflow checks
docs/               Architecture and server guide
```

- [ASL setup, submission, monitoring, and resume](docs/ASL_SETUP.md)
- [Architecture and existing inference limitations](docs/ARCHITECTURE.md)
- [Training defaults](configs/training.toml)

## Existing server checkout

```bash
cd ~/swinunet
git pull --ff-only
bash scripts/setup_conda.sh
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
sbatch slurm/gpu_check.sbatch   # optional ten-minute GPU test
bash scripts/submit.sh "$HOME/split_80_10_10"
```

Cancel or finish old jobs before updating their code checkout. Submission prints
commands for following its log and epoch metrics. Dataset and results are kept
out of Git. A pending `InvalidAccount` error requires scheduler investigation;
changing repository layout does not resolve it.

The layout draws on [pgat-length](https://github.com/Samir4456/pgat-length).
This project's Swin U-Net model and training algorithms are preserved.
The original `train_swin_unet_hairline.py` command remains supported.

Run workflow tests with `python -m unittest discover -s tests -v`.
