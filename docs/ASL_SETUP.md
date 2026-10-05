# Swin U-Net hairline crack training on ASL

## Project layout

```text
configs/training.toml       Core training defaults
src/swinunet/              Data, models, losses, evaluation, plots, training
scripts/                  Conda setup, submission, training CLI, epoch monitor
slurm/train_swin.sbatch    ASL CUDA training job
slurm/gpu_check.sbatch     Ten-minute GPU and environment check
docs/ARCHITECTURE.md       Module responsibilities and inference limitations
tests/                    Configuration and live CSV monitoring checks
inference/                Existing legacy inference/CAD code
train_swin_unet_hairline.py Compatibility training entry point
```

This organization takes inspiration from https://github.com/Samir4456/pgat-length.
The Swin U-Net training algorithms are preserved. See
[architecture](ARCHITECTURE.md) for details.

The Conda environment holds Python dependencies. Clone the repository into a
normal working directory, then activate the environment to run its code.
The dataset already on the server is passed by its absolute path and is not uploaded to GitHub.

## Upload to GitHub from Windows

Repository: https://github.com/poudelavik1/swinunet
For future local updates, run from this project directory:

```powershell
git add .gitignore .gitattributes README.md requirements.txt environment.yml pyproject.toml configs src scripts slurm docs tests train_swin_unet_hairline.py inference
git commit -m "Update training code"
git push origin main
```

Do not commit the dataset, ZIP, checkpoints, or credentials. A private repository
requires GitHub authentication when cloning (SSH key or credential helper).

## Server setup (once)

Connect using `ssh -p 44065 se-st126157@asl.ait.ac.th`.
Use AIT VPN when off campus, as documented by ASL.

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/poudelavik1/swinunet.git swinunet
cd swinunet
bash scripts/setup_conda.sh
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
```

The setup installs Python 3.11 and matched PyTorch 2.9.1 / torchvision 0.24.1
CUDA 12.6 wheels from the official PyTorch index. The GPU driver must support
these wheels; the job tests a CUDA tensor operation before training. Override
`TORCH_INDEX_URL` during setup if another CUDA build is required by the driver.
Set `CONDA_SH` or `ENV_NAME` if using a different Conda installation or name.
Installation requires internet access. Setup also downloads the default
pretrained Swin-T encoder into `~/.cache/torch`, so compute nodes need no
internet. Another `--encoder` needs its own weights cached from the login node.

## Check the GPU (once)

Run from the repository root, then read the result once the job has left the queue:

```bash
sbatch slurm/gpu_check.sbatch
squeue -u "$USER"
cat logs/gpu_check_JOBID.out logs/gpu_check_JOBID.err
```

The output shows the GPU, driver, and PyTorch CUDA build and ends with
`GPU check passed.` If CUDA is unavailable, rerun setup with a `TORCH_INDEX_URL`
matching the driver version reported by `nvidia-smi`.

## Submit CUDA training

Your uploaded dataset was extracted into `$HOME/split_80_10_10`. It must contain
`train/images`, `train/masks`, `val/images`, `val/masks`, `test/images`, and `test/masks`.

```bash
bash scripts/submit.sh "$HOME/split_80_10_10"
```

Core defaults live in `configs/training.toml`. CLI flags take precedence.
Direct training on an allocated compute node:

```bash
python scripts/train.py "$HOME/split_80_10_10" --output runs/manual --epochs 80
```

Default request: ASL-gpu, one GPU, four CPU cores, 32 GB host RAM, two days.
`sinfo -o "%P %l %m %c %G"` lists each partition's time limit, memory, cores,
and GPUs; request a shorter job with `SBATCH_TIMELIMIT=1-00:00:00` before the command.
Submission prints the job ID, result directory, and monitoring commands.
Each new job uses a separate output directory and does not resume old weights.
To customize training:

```bash
EPOCHS=100 BATCH_SIZE=8 bash scripts/submit.sh /absolute/path/to/split_80_10_10
```

An explicitly assigned account or other authorized partition can be supplied
without editing the job file:

```bash
SLURM_ACCOUNT=YOUR_ASSIGNED_ACCOUNT bash scripts/submit.sh "$HOME/split_80_10_10"
SLURM_PARTITION=YOUR_AUTHORIZED_PARTITION bash scripts/submit.sh "$HOME/split_80_10_10"
```

Only use account/partition values confirmed by the administrator. Restructuring
cannot fix a controller-side `InvalidAccount` error. Pending jobs produce no
epoch metrics until they start. On this server `sacct` may be unavailable because
accounting storage is not configured; use `scontrol show job JOBID` instead.

Reduce batch size if GPU memory is exhausted. Evaluation also uses tiled images;
if evaluation exhausts memory, reduce `--eval-tile` in the job script.
Early stopping defaults to 15 epochs without improvement, so training can stop
before the requested epoch count.

## Monitor performance

Use the actual job ID and output path printed by the submission script:

```bash
squeue -u "$USER"
tail -F logs/swin_unet_JOBID.out
python scripts/monitor_epochs.py /absolute/path/to/run --watch
cat logs/swin_unet_JOBID.err
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Ctrl+C stops the viewer; the submitted job keeps running after SSH disconnects.
The monitor prints a row only after training AND validation finish an epoch.
Train/validation loss and validation precision, recall, F1, IoU (fixed threshold
0.5), plus tolerant F1 at the selected threshold, are read from the flushed CSV.
The live monitor waits until interrupted; use `sacct` to check completion/failure.
All epoch values remain in `training_history.csv`. The trainer creates final
plots in `plots/` and saves `best.pt` and resumable `last.pt`. Jobs record the
Git commit and package versions in the output folder.

## Resume and update

Resume into the same run directory using its checkpoint:

```bash
RESUME=1 bash scripts/submit.sh /absolute/path/to/split_80_10_10 /absolute/path/to/existing/run
```

Never submit two jobs writing into the same output directory.
Cancel a job using `scancel JOBID`. Update the repo only after active jobs finish:

```bash
git pull --ff-only
```

Run setup again if dependencies changed. On Windows, commit and push your code
changes first. Retrieve results using PowerShell:

```powershell
scp -P 44065 -r se-st126157@asl.ait.ac.th:/absolute/path/to/run .
```

ASL instructions: https://asl.ait.ac.th/package-slurm.html
Slurm examples: https://asl.ait.ac.th/package-script.html
PyTorch wheels: https://pytorch.org/get-started/previous-versions/
