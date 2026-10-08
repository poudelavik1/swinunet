# Hairline crack segmentation architecture

Organization follows the separation of configs, src, scripts, slurm, and docs
in https://github.com/Samir4456/pgat-length. This project implements concrete
crack segmentation; its model, augmentations, losses, metrics, and training
logic come from the original Swin U-Net code in this repository.

| Module | Responsibility |
| --- | --- |
| `src/swinunet/common.py` | Constants, default paths, random seeds |
| `src/swinunet/data.py` | Image/mask pairing, native-resolution crops, augmentation, batching |
| `src/swinunet/models.py` | Swin encoder/decoder, line priors, EMA, checkpoint reconstruction |
| `src/swinunet/losses.py` | BCE, Tversky, clDice |
| `src/swinunet/evaluation.py` | Threshold sweeps, tiled inference, tolerant metrics, predictions |
| `src/swinunet/plots.py` | Epoch curves and final evaluation figures/tables |
| `src/swinunet/training.py` | Argument parsing, optimizer/scheduler, training/validation, resume |
| `scripts/train.py` | TOML defaults and training CLI |
| `slurm/train_swin.sbatch` | ASL GPU allocation and execution |

All original functions and classes were extracted without changing their AST.
The root `train_swin_unet_hairline.py` remains a compatibility entry point and
exports the Swin checkpoint loader. Existing training commands still work.
New runs use the TOML configuration through `scripts/train.py`.
Checkpoints retain the existing state-dict and architecture formats.

The dataset stays outside the repository and is supplied by absolute path.
Each fresh job writes into its own `runs/` directory. Resume uses the same
directory with `RESUME=1`. Do not run concurrent jobs into one output folder.

## Inference and AutoCAD

`inference/hairline_crack_detection.py` loads a Swin checkpoint through
`swinunet.models.load_model_from_checkpoint` and thins the mask with
`inference/zhang_suen_thinning.py`. It reads `inference/best.pt` unless
`--model` is given, and `inference/cracks_to_autocad.py` runs it before drawing
the crack polylines in AutoCAD. Its tile size, GPU batch sizing and line filters
were tuned on the earlier ResNet U-Net and have not been re-tuned for the Swin
model; the line filters are off by default. The threshold stored in `best.pt`
was calibrated on all sources pooled, which is too high for faint cracks on the
row/column tiles, so the detector uses 0.5 unless `--threshold` is given.

Repository restructuring does not address the server's `InvalidAccount`
scheduler error. CUDA execution must be verified on an allocated compute node.
