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

## Existing inference limitation

The separate legacy `inference/hairline_crack_detection.py` imports
`train_unet_hairline` and `zhang_suen_thinning`, which are not supplied here.
It describes a different U-Net inference pipeline and is not wired to the
Swin model. It is retained as existing work; it is not a verified Swin inference
entry point. Swin training's own evaluation and predictions use the packaged
evaluation code.

Repository restructuring does not address the server's `InvalidAccount`
scheduler error. CUDA execution must be verified on an allocated compute node.
