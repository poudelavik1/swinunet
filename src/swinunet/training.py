"""Train a hairline-crack Swin-Unet for thin and faint crack segmentation.

Expected dataset layout::

    dataset/
      train/images, train/masks
      val/images,   val/masks
      test/images,  test/masks

This is the Swin-Unet counterpart of ``train_unet_hairline.py``. Data pairing,
augmentation, loss, metrics, checkpoints and figures are identical, so the two
models can be compared directly; only the network and its optimiser settings
differ.

The network (Cao et al., "Swin-Unet: Unet-like Pure Transformer for Medical
Image Segmentation", 2021):

* Encoder: an ImageNet-pretrained Swin Transformer (``--encoder``, default
  Swin-T). 4x4 patch embedding, then four stages of shifted-window attention
  joined by patch merging, giving token maps at 1/4, 1/8, 1/16 and 1/32 scale.
* Decoder: three Swin up-stages, each a patch-expanding layer (the inverse of
  patch merging), concatenation with the encoder skip, a linear fusion layer
  and two Swin blocks. As in the paper, the decoder blocks start from the
  pretrained weights of the encoder stage they mirror.
* Hairline head (default): Swin tokens start at 1/4 scale, which is too coarse
  for 1-px cracks, so the last two stages are convolutional. A full-resolution
  stem, fed with fixed morphological black-hat and top-hat responses (classic
  dark/light thin-line detectors; many EH104 cracks are lighter than the
  concrete), supplies skips at 1/1 and 1/2 scale to two bilinear scSE decoder
  blocks, the same ones ``train_unet_hairline.py`` ends with.
  ``--pure-swin-unet`` replaces this head with the paper's 4x patch-expanding
  layer, for the pure-transformer baseline.
* The Swin decoder stages at 1/8 and 1/4 scale are deeply supervised.
* Images of any size are accepted: window attention pads each token map to a
  multiple of the window, so validation and test still run on the full tiles.

Kept from ``train_unet_hairline.py``:

* Training uses native-resolution random crops instead of resizing every image
  to 256 px, so 1-3 px cracks are never averaged away by downsampling.
  Validation and test run on the full, unresized images.
* Loss = BCE + recall-leaning Tversky + clDice. clDice rewards unbroken crack
  centerlines, which is what the downstream skeleton/DXF pipeline consumes.
* Faint-crack augmentation: crack fading toward the local background,
  crack-only copy-paste, and synthetic hairline cracks.
* Contrast enhancement and reduction at all levels: every training image is
  cycled through 21 contrast levels, -10 % to +10 % in 1 % steps including the
  original, so each image is trained at each level (``--contrast-mode all``,
  ``--contrast-min-change``, ``--contrast-max-change``, ``--contrast-step``).
  ``contrast_coverage.json`` records how many images saw every level.
  ``--contrast-mode random`` applies a random 1-10 % change instead.
* Model selection and threshold calibration use F1 with a 2-pixel tolerance
  (the usual crack-detection protocol). Strict pixel precision, recall, F1,
  and IoU are reported alongside.

Target and evaluation (changed from ``train_unet_hairline.py``):

* The target is the whole dataset, ``DATASETS/split_80_10_10``: selection and
  calibration pool the validation images of every source, and the test
  metrics are reported pooled and per source. ``--select-pattern '^EH\\d'``
  restricts selection to the EH104 tiles as before.
* That dataset mixes 256-544 px tiles with 12-megapixel photos. Evaluation
  batches are capped in pixels (``--eval-batch-megapixels``), and an image with
  a side above ``--eval-tile`` (1024 px) is evaluated as overlapping tiles whose
  logits are blended; smaller images still run whole.
* EMA weights, flip TTA, per-source metrics, and resumable checkpoints.
* ``validation_by_source.csv`` records every validation source at every epoch.
  The pooled metrics are summed over pixels, so the thick-crack sources dominate
  them and can hide a source the model fails on (the run of 2026-10-06: pooled
  test tol-F1 0.83, RAMA8 East 0.19).
* ``--init-weights best.pt`` fine-tunes an earlier model on new data, and
  ``--label-tolerance N`` leaves an N-pixel ring around every label out of the
  loss, for labels drawn a few pixels off the crack.

Figures and tables are written to ``<output>/plots`` after training:
loss curves, training-vs-validation precision/recall/F1/IoU/accuracy (both at
threshold 0.5), validation metrics at each epoch's best threshold, the
learning-rate schedule, the validation threshold sweep, precision-recall and
ROC curves, and pixel confusion matrices for validation and test, with their
numbers in ``evaluation_curves.csv`` and ``confusion_matrices.json``.
``--evaluate-only`` produces the evaluation figures for an existing
``best.pt`` without training and leaves every other file untouched.
"""

from __future__ import annotations
from .common import BASIC_METRICS, COUNT_NAMES, Counter, DEFAULT_DATASET_PATH, DEFAULT_OUTPUT_PATH, DataLoader, METRIC_NAMES, Path, SWIN_ENCODERS, WeightedRandomSampler, argparse, csv, defaultdict, json, math, nn, np, os, random, re, seed_everything, seed_worker, sys, time, torch
from .data import AugmentationOptions, CrackSegmentationDataset, collect_pairs, shape_grouped_batches, source_family
from .models import HairlineSwinUNet, ModelEMA
from .losses import HairlineCrackLoss
from .evaluation import ThresholdSweep, at_threshold, basic_metrics, evaluate, metrics_from_counts, prediction_writer
from .plots import save_evaluation_tables, save_plots

def parameter_groups(model, learning_rate, encoder_factor, weight_decay):
    groups = defaultdict(list)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            # No weight decay on normalisation weights, biases, or Swin's relative position terms.
            no_decay = parameter.ndim <= 1 or any(
                key in name for key in ("relative_position_bias_table", "logit_scale", "cpb_mlp")
            )
            groups[(name.startswith("encoder."), no_decay)].append(parameter)
    return [
        {"params": parameters, "lr": learning_rate * (encoder_factor if is_encoder else 1.0),
         "weight_decay": 0.0 if no_decay else weight_decay}
        for (is_encoder, no_decay), parameters in groups.items()
    ]


def warmup_cosine(total_steps: int, warmup_steps: int, minimum_ratio: float):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return factor


def json_safe(value):
    return json.loads(json.dumps(value, default=str))


def save_checkpoint(state, path: Path) -> None:
    """Write to a temporary file, then rename it over `path`.

    A job killed while saving (time limit, scancel) then leaves the previous
    checkpoint intact instead of a truncated file that cannot be resumed.
    """
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def open_epoch_log(path: Path, fields, first_epoch: int):
    """Open a per-epoch CSV (epoch in the first column) for the rows from `first_epoch` on.

    A fresh run starts the file. A resumed run keeps the rows of the epochs before
    `first_epoch` and drops later ones: a job killed between writing an epoch's
    row and saving last.pt repeats that epoch, which would otherwise be logged twice.
    """
    rows = []
    if first_epoch > 1 and path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        rows = rows[:1] + [row for row in rows[1:] if row and row[0].isdigit() and int(row[0]) < first_epoch]
    handle = path.open("w", newline="", encoding="utf-8")
    csv.writer(handle).writerows(rows or [list(fields)])
    handle.flush()
    return handle


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET_PATH,
                        help=f"80/10/10 dataset root (default: {DEFAULT_DATASET_PATH})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--resume", action="store_true", help="Continue from <output>/last.pt")
    parser.add_argument("--init-weights", type=Path,
                        help="Start from the weights of an earlier run's best.pt instead of ImageNet, to "
                             "fine-tune on new data. The run itself is fresh (new schedule, EMA and "
                             "history); not used when --resume continues from last.pt")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="No training: evaluate <output>/best.pt on validation and test and write the "
                             "figures, curves and confusion matrices to <output>/plots (other files untouched)")
    # Model
    parser.add_argument("--encoder", choices=SWIN_ENCODERS, default="swin_t",
                        help="Swin backbone. swin_t is the paper's; the swin_v2_* windows (8x8) divide a "
                             "256 px crop exactly")
    parser.add_argument("--no-pretrained", action="store_true", help="Random encoder initialisation")
    parser.add_argument("--pure-swin-unet", action="store_true",
                        help="The paper's pure-transformer network: replace the full-resolution "
                             "convolutional head with a 4x patch-expanding layer (no thin-line prior)")
    parser.add_argument("--no-line-prior", action="store_true",
                        help="Disable the black-hat/top-hat thin-line input channels")
    parser.add_argument("--no-deep-supervision", action="store_true")
    # Optimisation
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256,
                        help="Training crop size at native resolution; also the inference tile size")
    # Transformer fine-tuning settings: a lower rate and more weight decay than the ResNet U-Net.
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--encoder-lr-factor", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999, help="0 disables EMA")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else min(4, os.cpu_count() or 1))
    # Loss
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--tversky-weight", type=float, default=1.0)
    parser.add_argument("--cldice-weight", type=float, default=0.5)
    parser.add_argument("--cldice-warmup-epochs", type=int, default=5,
                        help="Ramp clDice in linearly; it is noisy while predictions are still blobs")
    parser.add_argument("--false-positive-weight", type=float, default=0.4)
    parser.add_argument("--false-negative-weight", type=float, default=0.6,
                        help="> false-positive weight favours recall of faint cracks")
    parser.add_argument("--skeleton-iterations", type=int, default=10)
    parser.add_argument("--aux-weights", type=float, nargs=2, default=(0.4, 0.2))
    parser.add_argument("--label-tolerance", type=int, default=0,
                        help="Leave the ring of this many pixels around every label out of the loss: "
                             "hand-drawn labels sit a few pixels off the crack, and the tol_* metrics "
                             "already accept a prediction that close. 0 = strict loss (default)")
    # Evaluation
    parser.add_argument("--tolerance", type=int, default=2, help="Pixel tolerance for tol_* metrics")
    parser.add_argument("--select-metric", choices=METRIC_NAMES, default="tol_f1",
                        help="Metric for checkpoint selection and threshold calibration")
    parser.add_argument("--select-pattern", default="",
                        help="Regex on validation file names defining the target domain (model selection "
                             r"and threshold), e.g. '^EH\d' for the EH104 tiles only. Default: empty = "
                             "all validation data, every source in the dataset")
    parser.add_argument("--no-tta", action="store_true", help="Disable flip TTA for calibration/test")
    parser.add_argument("--eval-batch-megapixels", type=float, default=2.0,
                        help="An evaluation batch holds at most this many megapixels "
                             "(and at most --eval-batch-size images)")
    parser.add_argument("--eval-tile", type=int, default=1024,
                        help="Validation/test images with a side above this are evaluated as overlapping "
                             "tiles of this size, because full-resolution photos do not fit in memory "
                             "whole. Smaller images still run whole; 0 disables tiling")
    parser.add_argument("--eval-tile-overlap", type=int, default=128, help="Overlap of those tiles in pixels")
    parser.add_argument("--save-predictions", type=int, default=40,
                        help="Test overlays to save, spread evenly over the sources; a number at least "
                             "the size of the test set saves every image")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Debug: subsample train")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="Debug: subsample val/test")
    # Augmentation
    defaults = AugmentationOptions()
    parser.add_argument("--target-weight", type=float, default=1.5,
                        help="Sampling weight of target-domain training images relative to others")
    parser.add_argument("--scale-p", type=float, default=defaults.scale_p)
    parser.add_argument("--focus-p", type=float, default=defaults.focus_p)
    parser.add_argument("--copy-paste-p", type=float, default=defaults.copy_paste_p)
    parser.add_argument("--synthetic-p", type=float, default=defaults.synthetic_p)
    parser.add_argument("--fade-p", type=float, default=defaults.fade_p)
    parser.add_argument("--affine-p", type=float, default=defaults.affine_p)
    parser.add_argument("--elastic-p", type=float, default=defaults.elastic_p)
    parser.add_argument("--cutout-p", type=float, default=defaults.cutout_p)
    parser.add_argument("--max-rotation", type=float, default=defaults.max_rotation)
    parser.add_argument("--synthetic-label-width", type=int, default=defaults.synthetic_label_width)
    parser.add_argument("--contrast-mode", choices=("all", "random"), default=defaults.contrast_mode,
                        help="all: every training image is cycled through every contrast level from "
                             "-max to +max in --contrast-step steps, including the original (default, "
                             "21 levels). random: a random change on a share --contrast-p of the images")
    parser.add_argument("--contrast-step", type=float, default=defaults.contrast_step,
                        help="Spacing of the contrast levels in 'all' mode (default 0.01 = 1 %%)")
    parser.add_argument("--contrast-p", type=float, default=defaults.contrast_p,
                        help="'random' mode only: probability of a contrast change per training image")
    parser.add_argument("--contrast-min-change", type=float, default=defaults.contrast_min_change,
                        help="Smallest contrast change, as a fraction (default 0.01 = 1 %%)")
    parser.add_argument("--contrast-max-change", type=float, default=defaults.contrast_max_change,
                        help="Largest contrast change, as a fraction (default 0.10 = 10 %%); "
                             "each change is an increase or a decrease with equal probability")
    # Jupyter injects ``-f <kernel.json>``. Because ``dataset`` is an optional
    # positional argument, remove the complete kernel pair before parsing.
    raw_arguments, filtered_arguments, index = sys.argv[1:], [], 0
    while index < len(raw_arguments):
        if raw_arguments[index] == "-f":
            index += 2
        else:
            filtered_arguments.append(raw_arguments[index])
            index += 1
    args = parser.parse_args(filtered_arguments)
    if args.image_size % 32:
        parser.error("--image-size must be divisible by 32 (4 px patches and three 2x patch mergings).")
    if args.eval_tile % 32 or not 0 <= args.eval_tile_overlap < max(1, args.eval_tile):
        parser.error("--eval-tile must be divisible by 32 and larger than --eval-tile-overlap.")
    if args.eval_batch_megapixels <= 0:
        parser.error("--eval-batch-megapixels must be greater than zero.")
    if not 0 <= args.contrast_min_change <= args.contrast_max_change < 1:
        parser.error("Contrast changes must satisfy 0 <= --contrast-min-change <= --contrast-max-change < 1.")
    if args.contrast_step <= 0:
        parser.error("--contrast-step must be greater than zero.")
    if args.label_tolerance < 0:
        parser.error("--label-tolerance must not be negative.")
    if args.init_weights is not None and not args.evaluate_only and not args.init_weights.is_file():
        parser.error(f"--init-weights file not found: {args.init_weights}")
    return args


def main():
    args = parse_arguments()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    args.output.mkdir(parents=True, exist_ok=True)
    last_path, best_path = args.output / "last.pt", args.output / "best.pt"
    if args.evaluate_only:
        if not best_path.is_file():
            raise SystemExit(f"--evaluate-only needs a trained model, but {best_path} does not exist.")
    else:
        (args.output / "configuration.json").write_text(json.dumps(json_safe(vars(args)), indent=2))

    splits = {name: collect_pairs(args.dataset / name) for name in ("train", "val", "test")}
    subsample = random.Random(args.seed)
    if args.max_train_samples:
        splits["train"] = subsample.sample(splits["train"], min(args.max_train_samples, len(splits["train"])))
    if args.max_eval_samples:
        for name in ("val", "test"):
            splits[name] = subsample.sample(splits[name], min(args.max_eval_samples, len(splits[name])))
    families = {name: [source_family(image) for image, _ in pairs] for name, pairs in splits.items()}
    domains = sorted(set().union(*families.values()))
    domain_index = {domain: index for index, domain in enumerate(domains)}

    pattern = re.compile(args.select_pattern) if args.select_pattern else None
    target_domains = sorted({
        family for family, (image, _) in zip(families["val"], splits["val"])
        if pattern is None or pattern.search(image.name)
    })
    if not target_domains:
        print(f"No validation file matches --select-pattern {args.select_pattern!r}; using all validation data.")
        target_domains = sorted(set(families["val"]))
    target_ids = [domain_index[domain] for domain in target_domains]

    counts = {name: Counter(values) for name, values in families.items()}
    print(f"{'Source family':30s} {'train':>6s} {'val':>6s} {'test':>6s}")
    for domain in domains:
        marker = "  <- target" if domain in target_domains else ""
        print(f"{domain:30s} {counts['train'][domain]:6d} {counts['val'][domain]:6d} "
              f"{counts['test'][domain]:6d}{marker}")

    options = AugmentationOptions(
        args.scale_p, args.focus_p, args.copy_paste_p, args.synthetic_p, args.fade_p,
        args.affine_p, args.elastic_p, args.cutout_p, args.max_rotation, args.synthetic_label_width,
        args.contrast_p, args.contrast_min_change, args.contrast_max_change,
        args.contrast_mode, args.contrast_step,
    )
    datasets = {
        name: CrackSegmentationDataset(
            splits[name], [domain_index[family] for family in families[name]],
            training=name == "train", crop_size=args.image_size, options=options,
        )
        for name in ("train", "val", "test")
    }
    pin = device.type == "cuda"
    loader_options = dict(num_workers=args.workers, pin_memory=pin, worker_init_fn=seed_worker,
                          persistent_workers=args.workers > 0)
    generator = torch.Generator().manual_seed(args.seed)
    sample_weights = [args.target_weight if family in target_domains else 1.0 for family in families["train"]]
    if args.target_weight != 1.0 and len(set(sample_weights)) > 1:
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)
        train_loader = DataLoader(datasets["train"], args.batch_size, sampler=sampler, drop_last=True, **loader_options)
    else:
        train_loader = DataLoader(datasets["train"], args.batch_size, shuffle=True, drop_last=True,
                                  generator=generator, **loader_options)
    # Evaluation memory is bounded twice: a batch holds at most --eval-batch-megapixels,
    # and an image with a side above --eval-tile is run as overlapping tiles.
    batch_pixels = args.eval_batch_megapixels * 1e6
    tiling = dict(tile=args.eval_tile, overlap=args.eval_tile_overlap,
                  tile_batch=max(1, int(batch_pixels // max(1, args.eval_tile) ** 2)))
    val_loader, test_loader = (
        DataLoader(datasets[name], batch_sampler=shape_grouped_batches(
            datasets[name].shapes, args.eval_batch_size, max_pixels=batch_pixels), **loader_options)
        for name in ("val", "test")
    )
    # The threshold is calibrated on the target-domain validation images only,
    # so the final passes do not need the other validation sources.
    target_set = set(target_ids)
    calibration_loader = DataLoader(datasets["val"], batch_sampler=shape_grouped_batches(
        datasets["val"].shapes, args.eval_batch_size,
        [index for index, domain in enumerate(datasets["val"].domain_ids) if domain in target_set],
        max_pixels=batch_pixels,
    ), **loader_options)
    if len(train_loader) == 0:
        raise SystemExit("Training set is smaller than one batch; lower --batch-size.")

    architecture = {"name": "hairline_swin_unet", "kwargs": {
        "encoder_name": args.encoder, "decoder_depth": 2, "hairline_head": not args.pure_swin_unet,
        "head_channels": [48, 32], "stem_channels": 32,
        "line_prior": not (args.no_line_prior or args.pure_swin_unet),
        "deep_supervision": not args.no_deep_supervision,
    }}
    if args.evaluate_only:  # rebuild exactly what was trained; its weights come from best.pt
        architecture = torch.load(best_path, map_location="cpu", weights_only=False)["architecture"]
    initial = None
    if args.init_weights is not None and not args.evaluate_only and not (args.resume and last_path.is_file()):
        initial = torch.load(args.init_weights, map_location="cpu", weights_only=False)
        if initial["architecture"] != architecture:
            raise SystemExit(f"--init-weights holds a different network ({initial['architecture']['kwargs']}); "
                             f"this run builds {architecture['kwargs']}. Pass the matching model options.")
    model = HairlineSwinUNet(pretrained=not (args.no_pretrained or args.evaluate_only or initial is not None),
                             **architecture["kwargs"])
    if initial is not None:
        model.load_state_dict(initial["model"])
        print(f"Initial weights: {args.init_weights} (epoch {initial.get('epoch')}, "
              f"{initial.get('select_metric')} {initial.get('best_val_score', float('nan')):.4f})")
        del initial
    model = model.to(device, memory_format=torch.channels_last)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    optimizer = torch.optim.AdamW(
        parameter_groups(model, args.learning_rate, args.encoder_lr_factor, args.weight_decay)
    )
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(
        args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch), args.min_lr_ratio
    ))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_function = HairlineCrackLoss(
        args.bce_weight, args.tversky_weight, args.cldice_weight,
        args.false_positive_weight, args.false_negative_weight,
        args.skeleton_iterations, args.aux_weights, args.label_tolerance,
    )
    thresholds = np.round(np.arange(0.05, 0.951, 0.025), 3)
    sweep = ThresholdSweep(thresholds, args.tolerance, len(domains), device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    print(f"Device: {device} | AMP: {use_amp} | parameters: {parameter_count:.1f} M | "
          f"train={len(datasets['train'])} val={len(datasets['val'])} test={len(datasets['test'])}")
    train_set = datasets["train"]
    if train_set.contrast_factors is not None:
        factors = train_set.contrast_factors
        print(f"Contrast: all {len(factors)} levels from {100 * (factors[0] - 1):+.0f} % to "
              f"{100 * (factors[-1] - 1):+.0f} % in {100 * args.contrast_step:g} % steps; every training image "
              f"cycles through all of them (needs at least {len(factors)} epochs per image).")
    else:
        print(f"Contrast: random change of {100 * args.contrast_min_change:g}-{100 * args.contrast_max_change:g} % "
              f"on {100 * args.contrast_p:g} % of the training images.")

    start_epoch, best_score, best_epoch, stale = 1, -1.0, 0, 0
    if args.resume and not args.evaluate_only and last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        if ema is not None and state.get("ema") is not None:
            ema.module.load_state_dict(state["ema"])
            ema.updates = state["ema_updates"]
        start_epoch = state["epoch"] + 1
        best_score, best_epoch, stale = state["best_score"], state["best_epoch"], state["stale"]
        draws = state.get("contrast_draws")
        if (draws is not None and train_set.contrast_factors is not None
                and draws[0].shape == train_set.image_draws.shape and draws[1].shape == train_set.level_draws.shape):
            train_set.image_draws.copy_(draws[0])  # continue each image's cycle where it stopped
            train_set.level_draws.copy_(draws[1])
        print(f"Resumed from epoch {state['epoch']} (best {args.select_metric} {best_score:.4f}).")

    # History columns: train_*/val_* use the fixed threshold 0.5 so the two curves are
    # comparable; the unprefixed metrics are validation at that epoch's best threshold.
    fields = ["epoch", "lr", "train_loss", "val_loss",
              *(f"train_{name}" for name in BASIC_METRICS), *(f"val_{name}" for name in BASIC_METRICS),
              "threshold", *METRIC_NAMES, "all_val_f1", "all_val_tol_f1", "seconds"]
    history_path = args.output / "training_history.csv"
    # One row per validation source and epoch, at that epoch's selected threshold; best_* is
    # the source's own optimum. The pooled history hides a source the model fails on.
    source_path = args.output / "validation_by_source.csv"
    source_fields = ["epoch", "source", "images", "target", "threshold", *METRIC_NAMES,
                     "best_threshold", f"best_{args.select_metric}"]
    val_sources = sorted(set(datasets["val"].domain_ids))
    labelled_column = COUNT_NAMES.index("labelled")
    half_index = int(np.argmin(np.abs(thresholds - 0.5)))
    if not args.evaluate_only:
        with open_epoch_log(history_path, fields, start_epoch) as handle, \
                open_epoch_log(source_path, source_fields, start_epoch) as source_handle:
            writer, source_writer = csv.DictWriter(handle, fieldnames=fields), csv.writer(source_handle)
            for epoch in range(start_epoch, args.epochs + 1):
                started = time.time()
                model.train()
                cldice_scale = min(1.0, epoch / max(1, args.cldice_warmup_epochs))
                running_loss, seen = 0.0, 0
                train_counts = torch.zeros(4, dtype=torch.float64, device=device)  # tp, fp, fn, pixels
                for images, masks, valid, _, _ in train_loader:
                    images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                    masks, valid = masks.to(device, non_blocking=True), valid.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        outputs = model(images)
                    loss = loss_function(outputs, masks, valid, cldice_scale)
                    if not torch.isfinite(loss):
                        print("Warning: non-finite loss; batch skipped.")
                        continue
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    if ema is not None:
                        ema.update(model)
                    running_loss += loss.item() * images.size(0)
                    seen += images.size(0)
                    with torch.no_grad():  # training metrics at threshold 0.5 (logit 0)
                        logits = outputs[0] if isinstance(outputs, tuple) else outputs
                        supervised = valid > 0.5
                        prediction, truth = (logits.detach() >= 0) & supervised, (masks >= 0.5) & supervised
                        train_counts += torch.stack((
                            (prediction & truth).sum(), (prediction & ~truth).sum(),
                            (~prediction & truth).sum(), supervised.sum(),
                        )).double()

                evaluation_model = ema.module if ema is not None else model
                val_loss = evaluate(evaluation_model, val_loader, device, sweep, use_amp, loss_function, **tiling)
                target_metrics, all_metrics = sweep.metrics(target_ids), sweep.metrics()
                best_index = int(np.argmax(target_metrics[args.select_metric]))
                score = float(target_metrics[args.select_metric][best_index])
                train_metrics = basic_metrics(*train_counts.tolist())
                row = {
                    "epoch": epoch, "lr": max(group["lr"] for group in optimizer.param_groups),
                    "train_loss": running_loss / max(1, seen), "val_loss": val_loss,
                    **{f"train_{name}": train_metrics[name] for name in BASIC_METRICS},
                    **{f"val_{name}": float(target_metrics[name][half_index]) for name in BASIC_METRICS},
                    "threshold": float(thresholds[best_index]), **at_threshold(target_metrics, best_index),
                    "all_val_f1": float(all_metrics["f1"].max()),
                    "all_val_tol_f1": float(all_metrics["tol_f1"].max()),
                    "seconds": time.time() - started,
                }
                writer.writerow(row)
                handle.flush()
                source_scores = []  # (score at the selected threshold, name) of target sources with labels
                for domain_id in val_sources:
                    name, source_counts = domains[domain_id], sweep.count_array([domain_id])
                    source_metrics = metrics_from_counts(source_counts)
                    own_index = int(np.argmax(source_metrics[args.select_metric]))
                    source_writer.writerow([
                        epoch, name, counts["val"][name], int(name in target_domains),
                        f"{thresholds[best_index]:.3f}",
                        *(f"{source_metrics[metric][best_index]:.5f}" for metric in METRIC_NAMES),
                        f"{thresholds[own_index]:.3f}", f"{source_metrics[args.select_metric][own_index]:.5f}",
                    ])
                    if name in target_domains and source_counts[0, labelled_column] > 0:
                        source_scores.append((float(source_metrics[args.select_metric][best_index]), name))
                source_handle.flush()
                improved = score > best_score
                print(
                    f"Epoch {epoch:03d}/{args.epochs} | loss {row['train_loss']:.4f}/{val_loss:.4f} | "
                    f"thr {row['threshold']:.3f} P {row['precision']:.4f} R {row['recall']:.4f} "
                    f"F1 {row['f1']:.4f} IoU {row['iou']:.4f} | tolF1 {row['tol_f1']:.4f} | "
                    f"{row['seconds']:.0f}s{'  *best*' if improved else ''}"
                )
                if len(source_scores) > 1:
                    print(f"    weakest sources ({args.select_metric}): "
                          + " | ".join(f"{name} {value:.3f}" for value, name in sorted(source_scores)[:3]))
                if improved:
                    best_score, best_epoch, stale = score, epoch, 0
                    save_checkpoint({
                        "model": evaluation_model.state_dict(), "architecture": architecture,
                        "arguments": json_safe(vars(args)), "epoch": epoch,
                        "best_val_score": best_score, "select_metric": args.select_metric,
                        "optimal_threshold": float(thresholds[best_index]),
                    }, best_path)
                else:
                    stale += 1
                save_checkpoint({
                    "model": model.state_dict(), "ema": ema.module.state_dict() if ema is not None else None,
                    "ema_updates": ema.updates if ema is not None else 0,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_score": best_score,
                    "best_epoch": best_epoch, "stale": stale, "architecture": architecture,
                    "arguments": json_safe(vars(args)),
                    "contrast_draws": None if train_set.contrast_factors is None else
                    [train_set.image_draws.clone(), train_set.level_draws.clone()],
                }, last_path)
                if stale >= args.patience:
                    print(f"Early stopping: no {args.select_metric} improvement for {args.patience} epochs.")
                    break
        coverage = train_set.contrast_coverage()
        if coverage is not None:
            (args.output / "contrast_coverage.json").write_text(json.dumps(coverage, indent=2))
            print(f"Contrast coverage: {coverage['images_trained_at_every_level']} of "
                  f"{coverage['training_images']} training images were trained at all {coverage['levels']} "
                  f"levels (draws per image: minimum {coverage['minimum_draws_per_image']}, "
                  f"mean {coverage['mean_draws_per_image']:.1f}).")

    # ---------------- Calibration on validation, then a single test pass ---------------- #
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"\nBest epoch {checkpoint['epoch']} ({args.select_metric} {checkpoint['best_val_score']:.4f}).")
    use_tta = not args.no_tta
    calibration, evaluation = {}, {}
    for tta in sorted({False, use_tta}):
        evaluate(model, calibration_loader, device, sweep, use_amp, tta=tta, **tiling)
        validation_counts = sweep.count_array(target_ids)
        metrics = metrics_from_counts(validation_counts)
        index = int(np.argmax(metrics[args.select_metric]))
        stored = checkpoint.get("optimal_threshold_tta" if tta else "optimal_threshold")
        if args.evaluate_only and stored is not None:
            # Report at the threshold the training run calibrated.
            # Recalibrating on another machine can move it by a grid step, because the
            # validation curve is flat near its maximum and CPU/GPU precision differs.
            index = int(np.argmin(np.abs(thresholds - stored)))
            print(f"Threshold stored in best.pt ({'flip TTA' if tta else 'no TTA'}): {thresholds[index]:.3f}")
        else:
            print(f"Validation-calibrated threshold ({'flip TTA' if tta else 'no TTA'}): {thresholds[index]:.3f}")
        calibration[tta] = (index, metrics)
        evaluation["tta" if tta else "no_tta"] = {"threshold_index": index, "val": validation_counts}
    if not args.evaluate_only:
        with (args.output / "val_threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
            sweep_writer = csv.writer(handle)
            sweep_writer.writerow(["tta", "threshold", *METRIC_NAMES])
            for tta, (_, metrics) in calibration.items():
                for index, threshold in enumerate(thresholds):
                    sweep_writer.writerow(
                        [tta, f"{threshold:.3f}", *(f"{metrics[name][index]:.5f}" for name in METRIC_NAMES)]
                    )
        checkpoint["optimal_threshold"] = float(thresholds[calibration[False][0]])
        checkpoint["optimal_threshold_tta"] = float(thresholds[calibration[use_tta][0]])
        checkpoint["threshold_metrics"] = {
            str(tta): {name: calibration[tta][1][name].tolist() for name in METRIC_NAMES} for tta in calibration
        }
        checkpoint["thresholds"] = thresholds.tolist()
        save_checkpoint(checkpoint, best_path)

    results = {
        "best_epoch": checkpoint["epoch"], "select_metric": args.select_metric,
        "best_validation_score": checkpoint["best_val_score"], "tolerance_px": args.tolerance,
        "target_sources": target_domains,
    }
    for tta in sorted(calibration):
        index, mode = calibration[tta][0], "tta" if tta else "no_tta"
        hook = None
        if tta == use_tta and args.save_predictions > 0 and not args.evaluate_only:
            hook = prediction_writer(datasets["test"], args.output / "test_predictions",
                                     args.save_predictions, float(thresholds[index]))
        test_loss = evaluate(model, test_loader, device, sweep, use_amp, loss_function, tta=tta, hook=hook, **tiling)
        evaluation[mode]["test"] = sweep.count_array()
        per_source = {domains[d]: at_threshold(sweep.metrics([d]), index)
                      for d in sorted(set(datasets["test"].domain_ids))}
        results[mode] = {
            "threshold": float(thresholds[index]), "loss": test_loss,
            **at_threshold(sweep.metrics(), index), "per_source": per_source,
        }
    if not args.evaluate_only:  # an evaluation-only run never overwrites the training run's files
        (args.output / "test_metrics.json").write_text(json.dumps(results, indent=2))
    print("Test:", json.dumps(results, indent=2))

    # ---------------- Tables and figures ---------------- #
    plot_folder = args.output / "plots"
    save_evaluation_tables(plot_folder, thresholds, evaluation)
    try:
        written = save_plots(args.output, thresholds, evaluation, checkpoint["epoch"], args.tolerance)
        print(f"Figures in {plot_folder}: {', '.join(written)}")
    except ImportError:
        print("matplotlib is not installed, so no figures were drawn (python -m pip install matplotlib). "
              f"The curve and confusion-matrix tables are in {plot_folder}.")



__all__ = ['parameter_groups', 'warmup_cosine', 'json_safe', 'save_checkpoint', 'open_epoch_log',
           'parse_arguments', 'main']
