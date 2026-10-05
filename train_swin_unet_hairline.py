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

import argparse
import copy
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# The target dataset: every source in it counts (EH104 tiles are no longer singled out).
DATASET_NAME = "split_80_10_10"
LOCAL_DATASET_PATH = Path(r"E:\STRUCTURAL ENGINEERING\thesis\crackdetection\DATASETS") / DATASET_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
# torchvision Swin Transformers. T and S embed 96 channels, B 128; V1 uses 7x7
# windows (the paper's backbone), V2 8x8 windows, which divide a 256 px crop exactly.
SWIN_ENCODERS = ("swin_t", "swin_s", "swin_b", "swin_v2_t", "swin_v2_s", "swin_v2_b")
METRIC_NAMES = ("precision", "recall", "f1", "iou", "tol_precision", "tol_recall", "tol_f1", "accuracy")
# Pixel counts accumulated per threshold; "near_*" are the 2-px tolerant hits.
COUNT_NAMES = ("tp", "fp", "fn", "near_label", "predicted", "near_prediction", "labelled", "pixels")
BASIC_METRICS = ("precision", "recall", "f1", "iou", "accuracy")  # logged for train and validation at 0.5


def find_dataset_root(base: Path = KAGGLE_INPUT_ROOT) -> Path:
    """Find the dataset among the Kaggle inputs.

    The folder named DATASET_NAME is preferred; otherwise the first input with
    the expected paired split layout is used.
    """
    if base.is_dir():
        candidates = [
            train_directory.parent for train_directory in sorted(base.rglob("train"))
            if (train_directory / "images").is_dir() and (train_directory / "masks").is_dir()
        ]
        named = [candidate for candidate in candidates if candidate.name == DATASET_NAME]
        if candidates:
            return (named or candidates)[0]
    return LOCAL_DATASET_PATH


DEFAULT_DATASET_PATH = find_dataset_root()
DEFAULT_OUTPUT_PATH = (
    KAGGLE_WORKING_ROOT / "swin_unet_hairline_output"
    if KAGGLE_WORKING_ROOT.is_dir() else Path("swin_unet_hairline_output")
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    # NumPy's global RNG is not reseeded per DataLoader worker by PyTorch.
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    cv2.setNumThreads(0)


# --------------------------------------------------------------------------- #
# Dataset pairing (same rules as train_unet_hairline.py)
# --------------------------------------------------------------------------- #
def normalized_stem(path: Path) -> str:
    stem = path.stem.lower().strip()
    # Remove repeated/common role suffixes while preserving the tile identity.
    previous = None
    while stem != previous:
        previous = stem
        stem = re.sub(r"(?:[_\- ](?:mask|label|image|img))$", "", stem)
    return stem


def tile_coordinate_key(path: Path) -> str | None:
    """Extract a stable row/column key when image and mask prefixes differ."""
    match = re.search(r"(?:^|[_\-])r(?:ow)?[_\-]?(\d+)[_\-]c(?:ol)?[_\-]?(\d+)", path.stem, re.IGNORECASE)
    if match:
        return f"r{int(match.group(1)):06d}_c{int(match.group(2)):06d}"
    return None


def unique_map(paths, key_function, description):
    result = {}
    duplicates = []
    for path in paths:
        key = key_function(path)
        if key is None:
            continue
        if key in result:
            duplicates.append((key, result[key].name, path.name))
        else:
            result[key] = path
    if duplicates:
        examples = "; ".join(f"{key}: {first}, {second}" for key, first, second in duplicates[:3])
        raise ValueError(f"Duplicate {description} tile identifiers: {examples}")
    return result


def collect_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    image_dir, mask_dir = split_dir / "images", split_dir / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise ValueError(f"Missing images/masks folders in {split_dir}")
    image_paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    mask_paths = sorted(p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    images = unique_map(image_paths, normalized_stem, "image")
    masks = unique_map(mask_paths, normalized_stem, "mask")
    if set(images) == set(masks):
        return [(images[key], masks[key]) for key in sorted(images)]

    # DXF tile datasets sometimes use different source prefixes for the image
    # and mask. Pair by the invariant r####_c#### coordinate in that case.
    coordinate_images = unique_map(image_paths, tile_coordinate_key, "image coordinate")
    coordinate_masks = unique_map(mask_paths, tile_coordinate_key, "mask coordinate")
    if coordinate_images and set(coordinate_images) == set(coordinate_masks):
        print(
            f"Pairing {len(coordinate_images)} tiles in {split_dir.name} by row/column "
            "because image and mask filename prefixes differ."
        )
        return [
            (coordinate_images[key], coordinate_masks[key])
            for key in sorted(coordinate_images)
        ]

    missing_mask_keys = sorted(set(images) - set(masks))
    missing_image_keys = sorted(set(masks) - set(images))
    image_examples = [images[key].name for key in missing_mask_keys[:5]]
    mask_examples = [masks[key].name for key in missing_image_keys[:5]]
    raise ValueError(
        f"Unpaired files in {split_dir}: {len(missing_mask_keys)} missing masks and "
        f"{len(missing_image_keys)} missing images. "
        f"Example unmatched images: {image_examples}. Example unmatched masks: {mask_examples}."
    )


def source_family(path: Path) -> str:
    """Group files by naming pattern, e.g. EH104_r0003_c0012 -> EH#_r#_c#."""
    return re.sub(r"\d+", "#", normalized_stem(path)) or "#"


def image_shape(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        return height, width
    except Exception:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape[:2]


def read_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return an RGB uint8 image and a {0, 1} uint8 mask of identical size."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or mask is None:
        raise ValueError(f"Could not read pair: {image_path}, {mask_path}")
    if image.shape[:2] != mask.shape:
        # Phone photos may carry an EXIF rotation that the mask did not get.
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image.shape[:2] != mask.shape:
        print(f"Warning: resizing {image_path.name} {image.shape[:2]} to its mask {mask.shape}.")
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    # JPEG masks (e.g. CRACK500) contain compression noise; 0/1 masks also occur.
    threshold = 1 if mask.max() <= 1 else 128
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (mask >= threshold).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Augmentation. `valid` marks supervised pixels; padding, warped-in borders and
# cutout regions are excluded from the loss instead of being labelled background.
# --------------------------------------------------------------------------- #
@dataclass
class AugmentationOptions:
    scale_p: float = 0.30
    focus_p: float = 0.60
    copy_paste_p: float = 0.20
    synthetic_p: float = 0.08
    fade_p: float = 0.25
    affine_p: float = 0.30
    elastic_p: float = 0.10
    cutout_p: float = 0.10
    max_rotation: float = 20.0
    synthetic_label_width: int = 4
    # Contrast enhancement / reduction of 1-10 %.
    #   "all":    every training image is cycled through every level, from the
    #             largest reduction to the largest enhancement in contrast_step
    #             steps: -10 %, -9 %, ..., 0 (original), ..., +10 % = 21 levels.
    #   "random": with probability contrast_p, a random change in that range,
    #             raised or lowered 50/50.
    contrast_p: float = 0.60
    contrast_min_change: float = 0.01
    contrast_max_change: float = 0.10
    contrast_mode: str = "all"
    contrast_step: float = 0.01


def rescale(image, mask, valid, factor: float):
    height, width = mask.shape
    size = (max(32, round(width * factor)), max(32, round(height * factor)))
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    image = cv2.resize(image, size, interpolation=interpolation)
    # Resample labels as coverage, then use a low cut so thin lines stay connected.
    mask = (cv2.resize(mask.astype(np.float32), size, interpolation=interpolation) >= 0.4).astype(np.uint8)
    valid = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST)
    return image, mask, valid


def pad_to_at_least(image, mask, valid, size: int, random_placement: bool = True):
    height, width = mask.shape
    pad_h, pad_w = max(0, size - height), max(0, size - width)
    if not pad_h and not pad_w:
        return image, mask, valid
    top = random.randint(0, pad_h) if random_placement else 0
    left = random.randint(0, pad_w) if random_placement else 0
    border = (top, pad_h - top, left, pad_w - left)
    image = cv2.copyMakeBorder(image, *border, cv2.BORDER_REFLECT_101)
    mask = cv2.copyMakeBorder(mask, *border, cv2.BORDER_CONSTANT, value=0)
    valid = cv2.copyMakeBorder(valid, *border, cv2.BORDER_CONSTANT, value=0)
    return image, mask, valid


def random_crop(image, mask, valid, size: int, focus_p: float):
    image, mask, valid = pad_to_at_least(image, mask, valid, size)
    height, width = mask.shape
    if random.random() < focus_p and mask.any():
        # Centre near a random crack pixel so large images still yield crack crops.
        ys, xs = np.nonzero(mask)
        pick = random.randrange(len(ys))
        centre_y = ys[pick] + random.randint(-size // 4, size // 4)
        centre_x = xs[pick] + random.randint(-size // 4, size // 4)
        y0 = int(np.clip(centre_y - size // 2, 0, height - size))
        x0 = int(np.clip(centre_x - size // 2, 0, width - size))
    else:
        y0, x0 = random.randint(0, height - size), random.randint(0, width - size)
    window = (slice(y0, y0 + size), slice(x0, x0 + size))
    return image[window], mask[window], valid[window]


def dihedral(image, mask, valid):
    if random.random() < 0.5:
        image, mask, valid = image[:, ::-1], mask[:, ::-1], valid[:, ::-1]
    if random.random() < 0.5:
        image, mask, valid = image[::-1], mask[::-1], valid[::-1]
    turns = random.randint(0, 3)
    if turns:
        image, mask, valid = np.rot90(image, turns), np.rot90(mask, turns), np.rot90(valid, turns)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask), np.ascontiguousarray(valid)


def random_affine(image, mask, valid, max_rotation: float):
    height, width = mask.shape
    matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), random.uniform(-max_rotation, max_rotation), random.uniform(0.9, 1.1)
    )
    matrix[:, 2] += (random.uniform(-0.05, 0.05) * width, random.uniform(-0.05, 0.05) * height)
    image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    # Linear warp + low cut avoids the staircase gaps nearest-neighbour leaves in 1-px lines.
    mask = cv2.warpAffine(mask.astype(np.float32), matrix, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpAffine(valid, matrix, (width, height), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, (mask >= 0.4).astype(np.uint8), valid


def elastic(image, mask, valid):
    """Low-magnitude elastic warp that bends cracks without breaking them."""
    height, width = mask.shape
    alpha, sigma = min(height, width) * 0.012, min(height, width) * 0.035
    dx = cv2.GaussianBlur(np.random.uniform(-1, 1, (height, width)).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(np.random.uniform(-1, 1, (height, width)).astype(np.float32), (0, 0), sigma) * alpha
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x, map_y = grid_x + dx, grid_y + dy
    image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.remap(mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.remap(valid, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, (mask >= 0.4).astype(np.uint8), valid


def local_background(image: np.ndarray, kernel: int = 9) -> np.ndarray:
    """Median filtering removes line structures narrower than about kernel/2."""
    return cv2.medianBlur(image, kernel)


def fade_cracks(image, mask, probability: float):
    """Blend labelled crack pixels toward the local background to mimic faint cracks."""
    if random.random() >= probability or not mask.any():
        return image
    background = local_background(image)
    region = cv2.dilate(mask, np.ones((3, 3), np.uint8)).astype(np.float32)
    alpha = cv2.GaussianBlur(region, (5, 5), 0)[..., None] * random.uniform(0.30, 0.65)
    return (image * (1 - alpha) + background * alpha).astype(np.uint8)


def add_synthetic_hairline(image, mask, probability: float, label_width: int):
    """Draw a faint meandering 1-2 px line, dark or light, labelled like the DXF masks."""
    if random.random() >= probability:
        return image, mask
    height, width = mask.shape
    x, y = random.uniform(0, width), random.uniform(0, height)
    angle = random.uniform(0, 2 * math.pi)
    points = [(x, y)]
    for _ in range(random.randint(8, 25)):
        angle += random.gauss(0, 0.35)
        step = random.uniform(6, 14)
        x, y = x + step * math.cos(angle), y + step * math.sin(angle)
        if not (0 <= x < width and 0 <= y < height):
            break
        points.append((x, y))
    if len(points) < 3:
        return image, mask
    polyline = [np.round(np.asarray(points)).astype(np.int32)]
    line = np.zeros(mask.shape, np.float32)
    cv2.polylines(line, polyline, False, 1.0, random.choice((1, 1, 2)), cv2.LINE_AA)
    line = (line * random.uniform(0.25, 0.55))[..., None]
    image = image.astype(np.float32)
    # EH104 cracks are often lighter than the concrete, so both polarities occur.
    image = image * (1 - line) if random.random() < 0.5 else image + (255 - image) * line
    image = image.astype(np.uint8)
    mask = mask.copy()
    cv2.polylines(mask, polyline, False, 1, label_width, cv2.LINE_8)
    return image, mask


def cutout(image, valid):
    height, width = valid.shape
    erase_h, erase_w = random.randint(8, max(9, height // 8)), random.randint(8, max(9, width // 8))
    y, x = random.randint(0, height - erase_h), random.randint(0, width - erase_w)
    image = image.copy()
    valid = valid.copy()
    image[y:y + erase_h, x:x + erase_w] = image.mean(axis=(0, 1)).astype(np.uint8)
    valid[y:y + erase_h, x:x + erase_w] = 0
    return image, valid


def motion_blur(image: np.ndarray) -> np.ndarray:
    size = random.choice((3, 5))
    kernel = np.zeros((size, size), np.float32)
    if random.random() < 0.5:
        kernel[size // 2, :] = 1 / size
    else:
        kernel[:, size // 2] = 1 / size
    return cv2.filter2D(image, -1, kernel)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale the deviation of every pixel from the per-channel mean: factor 1.10 is
    a 10 % contrast enhancement, 0.90 a 10 % reduction. Brightness is unchanged."""
    mean = image.mean(axis=(0, 1), keepdims=True)
    return (image - mean) * factor + mean


def random_contrast_factor(minimum_change: float, maximum_change: float) -> float:
    """1 +/- a random change in [minimum_change, maximum_change], sign chosen 50/50."""
    change = random.uniform(minimum_change, maximum_change)
    return 1.0 + change if random.random() < 0.5 else 1.0 - change


def contrast_level_factors(options: AugmentationOptions) -> list[float]:
    """Every contrast level of the "all" mode as a factor, lowest first.

    With the defaults: 0.90, 0.91, ..., 0.99, 1.00 (original), 1.01, ..., 1.10.
    """
    steps = int(round((options.contrast_max_change - options.contrast_min_change) / options.contrast_step)) + 1
    changes = [round(options.contrast_min_change + step * options.contrast_step, 6) for step in range(steps)]
    return [round(1 - change, 6) for change in reversed(changes)] + [1.0] + [round(1 + change, 6) for change in changes]


def photometric_augment(image: np.ndarray, options: AugmentationOptions | None = None,
                        contrast_factor: float | None = None) -> np.ndarray:
    """`contrast_factor` is the level chosen by the "all" mode; None means "random" mode."""
    options = options or AugmentationOptions()
    image = image.astype(np.float32)
    # Enhancement sharpens cracks; reduction compresses the crack/background
    # difference, which trains the model on fainter cracks.
    if contrast_factor is not None:
        if contrast_factor != 1.0:
            image = adjust_contrast(image, contrast_factor)
    elif random.random() < options.contrast_p:
        image = adjust_contrast(
            image, random_contrast_factor(options.contrast_min_change, options.contrast_max_change)
        )
    if random.random() < 0.5:
        image = image + random.uniform(-25, 25)  # brightness shift
    if random.random() < 0.3:
        image = 255.0 * np.clip(image / 255.0, 0, 1) ** random.uniform(0.7, 1.5)
    if random.random() < 0.2:
        # Smooth illumination gradient (shadow / uneven lighting on the surface).
        height, width = image.shape[:2]
        theta = random.uniform(0, 2 * math.pi)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        ramp = (xx * math.cos(theta) + yy * math.sin(theta))
        ramp = (ramp - ramp.min()) / max(1e-6, float(ramp.max() - ramp.min()))
        image *= (random.uniform(0.6, 0.9) + (1 - random.uniform(0.6, 0.9)) * ramp)[..., None]
    image = np.clip(image, 0, 255).astype(np.uint8)
    if random.random() < 0.15:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(random.uniform(1.5, 3.0), (8, 8)).apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if random.random() < 0.3:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-6, 6)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if random.random() < 0.10:
        image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    # Blur is kept mild: a 5x5 Gaussian can erase a 1-px crack while its label remains.
    blur_choice = random.random()
    if blur_choice < 0.08:
        image = cv2.GaussianBlur(image, (3, 3), random.uniform(0.4, 0.9))
    elif blur_choice < 0.12:
        image = motion_blur(image)
    if random.random() < 0.25:
        noise = np.random.normal(0, random.uniform(2.0, 8.0), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.25:
        quality = random.randint(40, 90)
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            image = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return image


class CrackSegmentationDataset(Dataset):
    def __init__(
        self, pairs, domain_ids, training: bool = False, crop_size: int = 256,
        options: AugmentationOptions | None = None,
    ):
        if not pairs:
            raise ValueError("Empty dataset split.")
        self.pairs, self.domain_ids = pairs, domain_ids
        self.training, self.crop_size = training, crop_size
        self.options = options or AugmentationOptions()
        # Copy-paste donors come from the same source, so label width conventions
        # (4 px DXF lines vs 18 px CRACK500 regions) are never mixed in one image.
        self.members = defaultdict(list)
        for index, domain in enumerate(domain_ids):
            self.members[domain].append(index)
        # Evaluation batches group equal image sizes, so read (h, w) headers once.
        self.shapes = None if training else [image_shape(mask) for _, mask in pairs]
        # "all" contrast mode: each image steps through every level as it is drawn.
        self.contrast_factors = None
        if training and self.options.contrast_mode == "all":
            self.contrast_factors = contrast_level_factors(self.options)
            levels, generator = len(self.contrast_factors), np.random.default_rng(2026)
            # One shuffled level order for all images, entered at a different point by
            # each image, so every epoch and batch holds a balanced mix of levels.
            self.level_order = generator.permutation(levels).tolist()
            self.level_offsets = (generator.permutation(len(pairs)) % levels).tolist()
            # Draw counters in shared memory, so DataLoader workers update the same arrays.
            self.image_draws = torch.zeros(len(pairs), dtype=torch.int64).share_memory_()
            self.level_draws = torch.zeros(levels, dtype=torch.int64).share_memory_()

    def __len__(self):
        return len(self.pairs)

    def next_contrast_factor(self, index: int) -> float | None:
        """Contrast factor for this draw of image `index`; None in "random" mode."""
        if self.contrast_factors is None:
            return None
        position = (self.level_offsets[index] + int(self.image_draws[index])) % len(self.contrast_factors)
        level = self.level_order[position]
        self.image_draws[index] += 1
        self.level_draws[level] += 1
        return self.contrast_factors[level]

    def contrast_coverage(self) -> dict | None:
        """How completely training has covered the contrast levels so far."""
        if self.contrast_factors is None:
            return None
        draws, levels = self.image_draws.clone(), len(self.contrast_factors)
        return {
            "mode": "all", "levels": levels,
            "contrast_change_percent": [round(100 * (factor - 1), 3) for factor in self.contrast_factors],
            "draws_per_level": self.level_draws.tolist(),
            "training_images": len(self.pairs),
            "images_trained_at_every_level": int((draws >= levels).sum()),
            "minimum_draws_per_image": int(draws.min()), "mean_draws_per_image": float(draws.float().mean()),
        }

    def paste_donor_crack(self, image, mask, domain):
        """Transfer another sample's crack *contrast* (dark or light), not its texture patch."""
        donor_image, donor_mask = read_pair(*self.pairs[random.choice(self.members[domain])])
        donor_image, donor_mask, donor_valid = random_crop(
            donor_image, donor_mask, np.ones_like(donor_mask), self.crop_size, focus_p=1.0
        )
        donor_mask = donor_mask * donor_valid
        if not donor_mask.any():
            return image, mask
        difference = donor_image.astype(np.float32) - local_background(donor_image, 15).astype(np.float32)
        if abs(difference[donor_mask > 0].mean()) < 4.0:
            return image, mask  # no visible contrast: pasting it would teach invisible labels
        region = cv2.dilate(donor_mask, np.ones((3, 3), np.uint8)).astype(np.float32)
        alpha = cv2.GaussianBlur(region, (5, 5), 0)[..., None]
        blended = image.astype(np.float32) + alpha * difference
        return np.clip(blended, 0, 255).astype(np.uint8), np.maximum(mask, donor_mask)

    def __getitem__(self, index):
        image, mask = read_pair(*self.pairs[index])
        valid = np.ones_like(mask)
        if self.training:
            options = self.options
            if random.random() < options.scale_p:
                image, mask, valid = rescale(image, mask, valid, random.uniform(0.75, 1.25))
            image, mask, valid = random_crop(image, mask, valid, self.crop_size, options.focus_p)
            if random.random() < options.copy_paste_p:
                image, mask = self.paste_donor_crack(image, mask, self.domain_ids[index])
            image, mask = add_synthetic_hairline(image, mask, options.synthetic_p, options.synthetic_label_width)
            image = fade_cracks(image, mask, options.fade_p)
            image, mask, valid = dihedral(image, mask, valid)
            if random.random() < options.affine_p:
                image, mask, valid = random_affine(image, mask, valid, options.max_rotation)
            if random.random() < options.elastic_p:
                image, mask, valid = elastic(image, mask, valid)
            image = photometric_augment(image, options, self.next_contrast_factor(index))
            if random.random() < options.cutout_p:
                image, valid = cutout(image, valid)
        else:
            # Full image at native resolution, padded bottom/right to a multiple of 32.
            height, width = mask.shape
            pad_h, pad_w = (-height) % 32, (-width) % 32
            if pad_h or pad_w:
                image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                valid = cv2.copyMakeBorder(valid, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)[None])
        valid_tensor = torch.from_numpy((valid > 0).astype(np.float32)[None])
        return image_tensor, mask_tensor * valid_tensor, valid_tensor, self.domain_ids[index], index


def shape_grouped_batches(shapes, batch_size: int, indices=None, max_pixels: float = math.inf) -> list[list[int]]:
    """Batches of equally sized images; `indices` restricts them to a subset of the dataset.

    A batch holds at most `batch_size` images and `max_pixels` pixels, so a
    full-resolution photo is evaluated on its own.
    """
    groups = defaultdict(list)
    for index in (range(len(shapes)) if indices is None else indices):
        height, width = shapes[index]
        groups[(height + (-height) % 32, width + (-width) % 32)].append(index)
    batches = []
    for (height, width), members in sorted(groups.items()):
        size = int(max(1, min(batch_size, max_pixels // (height * width))))
        batches.extend(members[start:start + size] for start in range(0, len(members), size))
    return batches


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ConvBNReLU(nn.Sequential):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, input_channels, output_channels):
        super().__init__(ConvBNReLU(input_channels, output_channels), ConvBNReLU(output_channels, output_channels))


class SCSE(nn.Module):
    """Concurrent spatial and channel squeeze-and-excitation."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.channel_gate(x) + x * self.spatial_gate(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.conv = DoubleConv(input_channels + skip_channels, output_channels)
        self.attention = SCSE(output_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.attention(self.conv(torch.cat((x, skip), 1)))


class ThinLineResponse(nn.Module):
    """Fixed morphological black-hat and white top-hat on grayscale input.

    Black-hat (closing - image) responds to dark structures narrower than the
    kernel; top-hat (image - opening) to bright ones. Both are needed: in EH104
    many cracks appear lighter than the concrete (deposits in the crack), not
    darker. The module has no parameters, so it is a thin-line prior at no data cost.
    """

    def __init__(self, kernel_sizes=(7, 15)):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.output_channels = 2 * len(self.kernel_sizes)
        self.register_buffer("luma", torch.tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1))

    def forward(self, rgb):
        gray = (rgb * self.luma).sum(1, keepdim=True)
        responses = []
        for size in self.kernel_sizes:
            closed = -F.max_pool2d(-F.max_pool2d(gray, size, 1, size // 2), size, 1, size // 2)
            opened = F.max_pool2d(-F.max_pool2d(-gray, size, 1, size // 2), size, 1, size // 2)
            responses.extend((closed - gray, gray - opened))
        return torch.cat(responses, 1)


class PatchExpand(nn.Module):
    """Swin-Unet patch expanding, the inverse of patch merging.

    A linear layer widens the channels, then every token is rearranged into
    scale x scale tokens. Tokens are channels-last: (N, H, W, C).
    """

    def __init__(self, input_channels: int, output_channels: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.expand = nn.Linear(input_channels, output_channels * scale * scale, bias=False)
        self.norm = nn.LayerNorm(output_channels)

    def forward(self, x):
        batch, height, width, _ = x.shape
        scale = self.scale
        x = self.expand(x).reshape(batch, height, width, scale, scale, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch, height * scale, width * scale, -1)
        return self.norm(x)


class SwinDecoderStage(nn.Module):
    """Swin-Unet up-stage: patch expanding, skip concatenation, linear fusion, Swin blocks."""

    def __init__(self, input_channels: int, blocks: nn.Sequential):
        super().__init__()
        output_channels = input_channels // 2
        self.expand = PatchExpand(input_channels, output_channels)
        self.fuse = nn.Linear(2 * output_channels, output_channels)
        self.blocks = blocks

    def forward(self, x, skip):
        # Patch merging pads odd sizes, so the expanded map can be one token larger than the skip.
        x = self.expand(x)[:, :skip.shape[1], :skip.shape[2]]
        return self.blocks(self.fuse(torch.cat((x, skip), -1)))


class HairlineSwinUNet(nn.Module):
    """Swin-Unet (Cao et al., 2021) with a full-resolution head for 1-4 px cracks.

    Input: RGB in [0, 1]. Eval mode returns logits (N, 1, H, W), the same
    interface as HairlineUNet in train_unet_hairline.py. Train mode with deep
    supervision returns (logits, [auxiliary logits]).

    ``hairline_head=False`` gives the paper's pure-transformer network: the
    convolutional stem and its two decoder blocks are replaced by a 4x
    patch-expanding layer.
    """

    def __init__(
        self, encoder_name: str = "swin_t", pretrained: bool = True, decoder_depth: int = 2,
        hairline_head: bool = True, head_channels=(48, 32), stem_channels: int = 32,
        line_prior: bool = True, deep_supervision: bool = True, prior_probability: float = 0.03,
    ):
        super().__init__()
        try:
            import torchvision
        except ImportError as exc:
            raise RuntimeError("The encoder requires torchvision: python -m pip install torchvision") from exc
        try:
            self.encoder = torchvision.models.get_model(encoder_name, weights="DEFAULT" if pretrained else None)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    f"Could not load ImageNet {encoder_name} weights. In Kaggle, enable Internet "
                    "for the first run, or pass --no-pretrained."
                ) from exc
            raise
        self.encoder.head = nn.Identity()
        # encoder.features = [patch embedding, stage 1, merge, stage 2, merge, stage 3, merge, stage 4]
        stages = self.encoder.features
        width = stages[0][0].out_channels  # C; the stages carry C, 2C, 4C, 8C channels

        def mirror(stage):
            # As in the paper, a decoder stage starts from the weights of the encoder stage it mirrors.
            return nn.Sequential(*copy.deepcopy(list(stage)[:decoder_depth]))

        self.up3 = SwinDecoderStage(8 * width, mirror(stages[5]))   # 1/16, 4C
        self.up2 = SwinDecoderStage(4 * width, mirror(stages[3]))   # 1/8,  2C
        self.up1 = SwinDecoderStage(2 * width, mirror(stages[1]))   # 1/4,  C
        self.decoder_norm = nn.LayerNorm(width)
        self.hairline_head = hairline_head
        self.thin_lines = ThinLineResponse() if hairline_head and line_prior else None
        if hairline_head:
            d1, d0 = head_channels
            stem_inputs = 3 + (self.thin_lines.output_channels if self.thin_lines is not None else 0)
            self.full_resolution = DoubleConv(stem_inputs, stem_channels)                                        # 1/1
            self.half_resolution = nn.Sequential(nn.MaxPool2d(2), DoubleConv(stem_channels, 2 * stem_channels))  # 1/2
            self.dec1 = DecoderBlock(width, 2 * stem_channels, d1)   # 1/2
            self.dec0 = DecoderBlock(d1, stem_channels, d0)          # 1/1
            self.head = nn.Conv2d(d0, 1, 1)
        else:
            self.final_expand = PatchExpand(width, width, scale=4)   # 1/4 -> 1/1
            self.head = nn.Conv2d(width, 1, 1)
        self.deep_supervision = deep_supervision
        # Auxiliary heads on the Swin decoder tokens at 1/8 and 1/4 scale.
        self.auxiliary_heads = (
            nn.ModuleList([nn.Linear(2 * width, 1), nn.Linear(width, 1)]) if deep_supervision else None
        )
        # Start near the crack prior so early training is not dominated by background.
        bias = -math.log((1 - prior_probability) / prior_probability)
        for head in [self.head, *(self.auxiliary_heads or [])]:
            nn.init.constant_(head.bias, bias)
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, x):
        size = x.shape[-2:]
        normalized = (x - self.image_mean) / self.image_std
        stages = self.encoder.features
        e1 = stages[1](stages[0](normalized))                    # 1/4, tokens (N, H/4, W/4, C)
        e2 = stages[3](stages[2](e1))                            # 1/8
        e3 = stages[5](stages[4](e2))                            # 1/16
        e4 = self.encoder.norm(stages[7](stages[6](e3)))         # 1/32, bottleneck
        d3 = self.up3(e4, e3)                                    # 1/16
        d2 = self.up2(d3, e2)                                    # 1/8
        d1 = self.decoder_norm(self.up1(d2, e1))                 # 1/4
        if self.hairline_head:
            stem_input = normalized if self.thin_lines is None else torch.cat((normalized, self.thin_lines(x)), 1)
            full = self.full_resolution(stem_input)
            half = self.half_resolution(full)
            logits = self.head(self.dec0(self.dec1(d1.permute(0, 3, 1, 2), half), full))
        else:
            logits = self.head(self.final_expand(d1).permute(0, 3, 1, 2))
            if logits.shape[-2:] != size:  # input size not divisible by the patch size
                logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        if self.training and self.deep_supervision:
            auxiliary = [
                F.interpolate(head(tokens).permute(0, 3, 1, 2), size=size, mode="bilinear", align_corners=False)
                for head, tokens in zip(self.auxiliary_heads, (d2, d1))
            ]
            return logits, auxiliary
        return logits


def load_model_from_checkpoint(path, device="cpu"):
    """Rebuild a HairlineSwinUNet from best.pt for inference scripts."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = HairlineSwinUNet(pretrained=False, **checkpoint["architecture"]["kwargs"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


class ModelEMA:
    """Exponential moving average of weights (and BN statistics)."""

    def __init__(self, model: nn.Module, decay: float = 0.999, ramp_steps: int = 2000):
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay, self.ramp_steps, self.updates = decay, ramp_steps, 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / self.ramp_steps))
        source = model.state_dict()
        for key, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source[key].detach(), alpha=1 - decay)
            else:
                value.copy_(source[key])


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def soft_erode(image):
    return torch.minimum(
        -F.max_pool2d(-image, (3, 1), 1, (1, 0)), -F.max_pool2d(-image, (1, 3), 1, (0, 1))
    )


def soft_skeleton(image, iterations: int):
    """Differentiable skeleton (Shit et al., clDice, CVPR 2021)."""
    opened = F.max_pool2d(soft_erode(image), 3, 1, 1)
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = soft_erode(image)
        opened = F.max_pool2d(soft_erode(image), 3, 1, 1)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(probability, target, iterations: int, smooth: float = 1.0):
    predicted_skeleton = soft_skeleton(probability, iterations)
    target_skeleton = soft_skeleton(target, iterations)
    topology_precision = ((predicted_skeleton * target).sum() + smooth) / (predicted_skeleton.sum() + smooth)
    topology_sensitivity = ((target_skeleton * probability).sum() + smooth) / (target_skeleton.sum() + smooth)
    return 1 - 2 * topology_precision * topology_sensitivity / (topology_precision + topology_sensitivity)


class HairlineCrackLoss(nn.Module):
    """BCE + Tversky (region) + clDice (centerline continuity), valid-pixel masked."""

    def __init__(
        self, bce_weight=1.0, tversky_weight=1.0, cldice_weight=0.5,
        false_positive_weight=0.4, false_negative_weight=0.6,
        skeleton_iterations=10, auxiliary_weights=(0.4, 0.2),
    ):
        super().__init__()
        self.bce_weight, self.tversky_weight, self.cldice_weight = bce_weight, tversky_weight, cldice_weight
        self.fp_weight, self.fn_weight = false_positive_weight, false_negative_weight
        self.skeleton_iterations = skeleton_iterations
        self.auxiliary_weights = tuple(auxiliary_weights)

    def region_loss(self, logits, target, valid):
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
        probability = torch.sigmoid(logits) * valid
        target = target * valid
        true_positive = (probability * target).sum()
        false_positive = (probability * (1 - target)).sum()
        false_negative = ((1 - probability) * target).sum()
        tversky = (true_positive + 1.0) / (
            true_positive + self.fp_weight * false_positive + self.fn_weight * false_negative + 1.0
        )
        return self.bce_weight * bce + self.tversky_weight * (1 - tversky), probability, target

    def forward(self, outputs, target, valid, cldice_scale: float = 1.0):
        logits, auxiliary = outputs if isinstance(outputs, tuple) else (outputs, ())
        loss, probability, target_valid = self.region_loss(logits, target, valid)
        weight = self.cldice_weight * cldice_scale
        if weight > 0:
            loss = loss + weight * soft_cldice_loss(probability, target_valid, self.skeleton_iterations)
        for auxiliary_weight, auxiliary_logits in zip(self.auxiliary_weights, auxiliary):
            loss = loss + auxiliary_weight * self.region_loss(auxiliary_logits, target, valid)[0]
        return loss


# --------------------------------------------------------------------------- #
# Metrics: every threshold at once, strict and with an N-pixel tolerance
# --------------------------------------------------------------------------- #
class ThresholdSweep:
    """Accumulate per-source counts for all thresholds in one pass.

    Tolerant precision = predicted pixels within `tolerance` px of a label;
    tolerant recall = label pixels within `tolerance` px of a prediction.
    """

    def __init__(self, thresholds, tolerance: int, domain_count: int, device):
        self.thresholds = [float(value) for value in thresholds]
        self.threshold_tensor = torch.tensor(self.thresholds, device=device).view(-1, 1, 1)
        self.tolerance = tolerance
        self.counts = torch.zeros(domain_count, len(self.thresholds), len(COUNT_NAMES),
                                  dtype=torch.float64, device=device)
        self.pool_dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32

    def reset(self):
        self.counts.zero_()

    def dilate(self, binary):
        if self.tolerance <= 0:
            return binary
        size = 2 * self.tolerance + 1
        return F.max_pool2d(binary[:, None].to(self.pool_dtype), size, 1, self.tolerance)[:, 0] > 0

    @torch.no_grad()
    def update(self, probabilities, targets, valid, domain_ids):
        for probability, target, valid_map, domain in zip(probabilities, targets, valid, domain_ids.tolist()):
            valid_map = valid_map[0] > 0.5
            truth = (target[0] >= 0.5) & valid_map
            prediction = (probability[0].unsqueeze(0) >= self.threshold_tensor) & valid_map
            true_positive = (prediction & truth).sum((1, 2))
            predicted = prediction.sum((1, 2))
            labelled = truth.sum().expand_as(true_positive)
            predicted_near_label = (prediction & self.dilate(truth[None])).sum((1, 2))
            label_near_prediction = (self.dilate(prediction) & truth).sum((1, 2))
            self.counts[domain] += torch.stack((
                true_positive, predicted - true_positive, labelled - true_positive,
                predicted_near_label, predicted, label_near_prediction, labelled,
                valid_map.sum().expand_as(true_positive),
            ), 1).double()

    def count_array(self, domain_ids=None) -> np.ndarray:
        """Counts summed over the given sources: array (thresholds, len(COUNT_NAMES))."""
        counts = self.counts if domain_ids is None else self.counts[list(domain_ids)]
        return counts.sum(0).cpu().numpy()

    def metrics(self, domain_ids=None) -> dict[str, np.ndarray]:
        return metrics_from_counts(self.count_array(domain_ids))


def metrics_from_counts(counts: np.ndarray) -> dict[str, np.ndarray]:
    """Per-threshold metrics from an array (thresholds, len(COUNT_NAMES)) of pixel counts."""
    tp, fp, fn, near_label, predicted, near_prediction, labelled, pixels = counts.T
    epsilon = 1e-9
    tn = pixels - tp - fp - fn
    precision, recall = tp / (tp + fp + epsilon), tp / (tp + fn + epsilon)
    tol_precision, tol_recall = near_label / (predicted + epsilon), near_prediction / (labelled + epsilon)
    return {
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + epsilon),
        "iou": tp / (tp + fp + fn + epsilon),
        "tol_precision": tol_precision, "tol_recall": tol_recall,
        "tol_f1": 2 * tol_precision * tol_recall / (tol_precision + tol_recall + epsilon),
        "accuracy": (tp + tn) / (pixels + epsilon),
        "false_positive_rate": fp / (fp + tn + epsilon),
    }


def basic_metrics(tp: float, fp: float, fn: float, pixels: float) -> dict[str, float]:
    """BASIC_METRICS from pixel counts at one threshold."""
    epsilon = 1e-9
    precision, recall = tp / (tp + fp + epsilon), tp / (tp + fn + epsilon)
    return {
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + epsilon),
        "iou": tp / (tp + fp + fn + epsilon), "accuracy": (pixels - fp - fn) / (pixels + epsilon),
    }


def confusion_from_counts(counts_row: np.ndarray) -> dict[str, int]:
    """Pixel confusion matrix at one threshold: crack is the positive class."""
    tp, fp, fn = (int(round(float(value))) for value in counts_row[:3])
    return {"tp": tp, "fp": fp, "fn": fn, "tn": int(round(float(counts_row[7]))) - tp - fp - fn}


def at_threshold(metrics: dict[str, np.ndarray], index: int) -> dict[str, float]:
    return {name: float(metrics[name][index]) for name in METRIC_NAMES}


def tile_origins(length: int, tile: int, stride: int) -> list[int]:
    """Offsets of tiles of size `tile` covering `length`; the last one ends flush with the edge."""
    if length <= tile:
        return [0]
    return [*range(0, length - tile, stride), length - tile]


@torch.no_grad()
def predict_logits(model, images, use_amp: bool, tile: int = 0, overlap: int = 0, tile_batch: int = 1):
    """Float logits (N, 1, H, W) of the main output.

    Images with a side longer than `tile` are run as overlapping tiles, and the
    tile logits are blended with weights that fade toward each tile's border.
    That bounds memory on full-resolution photos, which do not fit whole, and
    matches tiled inference. `tile` 0 always runs the whole image.
    """
    def run(batch):
        with torch.autocast(device_type=batch.device.type, dtype=torch.float16, enabled=use_amp):
            return model(batch).float()

    height, width = images.shape[-2:]
    if tile <= 0 or max(height, width) <= tile:
        return run(images)
    tile_h, tile_w, stride = min(tile, height), min(tile, width), tile - overlap
    windows = [(y, x) for y in tile_origins(height, tile_h, stride) for x in tile_origins(width, tile_w, stride)]

    def ramp(length):  # 1 in the tile's interior, falling linearly to its border
        position = torch.arange(length, device=images.device, dtype=torch.float32)
        return (torch.minimum(position, length - 1 - position) + 1).clamp(max=max(1, overlap)) / max(1, overlap)

    weight = ramp(tile_h)[:, None] * ramp(tile_w)[None, :]
    logits = torch.zeros((images.size(0), 1, height, width), device=images.device)
    weights = torch.zeros((height, width), device=images.device)
    for start in range(0, len(windows), tile_batch):
        chunk = windows[start:start + tile_batch]
        outputs = run(torch.cat([images[..., y:y + tile_h, x:x + tile_w] for y, x in chunk]))
        for (y, x), output in zip(chunk, outputs.split(images.size(0))):
            logits[..., y:y + tile_h, x:x + tile_w] += output * weight
            weights[y:y + tile_h, x:x + tile_w] += weight
    return logits / weights


@torch.no_grad()
def predict_probabilities(model, images, use_amp: bool, tta: bool, **tiling):
    """`tiling` holds the tile, overlap and tile_batch arguments of predict_logits."""
    logits = predict_logits(model, images, use_amp, **tiling)
    probabilities = torch.sigmoid(logits)
    if tta:
        for dims in ((3,), (2,), (2, 3)):
            flipped = predict_logits(model, torch.flip(images, dims), use_amp, **tiling)
            probabilities += torch.sigmoid(torch.flip(flipped, dims))
        probabilities /= 4
    return logits, probabilities


@torch.no_grad()
def evaluate(model, loader, device, sweep: ThresholdSweep, use_amp: bool,
             loss_function=None, tta: bool = False, hook=None, **tiling) -> float:
    model.eval()
    sweep.reset()
    total_loss, samples = 0.0, 0
    for images, masks, valid, domain_ids, indices in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        masks, valid = masks.to(device, non_blocking=True), valid.to(device, non_blocking=True)
        logits, probabilities = predict_probabilities(model, images, use_amp, tta, **tiling)
        if loss_function is not None:
            total_loss += loss_function(logits, masks, valid).item() * images.size(0)
            samples += images.size(0)
        sweep.update(probabilities, masks, valid, domain_ids)
        if hook is not None:
            hook(images, probabilities, masks, indices)
    return total_loss / max(1, samples)


def prediction_writer(dataset, directory: Path, limit: int, threshold: float):
    """Save probability maps and overlays: yellow = hit, red = false positive, green = missed."""
    directory.mkdir(parents=True, exist_ok=True)
    written = [0]

    def write(images, probabilities, masks, indices):
        for image, probability, mask, index in zip(images, probabilities, masks, indices.tolist()):
            if written[0] >= limit:
                return
            height, width = dataset.shapes[index]
            rgb = (image[:, :height, :width].float().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            probability = probability[0, :height, :width].cpu().numpy()
            truth, prediction = mask[0, :height, :width].cpu().numpy() > 0.5, probability >= threshold
            overlay = (rgb * 0.55).astype(np.uint8)
            overlay[truth & ~prediction] = (0, 220, 0)
            overlay[prediction & ~truth] = (230, 0, 0)
            overlay[prediction & truth] = (255, 220, 0)
            stem = dataset.pairs[index][0].stem
            cv2.imwrite(str(directory / f"{stem}_probability.png"), (probability * 255).astype(np.uint8))
            cv2.imwrite(str(directory / f"{stem}_overlay.png"),
                        cv2.cvtColor(np.hstack((rgb, overlay)), cv2.COLOR_RGB2BGR))
            written[0] += 1

    return write


# --------------------------------------------------------------------------- #
# Tables and figures: loss/metric curves, threshold sweep, PR, ROC, confusion matrix
# --------------------------------------------------------------------------- #
MODE_LABELS = {"no_tta": "no TTA", "tta": "flip TTA"}
PLOT_INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}
# Fixed colour order (blue, orange, aqua): distinguishable with colour-vision deficiency.
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
BLUE_RAMP = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")  # one hue, light -> dark
PLOT_STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.edgecolor": PLOT_INK["axis"], "axes.labelcolor": PLOT_INK["secondary"],
    "axes.titlecolor": PLOT_INK["primary"], "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": PLOT_INK["grid"], "grid.linewidth": 0.8,
    "grid.linestyle": "-", "xtick.color": PLOT_INK["muted"], "ytick.color": PLOT_INK["muted"],
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "legend.frameon": False,
    "text.color": PLOT_INK["primary"],
}


def read_history(path: Path) -> dict[str, np.ndarray]:
    """training_history.csv as {column: values}; empty when the file is missing."""
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    history = {}
    for name in (rows[0] if rows else ()):
        try:
            history[name] = np.array([float(row[name]) for row in rows])
        except (TypeError, ValueError):
            continue
    return history


def save_evaluation_tables(folder: Path, thresholds, evaluation: dict) -> None:
    """The numbers behind the figures: evaluation_curves.csv and confusion_matrices.json."""
    folder.mkdir(parents=True, exist_ok=True)
    curve_names = (*METRIC_NAMES, "false_positive_rate")
    confusion = {}
    with (folder / "evaluation_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mode", "split", "threshold", *COUNT_NAMES, "tn", *curve_names))
        for mode, data in evaluation.items():
            index = data["threshold_index"]
            confusion[mode] = {"threshold": float(thresholds[index])}
            for split, name in (("val", "validation"), ("test", "test")):
                counts, metrics = data[split], metrics_from_counts(data[split])
                for row, threshold in enumerate(thresholds):
                    matrix = confusion_from_counts(counts[row])
                    writer.writerow((mode, name, f"{threshold:.3f}", *(int(round(value)) for value in counts[row]),
                                     matrix["tn"], *(f"{metrics[key][row]:.6f}" for key in curve_names)))
                matrix = confusion_from_counts(counts[index])
                confusion[mode][name] = {
                    **matrix, **{key: float(metrics[key][index]) for key in curve_names},
                    "specificity": matrix["tn"] / max(1, matrix["tn"] + matrix["fp"]),
                }
    (folder / "confusion_matrices.json").write_text(json.dumps(confusion, indent=2))


def save_plots(output: Path, thresholds, evaluation: dict, best_epoch: int, tolerance: int) -> list[str]:
    """Write every figure to <output>/plots and return the file names.

    `evaluation` maps "no_tta"/"tta" to {"threshold_index", "val", "test"} with
    count arrays from ThresholdSweep.count_array(). Epoch curves come from
    training_history.csv; figures needing columns an older history lacks are skipped.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator

    folder = output / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    ink, written = PLOT_INK, []
    blue, orange, aqua = SERIES_COLOURS
    thresholds = np.asarray(thresholds, dtype=float)

    def finish(figure, name, epoch_axis=False):
        if epoch_axis:  # epochs are whole numbers
            for axis in figure.axes:
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        figure.savefig(folder / name)
        plt.close(figure)
        written.append(name)

    def note(axis, text, offset=-0.22):
        axis.text(0, offset, text, transform=axis.transAxes, fontsize=8, color=ink["muted"], va="top")

    def mark(axis, x, text):
        axis.axvline(x, color=ink["muted"], linewidth=1.0, zorder=1)
        axis.annotate(text, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -3),
                      textcoords="offset points", va="top", fontsize=8.5, color=ink["secondary"])

    def lines(axis, x, series, end_values=True, pad=0.10):
        """series = [(label, values, colour)]. End values are written only where they do
        not collide; the legend always carries the identity."""
        for label, values, colour in series:
            axis.plot(x, values, color=colour, label=label)
        span = (x[-1] - x[0]) or 1.0
        axis.set_xlim(x[0] - 0.01 * span, x[-1] + (pad if end_values else 0.01) * span)
        if end_values:
            low, high = axis.get_ylim()
            placed = []
            for _, values, _ in sorted(series, key=lambda item: -item[1][-1]):
                if all(abs(values[-1] - other) > 0.06 * (high - low) for other in placed):
                    axis.annotate(f"{values[-1]:.3f}", (x[-1], values[-1]), xytext=(5, 0),
                                  textcoords="offset points", va="center", fontsize=8.5, color=ink["secondary"])
                    placed.append(values[-1])

    def legend_in_title_row(axis, columns):
        """Single panel: legend right of the left-aligned title, clear of the data."""
        axis.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=columns, borderaxespad=0.2,
                    handlelength=1.6, columnspacing=1.4)

    def legend_in_heading_row(axes, source_axis, columns):
        """Several panels: one legend at the right of the figure heading."""
        axes[-1].legend(*source_axis.get_legend_handles_labels(), loc="lower right", bbox_to_anchor=(1.0, 1.09),
                        ncol=columns, borderaxespad=0, handlelength=1.6, columnspacing=1.4)

    def dot(axis, x, y, colour):
        axis.plot(x, y, "o", color=colour, markersize=8, markeredgecolor="white", markeredgewidth=2, zorder=4)

    with plt.rc_context(PLOT_STYLE):
        history = read_history(output / "training_history.csv")
        if "epoch" in history and "train_loss" in history:
            epochs = history["epoch"]
            show_best = bool(best_epoch) and epochs[0] <= best_epoch <= epochs[-1]

            figure, axis = plt.subplots(figsize=(7, 4.2))
            lines(axis, epochs, [("Training", history["train_loss"], blue), ("Validation", history["val_loss"], orange)])
            if show_best:
                mark(axis, best_epoch, f"best epoch {best_epoch}")
            axis.set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
            legend_in_title_row(axis, 2)
            note(axis, "Training loss also contains the auxiliary deep-supervision terms and is measured on augmented "
                       "crops,\nso it sits above the validation loss (full target images, main output only).")
            finish(figure, "loss_curves.png", epoch_axis=True)

            if all(f"{split}_{name}" in history for split in ("train", "val") for name in BASIC_METRICS):
                titles = {"precision": "Precision", "recall": "Recall", "f1": "F1 score", "iou": "IoU",
                          "accuracy": "Pixel accuracy"}
                figure, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), sharex=True)
                for axis, name in zip(axes.flat, BASIC_METRICS):
                    lines(axis, epochs, [("Training", history[f"train_{name}"], blue),
                                         ("Validation", history[f"val_{name}"], orange)], pad=0.17)
                    if show_best:
                        mark(axis, best_epoch, "best")
                    axis.set_title(titles[name])
                for axis in (axes[1, 0], axes[1, 1], axes[0, 2]):
                    axis.set_xlabel("Epoch")
                axes[0, 2].tick_params(labelbottom=True)
                axes[1, 2].axis("off")
                axes[1, 2].legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left",
                                  title="Both at threshold 0.5")
                note(axes[1, 2], "Training: augmented crops, all sources.\nValidation: full target images, EMA weights.\n"
                                 "Pixel accuracy is dominated by background\nand stays near 1; judge cracks by F1 and IoU.",
                     offset=0.62)
                figure.suptitle("Training and validation metrics per epoch", x=0.07, ha="left",
                                fontsize=12, fontweight="semibold")
                finish(figure, "metric_curves_train_val.png", epoch_axis=True)

            if all(name in history for name in ("precision", "recall", "f1", "iou", "tol_f1")):
                figure, axes = plt.subplots(1, 3, figsize=(14, 4))
                for axis, prefix, title in ((axes[0], "", "Strict pixel metrics"),
                                            (axes[1], "tol_", f"{tolerance}-px tolerance metrics")):
                    lines(axis, epochs, [("Precision", history[prefix + "precision"], blue),
                                         ("Recall", history[prefix + "recall"], orange),
                                         ("F1", history[prefix + "f1"], aqua)], pad=0.14)
                    axis.set(xlabel="Epoch", title=title)
                # One series: the title names it, and a neutral ink keeps it from reading as "Precision".
                lines(axes[2], epochs, [("IoU", history["iou"], ink["secondary"])], pad=0.14)
                axes[2].set(xlabel="Epoch", title="IoU")
                legend_in_heading_row(axes, axes[0], 3)
                if show_best:
                    for axis in axes:
                        mark(axis, best_epoch, f"best epoch {best_epoch}")
                figure.suptitle("Validation metrics per epoch (target images, at each epoch's best threshold)",
                                x=0.07, ha="left", fontsize=12, fontweight="semibold")
                finish(figure, "validation_metric_curves.png", epoch_axis=True)

            if "lr" in history:
                figure, axis = plt.subplots(figsize=(7, 3.6))
                lines(axis, epochs, [("Learning rate", history["lr"], blue)], end_values=False)
                axis.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
                axis.set(xlabel="Epoch", ylabel="Decoder learning rate", title="Learning-rate schedule (warm-up, cosine decay)")
                finish(figure, "learning_rate.png", epoch_axis=True)

        coverage_path = output / "contrast_coverage.json"
        if coverage_path.is_file():  # written by training in the "all" contrast mode
            coverage = json.loads(coverage_path.read_text())
            changes, draws = coverage["contrast_change_percent"], coverage["draws_per_level"]
            spacing = min(np.diff(changes)) if len(changes) > 1 else 1.0
            figure, axis = plt.subplots(figsize=(7.5, 3.8))
            axis.bar(changes, draws, width=0.6 * spacing, color=blue)
            axis.grid(axis="x", visible=False)
            ticks = changes[::2] if len(changes) > 12 else changes
            axis.set_xticks(ticks, [f"{value:+g}" if value else "0" for value in ticks])
            axis.set(xlabel="Contrast change (%)", ylabel="Training samples",
                     title=f"Training samples at each of the {coverage['levels']} contrast levels")
            note(axis, f"{coverage['images_trained_at_every_level']:,} of {coverage['training_images']:,} training "
                       f"images were trained at every level (mean {coverage['mean_draws_per_image']:.1f} draws per "
                       "image).", offset=-0.24)
            finish(figure, "contrast_level_coverage.png")

        colour_map = LinearSegmentedColormap.from_list("crack_blue", BLUE_RAMP)
        for mode, data in evaluation.items():
            label, index = MODE_LABELS.get(mode, mode), data["threshold_index"]
            selected = thresholds[index]
            splits = (("Validation", metrics_from_counts(data["val"]), orange),
                      ("Test", metrics_from_counts(data["test"]), aqua))
            validation = splits[0][1]

            figure, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
            for axis, prefix, title in ((axes[0], "", "Strict pixel metrics"),
                                        (axes[1], "tol_", f"{tolerance}-px tolerance metrics")):
                lines(axis, thresholds, [("Precision", validation[prefix + "precision"], blue),
                                         ("Recall", validation[prefix + "recall"], orange),
                                         ("F1", validation[prefix + "f1"], aqua)], end_values=False)
                mark(axis, selected, f"selected {selected:.3f}")
                axis.set(xlabel="Probability threshold", title=title, ylim=(0, 1))
            axes[0].set_ylabel("Score")
            legend_in_heading_row(axes, axes[0], 3)
            figure.suptitle(f"Validation threshold sweep ({label})", x=0.07, ha="left", fontsize=12,
                            fontweight="semibold")
            finish(figure, f"threshold_sweep_{mode}.png")

            figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
            for axis, prefix, title in ((axes[0], "", "Strict pixel metrics"),
                                        (axes[1], "tol_", f"{tolerance}-px tolerance metrics")):
                for name, metrics, colour in splits:
                    axis.plot(metrics[prefix + "recall"], metrics[prefix + "precision"], color=colour, label=name)
                    dot(axis, metrics[prefix + "recall"][index], metrics[prefix + "precision"][index], colour)
                axis.set(xlabel="Recall", title=title, xlim=(0, 1), ylim=(0, 1))
            axes[0].set_ylabel("Precision")
            legend_in_heading_row(axes, axes[0], 2)
            note(axes[0], f"Curves cover thresholds {thresholds[0]:.2f}-{thresholds[-1]:.2f}; "
                          f"dots mark the selected threshold {selected:.3f}.", offset=-0.2)
            figure.suptitle(f"Precision-recall curves ({label})", x=0.07, ha="left", fontsize=12,
                            fontweight="semibold")
            finish(figure, f"precision_recall_curve_{mode}.png")

            figure, axis = plt.subplots(figsize=(6.4, 4.4))
            for name, metrics, colour in splits:
                axis.plot(metrics["false_positive_rate"], metrics["recall"], color=colour, label=name)
                dot(axis, metrics["false_positive_rate"][index], metrics["recall"][index], colour)
            axis.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
                     title=f"ROC curves ({label})", xlim=(0, None), ylim=(0, 1))
            legend_in_title_row(axis, 2)
            background = 100 * (1 - data["test"][0, 6] / max(1.0, data["test"][0, 7]))
            note(axis, f"Thresholds {thresholds[0]:.2f}-{thresholds[-1]:.2f}; dots mark the selected threshold "
                       f"{selected:.3f}.\nBackground is {background:.1f} % of the test pixels, so the false "
                       "positive rate stays small.", offset=-0.19)
            finish(figure, f"roc_curve_{mode}.png")

            figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
            for axis, name, counts in ((axes[0], "Validation", data["val"]), (axes[1], "Test", data["test"])):
                cells = confusion_from_counts(counts[index])
                matrix = np.array([[cells["tp"], cells["fn"]], [cells["fp"], cells["tn"]]], dtype=float)
                share = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
                image = axis.imshow(share, cmap=colour_map, vmin=0, vmax=1)
                for row in range(2):
                    for column in range(2):
                        axis.text(column, row, f"{int(matrix[row, column]):,}\n{100 * share[row, column]:.2f} %",
                                  ha="center", va="center", fontsize=11,
                                  color="white" if share[row, column] > 0.5 else ink["primary"])
                axis.set_xticks([0, 1], ["Crack", "Background"])
                axis.set_yticks([0, 1], ["Crack", "Background"], rotation=90, va="center")
                axis.set(xlabel="Predicted", ylabel="Actual", title=f"{name} (threshold {selected:.3f})")
                axis.grid(False)
                axis.set_xticks([0.5], minor=True)
                axis.set_yticks([0.5], minor=True)
                axis.grid(which="minor", color="white", linewidth=2)  # surface gap between cells
                axis.tick_params(which="both", length=0, labelcolor=ink["secondary"])
                for spine in axis.spines.values():
                    spine.set_visible(False)
            bar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
            bar.set_label("Share of the actual class (row)", color=ink["secondary"])
            bar.outline.set_visible(False)
            figure.suptitle(f"Pixel confusion matrices ({label}): count and share of each actual class",
                            x=0.07, ha="left", fontsize=12, fontweight="semibold")
            finish(figure, f"confusion_matrix_{mode}.png")
    return written


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
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


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET_PATH,
                        help=f"80/10/10 dataset root (default: {DEFAULT_DATASET_PATH})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--resume", action="store_true", help="Continue from <output>/last.pt")
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
    parser.add_argument("--save-predictions", type=int, default=40, help="Test overlays to save")
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
    model = HairlineSwinUNet(pretrained=not (args.no_pretrained or args.evaluate_only), **architecture["kwargs"])
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
        args.skeleton_iterations, args.aux_weights,
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
    resuming_history = args.resume and start_epoch > 1 and history_path.is_file()
    half_index = int(np.argmin(np.abs(thresholds - 0.5)))
    if not args.evaluate_only:
        with history_path.open("a" if resuming_history else "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not resuming_history:
                writer.writeheader()
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
                improved = score > best_score
                print(
                    f"Epoch {epoch:03d}/{args.epochs} | loss {row['train_loss']:.4f}/{val_loss:.4f} | "
                    f"thr {row['threshold']:.3f} P {row['precision']:.4f} R {row['recall']:.4f} "
                    f"F1 {row['f1']:.4f} IoU {row['iou']:.4f} | tolF1 {row['tol_f1']:.4f} | "
                    f"{row['seconds']:.0f}s{'  *best*' if improved else ''}"
                )
                if improved:
                    best_score, best_epoch, stale = score, epoch, 0
                    torch.save({
                        "model": evaluation_model.state_dict(), "architecture": architecture,
                        "arguments": json_safe(vars(args)), "epoch": epoch,
                        "best_val_score": best_score, "select_metric": args.select_metric,
                        "optimal_threshold": float(thresholds[best_index]),
                    }, best_path)
                else:
                    stale += 1
                torch.save({
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
            # Report at the threshold the training run calibrated (and the detector uses).
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
        torch.save(checkpoint, best_path)

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


if __name__ == "__main__":
    main()
