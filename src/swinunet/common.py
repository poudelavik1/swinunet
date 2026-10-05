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

__all__ = ['argparse', 'copy', 'csv', 'json', 'math', 'os', 'random', 're', 'sys', 'time', 'Counter', 'defaultdict', 'dataclass', 'Path', 'cv2', 'np', 'torch', 'nn', 'F', 'DataLoader', 'Dataset', 'WeightedRandomSampler', 'IMAGE_EXTENSIONS', 'DATASET_NAME', 'LOCAL_DATASET_PATH', 'KAGGLE_INPUT_ROOT', 'KAGGLE_WORKING_ROOT', 'SWIN_ENCODERS', 'METRIC_NAMES', 'COUNT_NAMES', 'BASIC_METRICS', 'find_dataset_root', 'DEFAULT_DATASET_PATH', 'DEFAULT_OUTPUT_PATH', 'seed_everything', 'seed_worker']
