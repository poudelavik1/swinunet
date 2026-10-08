"""Detect cracks in an image of any size and export simplified crack polylines.

Pipeline
--------
1. Read the image (RGB/RGBA/gray, 8 or 16 bit). Transparent pixels (alpha = 0,
   e.g. outside an orthomosaic) are treated as "no data".
2. Plan the tiling. The tile size is fixed from the model's training crop
   (2 x 256 = 512 px) so every image is segmented at the same scale and
   context; the image size then sets the grid: an image that fits in one tile
   runs in a single pass, a larger one gets an evenly spaced grid of
   overlapping tiles blended with cosine weights (no seams). Memory sets the
   batch size. Images are never resized: the model was trained on
   native-resolution pixels, and downscaling would erase hairline cracks.
3. Run the HairlineSwinUNet trained by ``scripts/train.py`` (the ``swinunet``
   package in ``src``) and threshold the stitched probability map
   (``DEFAULT_THRESHOLD`` unless ``--threshold`` is given); a small
   morphological closing bridges 1-3 px gaps.
4. Zhang-Suen thinning (``zhang_suen_thinning.py``) in haloed tiles gives a
   one-pixel skeleton.
5. The skeleton is traced into a graph of branches. Burrs shorter than
   ``--spur-length`` and cracks shorter than ``--min-crack-length`` are
   removed. Two further filters remove lines that are not cracks. They are off
   by default: they were tuned on lab specimens with drawn grids, and on
   in-situ concrete (formwork board marks, straight cracks running along them)
   they deleted most of the real cracks. Turn them on for gridded specimens
   with ``--straight-line-length 500 --straight-segment-length 100
   --isolated-crack-length 150``.
   * Ruled lines (pencil grids, chalk lines, formwork and specimen edges).
     Long straight lines (>= ``--straight-line-length`` px) are found both in
     the crack mask (Hough transform) and in the photograph itself (thin dark
     or light lines that run straight along a row or column). A branch on such
     a line that is straight itself is removed at any length, so the grid's
     short leftover dashes go too; long straight stretches are also cut out of
     branches that join a crack to a ruled line. A crack that crosses a ruled
     line stays one polyline, and a crack that merely runs along one is kept
     because it wanders. ``removed_straight_lines.png`` shows what was removed.
   * Isolated short pieces. Cracks are regrouped after the ruled lines are
     gone, so pieces that were linked only through the grid are separate. A
     crack shorter than ``--isolated-crack-length`` is dropped unless it lies
     beside a longer crack or continues one across a gap (surface marks,
     recess edges, wood grain). ``removed_isolated_pieces.png`` shows them.
6. Ramer-Douglas-Peucker (RDP) reduces every branch to a few vertices.
7. CSV files hold the dense skeleton coordinates and the RDP polylines in CAD
   coordinates (origin bottom-left, ``--scale`` units per pixel), readable by
   ``draw_polylines_in_autocad.py``.
8. When run from the command line, detection hands straight over to
   ``cracks_to_autocad.py``: AutoCAD is detected (or started), the drawing is
   set to millimetres, and the image, crack polylines and crack-length labels
   are drawn. ``--no-autocad`` stops after step 7.

Run without arguments to pick an image in a file dialog::

    python hairline_crack_detection.py
    python hairline_crack_detection.py "type a front.png" --mm-per-pixel 0.30
    python hairline_crack_detection.py "type a front.png" --no-autocad
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
sys.path.insert(1, str(SCRIPT_DIRECTORY.parent / "src"))  # the swinunet package
from cracks_to_autocad import add_autocad_arguments  # noqa: E402
from swinunet.models import load_model_from_checkpoint  # noqa: E402
from zhang_suen_thinning import zhang_suen  # noqa: E402

DEFAULT_MODEL_PATH = SCRIPT_DIRECTORY / "best.pt"
# Probability threshold used unless --threshold is given. best.pt stores the value
# that maximised tolerance-F1 on the validation crops (0.875), but that curve is flat
# from 0.5 upwards (0.819 vs 0.828) and field images score lower. Against the manual
# crack drawings of two RAMA8 pylon orthomosaic tiles (1.9 mm/px, 10 px tolerance,
# ruled-line and isolated-piece filters off), 0.5 found 23-34 % of the manual crack
# length and 0.875 found 16-24 %, at the same F1 (0.20-0.23).
DEFAULT_THRESHOLD = 0.6
# Tile = CONTEXT_FACTOR x the model's training crop. The decoder's scSE attention
# pools over the whole input, so predictions depend on the tile size; it must
# therefore stay fixed rather than grow with the image. On a facade crop, 256 px
# tiles flagged pencil grid lines as cracks, whole-image passes dropped faint
# cracks, and 512 px (2 x 256) balanced the two. (Measured with the earlier
# HairlineUNet; the Swin model keeps the scSE head but was not re-tuned.)
CONTEXT_FACTOR = 2
# GPU memory per input pixel for fp16 HairlineUNet inference, with a safety
# margin (CPU fp32 peaks measured <= ~1 KB/px). Used only to size GPU batches;
# not re-measured for the Swin model, so pass --batch-size 1 if a GPU runs out.
CUDA_BYTES_PER_PIXEL = 1300
OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))


def choose_image() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Select an image for crack detection",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")],
        )
        root.destroy()
    except Exception as exc:
        raise SystemExit(f"Could not open the file chooser: {exc}") from exc
    if not selected:
        raise SystemExit("No image was selected.")
    return Path(selected)


def read_image(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return RGB uint8 and an optional valid-data mask from the alpha channel."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    if image.dtype != np.uint8:
        maximum = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else float(image.max())
        image = np.clip(image.astype(np.float32) * (255.0 / max(1.0, maximum)), 0, 255).astype(np.uint8)
    valid = None
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        alpha = image[:, :, 3]
        valid = alpha > 0 if alpha.min() == 0 else None
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    del image
    return rgb, valid


# --------------------------------------------------------------------------- #
# Tiling plan
# --------------------------------------------------------------------------- #
def round_up(value: int, multiple: int = 32) -> int:
    return int(math.ceil(value / multiple) * multiple)


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Evenly spaced tile origins covering [0, length) with at least `overlap` shared pixels."""
    if length <= tile:
        return [0]
    count = math.ceil((length - overlap) / (tile - overlap))
    return [round(index * (length - tile) / (count - 1)) for index in range(count)]


def plan_tiles(height: int, width: int, device: torch.device, training_crop: int,
               tile_size: int | None, overlap: int | None, batch_size: int | None) -> dict:
    """Decide the tiling from the model's training scale, the image size and memory.

    * Tile size: CONTEXT_FACTOR x the training crop (512 px for 256 px crops),
      the same for every image and machine, so a crack is segmented
      identically in a small photo and in a 100 MP mosaic. A 512 px tile needs
      well under 1 GB, so it is never shrunk automatically (use --tile-size).
    * Image size: images no larger than one tile run in a single pass (padded
      to a multiple of 32). Larger images get an evenly spaced grid with
      1/8-tile overlap (>= 32 px), blended with cosine weights.
    * Memory: sets the batch size on a GPU (free memory / per-tile need).
    """
    if tile_size is None:
        tile_size = round_up(CONTEXT_FACTOR * training_crop)
        reason = f"{CONTEXT_FACTOR} x the {training_crop} px training crop"
    else:
        tile_size, reason = round_up(tile_size), "set by --tile-size"
    tile_h, tile_w = min(tile_size, round_up(height)), min(tile_size, round_up(width))
    single = tile_h >= height and tile_w >= width
    if single:
        overlap, reason = 0, reason + "; image fits in one tile: single pass"
    else:
        overlap = min(overlap if overlap is not None else max(32, round_up(tile_size // 8)), tile_size // 2)
    ys, xs = tile_starts(height, tile_h, overlap), tile_starts(width, tile_w, overlap)
    if batch_size is None:
        # CPU throughput does not improve with batching (measured), so use 1.
        batch_size = 1
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            batch_size = int(max(1, min(8, 0.6 * free // (tile_h * tile_w * CUDA_BYTES_PER_PIXEL))))
    return {
        "image_height": height, "image_width": width, "tile_height": tile_h, "tile_width": tile_w,
        "overlap": overlap, "rows": len(ys), "columns": len(xs), "tiles": len(ys) * len(xs),
        "batch_size": batch_size, "reason": reason, "y_starts": ys, "x_starts": xs,
    }


def blend_weights(starts: list[int], tile: int, length: int, overlap: int):
    """1-D cosine ramps inside overlaps; the 2-D weight is their outer product.

    Because the tile grid is a Cartesian product, the per-pixel weight sum is
    also separable, so no full-size weight image has to be stored.
    """
    ramp = (0.5 - 0.5 * np.cos(np.pi * (np.arange(overlap) + 0.5) / overlap)).astype(np.float32) if overlap else None
    weights, total = [], np.zeros(length, np.float32)
    for index, start in enumerate(starts):
        span = min(tile, length - start)
        weight = np.ones(span, np.float32)
        if index > 0:
            weight[:overlap] = ramp
        if index < len(starts) - 1:
            weight[-overlap:] = np.minimum(weight[-overlap:], ramp[::-1])
        weights.append(weight)
        total[start:start + span] += weight
    return [weight / total[start:start + len(weight)] for weight, start in zip(weights, starts)]


@torch.inference_mode()
def predict_probability(model, rgb, valid, plan, device, tta: bool) -> np.ndarray:
    height, width = rgb.shape[:2]
    tile_h, tile_w, overlap = plan["tile_height"], plan["tile_width"], plan["overlap"]
    ys, xs = plan["y_starts"], plan["x_starts"]
    weight_y, weight_x = blend_weights(ys, tile_h, height, overlap), blend_weights(xs, tile_w, width, overlap)
    probability = np.zeros((height, width), np.float32)
    use_amp = device.type == "cuda"
    jobs = [(row, column) for row in range(len(ys)) for column in range(len(xs))]
    if valid is not None:
        jobs = [(r, c) for r, c in jobs if valid[ys[r]:ys[r] + tile_h, xs[c]:xs[c] + tile_w].any()]
    skipped = plan["tiles"] - len(jobs)
    started, done, report_every = time.time(), 0, max(1, len(jobs) // 20)

    def run(batch):
        tensor = torch.from_numpy(np.stack([tile for tile, _ in batch]).transpose(0, 3, 1, 2))
        tensor = tensor.to(device).float().div_(255).contiguous(memory_format=torch.channels_last)
        passes = ((),) + (((3,), (2,), (2, 3)) if tta else ())
        output = 0
        for dims in passes:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(torch.flip(tensor, dims) if dims else tensor)
            logits = torch.flip(logits, dims) if dims else logits
            output = output + torch.sigmoid(logits.float())
        output = (output / len(passes))[:, 0].cpu().numpy()
        for prediction, (_, (row, column)) in zip(output, batch):
            y, x = ys[row], xs[column]
            span_h, span_w = min(tile_h, height - y), min(tile_w, width - x)
            probability[y:y + span_h, x:x + span_w] += (
                prediction[:span_h, :span_w] * weight_y[row][:, None] * weight_x[column][None, :]
            )

    batch = []
    for row, column in jobs:
        y, x = ys[row], xs[column]
        tile = rgb[y:y + tile_h, x:x + tile_w]
        if tile.shape[0] != tile_h or tile.shape[1] != tile_w:  # image smaller than tile: pad
            tile = cv2.copyMakeBorder(tile, 0, tile_h - tile.shape[0], 0, tile_w - tile.shape[1],
                                      cv2.BORDER_REFLECT_101)
        batch.append((tile, (row, column)))
        if len(batch) == plan["batch_size"]:
            run(batch)
            done += len(batch)
            batch = []
            if done % report_every < plan["batch_size"] or done == len(jobs):
                elapsed = time.time() - started
                remaining = elapsed / done * (len(jobs) - done)
                print(f"  inference {done}/{len(jobs)} tiles ({100 * done / len(jobs):.0f}%) "
                      f"elapsed {elapsed:.0f}s, remaining ~{remaining:.0f}s", flush=True)
    if batch:
        run(batch)
    if skipped:
        print(f"  skipped {skipped} tiles with no valid (non-transparent) pixels")
    if valid is not None:
        probability[~valid] = 0
    return probability


# --------------------------------------------------------------------------- #
# Zhang-Suen thinning in haloed tiles
# --------------------------------------------------------------------------- #
def thin_in_tiles(mask: np.ndarray, core: int = 1024, halo: int = 32) -> np.ndarray:
    """Zhang-Suen per tile; the halo keeps results identical to whole-image thinning
    as long as fewer than `halo` iterations are needed (cracks < ~2*halo px wide)."""
    height, width = mask.shape
    skeleton = np.zeros((height, width), bool)
    most_iterations = 0
    for y0 in range(0, height, core):
        for x0 in range(0, width, core):
            y1, x1 = min(height, y0 + core), min(width, x0 + core)
            if not mask[y0:y1, x0:x1].any():
                continue
            hy0, hy1, hx0, hx1 = max(0, y0 - halo), min(height, y1 + halo), max(0, x0 - halo), min(width, x1 + halo)
            window = mask[hy0:hy1, hx0:hx1]
            rows, columns = np.flatnonzero(window.any(1)), np.flatnonzero(window.any(0))
            ry0, ry1, rx0, rx1 = rows[0], rows[-1] + 1, columns[0], columns[-1] + 1
            thinned, iterations = zhang_suen(np.pad(window[ry0:ry1, rx0:rx1], 1))
            most_iterations = max(most_iterations, iterations)
            local = np.zeros(window.shape, bool)
            local[ry0:ry1, rx0:rx1] = thinned[1:-1, 1:-1]
            skeleton[y0:y1, x0:x1] = local[y0 - hy0:y1 - hy0, x0 - hx0:x1 - hx0]
    if most_iterations >= halo:
        print(f"  note: a region needed {most_iterations} thinning iterations (> halo {halo}); "
              "very wide blobs may show small seams in the skeleton")
    return skeleton


# --------------------------------------------------------------------------- #
# Skeleton graph -> branches -> pruning -> RDP
# --------------------------------------------------------------------------- #
def trace_branches(skeleton: np.ndarray) -> list[list[tuple[int, int]]]:
    """Split one skeleton component into branches between endpoints/junctions.

    Diagonal steps are ignored when an orthogonal path exists, so staircase
    corners do not create false junctions (same rule as crack_mask_to_dxf.py).
    """
    height, width = skeleton.shape
    points = [tuple(map(int, point)) for point in np.argwhere(skeleton)]
    adjacency = {}
    for y, x in points:
        found = []
        for dy, dx in OFFSETS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and skeleton[ny, nx]:
                if dy and dx and (skeleton[y, nx] or skeleton[ny, x]):
                    continue
                found.append((ny, nx))
        adjacency[(y, x)] = found
    nodes = {point for point in points if len(adjacency[point]) != 2}
    visited, branches = set(), []

    def edge(a, b):
        return (a, b) if a < b else (b, a)

    def walk(start, following, stop_at_nodes):
        branch, previous, current = [start, following], start, following
        visited.add(edge(start, following))
        while not (stop_at_nodes and current in nodes) and current != start:
            candidates = [p for p in adjacency[current] if p != previous and edge(current, p) not in visited]
            if not candidates:
                break
            previous, current = current, candidates[0]
            visited.add(edge(previous, current))
            branch.append(current)
        return branch

    for start in sorted(nodes):
        for following in adjacency[start]:
            if edge(start, following) not in visited:
                branches.append(walk(start, following, True))
    for start in points:  # anything left is a closed loop without junctions
        for following in adjacency[start]:
            if edge(start, following) not in visited:
                branches.append(walk(start, following, False))
    if not branches and points:
        branches.append([points[0]])
    return branches


def polyline_length(points) -> float:
    if len(points) < 2:
        return 0.0
    array = np.asarray(points, np.float64)
    return float(np.hypot(*np.diff(array, axis=0).T).sum())


def prune_and_merge(branches, spur_length: float, shape, border_margin: int = 2):
    """Remove endpoint-to-junction burrs shorter than spur_length, then join branches
    that meet at a node of degree two. Burrs touching the image border are kept
    because the crack may continue outside the picture."""
    height, width = shape

    def at_border(point):
        y, x = point
        return y < border_margin or x < border_margin or y >= height - border_margin or x >= width - border_margin

    branches = [list(branch) for branch in branches if len(branch) >= 2]
    while True:
        degree = Counter(end for branch in branches for end in (branch[0], branch[-1]))
        shortest_spur = {}
        for index, branch in enumerate(branches):
            length = polyline_length(branch)
            if length >= spur_length:
                continue
            for tip, junction in ((branch[0], branch[-1]), (branch[-1], branch[0])):
                if degree[tip] == 1 and degree[junction] >= 3 and not at_border(tip):
                    # Only the shortest burr per junction per round, so a small
                    # cluster of burrs never deletes the crack it hangs from.
                    if junction not in shortest_spur or length < shortest_spur[junction][0]:
                        shortest_spur[junction] = (length, index)
                    break
        if not shortest_spur:
            break
        removed = {index for _, index in shortest_spur.values()}
        branches = [branch for index, branch in enumerate(branches) if index not in removed]
        branches = merge_degree_two(branches)
    return merge_degree_two(branches)


def merge_degree_two(branches):
    merged = True
    while merged:
        merged = False
        degree = Counter(end for branch in branches for end in (branch[0], branch[-1]))
        ends = {}
        for index, branch in enumerate(branches):
            for end in (branch[0], branch[-1]):
                ends.setdefault(end, []).append(index)
        for node, members in ends.items():
            if degree[node] != 2 or len(set(members)) != 2:
                continue  # junction, endpoint, or a closed loop through the node
            first, second = (branches[index] for index in members)
            if first[0] == node:
                first = first[::-1]
            if second[-1] == node:
                second = second[::-1]
            joined = first + second[1:]
            branches = [b for index, b in enumerate(branches) if index not in members] + [joined]
            merged = True
            break
    return branches


def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker with point-to-segment distances, iterative (no recursion limit)."""
    if len(points) < 3 or epsilon <= 0:
        return points
    return points[rdp_vertices(points, epsilon)]


def rdp_vertices(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Boolean mask of the points RDP keeps."""
    points = np.asarray(points, np.float64)
    count = len(points)
    keep = np.zeros(count, bool)
    keep[0] = keep[-1] = True
    stack = [(0, count - 1)]
    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue
        segment = points[end] - points[start]
        relative = points[start + 1:end] - points[start]
        denominator = float(segment @ segment)
        if denominator == 0:
            distances = np.hypot(relative[:, 0], relative[:, 1])
        else:
            t = np.clip(relative @ segment / denominator, 0.0, 1.0)
            offset = relative - t[:, None] * segment
            distances = np.hypot(offset[:, 0], offset[:, 1])
        farthest = int(np.argmax(distances))
        if distances[farthest] > epsilon:
            split = start + 1 + farthest
            keep[split] = True
            stack.extend(((start, split), (split, end)))
    return keep


def straight_runs(points: np.ndarray, tolerance: float) -> list[tuple[int, int]]:
    """Split a branch into maximal straight stretches.

    Returns inclusive index ranges; each stretch stays within max(tolerance, 1 %
    of its length) of its chord. Stretches are grown between RDP vertices.
    """
    vertices = np.flatnonzero(rdp_vertices(points, tolerance))
    runs, start = [], 0
    while start < len(vertices) - 1:
        end = start + 1
        while end + 1 < len(vertices):
            candidate = points[vertices[start]:vertices[end + 1] + 1]
            if chord_deviation(candidate, trim=0) > max(tolerance, 0.01 * polyline_length(candidate)):
                break
            end += 1
        runs.append((int(vertices[start]), int(vertices[end])))
        start = end
    return runs


def straight_line_band(mask: np.ndarray, min_length: int, radius: int = 4):
    """Band around long straight lines in the crack mask (ruled grid/reference lines).

    Cracks are tortuous, so only drawn lines, formwork edges and similar
    features stay straight within the mask width for `min_length` pixels. Each
    Hough segment is extended by `min_length` at both ends so shorter
    fragments of the same ruled line fall inside the band as well.
    """
    lines = cv2.HoughLinesP(mask, 1, np.pi / 720, threshold=int(0.6 * min_length),
                            minLineLength=min_length, maxLineGap=max(20, min_length // 8))
    band = np.zeros(mask.shape, np.uint8)
    if lines is None:
        return band.astype(bool), 0
    for x1, y1, x2, y2 in lines[:, 0].astype(np.float64):
        direction = np.array((x2 - x1, y2 - y1)) / max(1e-9, math.hypot(x2 - x1, y2 - y1))
        start = np.round((x1, y1) - direction * min_length).astype(int)
        end = np.round((x2, y2) + direction * min_length).astype(int)
        cv2.line(band, tuple(map(int, start)), tuple(map(int, end)), 1, 2 * radius + 1)
    return band.astype(bool), len(lines)


def ruled_line_mask(rgb: np.ndarray, valid: np.ndarray | None, min_length: int, radius: int = 4) -> np.ndarray:
    """Band around long straight thin lines drawn on the surface, found in the photograph itself.

    Pencil grids, chalk lines and joints are thin lines (dark or light against
    their surroundings) that run dead straight for a long way; cracks wander.
    A black-hat (dark) or top-hat (light) filter isolates thin lines, dotted
    pencil strokes are bridged, and only runs of at least `min_length` px along
    a row or a column survive. This finds the whole ruled line even where the
    network marked only short dashes of it, which the mask-based Hough test
    cannot. Lines must be within about 0.8 degrees of horizontal or vertical;
    tilted ones are left to the Hough test.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sample = gray[::8, ::8][valid[::8, ::8]] if valid is not None else gray[::8, ::8]
    # A pencil line on white paint is ~55 grey levels darker; paint texture is below ~15.
    threshold = max(15.0, 0.13 * float(np.median(sample))) if sample.size else 15.0
    length = int(min_length) | 1
    lines = np.zeros(gray.shape, np.uint8)
    for operation in (cv2.MORPH_BLACKHAT, cv2.MORPH_TOPHAT):
        thin = (cv2.morphologyEx(gray, operation, np.ones((9, 9), np.uint8)) >= threshold).astype(np.uint8)
        if valid is not None:
            thin[~valid] = 0
        for horizontal in (True, False):
            def strip(along, across=1):
                return np.ones((across, along) if horizontal else (along, across), np.uint8)
            run = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, strip(15))   # bridge dotted strokes
            run = cv2.dilate(run, strip(1, 5))                         # tolerate a slight tilt
            lines |= cv2.morphologyEx(run, cv2.MORPH_OPEN, strip(length))
    return cv2.dilate(lines, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) > 0


def connected_groups(branches) -> list[list[int]]:
    """Indices of branches that are connected through shared end pixels (junctions)."""
    parent = list(range(len(branches)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owner = {}
    for index, branch in enumerate(branches):
        for end in (tuple(branch[0]), tuple(branch[-1])):
            if end in owner:
                parent[root(index)] = root(owner[end])
            else:
                owner[end] = index
    groups = {}
    for index in range(len(branches)):
        groups.setdefault(root(index), []).append(index)
    return list(groups.values())


def drop_isolated_pieces(polylines, shape, min_length: float, distance: int, end_gap: int | None = None):
    """Drop cracks shorter than `min_length` unless they belong to a longer crack.

    A short crack is kept when it lies within `distance` px of a crack at least
    `min_length` long (a branch or parallel strand), or when one of its ends is
    within `end_gap` px (default 2.5 x distance) of an end of a kept crack: the
    crack continuing across a gap in the detection. The second test is repeated,
    so a chain of fragments bridging two long cracks survives. Short cracks on
    their own are mostly noise (surface marks, recess edges, leftover dashes of
    ruled lines). If no crack reaches `min_length`, nothing is dropped: there is
    then no way to tell fragments from noise.
    Returns (kept polylines renumbered 1..N, [points of dropped pieces]).
    """
    end_gap = int(2.5 * distance) if end_gap is None else end_gap
    lengths = {}
    for crack_id, points in polylines:
        lengths[crack_id] = lengths.get(crack_id, 0.0) + polyline_length(points)
    long_ids = {crack_id for crack_id, length in lengths.items() if length >= min_length}
    if not long_ids or len(long_ids) == len(lengths):
        return polylines, []
    beside = np.zeros(shape, np.uint8)
    for crack_id, points in polylines:
        if crack_id in long_ids:
            beside[points[:, 0].astype(int), points[:, 1].astype(int)] = 1
    beside = cv2.dilate(beside, np.ones((2 * distance + 1, 2 * distance + 1), np.uint8))
    keep = set(long_ids)
    for crack_id, points in polylines:
        if crack_id not in keep and beside[points[:, 0].astype(int), points[:, 1].astype(int)].any():
            keep.add(crack_id)
    del beside
    # Free ends (met by one polyline only) of every crack; junction pixels are shared.
    counts = {}
    for crack_id, points in polylines:
        for end in (tuple(points[0]), tuple(points[-1])):
            counts[(crack_id, end)] = counts.get((crack_id, end), 0) + 1
    ends = {}
    for (crack_id, end), count in counts.items():
        if count == 1:
            ends.setdefault(crack_id, []).append(end)
    added = True
    while added:  # grow along chains of fragments
        added = False
        kept_ends = np.array([end for crack_id in keep for end in ends.get(crack_id, [])], dtype=np.float64)
        if not len(kept_ends):
            break
        for crack_id, own in ends.items():
            if crack_id in keep:
                continue
            gaps = np.hypot(*(np.array(own, dtype=np.float64)[:, None, :] - kept_ends[None, :, :]).transpose(2, 0, 1))
            if gaps.min() <= end_gap:
                keep.add(crack_id)
                added = True
    renumber = {old: new for new, old in enumerate(dict.fromkeys(c for c, _ in polylines if c in keep), 1)}
    kept = [(renumber[crack_id], points) for crack_id, points in polylines if crack_id in keep]
    return kept, [points for crack_id, points in polylines if crack_id not in keep]


def chord_deviation(points: np.ndarray, trim: int = 6) -> float:
    """Largest distance of the points from the straight line through the two ends.

    `trim` points are ignored at each end: Zhang-Suen skeletons bend as they
    enter a junction, which would otherwise hide a straight ruled line.
    """
    trim = min(trim, len(points) // 5)
    if trim:
        points = points[trim:-trim]
    start, end = points[0], points[-1]
    dy, dx = end - start
    norm = math.hypot(dy, dx)
    if norm == 0:
        return math.inf  # closed loop: not a straight line
    return float(np.abs(dy * (points[:, 1] - start[1]) - dx * (points[:, 0] - start[0])).max() / norm)


def extract_polylines(skeleton: np.ndarray, spur_length: float, min_crack_length: float,
                      line_band: np.ndarray | None = None, straight_segment_length: float = 0.0,
                      band_fraction: float = 0.8, straight_tolerance: float = 1.5):
    """Return ([(component_id, dense (N, 2) array of (y, x) points)], dropped, removed_line_points).

    A crack is one connected set of branches *after* the ruled lines are
    removed, so cracks that were linked only through a grid line are counted
    separately and the grid's leftover dashes become short cracks of their own.

    A branch is a ruled line, not a crack, when it lies mostly (>= band_fraction)
    inside `line_band` (long straight lines, see straight_line_band and
    ruled_line_mask), or when it is at least
    `straight_segment_length` px long and within max(`straight_tolerance`, 1% of
    its length) of its chord; real cracks are rough at pixel scale and never
    that straight. (On the EH104 facade every such branch was a pencil grid
    line or a specimen edge; shorter limits started to catch real crack pieces
    along marker lines.) Ruled branches are removed before burr pruning,
    repeatedly with re-merging, so a crack crossing a grid line becomes one
    polyline instead of being cut.
    """
    # Measured on a gridded specimen: pencil-line pieces deviate 3.6-4.2 px (1.3-1.9 % of their
    # length) from their chord where they cross embossed lettering; real cracks running along
    # a grid line or a tape edge deviate 7.4-17 px (2.5-9 %).
    band_tolerance, run_tolerance = 3 * straight_tolerance, 2 * straight_tolerance
    connector_length = 15  # points; longest piece treated as a crack/line crossing
    compound_length = 1.5 * straight_segment_length if straight_segment_length > 0 else 150.0

    def ruled_parts(points):
        """Inclusive index ranges of the branch that are ruled lines (the whole branch or parts)."""
        whole = [(0, len(points) - 1)]
        length, deviation = polyline_length(points), chord_deviation(points)
        if (straight_segment_length > 0 and length >= straight_segment_length
                and deviation <= max(straight_tolerance, 0.01 * length)):
            return whole
        on_line = line_band[points[:, 0], points[:, 1]] if line_band is not None else np.zeros(len(points), bool)
        # On a ruled line and straight itself, at any length: the leftover dashes. A crack
        # that merely runs along a straight edge (tape, joint) wanders in the band and stays.
        if on_line.mean() >= band_fraction and deviation <= max(band_tolerance, 0.015 * length):
            return whole
        if length < compound_length:
            return []
        # Compound branch (a crack joined to a pencil line, two grid lines round a corner):
        # cut out its long straight stretches and keep the rest. A stretch counts when it
        # is ruler-straight on its own, or fairly straight and lying on a ruled line.
        floats = points.astype(np.float64)
        parts = [(start, end) for start, end in straight_runs(floats, straight_tolerance)
                 if straight_segment_length > 0 and polyline_length(floats[start:end + 1]) >= compound_length]
        if on_line.sum() >= band_fraction * compound_length:
            parts += [(start, end) for start, end in straight_runs(floats, run_tolerance)
                      if polyline_length(floats[start:end + 1]) >= compound_length
                      and on_line[start:end + 1].mean() >= band_fraction]
        merged = []
        for start, end in sorted(parts):  # union of overlapping ranges
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        return merged

    count, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton.astype(np.uint8), connectivity=8)
    polylines, dropped, removed_line_points, crack_id = [], 0, [], 0
    for label in range(1, count):
        x, y, w, h, pixels = stats[label]
        if pixels < 2:
            dropped += 1
            continue
        local = np.pad(labels[y:y + h, x:x + w] == label, 1)
        branches = [b for b in trace_branches(local) if len(b) >= 2]
        if line_band is not None or straight_segment_length > 0:
            removed_any = True
            while removed_any:
                kept, removed_any = [], False
                shared = Counter(end for branch in branches for end in (branch[0], branch[-1]))
                for branch in branches:
                    points = np.asarray(branch) + (y - 1, x - 1)
                    parts = ruled_parts(points)
                    # Where a crack crosses a ruled line obliquely, the two share a few pixels.
                    # Keep such a short piece while it still joins other branches at both ends;
                    # once the line around it is gone it either links the crack or hangs free
                    # and is removed on the next pass.
                    if (parts and len(branch) <= connector_length
                            and shared[branch[0]] > 1 and shared[branch[-1]] > 1):
                        parts = []
                    if not parts:
                        kept.append(branch)
                        continue
                    removed_any, cursor = True, 0
                    for start, end in parts:  # keep what lies between the ruled stretches
                        removed_line_points.append(points[start:end + 1])
                        if start > cursor:
                            kept.append(branch[cursor:start + 1])
                        cursor = end
                    if cursor < len(branch) - 1:
                        kept.append(branch[cursor:])
                branches = merge_degree_two(kept)
        branches = prune_and_merge(branches, spur_length, skeleton.shape) if spur_length > 0 else branches
        branches = [b for b in branches if len(b) >= 2]
        for group in connected_groups(branches):  # what is still connected is one crack
            if sum(polyline_length(branches[index]) for index in group) < min_crack_length:
                dropped += 1
                continue
            crack_id += 1
            for index in group:
                array = np.asarray(branches[index], np.float64) + (y - 1, x - 1)  # undo crop and pad
                polylines.append((crack_id, array))
    return polylines, dropped, removed_line_points


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def save_image(path: Path, image: np.ndarray, parameters=None) -> None:
    """cv2.imwrite fails silently (e.g. the file is briefly locked by a virus scanner or
    still open in a viewer), so check it, retry, and say so if the file is not written."""
    for attempt in range(3):
        try:
            if cv2.imwrite(str(path), image, parameters or []):
                return
        except cv2.error:
            pass
        time.sleep(0.5 * (attempt + 1))
    print(f"Warning: could not write {path} (is it open in another program?). The other results are unaffected.")


def to_cad(points_yx: np.ndarray, height: int, scale: float) -> np.ndarray:
    """(row, col) pixels -> CAD (X, Y): origin bottom-left, `scale` units per pixel."""
    return np.column_stack((points_yx[:, 1] * scale, (height - 1 - points_yx[:, 0]) * scale))


def write_vertex_csv(path: Path, polylines, height: int, scale: float) -> int:
    rows = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("polyline_id", "crack_id", "vertex_order", "x", "y", "x_px", "y_px"))
        for polyline_id, (crack_id, points) in enumerate(polylines, 1):
            cad = to_cad(points, height, scale)
            for order, ((row, column), (x, y)) in enumerate(zip(points, cad), 1):
                writer.writerow((polyline_id, crack_id, order, f"{x:.4f}", f"{y:.4f}", int(column), int(row)))
                rows += 1
    return rows


def write_summary_csv(path: Path, dense, simplified, height: int, scale: float):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("polyline_id", "crack_id", "skeleton_points", "rdp_points", "reduction_percent",
                         "length_px", "length_units", "start_x", "start_y", "end_x", "end_y"))
        for polyline_id, ((crack_id, raw), (_, reduced)) in enumerate(zip(dense, simplified), 1):
            length = polyline_length(raw)
            (sx, sy), (ex, ey) = to_cad(reduced[[0, -1]], height, scale)
            writer.writerow((polyline_id, crack_id, len(raw), len(reduced),
                             f"{100 * (1 - len(reduced) / len(raw)):.1f}", f"{length:.2f}",
                             f"{length * scale:.4f}", f"{sx:.4f}", f"{sy:.4f}", f"{ex:.4f}", f"{ey:.4f}"))


def save_preview(path: Path, rgb, simplified, max_side: int = 4000):
    """Downscaled overview: red RDP polylines, yellow RDP vertices."""
    height, width = rgb.shape[:2]
    factor = min(1.0, max_side / max(height, width))
    canvas = cv2.resize(rgb, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA) \
        if factor < 1 else rgb.copy()
    canvas = (canvas * 0.7).astype(np.uint8)
    thickness = max(1, round(2 * max(factor, 0.5)))
    for _, points in simplified:
        xy = np.round(points[:, ::-1] * factor).astype(np.int32)
        cv2.polylines(canvas, [xy], False, (255, 40, 40), thickness, cv2.LINE_AA)
        for x, y in xy:
            cv2.circle(canvas, (int(x), int(y)), thickness + 1, (255, 230, 0), -1, cv2.LINE_AA)
    save_image(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", type=Path, help="Input image; omit to choose in a dialog")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="best.pt from a Swin U-Net training run")
    parser.add_argument("--output", type=Path, help="Output folder (default: <image>_crack_detection)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Probability threshold (default: {DEFAULT_THRESHOLD}; the validation-calibrated "
                             "value stored in best.pt is printed for comparison)")
    parser.add_argument("--tta", action="store_true", help="Average 4 flipped predictions (4x slower)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tile-size", type=int, help="Override the automatic tile size (pixels)")
    parser.add_argument("--overlap", type=int, help="Override the automatic tile overlap (pixels)")
    parser.add_argument("--batch-size", type=int, help="Override the automatic batch size")
    parser.add_argument("--closing-kernel", type=int, default=3,
                        help="Elliptical closing to bridge small gaps before thinning; 0 disables (default: 3)")
    parser.add_argument("--spur-length", type=float, default=10.0,
                        help="Prune endpoint-to-junction burrs shorter than this many pixels (default: 10)")
    parser.add_argument("--min-crack-length", type=float, default=20.0,
                        help="Drop connected cracks shorter than this many pixels (default: 20)")
    parser.add_argument("--isolated-crack-length", type=float, default=0.0,
                        help="Drop cracks shorter than this many pixels unless they lie beside a crack at "
                             "least that long (surface marks, recess edges, leftover dashes). 0 disables "
                             "(default: 0; 150 suits gridded lab specimens)")
    parser.add_argument("--isolated-distance", type=int, default=60,
                        help="A short crack within this many pixels of a long one is kept as its "
                             "continuation across a gap (default: 60)")
    parser.add_argument("--straight-line-length", type=int, default=0,
                        help="Remove skeleton branches on straight lines at least this long (pencil grids, "
                             "reference lines, formwork edges), found both in the crack mask and in the "
                             "photograph itself; 0 disables (default: 0; 500 suits gridded lab specimens)")
    parser.add_argument("--straight-segment-length", type=float, default=0.0,
                        help="Also remove any branch at least this long that stays within 1.5 px (or 1%% "
                             "of its length) of a straight line: ruled-line fragments; 0 disables "
                             "(default: 0; 100 suits gridded lab specimens)")
    parser.add_argument("--epsilon", type=float, default=2.0, help="RDP tolerance in pixels (default: 2)")
    parser.add_argument("--scale", type=float, default=1.0, help="CAD units per pixel (default: 1)")
    parser.add_argument("--reuse-probability", action="store_true",
                        help="Skip inference and reuse probability.png in the output folder "
                             "(fast re-runs with another threshold, epsilon, or filter)")
    autocad = parser.add_argument_group("AutoCAD drawing after detection (cracks_to_autocad.py)")
    autocad.add_argument("--no-autocad", action="store_true", help="Stop after detection; do not draw in AutoCAD")
    add_autocad_arguments(autocad)
    return parser.parse_args(argv)


def main(args=None) -> Path:
    """Run the full pipeline; returns the output folder. `args` comes from parse_args()."""
    args = args if args is not None else parse_args()
    image_path = args.image or choose_image()
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")
    args.image = image_path.resolve()  # the AutoCAD step needs the image picked in the dialog
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}. Pass --model path\\to\\best.pt")
    if args.epsilon < 0 or args.scale <= 0:
        raise SystemExit("--epsilon must be >= 0 and --scale > 0.")
    output = args.output or image_path.with_name(f"{image_path.stem}_crack_detection")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    timings, started = {}, time.time()

    model, checkpoint = load_model_from_checkpoint(args.model, device)
    model = model.to(memory_format=torch.channels_last)
    stored = checkpoint.get("optimal_threshold_tta" if args.tta else "optimal_threshold")
    threshold = args.threshold

    rgb, valid = read_image(image_path)
    height, width = rgb.shape[:2]
    training_crop = int(checkpoint.get("arguments", {}).get("image_size", 256))
    plan = plan_tiles(height, width, device, training_crop, args.tile_size, args.overlap, args.batch_size)
    print(f"Image: {image_path.name}  {width} x {height} px ({width * height / 1e6:.1f} MP)"
          f"{'  with transparency mask' if valid is not None else ''}")
    print(f"Tiling: {plan['tile_width']} x {plan['tile_height']} px tiles, overlap {plan['overlap']} px, "
          f"{plan['columns']} x {plan['rows']} = {plan['tiles']} tiles, batch {plan['batch_size']} "
          f"on {device.type} ({plan['reason']})")
    print(f"Threshold: {threshold:.3f}{' (flip TTA)' if args.tta else ''}"
          f"{f'  (best.pt validation value: {float(stored):.3f})' if stored is not None else ''}")
    timings["load"] = time.time() - started

    step = time.time()
    saved_probability = output / "probability.png"
    if args.reuse_probability and saved_probability.is_file():
        probability_u8 = cv2.imread(str(saved_probability), cv2.IMREAD_GRAYSCALE)
        if probability_u8.shape != (height, width):
            raise SystemExit(f"{saved_probability} is {probability_u8.shape[::-1]}, image is {(width, height)}.")
        print(f"Reusing {saved_probability} (inference skipped)")
    else:
        probability = predict_probability(model, rgb, valid, plan, device, args.tta)
        probability_u8 = np.round(probability * 255).astype(np.uint8)
        del probability
        save_image(saved_probability, probability_u8)
    timings["inference"] = time.time() - step

    step = time.time()
    mask = (probability_u8 >= round(threshold * 255)).astype(np.uint8)
    if args.closing_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.closing_kernel, args.closing_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if valid is not None:
            mask[~valid] = 0
    save_image(output / "crack_mask.png", mask * 255)
    print(f"Crack pixels: {int(mask.sum()):,} ({100 * mask.mean():.3f}% of the image)")
    line_band, line_segments, ruled_share = None, 0, 0.0
    if args.straight_line_length > 0:
        line_band, line_segments = straight_line_band(mask, args.straight_line_length)
        image_lines = ruled_line_mask(rgb, valid, args.straight_line_length)
        ruled_share = float(image_lines.mean())
        line_band |= image_lines
        del image_lines

    skeleton = thin_in_tiles(mask.astype(bool))
    del mask
    timings["thinning"] = time.time() - step
    print(f"Zhang-Suen skeleton pixels: {int(skeleton.sum()):,}")

    step = time.time()
    dense, dropped, removed_lines = extract_polylines(
        skeleton, args.spur_length, args.min_crack_length,
        line_band if line_band is not None and line_band.any() else None, args.straight_segment_length,
    )
    removed_line_pixels = sum(len(points) for points in removed_lines)
    if removed_lines:
        print(f"Straight-line filter: {line_segments} Hough lines >= {args.straight_line_length} px in the mask, "
              f"ruled lines over {100 * ruled_share:.1f}% of the photograph; removed "
              f"{len(removed_lines)} ruled branches, {removed_line_pixels:,} skeleton px "
              "(see removed_straight_lines.png)")
        removed_image = np.zeros(skeleton.shape, np.uint8)
        for points in removed_lines:
            removed_image[points[:, 0], points[:, 1]] = 255
        save_image(output / "removed_straight_lines.png", removed_image)
        del removed_image
    del line_band
    isolated_length = args.isolated_crack_length
    isolated = []
    if isolated_length > 0:
        dense, isolated = drop_isolated_pieces(dense, skeleton.shape, isolated_length, args.isolated_distance)
    isolated_pixels = sum(len(points) for points in isolated)
    if isolated:
        print(f"Isolated-piece filter: removed {len(isolated)} polylines ({isolated_pixels:,} skeleton px) of "
              f"cracks shorter than {isolated_length:.0f} px with no longer crack within "
              f"{args.isolated_distance} px (see removed_isolated_pieces.png)")
        removed_image = np.zeros(skeleton.shape, np.uint8)
        for points in isolated:
            removed_image[points[:, 0].astype(int), points[:, 1].astype(int)] = 255
        save_image(output / "removed_isolated_pieces.png", removed_image)
        del removed_image
    kept_skeleton = np.zeros_like(skeleton, np.uint8)
    for _, points in dense:
        kept_skeleton[points[:, 0].astype(int), points[:, 1].astype(int)] = 255
    save_image(output / "skeleton.png", kept_skeleton)
    del skeleton, kept_skeleton
    simplified = [(crack_id, rdp(points, args.epsilon)) for crack_id, points in dense]
    timings["vectorization"] = time.time() - step

    dense_points = write_vertex_csv(output / "skeleton_coordinates.csv", dense, height, args.scale)
    rdp_points = write_vertex_csv(output / "crack_polylines_rdp.csv", simplified, height, args.scale)
    write_summary_csv(output / "crack_summary.csv", dense, simplified, height, args.scale)
    save_preview(output / "preview.jpg", rgb, simplified)
    total_length = sum(polyline_length(points) for _, points in dense)
    cracks = len({crack_id for crack_id, _ in dense})
    timings["total"] = time.time() - started

    summary = {
        "image": str(image_path.resolve()), "model": str(args.model.resolve()),
        "model_epoch": checkpoint.get("epoch"), "device": device.type, "threshold": threshold, "tta": args.tta,
        "tiling": {key: value for key, value in plan.items() if key not in ("y_starts", "x_starts")},
        "closing_kernel": args.closing_kernel, "spur_length_px": args.spur_length,
        "min_crack_length_px": args.min_crack_length, "rdp_epsilon_px": args.epsilon,
        "straight_line_length_px": args.straight_line_length, "straight_line_segments": line_segments,
        "straight_segment_length_px": args.straight_segment_length,
        "straight_line_skeleton_px_removed": removed_line_pixels,
        "ruled_lines_share_of_image": round(ruled_share, 5),
        "isolated_crack_length_px": isolated_length, "isolated_distance_px": args.isolated_distance,
        "isolated_polylines_removed": len(isolated), "isolated_skeleton_px_removed": isolated_pixels,
        "cad_units_per_pixel": args.scale, "coordinate_system": "CAD: origin bottom-left, Y up",
        "cracks": cracks, "polylines": len(dense), "small_components_dropped": dropped,
        "total_crack_length_px": round(total_length, 1), "total_crack_length_units": round(total_length * args.scale, 4),
        "skeleton_points": dense_points, "rdp_points": rdp_points,
        "point_reduction_percent": round(100 * (1 - rdp_points / dense_points), 2) if dense_points else 0.0,
        "seconds": {key: round(value, 1) for key, value in timings.items()},
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Cracks: {cracks}  polylines: {len(dense)}  total length: {total_length:,.0f} px "
          f"({total_length * args.scale:,.2f} units); {dropped} tiny fragments dropped")
    print(f"RDP (epsilon {args.epsilon} px): {dense_points:,} skeleton points -> {rdp_points:,} vertices "
          f"({summary['point_reduction_percent']}% fewer)")
    print(f"Done in {timings['total']:.0f}s. Results: {output.resolve()}")
    return output


if __name__ == "__main__":
    arguments = parse_args()
    results = main(arguments)
    if not arguments.no_autocad:
        # Hand over only from the command line: cracks_to_autocad.main() calls
        # main() above for detection, and must not be sent back to AutoCAD.
        from cracks_to_autocad import draw_in_autocad

        draw_in_autocad(arguments.image, results, arguments)
