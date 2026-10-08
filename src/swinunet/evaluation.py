'Evaluation for hairline crack segmentation.'

from __future__ import annotations
from .common import COUNT_NAMES, F, METRIC_NAMES, Path, cv2, defaultdict, np, torch

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


def spread_over_sources(domain_ids, limit: int) -> set[int]:
    """Up to `limit` dataset indices, shared evenly between the sources and evenly spaced within each.

    The evaluation loader groups images by size, so the first `limit` images all
    come from one source (40 of 40 overlays were EH104 tiles).
    """
    members = defaultdict(list)
    for index, domain in enumerate(domain_ids):
        members[domain].append(index)
    quotas = dict.fromkeys(members, 0)
    for _ in range(limit):  # one more for the source with the fewest so far, while it has images left
        unfilled = [domain for domain in members if quotas[domain] < len(members[domain])]
        if not unfilled:
            break
        quotas[min(unfilled, key=lambda domain: (quotas[domain], domain))] += 1
    return {members[domain][position * len(members[domain]) // quota]
            for domain, quota in quotas.items() for position in range(quota)}


def prediction_writer(dataset, directory: Path, limit: int, threshold: float):
    """Save probability maps and overlays: yellow = hit, red = false positive, green = missed.

    `limit` images are saved, spread over every source; a limit of at least the
    dataset size saves them all.
    """
    directory.mkdir(parents=True, exist_ok=True)
    selected = spread_over_sources(dataset.domain_ids, limit)

    def write(images, probabilities, masks, indices):
        for image, probability, mask, index in zip(images, probabilities, masks, indices.tolist()):
            if index not in selected:
                continue
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

    return write


# --------------------------------------------------------------------------- #
# Tables and figures: loss/metric curves, threshold sweep, PR, ROC, confusion matrix
# --------------------------------------------------------------------------- #

__all__ = ['ThresholdSweep', 'metrics_from_counts', 'basic_metrics', 'confusion_from_counts', 'at_threshold', 'tile_origins', 'predict_logits', 'predict_probabilities', 'evaluate', 'spread_over_sources', 'prediction_writer']
