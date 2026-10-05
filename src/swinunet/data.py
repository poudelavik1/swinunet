'Data for hairline crack segmentation.'

from __future__ import annotations
from .common import Dataset, IMAGE_EXTENSIONS, Path, cv2, dataclass, defaultdict, math, np, random, re, torch

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

__all__ = ['normalized_stem', 'tile_coordinate_key', 'unique_map', 'collect_pairs', 'source_family', 'image_shape', 'read_pair', 'AugmentationOptions', 'rescale', 'pad_to_at_least', 'random_crop', 'dihedral', 'random_affine', 'elastic', 'local_background', 'fade_cracks', 'add_synthetic_hairline', 'cutout', 'motion_blur', 'adjust_contrast', 'random_contrast_factor', 'contrast_level_factors', 'photometric_augment', 'CrackSegmentationDataset', 'shape_grouped_batches']
