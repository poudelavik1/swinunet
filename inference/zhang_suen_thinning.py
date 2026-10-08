"""Zhang-Suen skeletonization for an already segmented binary image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def zhang_suen(binary: np.ndarray, max_iterations: int | None = None) -> tuple[np.ndarray, int]:
    """Thin a 2-D boolean/0-1 image and return (skeleton, iterations)."""
    image = (binary > 0).astype(np.uint8)
    image[[0, -1], :] = 0
    image[:, [0, -1]] = 0
    iteration = 0

    while True:
        changed = False

        for step in (0, 1):
            p2 = image[:-2, 1:-1]
            p3 = image[:-2, 2:]
            p4 = image[1:-1, 2:]
            p5 = image[2:, 2:]
            p6 = image[2:, 1:-1]
            p7 = image[2:, :-2]
            p8 = image[1:-1, :-2]
            p9 = image[:-2, :-2]
            center = image[1:-1, 1:-1]

            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1))
                + ((p4 == 0) & (p5 == 1))
                + ((p5 == 0) & (p6 == 1))
                + ((p6 == 0) & (p7 == 1))
                + ((p7 == 0) & (p8 == 1))
                + ((p8 == 0) & (p9 == 1))
                + ((p9 == 0) & (p2 == 1))
            )

            remove = (center == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1)
            if step == 0:
                remove &= (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                remove &= (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

            if np.any(remove):
                center[remove] = 0
                changed = True

        iteration += 1
        if not changed or (max_iterations is not None and iteration >= max_iterations):
            break

    return image.astype(bool), iteration


def load_segmented(path: Path, threshold: int, foreground: str) -> tuple[np.ndarray, bool]:
    gray = np.asarray(Image.open(path).convert("L"))
    light_pixels = gray >= threshold

    if foreground == "light":
        crack = light_pixels
        output_white = True
    elif foreground == "dark":
        crack = ~light_pixels
        output_white = False
    else:
        # Cracks normally occupy less area than the background in a mask.
        crack_is_light = light_pixels.mean() <= 0.5
        crack = light_pixels if crack_is_light else ~light_pixels
        output_white = crack_is_light

    return crack, output_white


def save_skeleton(skeleton: np.ndarray, path: Path, white_foreground: bool) -> None:
    if white_foreground:
        pixels = np.where(skeleton, 255, 0)
    else:
        pixels = np.where(skeleton, 0, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8)).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Zhang-Suen thinning to a segmented crack mask."
    )
    parser.add_argument("image", type=Path, help="Input segmented/binary image")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG (default: <input>_skeleton.png)",
    )
    parser.add_argument("--threshold", type=int, default=128, help="Binary threshold (default: 128)")
    parser.add_argument(
        "--foreground",
        choices=("auto", "light", "dark"),
        default="auto",
        help="Crack color in the input (default: auto/minority class)",
    )
    parser.add_argument("--max-iterations", type=int, help="Optional iteration limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Input image not found: {args.image}")
    if not 0 <= args.threshold <= 255:
        raise SystemExit("Threshold must be between 0 and 255.")
    if args.max_iterations is not None and args.max_iterations < 1:
        raise SystemExit("Maximum iterations must be at least 1.")

    output = args.output or args.image.with_name(f"{args.image.stem}_skeleton.png")
    segmented, white_foreground = load_segmented(args.image, args.threshold, args.foreground)
    skeleton, iterations = zhang_suen(segmented, args.max_iterations)
    save_skeleton(skeleton, output, white_foreground)

    print(f"Skeleton saved to: {output.resolve()}")
    print(f"Iterations: {iterations}")
    print(f"Skeleton pixels: {int(skeleton.sum())}")


if __name__ == "__main__":
    main()
