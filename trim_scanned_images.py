#!/usr/bin/env python3
"""Auto-trim scanned image borders.

- Input:  ./image
- Output: ./trim

Usage:
  python trim_scanned_images.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def percentile_from_histogram(hist: list[int], percentile: float) -> int:
    """Return intensity value for percentile in [0, 100] from 256-bin grayscale histogram."""
    total = sum(hist)
    if total == 0:
        return 0

    threshold_count = max(1, int(total * (percentile / 100.0)))
    running = 0
    for value, count in enumerate(hist):
        running += count
        if running >= threshold_count:
            return value
    return 255


def _first_content_row(
    pixels, width: int, start: int, stop: int, step: int, darkness_threshold: int, min_ratio: float
) -> int:
    min_dark_pixels = max(1, int(width * min_ratio))
    for y in range(start, stop, step):
        dark_count = 0
        for x in range(width):
            if pixels[x, y] < darkness_threshold:
                dark_count += 1
                if dark_count >= min_dark_pixels:
                    return y
    return -1


def _first_content_col(
    pixels, height: int, start: int, stop: int, step: int, darkness_threshold: int, min_ratio: float
) -> int:
    min_dark_pixels = max(1, int(height * min_ratio))
    for x in range(start, stop, step):
        dark_count = 0
        for y in range(height):
            if pixels[x, y] < darkness_threshold:
                dark_count += 1
                if dark_count >= min_dark_pixels:
                    return x
    return -1


def detect_trim_box(image: Image.Image, margin: int = 2) -> tuple[int, int, int, int]:
    """Return PIL crop box (left, top, right, bottom) for visible scanned/photo region.

    This version aggressively removes outer white margins by scanning from each edge and
    selecting the first row/column that has enough non-white pixels.
    """
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()

    hist = gray.histogram()
    # White paper/background is generally concentrated in the high percentiles.
    p98 = percentile_from_histogram(hist, 98)
    p90 = percentile_from_histogram(hist, 90)
    # Pixel values below this threshold are treated as "content" (non-white).
    darkness_threshold = min(250, max(200, (p90 + p98) // 2))

    # At least ~0.7% dark pixels required in an edge row/column to be considered content.
    min_ratio = 0.007

    top = _first_content_row(pixels, width, 0, height, 1, darkness_threshold, min_ratio)
    bottom = _first_content_row(pixels, width, height - 1, -1, -1, darkness_threshold, min_ratio)
    left = _first_content_col(pixels, height, 0, width, 1, darkness_threshold, min_ratio)
    right = _first_content_col(pixels, height, width - 1, -1, -1, darkness_threshold, min_ratio)

    if min(top, bottom, left, right) < 0 or left >= right or top >= bottom:
        return (0, 0, width, height)

    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(width - 1, right + margin)
    bottom = min(height - 1, bottom + margin)

    return (left, top, right + 1, bottom + 1)


def process_image(src: Path, dst: Path) -> None:
    with Image.open(src) as img:
        box = detect_trim_box(img)
        cropped = img.crop(box)

        save_kwargs = {}
        if "dpi" in img.info:
            save_kwargs["dpi"] = img.info["dpi"]
        if "icc_profile" in img.info:
            save_kwargs["icc_profile"] = img.info["icc_profile"]

        dst.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(dst, **save_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-trim rectangular borders for images in ./image and save to ./trim."
    )
    parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "image"
    output_dir = base_dir / "trim"

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    images = list(iter_images(input_dir))
    if not images:
        print("No image files found.")
        return

    for src in images:
        dst = output_dir / src.name
        process_image(src, dst)
        print(f"Trimmed: {src.name} -> {dst}")


if __name__ == "__main__":
    main()
