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


def _edge_white_ratio_row(
    pixels, left: int, right: int, y: int, white_threshold: int
) -> float:
    width = right - left + 1
    white_count = 0
    for x in range(left, right + 1):
        if pixels[x, y] >= white_threshold:
            white_count += 1
    return white_count / width


def _edge_white_ratio_col(
    pixels, top: int, bottom: int, x: int, white_threshold: int
) -> float:
    height = bottom - top + 1
    white_count = 0
    for y in range(top, bottom + 1):
        if pixels[x, y] >= white_threshold:
            white_count += 1
    return white_count / height


def _shrink_white_edges(
    gray: Image.Image, left: int, top: int, right: int, bottom: int, white_threshold: int
) -> tuple[int, int, int, int]:
    """Remove remaining 1px-level white lines from all edges."""
    pixels = gray.load()
    white_ratio_threshold = 0.995

    while left < right and top < bottom:
        changed = False

        width = right - left + 1
        height = bottom - top + 1

        if height > 2 and _edge_white_ratio_row(pixels, left, right, top, white_threshold) >= white_ratio_threshold:
            top += 1
            changed = True

        if height > 2 and _edge_white_ratio_row(pixels, left, right, bottom, white_threshold) >= white_ratio_threshold:
            bottom -= 1
            changed = True

        if width > 2 and _edge_white_ratio_col(pixels, top, bottom, left, white_threshold) >= white_ratio_threshold:
            left += 1
            changed = True

        if width > 2 and _edge_white_ratio_col(pixels, top, bottom, right, white_threshold) >= white_ratio_threshold:
            right -= 1
            changed = True

        if not changed:
            break

    return left, top, right, bottom


def detect_trim_box(image: Image.Image) -> tuple[int, int, int, int]:
    """Return PIL crop box (left, top, right, bottom) for visible scanned/photo region.

    1) Find initial box by scanning inward for non-white content.
    2) Repeatedly shave off near-all-white outer lines to eliminate residual white edges.
    """
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()

    hist = gray.histogram()
    p98 = percentile_from_histogram(hist, 98)
    p90 = percentile_from_histogram(hist, 90)
    darkness_threshold = min(250, max(200, (p90 + p98) // 2))

    # For edge cleanup, treat very bright pixels as white.
    p99 = percentile_from_histogram(hist, 99)
    white_threshold = max(245, p99 - 2)

    min_ratio = 0.005

    top = _first_content_row(pixels, width, 0, height, 1, darkness_threshold, min_ratio)
    bottom = _first_content_row(pixels, width, height - 1, -1, -1, darkness_threshold, min_ratio)
    left = _first_content_col(pixels, height, 0, width, 1, darkness_threshold, min_ratio)
    right = _first_content_col(pixels, height, width - 1, -1, -1, darkness_threshold, min_ratio)

    if min(top, bottom, left, right) < 0 or left >= right or top >= bottom:
        return (0, 0, width, height)

    left, top, right, bottom = _shrink_white_edges(
        gray, left=left, top=top, right=right, bottom=bottom, white_threshold=white_threshold
    )

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
