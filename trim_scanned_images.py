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

    threshold_count = int(total * (percentile / 100.0))
    running = 0
    for value, count in enumerate(hist):
        running += count
        if running >= threshold_count:
            return value
    return 255


def detect_trim_box(image: Image.Image, margin: int = 2) -> tuple[int, int, int, int]:
    """Return PIL crop box (left, top, right, bottom) for visible scanned region."""
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()

    hist = gray.histogram()
    p5 = percentile_from_histogram(hist, 5)
    p95 = percentile_from_histogram(hist, 95)
    threshold = min(250, max(15, (p5 + p95) // 2))

    min_x, min_y = width, height
    max_x, max_y = -1, -1

    for y in range(height):
        for x in range(width):
            if pixels[x, y] < threshold:
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y

    if max_x == -1 or max_y == -1:
        return (0, 0, width, height)

    left = max(0, min_x - margin)
    top = max(0, min_y - margin)
    right = min(width - 1, max_x + margin)
    bottom = min(height - 1, max_y + margin)

    # PIL crop uses exclusive right/bottom bounds.
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
