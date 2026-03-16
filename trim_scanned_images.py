#!/usr/bin/env python3
"""Scan a directory for images, auto-detect rectangular borders, and save trimmed copies.

Usage:
  python trim_scanned_images.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def detect_trim_box(image: Image.Image, margin: int = 2) -> tuple[int, int, int, int]:
    """Return a crop box (left, top, right, bottom) for the visible scanned region.

    The algorithm targets dark borders/blank margins often seen in scanned images.
    It does not resize pixels; only computes a rectangle to crop.
    """

    gray = np.array(image.convert("L"), dtype=np.uint8)

    # Robust threshold from image statistics (works for common scanner borders).
    p5 = int(np.percentile(gray, 5))
    p95 = int(np.percentile(gray, 95))
    threshold = min(250, max(15, int((p5 + p95) / 2)))

    # Keep non-background content (typically darker than white page/background).
    mask = gray < threshold

    coords = np.argwhere(mask)
    if coords.size == 0:
        # If nothing detected, keep original.
        return (0, 0, image.width, image.height)

    top, left = coords.min(axis=0)
    bottom, right = coords.max(axis=0)

    left = max(0, int(left) - margin)
    top = max(0, int(top) - margin)
    right = min(image.width - 1, int(right) + margin)
    bottom = min(image.height - 1, int(bottom) + margin)

    # PIL crop uses exclusive right/bottom bounds.
    return (left, top, right + 1, bottom + 1)


def process_image(src: Path, dst: Path) -> None:
    with Image.open(src) as img:
        box = detect_trim_box(img)
        cropped = img.crop(box)

        # Preserve common metadata when possible (e.g., DPI/ICC).
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
    input_dir = (base_dir / "image").resolve()
    output_dir = (base_dir / "trim").resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    images = list(iter_images(input_dir))
    if not images:
        print("No image files found.")
        return

    for src in images:
        # Avoid reprocessing outputs if output is inside input dir.
        if output_dir in src.parents:
            continue

        dst = output_dir / src.name
        process_image(src, dst)
        print(f"Trimmed: {src.name} -> {dst}")


if __name__ == "__main__":
    main()
