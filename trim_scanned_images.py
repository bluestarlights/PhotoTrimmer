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
from statistics import median
from typing import Iterable

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def _sample_border_pixels(rgb: Image.Image, strip: int = 8) -> list[tuple[int, int, int]]:
    """Collect RGB pixels from image borders to estimate paper/background color."""
    width, height = rgb.size
    px = rgb.load()
    strip = max(1, min(strip, width // 4 if width >= 4 else 1, height // 4 if height >= 4 else 1))

    samples: list[tuple[int, int, int]] = []

    # Top / bottom strips
    for y in range(strip):
        for x in range(width):
            samples.append(px[x, y])
            samples.append(px[x, height - 1 - y])

    # Left / right strips
    for x in range(strip):
        for y in range(height):
            samples.append(px[x, y])
            samples.append(px[width - 1 - x, y])

    return samples


def _estimate_background_color(rgb: Image.Image) -> tuple[int, int, int]:
    """Estimate dominant border background color via per-channel median."""
    border_pixels = _sample_border_pixels(rgb)
    r = int(median(p[0] for p in border_pixels))
    g = int(median(p[1] for p in border_pixels))
    b = int(median(p[2] for p in border_pixels))
    return r, g, b


def _content_bbox_from_background(rgb: Image.Image, bg: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    """Return bounding box of non-background pixels using color-distance threshold."""
    width, height = rgb.size
    px = rgb.load()

    # Adaptive threshold: higher for bright paper scans, lower for darker backgrounds.
    bg_luma = (bg[0] * 299 + bg[1] * 587 + bg[2] * 114) // 1000
    tolerance = 18 if bg_luma < 200 else 24

    left, top = width, height
    right, bottom = -1, -1

    for y in range(height):
        for x in range(width):
            pr, pg, pb = px[x, y]
            dist = abs(pr - bg[0]) + abs(pg - bg[1]) + abs(pb - bg[2])
            if dist > tolerance:
                if x < left:
                    left = x
                if y < top:
                    top = y
                if x > right:
                    right = x
                if y > bottom:
                    bottom = y

    if right < left or bottom < top:
        return None
    return left, top, right, bottom


def _trim_residual_white_edges(rgb: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Iteratively shave off mostly-white border rows/cols from a candidate crop box."""
    px = rgb.load()
    left, top, right, bottom = box

    def row_white_ratio(y: int, l: int, r: int) -> float:
        white = 0
        total = r - l + 1
        for x in range(l, r + 1):
            rr, gg, bb = px[x, y]
            if rr >= 245 and gg >= 245 and bb >= 245:
                white += 1
        return white / total

    def col_white_ratio(x: int, t: int, b: int) -> float:
        white = 0
        total = b - t + 1
        for y in range(t, b + 1):
            rr, gg, bb = px[x, y]
            if rr >= 245 and gg >= 245 and bb >= 245:
                white += 1
        return white / total

    while left < right and top < bottom:
        changed = False
        # Aggressive values to remove remaining white lines on bottom/right too.
        if row_white_ratio(top, left, right) >= 0.992 and (bottom - top) > 2:
            top += 1
            changed = True
        if row_white_ratio(bottom, left, right) >= 0.992 and (bottom - top) > 2:
            bottom -= 1
            changed = True
        if col_white_ratio(left, top, bottom) >= 0.992 and (right - left) > 2:
            left += 1
            changed = True
        if col_white_ratio(right, top, bottom) >= 0.992 and (right - left) > 2:
            right -= 1
            changed = True

        if not changed:
            break

    return left, top, right, bottom


def detect_trim_box(image: Image.Image) -> tuple[int, int, int, int]:
    """Return PIL crop box (left, top, right, bottom exclusive) for content region.

    Strategy:
    1) Estimate background from border pixels.
    2) Build content bbox from color distance to background.
    3) Remove residual near-white 1px lines from all edges.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size

    bg = _estimate_background_color(rgb)
    initial_box = _content_bbox_from_background(rgb, bg)
    if initial_box is None:
        return (0, 0, width, height)

    left, top, right, bottom = _trim_residual_white_edges(rgb, initial_box)

    if left >= right or top >= bottom:
        return (0, 0, width, height)

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
