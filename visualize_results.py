from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from restoration.io import read_grayscale


def as_image(array: np.ndarray, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L")
    return image.resize(size, Image.Resampling.BICUBIC) if size else image


def main() -> None:
    parser = argparse.ArgumentParser(description="Create degraded/restored/ground-truth comparison panels.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--keys-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()

    keys = [line.strip() for line in args.keys_file.read_text(encoding="utf-8").splitlines() if line.strip()][:args.count]
    if not keys:
        raise ValueError("No keys supplied.")
    panels = []
    for key in keys:
        input_path = next((args.input_dir / f"{key}{suffix}" for suffix in (".npy", ".png", ".tif", ".tiff")), None)
        prediction_path = next((args.prediction_dir / f"{key}{suffix}" for suffix in (".npy", ".png", ".tif", ".tiff")), None)
        target_path = next((args.ground_truth_dir / f"{key}{suffix}" for suffix in (".npy", ".png", ".tif", ".tiff")), None)
        if not input_path or not prediction_path or not target_path or not input_path.exists() or not prediction_path.exists() or not target_path.exists():
            continue
        target = read_grayscale(target_path)
        target_size = (target.shape[1], target.shape[0])
        panels.append((key, as_image(read_grayscale(input_path), target_size), as_image(read_grayscale(prediction_path)), as_image(target)))
    if not panels:
        raise ValueError("No complete input/prediction/ground-truth triplets found.")

    width, height = panels[0][1].size
    header, gutter = 22, 8
    canvas = Image.new("L", (3 * width + 4 * gutter, len(panels) * (height + header + gutter) + gutter), color=255)
    draw = ImageDraw.Draw(canvas)
    for row, (key, degraded, restored, target) in enumerate(panels):
        y = gutter + row * (height + header + gutter)
        for col, (label, image) in enumerate((("Degraded", degraded), ("Restored", restored), ("Ground truth", target))):
            x = gutter + col * (width + gutter)
            draw.text((x, y), f"{key} — {label}", fill=0)
            canvas.paste(image, (x, y + header))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Wrote {args.output} with {len(panels)} comparison rows.")


if __name__ == "__main__":
    main()

