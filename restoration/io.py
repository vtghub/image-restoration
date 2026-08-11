from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def read_grayscale(path: str | Path) -> np.ndarray:
    """Read an image into float32 nominal intensity units without percentile clipping."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3].astype(np.float32).mean(axis=-1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got {array.shape} in {path}")
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        array = array.astype(np.float32) / float(info.max)
    else:
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite intensity values in {path}")
    return array


def write_grayscale(path: str | Path, image: np.ndarray) -> None:
    """Write a restored nominal-range image as an 8-bit grayscale PNG/TIFF/etc."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        np.save(path, image.astype(np.float32))
        return
    encoded = np.clip(image, 0.0, 1.0)
    Image.fromarray((encoded * 65535.0 + 0.5).astype(np.uint16), mode="I;16").save(path)

