from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from restoration.data import robust_normalize
from restoration.io import image_files, read_grayscale, write_grayscale
from restoration.model import JointRestorationNet


def output_path(source: Path, input_root: Path, output_root: Path) -> Path:
    relative = source.relative_to(input_root)
    if relative.suffix.lower() in {".jpg", ".jpeg"}:
        relative = relative.with_suffix(".png")
    return output_root / relative


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore all degraded images in a directory.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--keys-file", type=Path, help="Optional newline-delimited relative filename stems to restore.")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model = JointRestorationNet(**checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    files = image_files(args.input_dir)
    if args.keys_file:
        keys = {line.strip() for line in args.keys_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        files = [path for path in files if str(path.relative_to(args.input_dir).with_suffix("")).replace("\\", "/") in keys]
    if not files:
        raise FileNotFoundError(f"No supported image files found in {args.input_dir}")
    elapsed = []
    with torch.inference_mode():
        for index, path in enumerate(files):
            image = torch.from_numpy(read_grayscale(path)).unsqueeze(0)
            normalized, center, scale = robust_normalize(image)
            batch = normalized.unsqueeze(0).to(device)
            if index == 0:
                for _ in range(args.warmup):
                    _ = model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            start = time.perf_counter()
            restored = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter() - start)
            restored = (restored.squeeze(0).cpu() * scale + center).squeeze(0).numpy()
            write_grayscale(output_path(path, args.input_dir, args.output_dir), restored)
    print(f"Restored {len(files)} images. Mean model inference: {np.mean(elapsed) * 1000:.2f} ms/image")


if __name__ == "__main__":
    main()
