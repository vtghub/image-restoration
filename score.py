from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch

from restoration.io import image_files, read_grayscale


def main() -> None:
    parser = argparse.ArgumentParser(description="Score restored images against paired clean ground truth.")
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("artifacts/metrics.json"))
    parser.add_argument("--lpips", action="store_true", help="Also compute LPIPS; requires the lpips package and pretrained metric weights.")
    args = parser.parse_args()
    metric = None
    if args.lpips:
        try:
            import lpips
        except ImportError as error:
            raise SystemExit("LPIPS requested but unavailable. Install requirements.txt first.") from error
        metric = lpips.LPIPS(net="alex").eval()
    scores = []
    for prediction_path in image_files(args.prediction_dir):
        relative = prediction_path.relative_to(args.prediction_dir)
        target_path = args.ground_truth_dir / relative
        if not target_path.exists():
            continue
        prediction, target = read_grayscale(prediction_path), read_grayscale(target_path)
        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch for {relative}: {prediction.shape} vs {target.shape}")
        data_range = max(float(target.max() - target.min()), 1e-6)
        row = {"file": str(relative), "psnr": peak_signal_noise_ratio(target, prediction, data_range=data_range), "ssim": structural_similarity(target, prediction, data_range=data_range)}
        if metric is not None:
            pred_tensor = torch.from_numpy(prediction).float()[None, None].repeat(1, 3, 1, 1) * 2 - 1
            target_tensor = torch.from_numpy(target).float()[None, None].repeat(1, 3, 1, 1) * 2 - 1
            with torch.inference_mode():
                row["lpips"] = float(metric(pred_tensor, target_tensor).item())
        scores.append(row)
    if not scores:
        raise ValueError("No matched prediction/ground-truth files found.")
    summary = {"images": len(scores), "mean_psnr": float(np.mean([row["psnr"] for row in scores])), "mean_ssim": float(np.mean([row["ssim"] for row in scores])), "per_image": scores}
    if metric is not None:
        summary["mean_lpips"] = float(np.mean([row["lpips"] for row in scores]))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
