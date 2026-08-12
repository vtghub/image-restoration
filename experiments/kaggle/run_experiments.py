from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def resolve_layout(data_root: Path) -> tuple[Path, Path]:
    degraded = next((data_root / name for name in ("NoisyLR", "degraded") if (data_root / name).is_dir()), None)
    target = next((data_root / name for name in ("GT", "ground_truth") if (data_root / name).is_dir()), None)
    if degraded is None or target is None:
        raise FileNotFoundError("Expected NoisyLR/GT or degraded/ground_truth under --data-root.")
    return degraded, target


def gpu_metadata() -> dict[str, object]:
    try:
        import torch

        return {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        return {"torch": None, "cuda_available": False, "cuda_version": None, "device_name": None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeatable Kaggle GPU restoration experiments and produce a scorecard.")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing NoisyLR/GT or degraded/ground_truth.")
    parser.add_argument("--runs-root", type=Path, default=Path("/kaggle/working/runs"))
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiments.json"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true", help="Reuse a complete run directory when it already has metrics.json.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    degraded_root, target_root = resolve_layout(args.data_root)
    configs = json.loads(args.config.read_text(encoding="utf-8"))[args.profile]
    args.runs_root.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "audit.py", "--data-root", str(args.data_root), "--report", str(args.runs_root / "audit.json")], root)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "data_root": str(args.data_root),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": gpu_metadata(),
        "configs": configs,
    }
    (args.runs_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not metadata["gpu"]["cuda_available"]:
        print("Warning: CUDA is not available. Enable a Kaggle P100/T4 accelerator before trusting this run.")

    scorecard = []
    for config in configs:
        name = config["name"]
        run_dir = args.runs_root / name
        metrics_path = run_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            train_command = [
                sys.executable,
                "train.py",
                "--data-root", str(args.data_root),
                "--output-dir", str(run_dir),
                "--epochs", str(config["epochs"]),
                "--batch-size", str(config["batch_size"]),
                "--learning-rate", str(config["learning_rate"]),
                "--synthetic-probability", str(config["synthetic_probability"]),
                "--gradient-weight", str(config["gradient_weight"]),
                "--width", str(config["width"]),
                "--blocks", str(config["blocks"]),
                "--workers", str(args.workers),
                "--seed", str(args.seed),
            ]
            run(train_command, root)
            prediction_dir = run_dir / "validation_predictions"
            run([
                sys.executable, "evaluate.py",
                "--input-dir", str(degraded_root),
                "--output-dir", str(prediction_dir),
                "--weights", str(run_dir / "best.pt"),
                "--keys-file", str(run_dir / "validation_keys.txt"),
            ], root)
            run([
                sys.executable, "score.py",
                "--prediction-dir", str(prediction_dir),
                "--ground-truth-dir", str(target_root),
                "--report", str(metrics_path),
            ], root)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        scorecard.append({
            "name": name,
            "config": config,
            "best_validation_loss": min(row["validation_loss"] for row in history),
            "mean_psnr": metrics["mean_psnr"],
            "mean_ssim": metrics["mean_ssim"],
            "images": metrics["images"],
            "checkpoint": str(run_dir / "best.pt"),
        })

    scorecard.sort(key=lambda row: (row["mean_psnr"], row["mean_ssim"]), reverse=True)
    result = {"profile": args.profile, "baseline": {"mean_psnr": 27.19862, "mean_ssim": 0.7129819}, "ranked_runs": scorecard}
    (args.runs_root / "scoreboard.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
