from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


OFFICIAL_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6?usp=drive_link"


def has_layout(path: Path) -> bool:
    return (path / "NoisyLR").is_dir() and (path / "GT").is_dir()


def locate_layout(root: Path) -> Path | None:
    candidates = [root, *[path for path in root.rglob("*") if path.is_dir()]]
    return next((path for path in candidates if has_layout(path)), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the public official paired-training archive for a Kaggle notebook.")
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/data"))
    parser.add_argument("--drive-folder", default=OFFICIAL_DRIVE_FOLDER)
    args = parser.parse_args()

    existing = locate_layout(args.output_root)
    if existing is not None:
        print(existing)
        return

    try:
        import gdown
    except ImportError as error:
        raise SystemExit("gdown is required. Run `pip install gdown` first.") from error

    downloads = args.output_root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(url=args.drive_folder, output=str(downloads), quiet=False, remaining_ok=True)
    archives = sorted(downloads.rglob("train.zip"), key=lambda path: path.stat().st_size, reverse=True)
    if not archives:
        raise FileNotFoundError("Could not find train.zip after downloading the official Drive folder.")
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(args.output_root)

    extracted = locate_layout(args.output_root)
    if extracted is None:
        raise FileNotFoundError("The extracted archive did not contain a NoisyLR/GT paired-training layout.")
    target = args.output_root / "train"
    if extracted != target:
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(extracted), str(target))
    print(target)


if __name__ == "__main__":
    main()
