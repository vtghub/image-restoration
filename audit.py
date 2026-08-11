from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from restoration.data import discover_pairs
from restoration.io import read_grayscale


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit paired 2x semiconductor restoration images.")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing degraded/ and ground_truth/.")
    parser.add_argument("--report", type=Path, default=Path("artifacts/audit.json"))
    args = parser.parse_args()
    pairs, problems = discover_pairs(args.data_root)
    shapes, low_ranges, high_ranges, scale_failures = [], [], [], []
    for pair in pairs:
        low, high = read_grayscale(pair.degraded), read_grayscale(pair.target)
        shapes.append({"key": pair.key, "degraded": list(low.shape), "ground_truth": list(high.shape)})
        low_ranges.append([float(low.min()), float(low.max())])
        high_ranges.append([float(high.min()), float(high.max())])
        if high.shape != (low.shape[0] * 2, low.shape[1] * 2):
            scale_failures.append(pair.key)
    payload = {
        "pairs": len(pairs), "pairing_problems": problems, "scale_failures": scale_failures,
        "degraded_global_range": [min((x[0] for x in low_ranges), default=None), max((x[1] for x in low_ranges), default=None)],
        "ground_truth_global_range": [min((x[0] for x in high_ranges), default=None), max((x[1] for x in high_ranges), default=None)],
        "unique_degraded_shapes": sorted({tuple(item["degraded"]) for item in shapes}),
        "unique_ground_truth_shapes": sorted({tuple(item["ground_truth"]) for item in shapes}),
        "sample_shapes": shapes[:20],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if problems or scale_failures or not pairs:
        raise SystemExit("Audit failed: fix the reported data issues before training.")


if __name__ == "__main__":
    main()

