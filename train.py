from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from restoration.data import PairedRestorationDataset, discover_pairs
from restoration.metrics import restoration_loss
from restoration.model import JointRestorationNet


def split_pairs(pairs, validation_fraction: float, seed: int, split_by_parent: bool):
    if len(pairs) < 2:
        raise ValueError("At least two valid pairs are needed to create a validation split.")
    rng = random.Random(seed)
    if split_by_parent:
        groups: dict[str, list] = {}
        for pair in pairs:
            groups.setdefault(str(Path(pair.key).parent), []).append(pair)
        if len(groups) > 1:
            keys = list(groups)
            rng.shuffle(keys)
            count = max(1, round(len(keys) * validation_fraction))
            val_keys = set(keys[:count])
            validation = [pair for key in val_keys for pair in groups[key]]
            training = [pair for key in groups if key not in val_keys for pair in groups[key]]
            return training, validation
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    count = max(1, round(len(shuffled) * validation_fraction))
    return shuffled[count:], shuffled[:count]


@torch.no_grad()
def validate(model, loader, device, gradient_weight: float):
    model.eval()
    total, count = 0.0, 0
    for low, high, _ in loader:
        low, high = low.to(device), high.to(device)
        total += restoration_loss(model(low), high, gradient_weight).item() * low.shape[0]
        count += low.shape[0]
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the joint 2x semiconductor image-restoration network.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/run"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--synthetic-probability", type=float, default=0.35, help="Chance to replace a paired input with a synthetic degradation.")
    parser.add_argument("--gradient-weight", type=float, default=0.10, help="Weight of Sobel edge-fidelity loss.")
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-pairs", type=int, help="Optional deterministic subset for quick baseline experiments.")
    parser.add_argument("--split-by-parent", action="store_true", help="Hold out source folders when the dataset encodes source in directories.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    pairs, problems = discover_pairs(args.data_root)
    if problems:
        raise ValueError("Data pairing errors:\n" + "\n".join(problems[:20]))
    if args.max_pairs is not None:
        if args.max_pairs < 2:
            raise ValueError("--max-pairs must be at least 2.")
        sample_rng = random.Random(args.seed)
        sample_rng.shuffle(pairs)
        pairs = pairs[:args.max_pairs]
    train_pairs, val_pairs = split_pairs(pairs, args.validation_fraction, args.seed, args.split_by_parent)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_keys.txt").write_text("\n".join(pair.key for pair in train_pairs) + "\n", encoding="utf-8")
    (args.output_dir / "validation_keys.txt").write_text("\n".join(pair.key for pair in val_pairs) + "\n", encoding="utf-8")
    if not 0.0 <= args.synthetic_probability <= 1.0:
        raise ValueError("--synthetic-probability must be between 0 and 1.")
    train_data = PairedRestorationDataset(train_pairs, patch_size=args.patch_size, augment=True, synthetic_probability=args.synthetic_probability)
    val_data = PairedRestorationDataset(val_pairs, patch_size=args.patch_size, augment=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointRestorationNet(width=args.width, blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for low, high, _ in progress:
            low, high = low.to(device, non_blocking=True), high.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = restoration_loss(model(low), high, args.gradient_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * low.shape[0]
            seen += low.shape[0]
            progress.set_postfix(loss=f"{running / seen:.4f}")
        val_loss = validate(model, val_loader, device, args.gradient_weight)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": running / seen, "validation_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"], "synthetic_probability": args.synthetic_probability, "gradient_weight": args.gradient_weight}
        history.append(record)
        print(json.dumps(record))
        checkpoint = {"model": model.state_dict(), "model_config": {"width": args.width, "blocks": args.blocks}, "epoch": epoch, "validation_loss": val_loss}
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(checkpoint, args.output_dir / "best.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
