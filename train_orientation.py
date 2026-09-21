from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from orientation_core.config import (
    DEFAULT_BACKBONE,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATASET_DIR,
    FIRST_TRAIN_AT,
)
from orientation_core.dataset import OrientationDataset, load_records, split_records
from orientation_core.metrics import batch_circular_error_deg, summarize_errors
from orientation_core.model import OrientationModel


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    count = 0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(images)
        loss = (1.0 - F.cosine_similarity(pred, targets, dim=1)).mean()
        loss.backward()
        optimizer.step()

        total += float(loss.item()) * images.size(0)
        count += images.size(0)
    return total / max(1, count)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    errors = []
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        pred = model(images)
        batch_errors = batch_circular_error_deg(pred, targets)
        errors.extend(batch_errors.detach().cpu().tolist())
    return summarize_errors(errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--checkpoints", default=str(DEFAULT_CHECKPOINT_DIR))
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=["resnet18", "resnet50"])
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-samples", type=int, default=FIRST_TRAIN_AT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dataset_root = Path(args.dataset)
    ckpt_dir = Path(args.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(dataset_root)
    if len(records) < args.min_samples and not args.force:
        raise SystemExit(
            f"Only {len(records)} samples. Need at least {args.min_samples}. "
            "Use --force only for local debugging."
        )

    train_records, val_records = split_records(records, args.val_ratio, args.seed)
    train_ds = OrientationDataset(dataset_root, train_records, training=True)
    val_ds = OrientationDataset(dataset_root, val_records, training=False)

    device = get_device()
    model = OrientationModel(args.backbone, pretrained=args.pretrained).to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs) if val_records else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_score = math.inf
    best_payload = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device) if val_loader else {"mae": None}
        mae = metrics.get("mae")
        score = mae if mae is not None else train_loss

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_mae={mae if mae is not None else 'n/a'}"
        )

        if score < best_score:
            best_score = score
            best_payload = {
                "model_state_dict": model.state_dict(),
                "backbone": args.backbone,
                "sample_count": len(records),
                "epoch": epoch,
                "metrics": metrics,
            }

    milestone_path = ckpt_dir / f"orientation-{len(records)}.pt"
    best_path = ckpt_dir / "best.pt"
    torch.save(best_payload, milestone_path)
    torch.save(best_payload, best_path)

    summary = {
        "sample_count": len(records),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "device": str(device),
        "checkpoint": str(milestone_path),
        "best": str(best_path),
        "metrics": best_payload["metrics"],
    }
    dataset_root.mkdir(parents=True, exist_ok=True)
    with (dataset_root / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
