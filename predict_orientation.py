from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from orientation_core.config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_INPUT_SIZE,
    FIRST_TRAIN_AT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RETRAIN_EVERY,
)
from orientation_core.metrics import vector_to_angle
from orientation_core.model import load_orientation_checkpoint
from orientation_core.sample_store import SampleStore


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def milestone_due(count: int) -> bool:
    return count >= FIRST_TRAIN_AT and (count - FIRST_TRAIN_AT) % RETRAIN_EVERY == 0


def maybe_auto_train(dataset: Path, checkpoints: Path, count: int):
    if not milestone_due(count):
        return
    milestone = checkpoints / f"orientation-{count}.pt"
    if milestone.exists():
        return

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "train_orientation.py"),
            "--dataset",
            str(dataset),
            "--checkpoints",
            str(checkpoints),
        ],
        check=True,
    )


def preprocess(image: Image.Image):
    tfm = transforms.Compose([
        transforms.Resize((DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tfm(image.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def predict(image_path: Path, model_path: Path):
    dev = device()
    model, checkpoint = load_orientation_checkpoint(model_path, dev)
    x = preprocess(Image.open(image_path)).to(dev)
    y = model(x)[0].detach().cpu()
    angle = vector_to_angle(float(y[0]), float(y[1]))
    return angle, checkpoint


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect")
    collect.add_argument("--image", required=True)
    collect.add_argument("--upright", default="")
    collect.add_argument("--angle", type=float, required=True)
    collect.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR))
    collect.add_argument("--checkpoints", default=str(DEFAULT_CHECKPOINT_DIR))
    collect.add_argument("--auto-train", action="store_true")
    collect.add_argument("--metadata-json", default="")

    pred = sub.add_parser("predict")
    pred.add_argument("--image", required=True)
    pred.add_argument("--model", default=str(DEFAULT_CHECKPOINT_DIR / "best.pt"))
    pred.add_argument("--save-corrected", default="")

    status = sub.add_parser("status")
    status.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR))

    args = ap.parse_args()

    if args.command == "collect":
        metadata = {}
        if args.metadata_json:
            metadata = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))

        store = SampleStore(args.dataset)
        rec = store.save(
            initial_image=args.image,
            upright_image=args.upright or None,
            correction_angle=args.angle,
            metadata=metadata or None,
        )
        count = store.count()
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        print(f"sample_count={count}")

        if args.auto_train:
            maybe_auto_train(Path(args.dataset), Path(args.checkpoints), count)

    elif args.command == "predict":
        angle, checkpoint = predict(Path(args.image), Path(args.model))
        print(f"predicted_correction_angle={angle:.3f}")
        print(f"checkpoint_samples={checkpoint.get('sample_count', 'unknown')}")

        if args.save_corrected:
            image = Image.open(args.image).convert("RGB")
            corrected = image.rotate(
                angle,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=(127, 127, 127),
            )
            out = Path(args.save_corrected)
            out.parent.mkdir(parents=True, exist_ok=True)
            corrected.save(out)
            print(f"saved={out}")

    elif args.command == "status":
        store = SampleStore(args.dataset)
        count = store.count()
        print(f"sample_count={count}")
        print(f"first_train_at={FIRST_TRAIN_AT}")
        if count >= FIRST_TRAIN_AT:
            remainder = (count - FIRST_TRAIN_AT) % RETRAIN_EVERY
            next_count = count if remainder == 0 else count + (RETRAIN_EVERY - remainder)
            print(f"next_milestone={next_count}")


if __name__ == "__main__":
    main()
