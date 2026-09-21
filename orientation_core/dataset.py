from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from .config import DEFAULT_INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD
from .metrics import angle_to_vector


def load_records(dataset_root: str | Path) -> list[dict]:
    root = Path(dataset_root)
    labels = root / "labels.jsonl"
    if not labels.exists():
        return []

    records = []
    with labels.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("human_confirmed_upright") is True:
                records.append(item)
    return records


def split_records(records: list[dict], val_ratio: float = 0.2, seed: int = 42):
    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)

    if len(records) <= 1:
        return records, []

    val_count = max(1, int(round(len(records) * val_ratio)))
    val_count = min(val_count, len(records) - 1)
    return records[val_count:], records[:val_count]


class OrientationDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        records: list[dict],
        training: bool,
        input_size: int = DEFAULT_INPUT_SIZE,
        use_upright_synthetic: bool = True,
    ):
        self.root = Path(dataset_root)
        self.records = list(records)
        self.training = training
        self.input_size = input_size
        self.use_upright_synthetic = use_upright_synthetic
        self.to_tensor = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.records)

    def _augment_visual(self, image: Image.Image) -> Image.Image:
        if not self.training:
            return image
        if random.random() < 0.35:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.35:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.8)))
        return image

    def __getitem__(self, index):
        rec = self.records[index]
        synthetic = self.training and self.use_upright_synthetic and random.random() < 0.5

        if synthetic:
            image = Image.open(self.root / rec["upright_image"]).convert("RGB")
            extra = random.uniform(0.0, 360.0)
            image = image.rotate(
                extra,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=(127, 127, 127),
            )
            correction = (-extra) % 360.0
        else:
            image = Image.open(self.root / rec["initial_image"]).convert("RGB")
            correction = float(rec["correction_angle"]) % 360.0
            if self.training:
                extra = random.uniform(-180.0, 180.0)
                image = image.rotate(
                    extra,
                    resample=Image.Resampling.BICUBIC,
                    expand=False,
                    fillcolor=(127, 127, 127),
                )
                correction = (correction - extra) % 360.0

        image = ImageOps.exif_transpose(image)
        image = self._augment_visual(image)
        sin_v, cos_v = angle_to_vector(correction)
        target = torch.tensor([sin_v, cos_v], dtype=torch.float32)
        return self.to_tensor(image), target, int(rec["sample_id"])
