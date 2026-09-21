from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .config import DEFAULT_DATASET_DIR


class SampleStore:
    def __init__(self, root: str | Path = DEFAULT_DATASET_DIR):
        self.root = Path(root)
        self.samples_dir = self.root / "samples"
        self.labels_path = self.root / "labels.jsonl"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def count(self) -> int:
        if not self.labels_path.exists():
            return 0
        with self.labels_path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def next_sample_id(self) -> int:
        return self.count() + 1

    def save(
        self,
        initial_image: str | Path,
        correction_angle: float,
        upright_image: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        angle = float(correction_angle) % 360.0
        sample_id = self.next_sample_id()
        sample_name = f"{sample_id:06d}"
        sample_dir = self.samples_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=False)

        initial_dst = sample_dir / "initial.png"
        upright_dst = sample_dir / "upright.png"

        initial = Image.open(initial_image).convert("RGB")
        initial.save(initial_dst)

        if upright_image:
            Image.open(upright_image).convert("RGB").save(upright_dst)
        else:
            initial.rotate(
                angle,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=(127, 127, 127),
            ).save(upright_dst)

        record: dict[str, Any] = {
            "sample_id": sample_id,
            "sample_name": sample_name,
            "initial_image": str(initial_dst.relative_to(self.root)).replace("\\", "/"),
            "upright_image": str(upright_dst.relative_to(self.root)).replace("\\", "/"),
            "correction_angle": angle,
            "human_confirmed_upright": True,
            "created_at": time.time(),
        }
        if metadata:
            record["interaction"] = metadata

        with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        with self.labels_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record
