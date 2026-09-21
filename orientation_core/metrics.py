import math
from typing import Iterable

import torch


def normalize_angle_deg(angle: float) -> float:
    return float(angle) % 360.0


def angle_to_vector(angle_deg: float) -> tuple[float, float]:
    r = math.radians(normalize_angle_deg(angle_deg))
    return math.sin(r), math.cos(r)


def vector_to_angle(sin_value: float, cos_value: float) -> float:
    return math.degrees(math.atan2(sin_value, cos_value)) % 360.0


def circular_error_deg(pred_deg: float, target_deg: float) -> float:
    diff = abs((float(pred_deg) - float(target_deg)) % 360.0)
    return min(diff, 360.0 - diff)


def batch_circular_error_deg(pred_vectors: torch.Tensor, target_vectors: torch.Tensor) -> torch.Tensor:
    pred_angles = torch.atan2(pred_vectors[:, 0], pred_vectors[:, 1])
    target_angles = torch.atan2(target_vectors[:, 0], target_vectors[:, 1])
    diff = torch.remainder(pred_angles - target_angles + math.pi, 2 * math.pi) - math.pi
    return torch.rad2deg(diff.abs())


def summarize_errors(errors: Iterable[float]) -> dict:
    values = sorted(float(x) for x in errors)
    if not values:
        return {
            "count": 0,
            "mae": None,
            "median": None,
            "p90": None,
            "within_3": None,
            "within_5": None,
            "within_8": None,
            "within_10": None,
            "within_15": None,
        }

    def percentile(p: float) -> float:
        if len(values) == 1:
            return values[0]
        idx = (len(values) - 1) * p
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return values[lo]
        w = idx - lo
        return values[lo] * (1 - w) + values[hi] * w

    n = len(values)
    return {
        "count": n,
        "mae": sum(values) / n,
        "median": percentile(0.5),
        "p90": percentile(0.9),
        "within_3": sum(v <= 3 for v in values) / n,
        "within_5": sum(v <= 5 for v in values) / n,
        "within_8": sum(v <= 8 for v in values) / n,
        "within_10": sum(v <= 10 for v in values) / n,
        "within_15": sum(v <= 15 for v in values) / n,
    }
