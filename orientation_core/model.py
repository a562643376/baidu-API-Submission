from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class OrientationModel(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = False):
        super().__init__()

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        in_features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )
        self.backbone_name = backbone
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


def load_orientation_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    backbone = checkpoint.get("backbone", "resnet18")
    model = OrientationModel(backbone=backbone, pretrained=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, checkpoint
