from __future__ import annotations

import torch
from torch import nn


class BaselineHRTFEstimator(nn.Module):
    """Paper-shaped three-subnetwork model."""

    def __init__(
        self,
        anthropometric_dim: int = 17,
        hrtf_dim: int = 200,
        ear_image_size: int = 64,
        ear_side_dim: int = 0,
    ) -> None:
        super().__init__()
        self.ear_side_dim = ear_side_dim
        self.subnet_a = nn.Sequential(
            nn.Linear(anthropometric_dim + ear_side_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
        )
        self.subnet_b = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(self._conv_feature_dim(ear_image_size), 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
        )
        self.subnet_c = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, hrtf_dim),
        )

    @staticmethod
    def _conv_feature_dim(ear_image_size: int) -> int:
        with torch.no_grad():
            features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )(torch.zeros(1, 1, ear_image_size, ear_image_size))
        return int(features.numel())

    def forward(
        self,
        anthropometrics: torch.Tensor,
        ear_image: torch.Tensor,
        ear_side: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ear_image.ndim == 3:
            ear_image = ear_image.unsqueeze(1)
        anthro_inputs = anthropometrics
        if self.ear_side_dim:
            if ear_side is None:
                raise ValueError("ear_side must be provided when ear_side_dim is enabled.")
            anthro_inputs = torch.cat([anthropometrics, ear_side], dim=-1)
        anthro_features = self.subnet_a(anthro_inputs)
        image_features = self.subnet_b(ear_image)
        fused = torch.cat([anthro_features, image_features], dim=-1)
        return self.subnet_c(fused)
