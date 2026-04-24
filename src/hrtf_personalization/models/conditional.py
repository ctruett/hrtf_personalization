from __future__ import annotations

import torch
from torch import nn


class ConditionalHRTFEstimator(nn.Module):
    """Notebook-style conditional model from HRTFCNN.

    This mirrors the original layout:
    - main branch on [anthropometrics + direction]
    - CNN branch on ear image
    - fusion MLP for 200-sample HRIR prediction
    """

    def __init__(
        self,
        anthropometric_dim: int = 17,
        direction_dim: int = 2,
        hrtf_dim: int = 200,
        ear_image_size: int = 64,
        ear_side_dim: int = 0,
    ) -> None:
        super().__init__()
        self.ear_side_dim = ear_side_dim
        self.main_branch = nn.Sequential(
            nn.Linear(anthropometric_dim + direction_dim + ear_side_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
        )
        self.image_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(self._conv_feature_dim(ear_image_size), 8),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, hrtf_dim),
        )

    @staticmethod
    def _conv_feature_dim(ear_image_size: int) -> int:
        with torch.no_grad():
            features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 16, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )(torch.zeros(1, 1, ear_image_size, ear_image_size))
        return int(features.numel())

    def forward(
        self,
        anthropometrics: torch.Tensor,
        ear_image: torch.Tensor,
        direction: torch.Tensor,
        ear_side: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ear_image.ndim == 3:
            ear_image = ear_image.unsqueeze(1)
        main_inputs = [anthropometrics, direction]
        if self.ear_side_dim:
            if ear_side is None:
                raise ValueError("ear_side must be provided when ear_side_dim is enabled.")
            main_inputs.append(ear_side)
        main_features = self.main_branch(torch.cat(main_inputs, dim=-1))
        image_features = self.image_branch(ear_image)
        fused = torch.cat([main_features, image_features], dim=-1)
        return self.fusion(fused)
