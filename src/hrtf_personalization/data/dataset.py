from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import PreparedDatasetManifest


@dataclass
class PreparedSample:
    subject_id: str
    anthropometrics: torch.Tensor
    ear_image: torch.Tensor
    ear_side: torch.Tensor
    hrtf: torch.Tensor
    direction: torch.Tensor


@dataclass
class PreparedBatch:
    subject_ids: list[str]
    anthropometrics: torch.Tensor
    ear_image: torch.Tensor
    ear_side: torch.Tensor
    hrtf: torch.Tensor
    direction: torch.Tensor


class CIPICPreparedDataset(Dataset[PreparedSample]):
    """Dataset backed by prepared numpy artifacts.

    The preparation step is expected to create one `.npz` file per sample with
    `anthropometrics`, `ear_image`, `hrtf`, and `direction`.
    """

    def __init__(self, prepared_root: str | Path) -> None:
        self.prepared_root = Path(prepared_root)
        self.files = sorted(self.prepared_root.glob("samples/*.npz"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> PreparedSample:
        sample_path = self.files[index]
        data = np.load(sample_path)
        return PreparedSample(
            subject_id=str(data["subject_id"]),
            anthropometrics=torch.tensor(data["anthropometrics"], dtype=torch.float32),
            ear_image=torch.tensor(data["ear_image"], dtype=torch.float32),
            ear_side=torch.tensor(_load_ear_side(data), dtype=torch.float32),
            hrtf=torch.tensor(data["hrtf"], dtype=torch.float32),
            direction=torch.tensor(data["direction"], dtype=torch.float32),
        )

    @classmethod
    def from_manifest(cls, manifest: PreparedDatasetManifest) -> "CIPICPreparedDataset":
        return cls(manifest.prepared_root)


def collate_prepared_samples(samples: list[PreparedSample]) -> PreparedBatch:
    return PreparedBatch(
        subject_ids=[sample.subject_id for sample in samples],
        anthropometrics=torch.stack([sample.anthropometrics for sample in samples], dim=0),
        ear_image=torch.stack([sample.ear_image for sample in samples], dim=0),
        ear_side=torch.stack([sample.ear_side for sample in samples], dim=0),
        hrtf=torch.stack([sample.hrtf for sample in samples], dim=0),
        direction=torch.stack([sample.direction for sample in samples], dim=0),
    )


def _load_ear_side(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "ear_side" not in data:
        return np.asarray([0.0], dtype=np.float32)
    ear_side = np.asarray(data["ear_side"], dtype=np.float32)
    if ear_side.ndim == 0:
        ear_side = ear_side.reshape(1)
    return ear_side
