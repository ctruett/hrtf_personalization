from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SubjectRecord:
    subject_id: str
    anthropometrics_path: Path
    left_ear_image_path: Path
    right_ear_image_path: Path
    hrtf_path: Path


@dataclass
class PreparedDatasetManifest:
    prepared_root: Path
    subjects: list[SubjectRecord] = field(default_factory=list)

