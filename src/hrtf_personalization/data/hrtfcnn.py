from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import scipy.io

from hrtf_personalization.preprocessing import EarImagePreprocessor


@dataclass
class HRTFCNNSampleSummary:
    subject_number: int
    subject_id: str
    ear_index: int
    ear_image_path: str
    sofa_path: str
    num_directions: int


@dataclass
class HRTFCNNPaths:
    repo_root: Path
    sofa_dir: Path
    anthropometry_mat: Path
    ear_photos_dir: Path


def resolve_hrtfcnn_paths(repo_root: str | Path) -> HRTFCNNPaths:
    root = Path(repo_root)
    return HRTFCNNPaths(
        repo_root=root,
        sofa_dir=root / "data" / "cipic_hrtf_sofa",
        anthropometry_mat=root / "data" / "CIPIC_hrtf_database" / "anthropometry" / "anthro.mat",
        ear_photos_dir=root / "data" / "ear_photos",
    )


def prepare_from_hrtfcnn_repo(
    repo_root: str | Path,
    prepared_root: str | Path,
    anthropometric_dim: int,
    ear_preprocessor: EarImagePreprocessor,
) -> list[HRTFCNNSampleSummary]:
    paths = resolve_hrtfcnn_paths(repo_root)
    _validate_hrtfcnn_paths(paths)
    prepared = Path(prepared_root)
    samples_dir = prepared / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    anthropometrics, anthropometric_row_by_subject_id = _load_anthropometrics(
        paths.anthropometry_mat,
        anthropometric_dim,
    )
    summaries: list[HRTFCNNSampleSummary] = []

    for sofa_path in sorted(paths.sofa_dir.glob("subject_*.sofa")):
        subject_number = _extract_subject_number(sofa_path)
        anthropometric_row_index = anthropometric_row_by_subject_id.get(subject_number)
        if anthropometric_row_index is None:
            continue

        anthropometric_row = anthropometrics[anthropometric_row_index]
        if np.isnan(anthropometric_row).any():
            continue

        image_match = _find_ear_image(paths.ear_photos_dir, subject_number)
        if image_match is None:
            continue
        image_path, ear_index = image_match

        ear_image = _load_image(image_path)
        mirrored_ear_image = cv2.flip(ear_image, 1)
        ear_edges = ear_preprocessor.preprocess(ear_image)
        mirrored_ear_edges = ear_preprocessor.preprocess(mirrored_ear_image)
        impulses, directions = _load_sofa_impulses_and_directions(sofa_path)

        if impulses.shape[0] != directions.shape[0]:
            msg = (
                f"Mismatch between impulse rows ({impulses.shape[0]}) "
                f"and direction rows ({directions.shape[0]}) for {sofa_path}."
            )
            raise ValueError(msg)
        opposite_ear_index = 1 - ear_index
        ear_variants = [
            (ear_index, ear_edges, _ear_side_value(ear_index)),
            (opposite_ear_index, mirrored_ear_edges, _ear_side_value(opposite_ear_index)),
        ]
        for variant_ear_index, variant_ear_edges, ear_side in ear_variants:
            for direction_index, direction in enumerate(directions):
                sample_path = samples_dir / f"{subject_number:03d}__{variant_ear_index}__{direction_index:04d}.npz"
                np.savez_compressed(
                    sample_path,
                    subject_id=f"{subject_number:03d}",
                    anthropometrics=anthropometric_row.astype(np.float32),
                    ear_image=variant_ear_edges.astype(np.float32),
                    ear_side=np.asarray([ear_side], dtype=np.float32),
                    hrtf=impulses[direction_index, variant_ear_index, :].astype(np.float32),
                    direction=direction.astype(np.float32),
                )

        summaries.append(
            HRTFCNNSampleSummary(
                subject_number=subject_number,
                subject_id=f"{subject_number:03d}",
                ear_index=ear_index,
                ear_image_path=str(image_path),
                sofa_path=str(sofa_path),
                num_directions=int(impulses.shape[0]),
            )
        )

    manifest_path = prepared / "manifest.json"
    manifest_payload = {
        "source_layout": "hrtfcnn",
        "repo_root": str(paths.repo_root),
        "num_subjects": len(summaries),
        "num_samples": sum(item.num_directions * 2 for item in summaries),
        "subjects": [asdict(summary) for summary in summaries],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return summaries


def _validate_hrtfcnn_paths(paths: HRTFCNNPaths) -> None:
    missing: list[str] = []
    if not paths.repo_root.exists():
        missing.append(f"repo root: {paths.repo_root}")
    if not paths.sofa_dir.exists():
        missing.append(f"SOFA directory: {paths.sofa_dir}")
    if not paths.anthropometry_mat.exists():
        missing.append(f"anthropometry file: {paths.anthropometry_mat}")
    if not paths.ear_photos_dir.exists():
        missing.append(f"ear photo directory: {paths.ear_photos_dir}")
    if missing:
        details = "\n".join(f"- {item}" for item in missing)
        msg = (
            "Missing HRTFCNN dataset assets.\n"
            "Expected the notebook-prepared layout under the configured repo root:\n"
            f"{details}"
        )
        raise FileNotFoundError(msg)


def _load_anthropometrics(path: Path, anthropometric_dim: int) -> tuple[np.ndarray, dict[int, int]]:
    mat = scipy.io.loadmat(path)
    raw = np.asarray(mat["X"], dtype=np.float32)
    subject_ids = np.asarray(mat["id"]).reshape(-1)
    if raw.shape[1] < anthropometric_dim:
        msg = f"anthro.mat only has {raw.shape[1]} columns, expected at least {anthropometric_dim}."
        raise ValueError(msg)
    if raw.shape[0] != subject_ids.shape[0]:
        msg = (
            "anthro.mat subject id count does not match anthropometry row count: "
            f"{subject_ids.shape[0]} ids vs {raw.shape[0]} rows."
        )
        raise ValueError(msg)
    row_by_subject_id = _build_subject_row_index(subject_ids)
    return raw[:, :anthropometric_dim], row_by_subject_id


def _build_subject_row_index(subject_ids: np.ndarray) -> dict[int, int]:
    return {int(subject_id): index for index, subject_id in enumerate(subject_ids.reshape(-1))}


def _extract_subject_number(sofa_path: Path) -> int:
    match = re.search(r"subject_(\d+)\.sofa$", sofa_path.name)
    if match is None:
        msg = f"Unable to extract subject number from {sofa_path.name}."
        raise ValueError(msg)
    return int(match.group(1))


def _find_ear_image(ear_photos_dir: Path, subject_number: int) -> tuple[Path, int] | None:
    subject_id = f"{subject_number:03d}"
    subject_dir = ear_photos_dir / f"Subject_{subject_id}"
    if not subject_dir.exists():
        return None

    files = sorted(candidate for candidate in subject_dir.iterdir() if candidate.is_file())
    for token, ear_index in (
        ("left_side", 0),
        ("right_side", 1),
        ("left_", 0),
        ("right_", 1),
        ("left", 0),
        ("right", 1),
    ):
        for candidate in files:
            if token in candidate.name.lower():
                return candidate, ear_index
    return None


def _load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        msg = f"Unable to read image at {image_path}."
        raise FileNotFoundError(msg)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _load_sofa_impulses_and_directions(sofa_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(sofa_path, "r") as handle:
        impulses = np.asarray(handle["Data.IR"])
        source_positions = np.asarray(handle["SourcePosition"])
    vertical_azimuth = source_positions[:, 0]
    vertical_elevation = source_positions[:, 1]
    interaural_elevation, interaural_azimuth = _vertical_to_interaural(
        vertical_elevation,
        vertical_azimuth,
    )
    directions = np.stack([interaural_elevation, interaural_azimuth], axis=1)
    return impulses, directions


def _ear_side_value(ear_index: int) -> float:
    return -1.0 if ear_index == 0 else 1.0


def _vertical_to_interaural(elevation_deg: np.ndarray, azimuth_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    elevation_rad = np.deg2rad(elevation_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)
    x = np.cos(elevation_rad) * np.sin(azimuth_rad)
    y = np.cos(elevation_rad) * np.cos(azimuth_rad)
    z = np.sin(elevation_rad)
    interaural_azimuth = np.rad2deg(np.arcsin(x))
    interaural_elevation = np.rad2deg(np.arctan2(z, y))
    interaural_elevation = np.where(interaural_elevation < -46.0, interaural_elevation + 360.0, interaural_elevation)
    return interaural_elevation.astype(np.float32), interaural_azimuth.astype(np.float32)


def vertical_polar_to_interaural(elevation_deg: np.ndarray, azimuth_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _vertical_to_interaural(elevation_deg, azimuth_deg)
