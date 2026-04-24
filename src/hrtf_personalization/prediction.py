from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import cv2
import h5py
import numpy as np
import torch

from hrtf_personalization.data.hrtfcnn import vertical_polar_to_interaural
from hrtf_personalization.preprocessing import EarImagePreprocessor
from hrtf_personalization.sofa import export_simple_free_field_hrir


@dataclass
class PredictionInputs:
    checkpoint_path: Path
    image_path: Path | None
    left_image_path: Path | None
    right_image_path: Path | None
    anthropometrics: np.ndarray
    template_sofa_path: Path
    output_sofa_path: Path
    image_ear: str = "auto"
    model_type: str = "auto"
    device: str = "cpu"


def predict_sofa_from_image(
    inputs: PredictionInputs,
    model: torch.nn.Module,
    image_preprocessor: EarImagePreprocessor,
) -> Path:
    model = model.to(inputs.device)
    if (inputs.left_image_path or inputs.right_image_path) and not getattr(model, "ear_side_dim", 0):
        raise ValueError(
            "This checkpoint is not ear-side-aware, so it cannot generate distinct left/right outputs from "
            "separate ear images. Retrain with ear_side_dim enabled first."
        )
    left_ear_image, right_ear_image = _resolve_binaural_prediction_images(inputs, image_preprocessor)
    directions_deg, source_positions = _load_template_directions(inputs.template_sofa_path)

    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        anthropometrics = torch.tensor(inputs.anthropometrics, dtype=torch.float32, device=inputs.device).unsqueeze(0).repeat(2, 1)
        ear_side_tensor = torch.tensor([[-1.0], [1.0]], dtype=torch.float32, device=inputs.device)
        ear_images = np.stack([left_ear_image, right_ear_image], axis=0)
        image_tensor = torch.tensor(ear_images, dtype=torch.float32, device=inputs.device).unsqueeze(1)
        for direction in directions_deg:
            direction_tensor = torch.tensor(direction, dtype=torch.float32, device=inputs.device).unsqueeze(0).repeat(2, 1)
            output = _predict_binaural_batch(
                model=model,
                anthropometrics=anthropometrics,
                image_tensor=image_tensor,
                direction_tensor=direction_tensor,
                ear_side_tensor=ear_side_tensor,
                model_type=inputs.model_type,
            )
            predictions.append(output.cpu().numpy())

    hrir = np.asarray(predictions, dtype=np.float64)
    hrir = _postprocess_hrir(hrir)
    if np.allclose(hrir[:, 0, :], hrir[:, 1, :]):
        warnings.warn(
            "Predicted left/right HRIR channels are identical. The current checkpoint is not producing a true "
            "binaural output yet.",
            RuntimeWarning,
            stacklevel=2,
        )
    return export_simple_free_field_hrir(
        inputs.output_sofa_path,
        hrir=hrir,
        source_positions_deg=source_positions,
        sampling_rate_hz=44100.0,
    )


def _load_and_preprocess_image(image_path: Path, preprocessor: EarImagePreprocessor) -> np.ndarray:
    image = _load_rgb_image(image_path)
    return preprocessor.preprocess(image)


def _load_and_preprocess_binaural_images(
    image_path: Path,
    preprocessor: EarImagePreprocessor,
    image_ear: str,
) -> tuple[np.ndarray, np.ndarray]:
    image = _load_rgb_image(image_path)
    mirrored = cv2.flip(image, 1)
    original_processed = preprocessor.preprocess(image)
    mirrored_processed = preprocessor.preprocess(mirrored)
    resolved_ear = _resolve_image_ear_side(image_path, image_ear)
    if resolved_ear == "left":
        return original_processed, mirrored_processed
    return mirrored_processed, original_processed


def _resolve_binaural_prediction_images(
    inputs: PredictionInputs,
    preprocessor: EarImagePreprocessor,
) -> tuple[np.ndarray, np.ndarray]:
    if inputs.left_image_path and inputs.right_image_path:
        return (
            _load_and_preprocess_image(inputs.left_image_path, preprocessor),
            _load_and_preprocess_image(inputs.right_image_path, preprocessor),
        )
    if inputs.image_path is None:
        raise ValueError("Provide either --image or both --left-image and --right-image.")
    return _load_and_preprocess_binaural_images(inputs.image_path, preprocessor, image_ear=inputs.image_ear)


def _load_rgb_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        msg = f"Unable to read image at {image_path}."
        raise FileNotFoundError(msg)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _resolve_image_ear_side(image_path: Path, image_ear: str) -> str:
    if image_ear in {"left", "right"}:
        return image_ear
    stem = image_path.stem.lower()
    tokens = stem.replace("-", "_").split("_")
    if "left" in tokens or stem.endswith("l"):
        return "left"
    if "right" in tokens or stem.endswith("r"):
        return "right"
    return "right"


def _predict_binaural_batch(
    model: torch.nn.Module,
    anthropometrics: torch.Tensor,
    image_tensor: torch.Tensor,
    direction_tensor: torch.Tensor,
    ear_side_tensor: torch.Tensor,
    model_type: str,
) -> torch.Tensor:
    uses_ear_side = bool(getattr(model, "ear_side_dim", 0))
    if model_type == "conditional":
        if uses_ear_side:
            return model(anthropometrics, image_tensor, direction_tensor, ear_side_tensor)
        return model(anthropometrics, image_tensor, direction_tensor)
    if uses_ear_side:
        return model(anthropometrics, image_tensor, ear_side_tensor)
    return model(anthropometrics, image_tensor)


def _postprocess_hrir(hrir: np.ndarray, target_peak: float = 1.0) -> np.ndarray:
    centered = hrir - np.mean(hrir, axis=-1, keepdims=True)
    peak = float(np.max(np.abs(centered)))
    if peak == 0.0:
        return centered
    return centered * (target_peak / peak)


def _load_template_directions(template_sofa_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(template_sofa_path, "r") as handle:
        source_positions = np.asarray(handle["SourcePosition"])
    vertical_azimuth = source_positions[:, 0]
    vertical_elevation = source_positions[:, 1]
    interaural_elevation, interaural_azimuth = vertical_polar_to_interaural(vertical_elevation, vertical_azimuth)
    directions = np.stack([interaural_elevation, interaural_azimuth], axis=1)
    return directions.astype(np.float32), source_positions.astype(np.float64)
