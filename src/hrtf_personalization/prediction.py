from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import cv2
import h5py
import numpy as np
import torch
from scipy.signal import resample_poly

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
    hrir = _diffuse_field_equalize(hrir)
    hrir = _postprocess_hrir(hrir)
    hrir = _resample_hrir(hrir, from_hz=44100, to_hz=192000)
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
        sampling_rate_hz=192000.0,
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


def _diffuse_field_equalize(hrir: np.ndarray) -> np.ndarray:
    # Average power spectrum across all directions and ears → diffuse field response.
    # Shape: (M, R, N) → spec (M, R, K) → mean power (K,)
    spec = np.fft.rfft(hrir, axis=-1)
    df_power = np.mean(np.abs(spec) ** 2, axis=(0, 1))
    df_mag = np.sqrt(df_power)
    # Floor at 1 % of the peak to avoid division blow-up at nulls.
    floor = 0.01 * np.max(df_mag)
    df_mag = np.where(df_mag < floor, floor, df_mag)
    equalized_spec = spec / df_mag[np.newaxis, np.newaxis, :]
    equalized = np.fft.irfft(equalized_spec, n=hrir.shape[-1], axis=-1)
    # Tukey window to suppress ringing at the edges introduced by spectral division.
    window = _tukey_window(hrir.shape[-1], alpha=0.1)
    return equalized * window[np.newaxis, np.newaxis, :]


def _tukey_window(n: int, alpha: float = 0.1) -> np.ndarray:
    # Flat in the centre, cosine tapers over alpha/2 of each end.
    if alpha <= 0.0:
        return np.ones(n)
    taper = int(alpha * n / 2)
    window = np.ones(n)
    t = np.arange(taper)
    window[:taper] = 0.5 * (1 - np.cos(np.pi * t / taper))
    window[-taper:] = window[:taper][::-1]
    return window


def _resample_hrir(hrir: np.ndarray, from_hz: int, to_hz: int) -> np.ndarray:
    from math import gcd
    g = gcd(to_hz, from_hz)
    up, down = to_hz // g, from_hz // g
    return resample_poly(hrir, up, down, axis=-1)


def _postprocess_hrir(hrir: np.ndarray, target_gain: float = 0.9) -> np.ndarray:
    centered = hrir - np.mean(hrir, axis=-1, keepdims=True)
    # Normalize by frequency-domain magnitude so convolution output won't clip.
    # Time-domain peak normalization is wrong for filters: the freq response gain
    # determines output level, not the IR peak, and can be far higher.
    # max_mag shape: (1, R, 1) — max magnitude across all directions and frequencies per channel.
    mag = np.abs(np.fft.rfft(centered, axis=-1))
    max_mag = np.max(mag, axis=(0, 2), keepdims=True)
    max_mag = np.where(max_mag == 0.0, 1.0, max_mag)
    return centered * (target_gain / max_mag)


def _load_template_directions(template_sofa_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(template_sofa_path, "r") as handle:
        source_positions = np.asarray(handle["SourcePosition"])
    vertical_azimuth = source_positions[:, 0]
    vertical_elevation = source_positions[:, 1]
    interaural_elevation, interaural_azimuth = vertical_polar_to_interaural(vertical_elevation, vertical_azimuth)
    directions = np.stack([interaural_elevation, interaural_azimuth], axis=1)
    return directions.astype(np.float32), source_positions.astype(np.float64)
