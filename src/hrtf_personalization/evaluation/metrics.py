from __future__ import annotations

import numpy as np


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(reference - estimate))))


def log_spectral_distance(reference: np.ndarray, estimate: np.ndarray, fft_size: int = 512) -> float:
    ref_spec = np.fft.rfft(reference, n=fft_size)
    est_spec = np.fft.rfft(estimate, n=fft_size)
    ref_mag = np.maximum(np.abs(ref_spec), 1.0e-12)
    est_mag = np.maximum(np.abs(est_spec), 1.0e-12)
    distance = 20.0 * np.log10(ref_mag / est_mag)
    return float(np.sqrt(np.mean(np.square(distance))))

