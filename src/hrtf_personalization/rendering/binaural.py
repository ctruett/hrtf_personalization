from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def convolve_mono_with_hrir(mono: np.ndarray, hrir_left: np.ndarray, hrir_right: np.ndarray) -> np.ndarray:
    left = fftconvolve(mono, hrir_left, mode="full")
    right = fftconvolve(mono, hrir_right, mode="full")
    return np.stack([left, right], axis=0)

