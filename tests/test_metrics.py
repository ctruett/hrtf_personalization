from __future__ import annotations

import numpy as np

from hrtf_personalization.evaluation import log_spectral_distance, rmse


def test_rmse_zero_for_identical_arrays() -> None:
    values = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    assert rmse(values, values) == 0.0


def test_lsd_zero_for_identical_arrays() -> None:
    values = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float64)
    assert log_spectral_distance(values, values) == 0.0

