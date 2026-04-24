from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AnthropometricNormalizer:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "AnthropometricNormalizer":
        self.mean = values.mean(axis=0)
        self.std = values.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            msg = "Normalizer must be fit before transform."
            raise RuntimeError(msg)
        return (values - self.mean) / self.std

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)

