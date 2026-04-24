from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EarImagePreprocessor:
    image_size: int = 64
    crop_x: int = 50
    crop_y: int = 50
    crop_width: int = 450
    crop_height: int = 450
    blur_kernel: int = 5
    low_threshold: int = 50
    high_threshold: int = 100

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = self._to_grayscale(image)
        cropped = gray[
            self.crop_y : self.crop_y + self.crop_height,
            self.crop_x : self.crop_x + self.crop_width,
        ]
        resized = cv2.resize(cropped, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(resized, (self.blur_kernel, self.blur_kernel), 0)
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
        return edges.astype(np.float32) / 255.0

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
