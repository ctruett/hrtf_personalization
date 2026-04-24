from __future__ import annotations

import torch

from hrtf_personalization.models import BaselineHRTFEstimator, ConditionalHRTFEstimator


def test_baseline_forward_shape() -> None:
    model = BaselineHRTFEstimator(ear_image_size=32)
    anthropometrics = torch.randn(2, 17)
    ear_image = torch.randn(2, 1, 32, 32)
    output = model(anthropometrics, ear_image)
    assert output.shape == (2, 200)


def test_conditional_forward_shape() -> None:
    model = ConditionalHRTFEstimator(ear_image_size=32)
    anthropometrics = torch.randn(2, 17)
    ear_image = torch.randn(2, 1, 32, 32)
    direction = torch.randn(2, 2)
    output = model(anthropometrics, ear_image, direction)
    assert output.shape == (2, 200)


def test_conditional_forward_shape_for_hrtfcnn_image_size() -> None:
    model = ConditionalHRTFEstimator(ear_image_size=64)
    anthropometrics = torch.randn(2, 17)
    ear_image = torch.randn(2, 1, 64, 64)
    direction = torch.randn(2, 2)
    output = model(anthropometrics, ear_image, direction)
    assert output.shape == (2, 200)


def test_baseline_forward_shape_with_ear_side() -> None:
    model = BaselineHRTFEstimator(ear_image_size=32, ear_side_dim=1)
    anthropometrics = torch.randn(2, 17)
    ear_image = torch.randn(2, 1, 32, 32)
    ear_side = torch.tensor([[-1.0], [1.0]])
    output = model(anthropometrics, ear_image, ear_side)
    assert output.shape == (2, 200)


def test_conditional_forward_shape_with_ear_side() -> None:
    model = ConditionalHRTFEstimator(ear_image_size=32, ear_side_dim=1)
    anthropometrics = torch.randn(2, 17)
    ear_image = torch.randn(2, 1, 32, 32)
    direction = torch.randn(2, 2)
    ear_side = torch.tensor([[-1.0], [1.0]])
    output = model(anthropometrics, ear_image, direction, ear_side)
    assert output.shape == (2, 200)
