from __future__ import annotations

import json

import numpy as np

from hrtf_personalization.measurement import (
    MEASUREMENT_SPECS,
    CalibratedImage,
    Landmark,
    _capture_center,
    _capture_center_distance,
    _capture_distance,
    load_anthropometrics_json,
)


def test_measurement_spec_order_has_17_entries() -> None:
    assert len(MEASUREMENT_SPECS) == 17
    assert MEASUREMENT_SPECS[0].key == "x1"
    assert MEASUREMENT_SPECS[-1].key == "x17"


def test_load_anthropometrics_json_reads_values_cm(tmp_path) -> None:
    payload = {"values_cm": [float(i) for i in range(17)]}
    path = tmp_path / "anthro.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    values = load_anthropometrics_json(path)
    assert values.shape == (17,)
    assert values[0] == 0.0
    assert values[-1] == 16.0


def test_capture_center_returns_transformed_point(monkeypatch) -> None:
    image = CalibratedImage(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        transform=np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    seen: dict[str, tuple[str, ...]] = {}

    monkeypatch.setattr(
        "hrtf_personalization.measurement._click_points",
        lambda _image, count, _title, existing_landmarks=None, point_labels=None: (
            seen.setdefault("point_labels", point_labels or ()),
            np.array([[4.0, 5.0]], dtype=np.float32),
        )[1],
    )

    center, landmark = _capture_center(image, "Side image", "Click the center of the head.")

    np.testing.assert_allclose(center, np.array([8.0, 15.0], dtype=np.float32))
    np.testing.assert_allclose(landmark.point, np.array([4.0, 5.0], dtype=np.float32))
    assert landmark.label == "Head center"
    assert seen["point_labels"] == ("Head center",)


def test_capture_distance_uses_calibration_transform(monkeypatch) -> None:
    image = CalibratedImage(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        transform=np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )

    monkeypatch.setattr(
        "hrtf_personalization.measurement._click_points",
        lambda _image, count, _title, existing_landmarks=None, point_labels=None: np.array(
            [[1.0, 2.0], [4.0, 6.0]], dtype=np.float32
        ),
    )

    distance, landmarks = _capture_distance(
        image,
        "Head width",
        "Click two points.",
        point_labels=("Head left", "Head right"),
    )

    assert np.isclose(distance, np.hypot(6.0, 12.0) * 2.54 / 100.0)
    assert [landmark.label for landmark in landmarks] == ["Head left", "Head right"]


def test_capture_center_distance_uses_transformed_points(monkeypatch) -> None:
    image = CalibratedImage(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        transform=np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )

    monkeypatch.setattr(
        "hrtf_personalization.measurement._click_points",
        lambda _image, count, _title, existing_landmarks=None, point_labels=None: np.array([[5.0, 7.0]], dtype=np.float32),
    )

    distance, landmark = _capture_center_distance(
        image,
        np.array([8.0, 15.0], dtype=np.float32),
        "Pinna offset down",
        "Click the target point from the head center.",
        point_label="Ear landmark",
    )

    assert np.isclose(distance, np.hypot(2.0, 6.0) * 2.54 / 100.0)
    assert landmark.label == "Ear landmark"


def test_capture_center_distance_prompt_mentions_head_center(monkeypatch) -> None:
    image = CalibratedImage(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        transform=np.eye(3, dtype=np.float32),
    )
    seen: dict[str, object] = {}
    existing_landmarks = [Landmark("Head center", np.array([1.0, 2.0], dtype=np.float32))]

    def fake_click_points(_image, count, title, existing_landmarks=None, point_labels=None):
        seen["title"] = title
        seen["labels"] = [landmark.label for landmark in existing_landmarks or []]
        seen["point_labels"] = point_labels
        return np.array([[5.0, 7.0]], dtype=np.float32)

    monkeypatch.setattr("hrtf_personalization.measurement._click_points", fake_click_points)

    _capture_center_distance(
        image,
        np.array([8.0, 15.0], dtype=np.float32),
        "Pinna offset down",
        "Click the ear landmark.",
        existing_landmarks=existing_landmarks,
    )

    assert "head center you already clicked" in seen["title"]
    assert seen["labels"] == ["Head center"]
    assert seen["point_labels"] == ("Target point",)
