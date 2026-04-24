from __future__ import annotations

from pathlib import Path

import numpy as np

from hrtf_personalization.prediction import _postprocess_hrir, _resolve_image_ear_side


def test_resolve_image_ear_side_honors_explicit_value() -> None:
    assert _resolve_image_ear_side(Path("images/ear.jpg"), "left") == "left"
    assert _resolve_image_ear_side(Path("images/ear.jpg"), "right") == "right"


def test_resolve_image_ear_side_infers_from_filename() -> None:
    assert _resolve_image_ear_side(Path("images/my_left_ear.jpg"), "auto") == "left"
    assert _resolve_image_ear_side(Path("images/my-right-ear.jpg"), "auto") == "right"


def test_resolve_image_ear_side_defaults_to_right_when_unknown() -> None:
    assert _resolve_image_ear_side(Path("images/ear.jpg"), "auto") == "right"


def test_postprocess_hrir_removes_dc_offset_and_caps_peak() -> None:
    hrir = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            [[-3.0, -1.0, 1.0, 3.0], [-4.0, -2.0, 0.0, 2.0]],
        ],
        dtype=np.float64,
    )

    processed = _postprocess_hrir(hrir, target_peak=0.2)

    assert np.allclose(np.mean(processed, axis=-1), 0.0)
    assert np.max(np.abs(processed)) <= 0.2 + 1e-9
    assert not np.allclose(processed, 0.0)
