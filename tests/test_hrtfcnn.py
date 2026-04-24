from __future__ import annotations

import numpy as np

from hrtf_personalization.data.hrtfcnn import _build_subject_row_index, _find_ear_image


def test_build_subject_row_index_uses_subject_ids() -> None:
    subject_ids = np.array([[3], [8], [119]], dtype=np.int32)
    assert _build_subject_row_index(subject_ids) == {3: 0, 8: 1, 119: 2}


def test_find_ear_image_prefers_left_over_rear(tmp_path) -> None:
    subject_dir = tmp_path / "Subject_010"
    subject_dir.mkdir()
    (subject_dir / "0010_back.jpg").write_text("back", encoding="utf-8")
    (subject_dir / "0010_left.jpg").write_text("left", encoding="utf-8")
    (subject_dir / "0010_right_rear.jpg").write_text("right", encoding="utf-8")

    image_path, ear_index = _find_ear_image(tmp_path, 10)

    assert image_path.name == "0010_left.jpg"
    assert ear_index == 0


def test_find_ear_image_falls_back_to_right(tmp_path) -> None:
    subject_dir = tmp_path / "Subject_011"
    subject_dir.mkdir()
    (subject_dir / "011_right_side.jpg").write_text("right", encoding="utf-8")

    image_path, ear_index = _find_ear_image(tmp_path, 11)

    assert image_path.name == "011_right_side.jpg"
    assert ear_index == 1
