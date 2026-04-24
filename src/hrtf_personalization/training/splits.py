from __future__ import annotations

from collections import defaultdict

from hrtf_personalization.data import CIPICPreparedDataset


def leave_one_subject_out(dataset: CIPICPreparedDataset) -> list[tuple[list[int], list[int]]]:
    subject_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, sample_path in enumerate(dataset.files):
        subject_id = sample_path.stem.split("__")[0]
        subject_to_indices[subject_id].append(index)

    splits: list[tuple[list[int], list[int]]] = []
    all_indices = set(range(len(dataset)))
    for test_indices in subject_to_indices.values():
        test_set = set(test_indices)
        train_indices = sorted(all_indices - test_set)
        splits.append((train_indices, sorted(test_set)))
    return splits

