from __future__ import annotations

from typing import Sequence

from diffshape.data.registry import CaseRecord


def split_by_ratio(
    cases: Sequence[CaseRecord],
    train_ratio: float,
) -> dict[str, list[CaseRecord]]:
    n_train = int(len(cases) * train_ratio)
    return {
        "train": list(cases[:n_train]),
        "test": list(cases[n_train:]),
    }


def split_by_kfold(
    cases: Sequence[CaseRecord],
    n_folds: int,
    fold_size: int,
    train_per_fold: int,
    test_per_fold: int,
    active_folds: list[int] | None = None,
) -> dict[str, list[CaseRecord]]:
    """K-fold split with fixed train/test partition per fold.
    active_folds: 0-indexed fold indices to include in train/test.
    Remaining folds go to 'unseen'."""
    if active_folds is None:
        active_folds = []

    all_fold_indices = set(range(n_folds))
    unseen_fold_indices = all_fold_indices - set(active_folds)

    train: list[CaseRecord] = []
    test: list[CaseRecord] = []
    unseen: list[CaseRecord] = []

    for fold_idx in active_folds:
        start = fold_idx * fold_size
        fold_cases = list(cases[start : start + fold_size])
        train.extend(fold_cases[:train_per_fold])
        test.extend(fold_cases[train_per_fold : train_per_fold + test_per_fold])

    for fold_idx in sorted(unseen_fold_indices):
        start = fold_idx * fold_size
        unseen.extend(cases[start : start + fold_size])

    return {"train": train, "test": test, "unseen": unseen}


def split_all_test(
    cases: Sequence[CaseRecord],
) -> dict[str, list[CaseRecord]]:
    return {"test": list(cases)}


def apply_split(
    cases: Sequence[CaseRecord],
    split_config: dict,
    active_folds: list[int] | None = None,
) -> dict[str, list[CaseRecord]]:
    method = split_config["method"]
    if method == "ratio":
        return split_by_ratio(cases, split_config["train_ratio"])
    elif method == "kfold":
        return split_by_kfold(
            cases,
            n_folds=split_config["n_folds"],
            fold_size=split_config["fold_size"],
            train_per_fold=split_config["train_per_fold"],
            test_per_fold=split_config["test_per_fold"],
            active_folds=active_folds,
        )
    elif method == "all_test":
        return split_all_test(cases)
    else:
        raise ValueError(f"Unknown split method: {method}")
