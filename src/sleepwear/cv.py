"""Leave-one-subject-out folds, grouped inner folds for tuning, and the paper's Monte Carlo
splits (reproduction gate only). Index arrays only: no scaling, imputation or model code."""

import numpy as np
from sklearn.model_selection import GroupKFold

from sleepwear import config as C


def loso_folds(subjects) -> list[tuple[np.ndarray, np.ndarray]]:
    """One (train_idx, test_idx) pair per subject, as positional indices, in sorted subject order."""
    subjects = np.asarray(subjects)
    folds = []
    for sid in np.unique(subjects):
        test = subjects == sid
        folds.append((np.flatnonzero(~test), np.flatnonzero(test)))
    return folds


def inner_folds(subjects, k: int = C.INNER_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped (fit_idx, val_idx) pairs over one training split's rows, positional within it.

    Every model, the transformer included, is tuned on these same folds.
    """
    subjects = np.asarray(subjects)
    return list(GroupKFold(k).split(subjects, groups=subjects))


def monte_carlo_splits(subjects, n_splits: int, test_fraction: float = 0.3,
                       seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random subject-wise splits as in the reference. Gate only: results use LOSO."""
    subjects = np.asarray(subjects)
    ids = np.unique(subjects)
    n_test = int(np.round(test_fraction * len(ids)))
    rng = np.random.default_rng(seed)
    splits = []
    for _ in range(n_splits):
        test = np.isin(subjects, rng.choice(ids, n_test, replace=False))
        splits.append((np.flatnonzero(~test), np.flatnonzero(test)))
    return splits
