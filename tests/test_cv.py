import numpy as np
import pandas as pd
import pytest

from sleepwear import config as C
from sleepwear.cv import inner_folds, loso_folds

needs_data = pytest.mark.skipif(not C.FEATURES_PATH.exists(), reason="run `make features` first")


def _check_folds(subjects: np.ndarray) -> None:
    folds = loso_folds(subjects)
    assert len(folds) == len(np.unique(subjects))
    seen = np.zeros(len(subjects), dtype=int)
    for train, test in folds:
        train_ids, test_ids = set(subjects[train]), set(subjects[test])
        assert train_ids.isdisjoint(test_ids)
        assert len(test_ids) == 1
        assert len(train) + len(test) == len(subjects)
        seen[test] += 1
    assert np.all(seen == 1)  # every row is tested exactly once


def test_synthetic_folds_disjoint() -> None:
    _check_folds(np.array(["b", "a", "b", "c", "a", "c", "c"]))


@needs_data
def test_real_folds_disjoint_and_31() -> None:
    subjects = pd.read_parquet(C.FEATURES_PATH, columns=["subject"])["subject"].to_numpy()
    assert len(loso_folds(subjects)) == C.N_SUBJECTS
    _check_folds(subjects)


def test_inner_folds_stay_inside_training_subjects() -> None:
    subjects = np.repeat(np.array(list("abcdefgh")), 5)
    for train, test in loso_folds(subjects):
        inner = inner_folds(subjects[train])
        assert len(inner) == 3
        for fit, val in inner:
            fit_ids, val_ids = set(subjects[train][fit]), set(subjects[train][val])
            assert fit_ids.isdisjoint(val_ids)
            assert subjects[test[0]] not in fit_ids | val_ids  # outer test subject never seen


def test_inner_folds_do_not_depend_on_row_order() -> None:
    # models.py shuffles training rows, transformer.py does not: the subject-to-fold map must agree.
    subjects = np.repeat(np.array(list("abcdefghij")), np.arange(10, 20))
    perm = np.random.default_rng(0).permutation(len(subjects))
    a = [set(subjects[val]) for _, val in inner_folds(subjects)]
    b = [set(subjects[perm][val]) for _, val in inner_folds(subjects[perm])]
    assert a == b
