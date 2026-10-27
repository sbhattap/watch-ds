import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score, f1_score

from sleepwear import evaluate as E


def test_kappa_and_macro_f1_match_sklearn() -> None:
    rng = np.random.default_rng(0)
    y, pred = rng.integers(0, 3, 500), rng.integers(0, 3, 500)
    cm = E._confusion(y, pred, 3)
    assert E.kappa(cm) == pytest.approx(cohen_kappa_score(y, pred))
    proba = np.eye(3)[pred]
    m = E.subject_metrics(y, proba, 3)
    assert m["kappa"] == pytest.approx(cohen_kappa_score(y, pred))
    assert m["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))


def test_sleep_wake_threshold_hits_target_sensitivity() -> None:
    wake = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.33, 0.45, 0.5]
    sleep = [0.35, 0.40, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    y, p = np.r_[np.zeros(10, int), np.ones(10, int)], np.r_[wake, sleep]
    r = E.sleep_wake_at(y, p, 0.90)
    # TPR first reaches 0.9 at the 9th-highest sleep score, 0.40; wake 0.45 and 0.5 lie above it.
    assert r["wake"] == pytest.approx(0.8)


def test_absent_stage_is_undefined_not_zero() -> None:
    # 3-class subject with no REM: REM recall is NaN and macro-F1 averages W and NREM.
    y = np.array([0, 0, 1, 1, 1, 1])
    pred = np.array([0, 1, 1, 1, 2, 1])
    m = E.subject_metrics(y, np.eye(3)[pred], 3)
    assert np.isnan(m["recall_REM"])
    assert m["recall_W"] == 0.5 and m["recall_NREM"] == 0.75
    assert m["macro_f1"] == pytest.approx(np.mean([2 / 3, 2 * 3 / (3 + 4 + 1)]))  # 2TP/(2TP+FP+FN)


def test_two_class_auc_and_wake_at_93_per_subject() -> None:
    y = np.r_[np.zeros(10, int), np.ones(90, int)]
    p1 = np.r_[np.linspace(0, 0.5, 10), np.linspace(0.3, 1, 90)]
    m = E.subject_metrics(y, np.c_[1 - p1, p1], 2)
    assert m["auc"] == pytest.approx(E.roc_auc_score(y, p1))
    assert 0 <= m["wake_at_0.93"] <= 1


def test_kappa_undefined_for_single_class() -> None:
    assert np.isnan(E.kappa(E._confusion(np.zeros(5, int), np.zeros(5, int), 2)))


def test_boot_mean_ignores_nan_and_counts_contributors() -> None:
    mean, lo, hi, n = E.boot_mean(np.array([0.2, np.nan, 0.4, 0.6]), n_boot=500)
    assert n == 3 and mean == pytest.approx(0.4) and 0.2 <= lo <= mean <= hi <= 0.6
    assert E.boot_mean(np.array([np.nan]))[3] == 0


def test_per_subject_never_pools_and_averages_seeds() -> None:
    # Subject a is perfect, subject b all wrong: the mean is 0.5, whatever their sizes.
    y = np.array([0, 1] * 51)
    pred = np.r_[[0, 1] * 50, [1, 0]]
    oof = pd.DataFrame({"model": "m", "seed": 0, "subject": ["a"] * 100 + ["b"] * 2, "y": y,
                        "p0": 1 - pred, "p1": pred})
    per = E.per_subject(oof, 2)
    assert per.set_index("subject")["accuracy"].to_dict() == {"a": 1.0, "b": 0.0}
    assert E.boot_mean(per["accuracy"].to_numpy())[0] == 0.5
    # A second seed that gets b right: b's accuracy is the mean over seeds, 0.5.
    two = pd.concat([oof, oof.assign(seed=1, p0=1 - y, p1=y)])
    assert E.per_subject(two, 2).set_index("subject")["accuracy"]["b"] == 0.5


def test_wake_at_93_undefined_for_constant_score() -> None:
    y = np.r_[np.zeros(10, int), np.ones(90, int)]
    m = E.subject_metrics(y, np.tile([0.1, 0.9], (100, 1)), 2)
    assert np.isnan(m["wake_at_0.93"]) and m["auc"] == 0.5
