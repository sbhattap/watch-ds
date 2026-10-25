import numpy as np
import pandas as pd
import pytest

from sleepwear import config as C
from sleepwear import data as D
from sleepwear import features as F

needs_data = pytest.mark.skipif(not C.FEATURES_PATH.exists(), reason="run `make features` first")


def _subject(sid: str, epochs, value: float) -> pd.DataFrame:
    """Synthetic epochs.parquet rows with distinguishable values per epoch and subject."""
    e = np.asarray(epochs)
    return pd.DataFrame({"subject": sid, "epoch": e, "t": e * C.EPOCH_S, "stage": 2,
                         "act_count": value + e, "walch_hr": value + e})


def test_activity_is_gaussian_weighted_per_15s_count() -> None:
    w = np.exp(-0.5 * (np.arange(-9, 10) * 30 / 50) ** 2)
    df = _subject("A", range(30), 0.0)
    df["act_count"] = 2.0  # per 30 s -> 1.0 per 15 s
    assert np.allclose(F.activity(df), 1.0)
    df["act_count"] = 0.0
    df.loc[10, "act_count"] = 2.0
    got = F.activity(df)
    assert got[10] == pytest.approx(1 / w.sum())
    assert got[11] == pytest.approx(np.exp(-0.5 * (30 / 50) ** 2) / w.sum())


def test_activity_treats_dropped_epochs_as_holes() -> None:
    df = _subject("A", [0, 1, 3, 4], 0.0)  # epoch 2 dropped
    df["act_count"] = [2.0, 2.0, 4.0, 4.0]
    w = np.exp(-0.5 * (np.arange(-9, 10) * 30 / 50) ** 2)
    # Epoch 1 sees epochs 0, 1 (value 1) at offsets -1, 0 and epochs 3, 4 (value 2) at +2, +3.
    num = w[8] + w[9] + 2 * (w[11] + w[12])
    assert F.activity(df)[1] == pytest.approx(num / (w[8] + w[9] + w[11] + w[12]))


def test_clock_proxy_minimum_at_5h() -> None:
    assert F.clock(pd.Series([5 * 3600])).iloc[0] == pytest.approx(-1.0)
    assert F.clock(pd.Series([17 * 3600])).iloc[0] == pytest.approx(1.0)


def test_hr_std_zero_for_flat_hr_and_invariant_to_scale() -> None:
    t = np.arange(0.0, 3600.0, 5.0)  # HR every 5 s for an hour
    flat = D.walch_hr_epochs(pd.DataFrame({"t": t, "hr": 60.0 + 0 * t}), 0.0, 3600.0)
    wavy = 60 + 5 * np.sin(t / 200)
    base = D.walch_hr_epochs(pd.DataFrame({"t": t, "hr": wavy}), 0.0, 3600.0)
    moved = D.walch_hr_epochs(pd.DataFrame({"t": t, "hr": 3 * wavy + 7}), 0.0, 3600.0)
    assert np.allclose(flat, 0.0)
    assert np.allclose(base, moved, equal_nan=True)  # per-night normalisation removes scale
    assert base.notna().sum() > 100 and (base > 0).any()


def test_hr_std_leaves_long_gaps_out() -> None:
    t = np.r_[np.arange(0.0, 1800.0, 5.0), np.arange(3000.0, 4800.0, 5.0)]  # 20 min gap
    f = D.walch_hr_epochs(pd.DataFrame({"t": t, "hr": 60 + 5 * np.sin(t / 200)}), 0.0, 4800.0)
    # Windows (t - 285, t + 315) wholly inside the gap: t = 2100 ... 2670, epochs 70-89.
    assert f.loc[70:89].isna().all() and f.loc[:65].notna().all() and f.loc[95:155].notna().all()


def test_features_never_cross_subject_boundary() -> None:
    # Same epoch numbers in both subjects, B adjacent in row order, values 1e6 apart.
    a, b = _subject("A", range(20), 0.0), _subject("B", range(20), 1e6)
    both = F.build_features(pd.concat([a, b], ignore_index=True))
    pd.testing.assert_frame_equal(both.iloc[:20].reset_index(drop=True), F.build_features(a))
    pd.testing.assert_frame_equal(both.iloc[20:].reset_index(drop=True), F.build_features(b))
    assert (both.iloc[:20]["activity"] < 1e5).all()


@needs_data
def test_row_count_and_order_match_epochs() -> None:
    ep = pd.read_parquet(C.EPOCHS_PATH)
    feats = pd.read_parquet(C.FEATURES_PATH)
    assert len(feats) == len(ep)
    pd.testing.assert_frame_equal(feats[F.KEYS], ep[F.KEYS])
    assert list(feats.columns) == F.KEYS + F.FEATURES


@needs_data
def test_features_complete_except_hr_gaps() -> None:
    feats = pd.read_parquet(C.FEATURES_PATH)
    assert feats["activity"].notna().all() and feats["clock"].notna().all()
    assert feats["hr_std"].isna().sum() < 20  # only epochs whose 10-min window is all HR gap
