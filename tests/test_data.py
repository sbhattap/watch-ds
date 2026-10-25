import numpy as np
import pandas as pd
import pytest

from sleepwear import config as C
from sleepwear import data as D

needs_data = pytest.mark.skipif(not C.EPOCHS_PATH.exists(), reason="run `make data` first")


@pytest.fixture(scope="module")
def epochs() -> pd.DataFrame:
    return pd.read_parquet(C.EPOCHS_PATH)


@pytest.fixture(scope="module")
def raw() -> dict:
    """Per subject: cropped label grid, and the set of epochs holding any raw motion sample."""
    out = {}
    for sid in D.subject_ids():
        labels, motion, hr = D.load_labels(sid), D.load_motion(sid), D.load_hr(sid)
        start, end = D.valid_window(labels, motion, hr)
        t = motion.loc[motion["t"] >= 0, "t"].to_numpy()
        out[sid] = (labels, (start, end), set(np.floor(t / C.EPOCH_S).astype(int)))
    return out


# Synthetic: cropping and signal reductions on known input.

def test_crop_keeps_only_whole_epochs_inside_window() -> None:
    labels = pd.DataFrame({"t": np.arange(0, 300, 30), "stage": 2})
    kept = D.crop(labels, start=10.0, end=200.0)
    assert kept["t"].tolist() == [30, 60, 90, 120, 150]


def test_window_is_intersection_of_signals() -> None:
    labels = pd.DataFrame({"t": [0, 30, 60]})
    motion = pd.DataFrame({"t": [-50.0, 85.0]})
    hr = pd.DataFrame({"t": [5.0, 200.0]})
    assert D.valid_window(labels, motion, hr) == (5.0, 85.0)


def test_motion_epochs_count_samples_and_flag_gaps() -> None:
    # Epoch 0 at 2 Hz with seconds 10-14 missing; epoch 1 full. The t < 0 sample is excluded.
    t = np.arange(120) / 2
    t = np.r_[-1.0, t[(t < 10) | (t >= 15)]]
    m = D.motion_epochs(pd.DataFrame({"t": t, "x": 0.0, "y": 0.0, "z": 1.0}))
    assert m.loc[0, "acc_n"] == 50 and m.loc[0, "acc_gap"] == pytest.approx(5 / 30)
    assert m.loc[1, "acc_n"] == 60 and m.loc[1, "acc_gap"] == 0.0


def test_activity_counts_hand_computed() -> None:
    # 1 g cosine at 5 Hz, sampled at 50 Hz: |filtered| peaks at 1.0 -> bin 26 every second.
    t = np.arange(-60 * 50, 150 * 50) / 50
    c = D.count_epochs(pd.DataFrame({"t": t, "z": np.cos(2 * np.pi * 5 * t)}), end=150.0)
    assert c.loc[2] == pytest.approx(2 * (15 * 26 - 18) * 3.07)
    assert (D.count_epochs(pd.DataFrame({"t": t, "z": np.ones_like(t)}), end=150.0) == 0).all()


def test_hr_counts_per_epoch() -> None:
    n = D.hr_counts(pd.DataFrame({"t": [-3.0, 1.0, 6.0, 29.9, 31.0], "hr": 60.0}))
    assert n.to_dict() == {0: 3, 1: 1}


# Real data: epochs.parquet against the raw files.

@needs_data
def test_all_subjects_present(epochs: pd.DataFrame) -> None:
    assert epochs["subject"].nunique() == C.N_SUBJECTS


@needs_data
def test_kept_epochs_are_exactly_labeled_in_window_with_motion(epochs: pd.DataFrame, raw: dict) -> None:
    for sid, (labels, (start, end), motion_epochs) in raw.items():
        grid = D.crop(labels, start, end)
        expect = grid[(grid["stage"] != C.UNSCORED) & (grid["t"] // C.EPOCH_S).isin(motion_epochs)]
        got = epochs.loc[epochs["subject"] == sid, "t"]
        assert got.tolist() == expect["t"].tolist(), sid


@needs_data
def test_every_dropped_row_has_a_reason(epochs: pd.DataFrame, raw: dict) -> None:
    for sid, (labels, (start, end), motion_epochs) in raw.items():
        kept = set(epochs.loc[epochs["subject"] == sid, "t"])
        d = labels[~labels["t"].isin(kept)]
        outside = (d["t"] < start) | (d["t"] + C.EPOCH_S > end)
        no_motion = ~(d["t"] // C.EPOCH_S).isin(motion_epochs)
        assert (outside | (d["stage"] == C.UNSCORED) | no_motion).all(), sid


@needs_data
def test_timestamps_increasing_on_30s_grid(epochs: pd.DataFrame) -> None:
    for sid, g in epochs.groupby("subject"):
        t = g["t"].to_numpy()
        assert np.all(np.diff(t) > 0) and np.all(t % C.EPOCH_S == 0), sid
        assert np.all(t == g["epoch"].to_numpy() * C.EPOCH_S), sid


@needs_data
def test_every_label_comes_from_raw_file(epochs: pd.DataFrame, raw: dict) -> None:
    for sid, (labels, _, _) in raw.items():
        g = epochs[epochs["subject"] == sid]
        merged = g[["t", "stage"]].merge(labels, on="t", how="left", suffixes=("", "_raw"))
        assert (merged["stage"] == merged["stage_raw"]).all(), sid
    assert set(epochs["stage"].unique()) <= set(C.LABELED)


@needs_data
def test_signal_columns_consistent(epochs: pd.DataFrame) -> None:
    assert (epochs["acc_n"] > 0).all()
    assert epochs["acc_gap"].between(0, 1).all()
    assert epochs["walch_hr"].notna().mean() > 0.95


@needs_data
def test_counts_everywhere_and_band_flag_only_for_10hz(epochs: pd.DataFrame) -> None:
    subj = pd.read_csv(C.SUBJECTS_PATH, dtype={"subject": str}).set_index("subject")
    assert set(subj.index[~subj["counts_full_band"]]) == {"5383425", "8258170"}
    assert epochs["act_count"].notna().all()
    assert "acc_hz" not in epochs.columns  # device rate is metadata, never a feature
