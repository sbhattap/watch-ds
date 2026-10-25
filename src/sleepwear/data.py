"""Load raw sleep-accel files, reduce every signal onto the 30 s PSG grid, write epochs.parquet."""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from sleepwear import config as C


def subject_ids() -> list[str]:
    """Subject ids, taken from the label files present in data/raw."""
    return sorted(p.name.split("_")[0] for p in (C.RAW / "labels").glob("*_labeled_sleep.txt"))


def _read(path, sep: str, names: list[str]) -> pd.DataFrame:
    # header=None matters: the files have no header row, and the default would drop row 0.
    df = pd.read_csv(path, sep=sep, header=None, names=names, engine="pyarrow")
    # Some raw files have out-of-order and repeated timestamps.
    return df.sort_values("t", kind="stable").drop_duplicates("t").reset_index(drop=True)


def load_labels(sid: str) -> pd.DataFrame:
    """PSG stages. Checks that t starts at 0, steps by one epoch, and uses known codes."""
    df = pd.read_csv(C.RAW / C.LABEL_FILE.format(sid=sid), sep=" ", header=None, names=["t", "stage"])
    t = df["t"].to_numpy()
    if t[0] != 0 or not np.all(np.diff(t) == C.EPOCH_S):
        raise ValueError(f"{sid}: label timestamps are not a 30 s grid starting at 0")
    unknown = set(df["stage"]) - set(C.STAGE_CODES)
    if unknown:
        raise ValueError(f"{sid}: unknown stage codes {sorted(unknown)}")
    return df.astype({"t": "int64", "stage": "int8"})


def load_motion(sid: str) -> pd.DataFrame:
    """Accelerometer, t in s from PSG start, x/y/z in g."""
    return _read(C.RAW / C.MOTION_FILE.format(sid=sid), " ", ["t", "x", "y", "z"])


def load_hr(sid: str) -> pd.DataFrame:
    """PPG heart rate, t in s from PSG start, bpm. Irregularly sampled."""
    return _read(C.RAW / C.HR_FILE.format(sid=sid), ",", ["t", "hr"])


def load_steps(sid: str) -> pd.DataFrame:
    """Step counts per bin; t is the bin start. Used only to report coverage."""
    return _read(C.RAW / C.STEPS_FILE.format(sid=sid), ",", ["t", "steps"])


def steps_coverage(steps: pd.DataFrame, t: np.ndarray) -> float:
    """Share of epoch start times that fall inside some step-count bin."""
    st = steps["t"].to_numpy()
    width = np.median(np.diff(st))
    i = np.searchsorted(st, t, side="right") - 1
    return float(np.mean((i >= 0) & (t < st[np.clip(i, 0, None)] + width)))


def valid_window(labels: pd.DataFrame, motion: pd.DataFrame, hr: pd.DataFrame) -> tuple[float, float]:
    """[start, end) where PSG, accelerometer and heart rate all have data."""
    start = max(labels["t"].iloc[0], motion["t"].iloc[0], hr["t"].iloc[0])
    end = min(labels["t"].iloc[-1] + C.EPOCH_S, motion["t"].iloc[-1], hr["t"].iloc[-1])
    return float(start), float(end)


def crop(labels: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    """Keep only epochs lying wholly inside [start, end)."""
    inside = (labels["t"] >= start) & (labels["t"] + C.EPOCH_S <= end)
    return labels[inside].reset_index(drop=True)


def sampling_rate(motion: pd.DataFrame) -> float:
    """Accelerometer rate in Hz, from the median sample interval during the night."""
    return float(1 / np.median(np.diff(motion.loc[motion["t"] >= 0, "t"].to_numpy())))


def motion_epochs(motion: pd.DataFrame) -> pd.DataFrame:
    """Accelerometer samples per epoch, and the share of the epoch's seconds with no sample.

    Gaps between samples are bridged by interpolation in count_epochs; acc_gap flags them.
    """
    sec = np.floor(motion.loc[motion["t"] >= 0, "t"]).astype("int64")
    g = pd.DataFrame({"epoch": sec // C.EPOCH_S, "sec": sec}).groupby("epoch")["sec"]
    return pd.DataFrame({"acc_n": g.size(), "acc_gap": 1 - g.nunique() / C.EPOCH_S})


def count_epochs(motion: pd.DataFrame, end: float) -> pd.Series:
    """Reference activity counts per epoch, z axis only."""
    fs, (lo, hi) = C.ACC_FS, C.COUNT_BAND_HZ
    # Start up to 60 s before PSG so the filter's edge transient falls outside the night.
    k0 = min(0, int(np.ceil(max(motion["t"].iloc[0], -60) * fs)))
    t = np.arange(k0, int(end * fs)) / fs
    z = np.interp(t, motion["t"], motion["z"])
    b, a = butter(5, [lo / (fs / 2), hi / (fs / 2)], "bandpass")
    zf = filtfilt(b, a, z)[-k0:]  # sample 0 is t = 0, so seconds and epochs align to the grid
    n_ep = len(zf) // (fs * C.EPOCH_S)
    binned = np.digitize(np.abs(zf[: n_ep * fs * C.EPOCH_S]), np.linspace(0, 5, 129))
    per_15s = binned.reshape(-1, fs).max(axis=1).reshape(-1, 15).sum(axis=1)
    counts = np.clip((per_15s - 18) * 3.07, 0, None).reshape(-1, 2).sum(axis=1)
    return pd.Series(counts, index=pd.RangeIndex(n_ep, name="epoch"), name="act_count")


def hr_counts(hr: pd.DataFrame) -> pd.Series:
    """Heart-rate samples per epoch."""
    return (hr.loc[hr["t"] >= 0, "t"] // C.EPOCH_S).astype("int64").value_counts().rename("hr_n")


def walch_hr_epochs(hr: pd.DataFrame, start: float, end: float) -> pd.Series:
    """Reference HR feature: HR interpolated to 1 Hz, difference-of-Gaussians filtered over the
    night, divided by its 90th percentile of |value|, then SD over (t - 285, t + 315) s per epoch.

    Gaps longer than 5 min are not bridged (the reference bridges any gap); they are excluded.
    """
    h = hr[(hr["t"] >= start) & (hr["t"] < end)]
    ht = h["t"].to_numpy()
    t = np.arange(np.ceil(ht[0]), ht[-1])
    seg = np.clip(np.searchsorted(ht, t, side="right") - 1, 0, len(ht) - 2)
    missing = np.diff(ht)[seg] > C.HR_MAX_GAP_S
    v = np.interp(t, ht, h["hr"].to_numpy())
    v = np.where(missing, 0.0, v - v[~missing].mean())
    k = np.arange(285) - 142
    box = np.exp(-0.5 * (k / 120) ** 2) - 0.75 * np.exp(-0.5 * (k / 600) ** 2)
    f = np.convolve(np.r_[v[142:0:-1], v, v[-2:-144:-1]], box, "valid")  # reflection padding
    f = np.where(missing, np.nan, f / (np.percentile(np.abs(f[~missing]), 90) or 1.0))
    ok = ~np.isnan(f)
    c0, c1, c2 = (np.r_[0, np.cumsum(a)] for a in (ok, np.where(ok, f, 0), np.where(ok, f, 0) ** 2))
    e = np.arange(int(start // C.EPOCH_S), int(end // C.EPOCH_S) + 1)
    lo, hi = np.searchsorted(t, e * C.EPOCH_S - 285, "right"), np.searchsorted(t, e * C.EPOCH_S + 315)
    n = c0[hi] - c0[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(np.maximum((c2[hi] - c2[lo]) / n - ((c1[hi] - c1[lo]) / n) ** 2, 0))
    return pd.Series(sd, index=pd.Index(e, name="epoch"), name="walch_hr")


def epochs_for_subject(sid: str) -> tuple[pd.DataFrame, dict]:
    """Labeled epochs with motion inside the valid window, and per-subject metadata."""
    labels, motion, hr = load_labels(sid), load_motion(sid), load_hr(sid)
    start, end = valid_window(labels, motion, hr)
    ep = crop(labels, start, end)
    ep["epoch"] = ep["t"] // C.EPOCH_S
    fs = sampling_rate(motion)
    # Below 22 Hz the 3-11 Hz band is cut at Nyquist; counts are still computed, as in the
    # reference, from the signal resampled to 50 Hz (in effect a 3 Hz-to-Nyquist band).
    full_band = fs / 2 > C.COUNT_BAND_HZ[1]
    counts = count_epochs(motion, end)
    ep = ep.join(motion_epochs(motion), on="epoch").join(counts, on="epoch").join(hr_counts(hr), on="epoch")
    ep = ep.join(walch_hr_epochs(hr, start, end), on="epoch")
    ep[["acc_n", "hr_n"]] = ep[["acc_n", "hr_n"]].fillna(0).astype("int32")

    unscored, no_motion = ep["stage"] == C.UNSCORED, ep["acc_n"] == 0
    keep = ~unscored & ~no_motion
    meta = {"subject": sid, "acc_hz": round(fs, 1), "counts_full_band": full_band,
            "window_epochs": len(ep), "unscored": int(unscored.sum()), "no_motion": int(no_motion.sum()),
            "dropped": int((~keep).sum()), "kept": int(keep.sum()),
            "acc_gap_epochs": int((ep["acc_gap"][keep] > 0).sum())}
    ep = ep[keep].reset_index(drop=True)
    meta.update({f"n_{C.STAGE_CODES[k]}": int((ep["stage"] == k).sum()) for k in C.LABELED})
    meta["steps_covered"] = round(steps_coverage(load_steps(sid), ep["t"].to_numpy()), 3)
    ep["subject"] = sid
    return ep[["subject", "epoch", "t", "stage", "acc_n", "acc_gap", "act_count", "hr_n", "walch_hr"]], meta


def main() -> None:
    """Write data/interim/epochs.parquet and results/tables/subjects.csv."""
    sids = subject_ids()
    if len(sids) != C.N_SUBJECTS:
        raise ValueError(f"expected {C.N_SUBJECTS} subjects in {C.RAW}, found {len(sids)}")
    frames, metas = [], []
    for sid in sids:
        ep, meta = epochs_for_subject(sid)
        frames.append(ep), metas.append(meta)
        print(f"{sid}: kept {meta['kept']}, dropped {meta['dropped']} "
              f"(unscored {meta['unscored']}, no motion {meta['no_motion']}), {meta['acc_hz']} Hz")
    ep = pd.concat(frames, ignore_index=True)
    if not set(ep["stage"].unique()) <= set(C.LABELED):
        raise ValueError("unlabeled stage code survived filtering")
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    ep.to_parquet(C.EPOCHS_PATH, index=False)
    pd.DataFrame(metas).to_csv(C.SUBJECTS_PATH, index=False)
    print(f"wrote {len(ep)} epochs, {ep['subject'].nunique()} subjects -> {C.EPOCHS_PATH}")


if __name__ == "__main__":
    main()
