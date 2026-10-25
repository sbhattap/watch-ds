"""The paper's three per-epoch features from epochs.parquet: activity counts, HR standard
deviation, clock proxy.

Raw values only. No scaling or imputation here; that happens inside the fold. hr_std (computed
at 1 Hz in data.py) carries the reference's per-night normalisation.
"""

import numpy as np
import pandas as pd

from sleepwear import config as C

KEYS = ["subject", "epoch", "t", "stage"]
FEATURES = ["activity", "hr_std", "clock"]
# The paper's feature sets, for the reproduction gate.
FEATURE_SETS = {
    "motion": ["activity"],
    "hr": ["hr_std"],
    "motion_hr": ["activity", "hr_std"],
    "motion_hr_clock": ["activity", "hr_std", "clock"],
}


def activity(df: pd.DataFrame) -> pd.Series:
    """Gaussian (sigma 50 s) weighted mean of per-15 s counts over +/-9 epochs, within each
    subject, on the 30 s grid. Dropped epochs are holes, not neighbours; weights renormalise."""
    k = np.arange(-9, 10) * C.EPOCH_S / C.COUNT_SIGMA_S
    gauss = np.exp(-0.5 * k**2)
    out = []
    for _, g in df.groupby("subject", sort=False):
        grid = g.set_index("epoch")["act_count"].reindex(range(g["epoch"].min(), g["epoch"].max() + 1))
        c = grid / 2  # reference counts are per 15 s
        # "full" then trim: "same" returns the kernel's length for nights shorter than it.
        den = np.convolve(c.notna(), gauss)[9:9 + len(c)]
        smooth = np.convolve(c.fillna(0), gauss)[9:9 + len(c)] / np.where(den > 0, den, np.nan)
        out.append(pd.Series(smooth, index=c.index).loc[g["epoch"]].set_axis(g.index))
    return pd.concat(out).rename("activity")


def clock(t: pd.Series) -> pd.Series:
    """The reference's cosine clock proxy, t in seconds from PSG start (no wall clock exists)."""
    return (-np.cos((t.astype("float64") - 5 * 3600) * 2 * np.pi / 86400)).rename("clock")


def build_features(ep: pd.DataFrame) -> pd.DataFrame:
    """Feature table, one row per row of epochs.parquet, in the same order."""
    ep = ep.reset_index(drop=True)
    return pd.concat([ep[KEYS], activity(ep), ep["walch_hr"].rename("hr_std"), clock(ep["t"])], axis=1)


def main() -> None:
    """Write data/processed/features.parquet."""
    feats = build_features(pd.read_parquet(C.EPOCHS_PATH))
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(C.FEATURES_PATH, index=False)
    print(f"wrote {len(feats)} rows x {len(FEATURES)} features -> {C.FEATURES_PATH}")


if __name__ == "__main__":
    main()
