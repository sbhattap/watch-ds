"""Raw PSG stage codes to class targets at 2 and 3 classes. Code 4 (R&K S4) is NREM sleep."""

import numpy as np
import pandas as pd

from sleepwear import config as C

CLASSES = {2: ["W", "Sleep"], 3: ["W", "NREM", "REM"]}  # index = y


def _map(stage: pd.Series, mapping: dict[int, int]) -> np.ndarray:
    unexpected = set(stage.unique()) - set(C.LABELED)
    if unexpected:
        raise ValueError(f"unexpected stage codes {sorted(unexpected)}")
    return stage.map(mapping).to_numpy(dtype="int8")


def y2(stage: pd.Series) -> np.ndarray:
    """Wake 0, Sleep 1."""
    return _map(stage, {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1})


def y3(stage: pd.Series) -> np.ndarray:
    """Wake 0, NREM 1, REM 2."""
    return _map(stage, {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2})


TARGETS = {2: y2, 3: y3}
