import numpy as np
import pytest
import torch

from sleepwear import transformer as T


@pytest.mark.parametrize("length", [5, 64, 65, 200, 950])
def test_windows_cover_every_epoch_and_stay_in_the_night(length: int) -> None:
    W = 64
    s = T.starts(length, W)
    covered = np.zeros(length, int)
    for a in s:
        covered[a:a + W] += 1
    assert (covered > 0).all() and min(s) >= 0 and (length < W or max(s) + W == length)


def test_short_night_is_padded_and_masked() -> None:
    X, y = np.ones((5, 3), "float32"), np.array([0, 1, 1, 2, 2])
    x, t, pad = T.windows([(X, y)], 8)
    assert x.shape == (1, 8, 3) and pad[0].tolist() == [False] * 5 + [True] * 3
    assert t[0, 5:].tolist() == [-100] * 3


def test_prediction_is_a_distribution_per_epoch_and_ignores_padding() -> None:
    torch.manual_seed(0)
    model = T.Encoder(3, 3, 16, 0.1).to(T.DEVICE)
    X = np.random.default_rng(0).normal(size=(40, 3)).astype("float32")
    p = T.predict(model, [(X, np.zeros(40))], 32, 3)[0]
    assert p.shape == (40, 3) and np.allclose(p.sum(1), 1)
    # A night shorter than the window: padding must not change the real epochs' output.
    short = T.predict(model, [(X[:10], np.zeros(10))], 32, 3)[0]
    x = torch.tensor(X[None, :10], device=T.DEVICE)
    with torch.no_grad():
        direct = torch.softmax(model(x, torch.zeros(1, 10, dtype=torch.bool, device=T.DEVICE)), -1)
    assert np.allclose(short, direct[0].cpu().numpy(), atol=1e-5)
