"""Small transformer encoder over windows of per-epoch features, one class per epoch. Same
features, preprocessing, LOSO folds and inner tuning folds as the paper models (invariant 4)."""

import itertools
import sys
import time

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn

from sleepwear import config as C
from sleepwear import features as F
from sleepwear import labels as L
from sleepwear import models as M
from sleepwear.cv import inner_folds, loso_folds

GRID = {"d_model": [32, 64], "window": [32, 64, 128], "dropout": [0.1, 0.3], "lr": [3e-4, 1e-3]}
CONFIGS = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
MAX_EPOCHS, PATIENCE, BATCH, N_VAL = 100, 10, 64, 3  # N_VAL: early-stopping subjects in the final fit
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = C.MODELS_DIR / "transformer"


class Encoder(nn.Module):
    """Linear projection, sinusoidal positions, 2 encoder layers of 4 heads, per-epoch head."""

    def __init__(self, n_in: int, n_classes: int, d_model: int, dropout: float, max_len: int = 512):
        super().__init__()
        self.proj = nn.Linear(n_in, d_model)
        angle = torch.arange(max_len)[:, None] * torch.exp(-np.log(1e4) * torch.arange(0, d_model, 2) / d_model)
        self.register_buffer("pe", torch.stack([torch.sin(angle), torch.cos(angle)], -1).flatten(1))
        layer = nn.TransformerEncoderLayer(d_model, 4, 4 * d_model, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + self.pe[: x.shape[1]]

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.embed(x), src_key_padding_mask=pad))


def starts(T: int, W: int) -> list[int]:
    """Window starts at stride W/4; the last window is aligned to the night's end."""
    s = list(range(0, max(T - W, 0) + 1, W // 4))
    return s + [T - W] if s[-1] + W < T else s


def windows(nights: list[tuple[np.ndarray, np.ndarray]], W: int) -> tuple[torch.Tensor, ...]:
    """Stack every window of every night. Nights shorter than W are zero-padded and masked."""
    xs, ys = [], []
    for X, y in nights:
        for s in starts(len(y), W):
            n = min(W, len(y) - s)
            xs.append(np.pad(X[s:s + n], ((0, W - n), (0, 0))))
            ys.append(np.pad(y[s:s + n].astype("int64"), (0, W - n), constant_values=-100))
    x, y = torch.tensor(np.stack(xs), device=DEVICE), torch.tensor(np.stack(ys), device=DEVICE)
    return x, y, y == -100


@torch.no_grad()
def predict(model: Encoder, nights: list, W: int, n: int, data: tuple | None = None) -> list[np.ndarray]:
    """Per-epoch class probabilities for each night, averaged over overlapping windows.
    `data` is windows(nights, W), passed in when the same nights are scored repeatedly."""
    model.eval()
    x, _, pad = data or windows(nights, W)
    p, k, out = torch.softmax(model(x, pad), -1).double().cpu().numpy(), 0, []
    for _, y in nights:
        total, count = np.zeros((len(y), n)), np.zeros(len(y))
        for s in starts(len(y), W):
            m = min(W, len(y) - s)
            total[s:s + m] += p[k, :m]
            count[s:s + m] += 1
            k += 1
        out.append(total / count[:, None])
    return out


@torch.no_grad()
def attention(model: Encoder, x: torch.Tensor) -> np.ndarray:
    """Last layer's self-attention weights (mean over heads) for one window, (W, W)."""
    model.eval()
    h = model.embed(x)
    for layer in model.encoder.layers:  # post-norm layers: attention reads the layer input
        w = layer.self_attn(h, h, h, need_weights=True)[1]
        h = layer(h)
    return w[0].cpu().numpy()


def score(model: Encoder, nights: list, W: int, n: int, data: tuple) -> float:
    """Class-balanced log loss over the given nights: the same criterion as models.py tuning."""
    p = np.concatenate(predict(model, nights, W, n, data))
    y = np.concatenate([y for _, y in nights])
    return log_loss(y, p, sample_weight=compute_sample_weight("balanced", y), labels=range(n))


def train(fit: list, val: list, cfg: dict, n: int, seed: int) -> tuple[Encoder, float]:
    """AdamW on weighted cross-entropy, early stopping on the validation nights' score."""
    torch.manual_seed(seed)
    W = cfg["window"]
    x, y, pad = windows(fit, W)
    val_data = windows(val, W)  # built once, scored after every pass
    counts = torch.bincount(y[~pad], minlength=n).float()
    loss_fn = nn.CrossEntropyLoss(weight=counts.sum() / (n * counts), ignore_index=-100)
    model = Encoder(x.shape[2], n, cfg["d_model"], cfg["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    best, best_state, since = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for b in torch.randperm(len(x), device=DEVICE).split(BATCH):
            opt.zero_grad()
            loss_fn(model(x[b], pad[b]).reshape(-1, n), y[b].reshape(-1)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_score = score(model, val, W, n, val_data)
        if val_score < best:
            best, since = val_score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif (since := since + 1) >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best


def nights(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, ids) -> list:
    """(features, labels) per subject night, rows in epoch order; dropped epochs are skipped."""
    return [(X[subjects == s], y[subjects == s]) for s in ids]


def fold(df: pd.DataFrame, y: np.ndarray, n: int, test_sid: str) -> tuple[pd.DataFrame, dict]:
    """Tune on the inner folds, then fit 3 seeds and predict the held-out subject."""
    torch.set_num_threads(2)
    subjects, X = df["subject"].to_numpy(), df[F.FEATURES].to_numpy("float64")
    is_train = subjects != test_sid
    sub_tr = subjects[is_train]
    losses = np.zeros(len(CONFIGS))
    for fit_idx, val_idx in inner_folds(sub_tr):  # identical folds to the paper models' tuning
        fit_ids, val_ids = np.unique(sub_tr[fit_idx]), np.unique(sub_tr[val_idx])
        prep = Pipeline(M.preprocess()).fit(X[is_train][fit_idx])  # scaling inside the inner fold too
        Xs = prep.transform(X).astype("float32")
        for k, cfg in enumerate(CONFIGS):
            losses[k] += train(nights(Xs, y, subjects, fit_ids), nights(Xs, y, subjects, val_ids),
                               cfg, n, seed=0)[1] / C.INNER_FOLDS
    cfg = CONFIGS[int(np.argmin(losses))]
    prep = Pipeline(M.preprocess()).fit(X[is_train])
    Xs = prep.transform(X).astype("float32")
    test = nights(Xs, y, subjects, [test_sid])[0]
    rows = []
    for seed in C.SEEDS:
        val_ids = np.random.default_rng(seed).choice(np.unique(sub_tr), N_VAL, replace=False)
        fit_ids = np.setdiff1d(np.unique(sub_tr), val_ids)
        model, _ = train(nights(Xs, y, subjects, fit_ids), nights(Xs, y, subjects, val_ids), cfg, n, seed)
        torch.save({"state": model.state_dict(), "cfg": cfg, "prep": prep},
                   CKPT / f"{n}class_{test_sid}_seed{seed}.pt")
        p = predict(model, [test], cfg["window"], n)[0]
        out = df.loc[~is_train, ["subject", "epoch", "stage"]].assign(
            y=test[1], model="transformer", seed=seed, pred=p.argmax(1))
        rows.append(out.assign(**{f"p{k}": p[:, k] for k in range(n)}))
    return pd.concat(rows), {"classes": n, "model": "transformer", "subject": test_sid,
                             "params": str(cfg), "inner_loss": float(losses.min())}


def main() -> None:
    """Write models/oof_transformer_{n}class.parquet and models/params_transformer_{n}class.csv."""
    df = pd.read_parquet(C.FEATURES_PATH)
    CKPT.mkdir(parents=True, exist_ok=True)
    for n in [int(a) for a in sys.argv[1:]] or list(L.TARGETS):
        start, y = time.time(), L.TARGETS[n](df["stage"])
        sids = [df["subject"].iloc[te[0]] for _, te in loso_folds(df["subject"])]
        res = Parallel(n_jobs=4, verbose=5)(delayed(fold)(df, y, n, s) for s in sids)
        pd.concat([r[0] for r in res]).to_parquet(C.MODELS_DIR / f"oof_transformer_{n}class.parquet")
        pd.DataFrame([r[1] for r in res]).to_csv(C.MODELS_DIR / f"params_transformer_{n}class.csv", index=False)
        print(f"{n}-class transformer {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
