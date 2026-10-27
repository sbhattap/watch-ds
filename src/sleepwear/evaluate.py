"""Reproduction gate, per-subject metrics with subject-bootstrap CIs, results table, figures.
Never pools out-of-fold predictions across LOSO folds (each subject is scored by a different
model; a null model scores pooled AUC 0.348): metrics are per subject, then averaged."""

import itertools
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from sleepwear import config as C
from sleepwear import features as F
from sleepwear import labels as L
from sleepwear import models as M
from sleepwear import transformer as T
from sleepwear.cv import monte_carlo_splits

N_BOOT = 2000
ORDER = M.MODELS + ["transformer"]


def _confusion(y: np.ndarray, pred: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(n * y + pred, minlength=n * n).reshape(n, n)


def kappa(cm: np.ndarray) -> float:
    """Cohen's kappa from a confusion matrix (rows true, columns predicted)."""
    total = cm.sum()
    po, pe = np.trace(cm) / total, (cm.sum(0) @ cm.sum(1)) / total**2
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan  # one class only: undefined


def sleep_wake_at(y: np.ndarray, p_sleep: np.ndarray, sensitivity: float) -> dict:
    """The paper's operating point: interpolate the ROC threshold giving this sleep correct."""
    _, tpr, thr = roc_curve(y, p_sleep, drop_intermediate=False)
    cm = _confusion(y, (p_sleep >= np.interp(sensitivity, tpr, thr)).astype(int), 2)
    return {"acc": np.trace(cm) / cm.sum(), "wake": cm[0, 0] / cm[0].sum(), "kappa": kappa(cm)}


def gate() -> pd.DataFrame:
    """Paper protocol: 50 random 70/30 subject splits, metrics on each split's test subjects
    (one model scores them all), averaged over splits. Rule pre-registered in decisions.md."""
    df = pd.read_parquet(C.FEATURES_PATH)
    y, splits = L.y2(df["stage"]), monte_carlo_splits(df["subject"], 50, seed=2)
    rows = []
    for name in M.MODELS[1:]:
        for fs, cols in F.FEATURE_SETS.items():
            for (te_p, _), (_, te) in zip(M.fit_splits(df, cols, y, name, 2, splits), splits):
                ops = {s: sleep_wake_at(y[te], te_p[:, 1], s) for s in (0.90, 0.93)}
                rows.append({"classifier": name, "features": fs, "auc": roc_auc_score(y[te], te_p[:, 1]),
                             "wake_at_0.90": ops[0.90]["wake"], "wake_at_0.93": ops[0.93]["wake"],
                             "acc_at_0.93": ops[0.93]["acc"], "kappa_at_0.93": ops[0.93]["kappa"]})
            print(f"gate {name:7s} {fs:16s} done")
    mc = pd.DataFrame(rows).melt(["classifier", "features"], var_name="metric")
    mc = mc.groupby(["classifier", "features", "metric"])["value"].agg(mc_mean="mean", mc_sd="std")
    out = pd.read_csv(C.TABLES / "published_walch2019.csv").join(mc, on=["classifier", "features", "metric"])
    out["diff"] = out["mc_mean"] - out["published"]
    out["within_tol"] = out["diff"].abs() <= out["tolerance"]
    auc = out[out["metric"] == "auc"]
    # Pairs the paper separates by more than the tolerance must come out in the same order.
    order = pd.DataFrame([{"classifier": c, "lower": a, "higher": b, "holds": bool(g.loc[b, "mc_mean"] > g.loc[a, "mc_mean"])}
                          for c, g in auc.set_index("features").groupby("classifier", sort=False)
                          for a, b in itertools.permutations(g.index, 2)
                          if g.loc[b, "published"] - g.loc[a, "published"] > g.loc[a, "tolerance"]])
    out.to_csv(C.TABLES / "gate.csv", index=False, float_format="%.4f")
    order.to_csv(C.TABLES / "gate_ordering.csv", index=False)
    gated = out[out["gate"]]
    print(f"gate: {int(gated['within_tol'].sum())}/{len(gated)} AUC rows within ±0.03, "
          f"{int(order['holds'].sum())}/{len(order)} ordering pairs hold")
    return out


def subject_metrics(y: np.ndarray, proba: np.ndarray, n: int) -> dict:
    """Kappa, macro-F1 and per-class recall at argmax; AUC and wake correct at 93% sleep
    correct for 2-class. A class the subject lacks has undefined (NaN) recall and F1, and
    macro-F1 is the mean over the classes the subject has."""
    cm = _confusion(y, proba.argmax(1), n)
    tp, true_n = np.diag(cm), cm.sum(1)
    f1 = np.where(true_n > 0, 2 * tp / np.maximum(true_n + cm.sum(0), 1), np.nan)
    out = {"kappa": kappa(cm), "macro_f1": np.nanmean(f1), "accuracy": tp.sum() / cm.sum()}
    out |= {f"recall_{c}": tp[i] / true_n[i] if true_n[i] else np.nan for i, c in enumerate(L.CLASSES[n])}
    if n == 2:
        both = len(set(y)) == 2
        out["auc"] = roc_auc_score(y, proba[:, 1]) if both else np.nan
        # A constant score (the baseline) has no threshold giving 93% sleep correct: undefined.
        ranked = both and np.ptp(proba[:, 1]) > 0
        out["wake_at_0.93"] = sleep_wake_at(y, proba[:, 1], 0.93)["wake"] if ranked else np.nan
    return out


def per_subject(oof: pd.DataFrame, n: int) -> pd.DataFrame:
    """Metrics per model x subject; for the transformer, the mean over its seeds."""
    rows = [{"model": m, "seed": s, "subject": sid,
             **subject_metrics(g["y"].to_numpy(), g[[f"p{k}" for k in range(n)]].to_numpy(), n)}
            for (m, s, sid), g in oof.groupby(["model", "seed", "subject"], sort=False)]
    return pd.DataFrame(rows).drop(columns="seed").groupby(["model", "subject"], sort=False).mean().reset_index()


def boot_mean(values: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float, int]:
    """nanmean over subjects with a 95% percentile bootstrap CI over subjects, and the count."""
    v = np.asarray(values, dtype="float64")
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0
    idx = np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v)))
    lo, hi = np.percentile(v[idx].mean(1), [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi), len(v)


def summarise(per: pd.DataFrame, n: int) -> pd.DataFrame:
    """One row per model: every metric's mean and 95% CI, and the subjects contributing."""
    rows = []
    for model, g in per.groupby("model", sort=False):
        r = {"classes": n, "model": model}
        for m in per.columns.drop(["model", "subject"]):
            mean, lo, hi, k = boot_mean(g[m].to_numpy())
            r |= {m: mean, f"{m}_lo": lo, f"{m}_hi": hi, f"{m}_n": k}
        rows.append(r)
    return pd.DataFrame(rows)


def paired(per: pd.DataFrame, n: int) -> pd.DataFrame:
    """Per-subject transformer minus each paper model (kappa, macro-F1, 2-class AUC), bootstrap CI."""
    rows = []
    for metric in ("kappa", "macro_f1", "auc")[: 3 if n == 2 else 2]:
        wide = per.pivot(index="subject", columns="model", values=metric)
        for m in M.MODELS:
            d = boot_mean((wide["transformer"] - wide[m]).to_numpy())
            rows.append({"classes": n, "metric": metric, "vs": m, "diff": d[0], "ci_lo": d[1],
                         "ci_hi": d[2], "n_subjects": d[3], "transformer_better_in": int((wide["transformer"] > wide[m]).sum())})
    return pd.DataFrame(rows)


def plot_confusion(oof: pd.DataFrame, model: str, path) -> None:
    """Mean over subjects of the row-normalised 3-class confusion matrix for one model."""
    names, mats = L.CLASSES[3], []
    for _, g in oof[oof["model"] == model].groupby(["seed", "subject"]):
        cm = _confusion(g["y"].to_numpy(), g["pred"].to_numpy(), 3).astype(float)
        mats.append(cm / np.where(cm.sum(1, keepdims=True) > 0, cm.sum(1, keepdims=True), np.nan))
    mat = np.nanmean(mats, axis=0)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.6 else "black")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=names, yticklabels=names,
           xlabel="predicted", ylabel="scored", title=f"3-class {model}, mean over subjects")
    fig.tight_layout(), fig.savefig(path, dpi=120), plt.close(fig)


def plot_box(pers: dict, path) -> None:
    """Per-subject kappa for every model, 2-class and 3-class."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (n, per) in zip(axes, pers.items()):
        data = [per.loc[per["model"] == m, "kappa"].dropna() for m in ORDER]
        ax.boxplot(data, tick_labels=ORDER, showmeans=True)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set(title=f"{n}-class: per-subject kappa (31 subjects)", ylabel="kappa")
    fig.tight_layout(), fig.savefig(path, dpi=120), plt.close(fig)


def plot_night(oof: pd.DataFrame, sid: str, path) -> None:
    """One subject's 3-class hypnogram: scored, MLP, transformer (seed 0)."""
    level = {0: 2, 2: 1, 1: 0}  # W top, REM, NREM bottom
    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True)
    g = oof[(oof["subject"] == sid) & (oof["seed"] == 0)]
    for ax, (label, col, model) in zip(axes, [("scored", "y", "mlp"), ("MLP", "pred", "mlp"),
                                              ("transformer", "pred", "transformer")]):
        h = g[g["model"] == model]
        ax.step(h["epoch"] * C.EPOCH_S / 3600, h[col].map(level), where="post", lw=1)
        ax.set(yticks=[0, 1, 2], yticklabels=["NREM", "REM", "W"], ylabel=label)
    axes[0].set_title(f"subject {sid}, 3-class")
    axes[-1].set_xlabel("hours from PSG start")
    fig.tight_layout(), fig.savefig(path, dpi=120), plt.close(fig)


def plot_attention(sid: str, path) -> None:
    """Last-layer attention (head mean) of the 3-class transformer, seed 0, around the subject's
    first NREM to REM transition."""
    ck = torch.load(C.MODELS_DIR / "transformer" / f"3class_{sid}_seed0.pt", weights_only=False)
    df = pd.read_parquet(C.FEATURES_PATH).query("subject == @sid")
    y, W = L.y3(df["stage"]), ck["cfg"]["window"]
    model = T.Encoder(3, 3, ck["cfg"]["d_model"], ck["cfg"]["dropout"])
    model.load_state_dict(ck["state"])
    at = int(np.flatnonzero((y[:-1] == 1) & (y[1:] == 2))[0]) + 1
    s = int(np.clip(at - W // 2, 0, len(y) - W))
    x = torch.tensor(ck["prep"].transform(df[F.FEATURES].to_numpy("float64"))[s:s + W][None], dtype=torch.float32)
    w = T.attention(model, x)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(6, 7), height_ratios=[1, 6], sharex=True)
    a0.step(range(W), y[s:s + W], where="mid"), a0.set(yticks=[0, 1, 2], yticklabels=L.CLASSES[3])
    a1.imshow(w, cmap="viridis", aspect="auto")
    a1.set(xlabel="attended epoch (in window)", ylabel="query epoch",
           title=f"subject {sid}: attention around NREM to REM at epoch {at - s}")
    fig.tight_layout(), fig.savefig(path, dpi=120), plt.close(fig)


def main() -> None:
    """main.csv, per_subject.csv, transformer_vs_paper.csv, tuning.csv, and the figures."""
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    pers, sums, diffs, oofs = {}, [], [], {}
    for n in L.TARGETS:
        oofs[n] = pd.concat([pd.read_parquet(C.MODELS_DIR / f"oof_{n}class.parquet"),
                             pd.read_parquet(C.MODELS_DIR / f"oof_transformer_{n}class.parquet")])
        pers[n] = per_subject(oofs[n], n)
        sums.append(summarise(pers[n], n)), diffs.append(paired(pers[n], n))
    pd.concat(sums).to_csv(C.TABLES / "main.csv", index=False, float_format="%.4f")
    pd.concat(diffs).to_csv(C.TABLES / "transformer_vs_paper.csv", index=False, float_format="%.4f")
    pd.concat([p.assign(classes=n) for n, p in pers.items()]).to_csv(
        C.TABLES / "per_subject.csv", index=False, float_format="%.4f")
    pd.concat([pd.read_csv(C.MODELS_DIR / f"params_{m}{n}class.csv") for n in L.TARGETS
               for m in ("", "transformer_")]).to_csv(C.TABLES / "tuning.csv", index=False)
    best = sums[1].set_index("model")["kappa"].drop("baseline").idxmax()
    plot_confusion(oofs[3], best, C.FIGURES / "confusion_3class_best.png")
    plot_box(pers, C.FIGURES / "kappa_per_subject.png")
    # The median subject by transformer kappa, not a showcase, among nights with an NREM to REM
    # transition (the attention figure needs one; 7749105 and 5132496 have no REM).
    y = oofs[3].query("model == 'mlp'").groupby("subject")["y"].agg(lambda v: ((v.values[:-1] == 1) & (v.values[1:] == 2)).any())
    k = pers[3].query("model == 'transformer'").set_index("subject")["kappa"][y]
    sid = (k - k.median()).abs().idxmin()
    plot_night(oofs[3], sid, C.FIGURES / f"hypnogram_3class_{sid}.png")
    plot_attention(sid, C.FIGURES / f"attention_3class_{sid}.png")


if __name__ == "__main__":
    gate() if "--gate" in sys.argv else main()
