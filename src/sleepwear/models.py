"""Majority baseline and the paper's four classifiers, each tuned inside the training fold, and
the leave-one-subject-out loop. Writes out-of-fold class probabilities.

Estimator settings and grids follow the reference code (sleep_classifiers, ParameterSearch).
"""

import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from sleepwear import config as C
from sleepwear import features as F
from sleepwear import labels as L
from sleepwear.cv import inner_folds, loso_folds


def preprocess() -> list:
    """Median imputation and standard scaling, fitted on the training fold only (invariant 2).
    The transformer uses the same two steps."""
    return [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]


def make(name: str, n_classes: int) -> tuple[Pipeline, dict]:
    """Pipeline and tuning grid for one model. Balanced class weighting wherever the
    estimator supports it; k-NN is balanced at prediction time (see predict_proba)."""
    if name == "baseline":  # predicts the training-fold majority class
        clf, grid = DummyClassifier(strategy="prior"), {}
    elif name == "logreg":
        lr = LogisticRegression(solver="liblinear", class_weight="balanced")
        # liblinear has no native multiclass in current sklearn; the reference relied on its OvR.
        clf, key = (lr, "clf") if n_classes == 2 else (OneVsRestClassifier(lr), "clf__estimator")
        # l1_ratio 1 / 0 is the reference's penalty l1 / l2 (`penalty` is deprecated in sklearn).
        grid = {f"{key}__C": [0.001, 0.01, 0.1, 1, 10, 100], f"{key}__l1_ratio": [1.0, 0.0]}
    elif name == "knn":
        clf, grid = KNeighborsClassifier(weights="distance"), {"clf__n_neighbors": [500, 1000]}
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=100, max_features=1.0, min_samples_split=10,
                                     min_samples_leaf=32, class_weight="balanced", n_jobs=1,
                                     random_state=0)
        grid = {"clf__max_depth": [10, 50, 100]}
    elif name == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(15, 15, 15), activation="relu", solver="adam",
                            max_iter=2000, n_iter_no_change=20, random_state=0)
        grid = {"clf__alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001]}
    else:
        raise ValueError(name)
    return Pipeline(preprocess() + [("clf", clf)]), grid


MODELS = ["baseline", "logreg", "knn", "rf", "mlp"]


def predict_proba(est: Pipeline, X: np.ndarray) -> np.ndarray:
    """Class probabilities. k-NN votes are divided by the training class counts, the
    prediction-time equivalent of balanced class weights."""
    p = est.predict_proba(X)
    if isinstance(est[-1], KNeighborsClassifier):
        p = p / np.bincount(est[-1]._y)
        p /= p.sum(1, keepdims=True)
    return p


def balanced_log_loss(est: Pipeline, X: np.ndarray, y: np.ndarray) -> float:
    """Tuning score for every model: class-balanced log loss (negated, higher is better). It
    equals the transformer's weighted cross-entropy, so all models are selected alike."""
    return -log_loss(y, predict_proba(est, X), sample_weight=compute_sample_weight("balanced", y),
                     labels=est.classes_)


def fit_predict(name: str, n_classes: int, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, dict]:
    """Tune on grouped inner folds of the training subjects, refit, predict the test rows."""
    pipe, grid = make(name, n_classes)
    # Random row order: k-NN breaks distance ties (half of all activity values are exactly 0)
    # by storage order, which would otherwise pick tied neighbours from a few subjects.
    train = np.random.default_rng(0).permutation(train)
    # MLPClassifier has no class_weight; balanced sample weights do the same job.
    kw = {"clf__sample_weight": compute_sample_weight("balanced", y[train])} if name == "mlp" else {}
    if not grid:
        return predict_proba(pipe.fit(X[train], y[train]), X[test]), {}
    search = GridSearchCV(pipe, grid, scoring=balanced_log_loss, cv=inner_folds(groups[train]))
    search.fit(X[train], y[train], **kw)
    return predict_proba(search.best_estimator_, X[test]), search.best_params_


def fit_splits(df: pd.DataFrame, cols: list[str], y: np.ndarray, name: str, n_classes: int,
               splits: list) -> list[tuple[np.ndarray, dict]]:
    """fit_predict on every (train, test) split, in parallel over splits."""
    X, groups = df[cols].to_numpy(dtype="float64"), df["subject"].to_numpy()
    return list(Parallel(n_jobs=-1)(delayed(fit_predict)(name, n_classes, X, y, groups, tr, te)
                                    for tr, te in splits))


def run(df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LOSO out-of-fold probabilities for every model, and the tuned parameters per fold."""
    y = L.TARGETS[n](df["stage"])
    folds = loso_folds(df["subject"])
    frames, params = [], []
    for name in MODELS:
        start = time.time()
        res = fit_splits(df, F.FEATURES, y, name, n, folds)
        proba = np.full((len(df), n), np.nan)
        for (p, best), (_, te) in zip(res, folds):
            proba[te] = p
            params.append({"classes": n, "model": name, "subject": df["subject"].iloc[te[0]],
                           "params": str({k.split("__")[-1]: v for k, v in best.items()})})
        out = df[["subject", "epoch", "stage"]].assign(y=y, model=name, seed=0, pred=proba.argmax(1))
        frames.append(out.assign(**{f"p{k}": proba[:, k] for k in range(n)}))
        print(f"{n}-class {name:8s} {time.time() - start:6.1f} s")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(params)


def main() -> None:
    """Write models/oof_{n}class.parquet and models/params_{n}class.csv for the tasks given."""
    df = pd.read_parquet(C.FEATURES_PATH)
    C.MODELS_DIR.mkdir(exist_ok=True)
    for n in [int(a) for a in sys.argv[1:]] or list(L.TARGETS):
        oof, params = run(df, n)
        oof.to_parquet(C.MODELS_DIR / f"oof_{n}class.parquet", index=False)
        params.to_csv(C.MODELS_DIR / f"params_{n}class.csv", index=False)


if __name__ == "__main__":
    main()
