"""TSB-AD-U dataset loading: parse filenames, split train/test, standardize by the
training segment.

Filenames look like ``001_NAB_id_1_Facility_tr_1007_1st_2014.csv``:

* ``tr_1007`` -- training segment length (the first 1007 points are the normal
  period);
* ``1st_2014`` -- position of the first anomaly within the test segment (relative
  to the test segment start, 1-based).

Honors the "protocol discipline": the standardization statistics (mean/std) are
fit only on the training segment and then applied to the test segment, so no test
distribution information leaks into preprocessing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

_FN_RE = re.compile(
    r"^(?P<id>\d+)_(?P<src>\w+)_id_\d+_\w+_tr_(?P<tr>\d+)_1st_(?P<fst>\d+)\.csv$"
)


def parse_filename(name: str) -> Dict[str, Any]:
    """Parse a TSB-AD-U filename and return id / data source / training length / first anomaly position."""
    m = _FN_RE.match(Path(name).name)
    if not m:
        return {"id": name, "src": "unknown", "train_len": None, "first_anomaly": None}
    return {
        "id": int(m.group("id")),
        "src": m.group("src"),
        "train_len": int(m.group("tr")),
        "first_anomaly": int(m.group("fst")),
    }


def load_series(path: str | Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Read a univariate CSV and return (series, labels, metadata)."""
    df = pd.read_csv(path)
    if {"Data", "Label"} <= set(df.columns):
        x = df["Data"].to_numpy(dtype=float)
        y = df["Label"].to_numpy(dtype=float)
    else:  # fallback: treat the first column as the series and the last column as labels
        x = df.iloc[:, 0].to_numpy(dtype=float)
        y = df.iloc[:, -1].to_numpy(dtype=float)
    return x, y, parse_filename(Path(path).name)


def split_standardize(path: str | Path, standardize: bool = True) -> Dict[str, Any]:
    """Load and split train/test by filename, standardizing the test segment with
    training-segment statistics.

    Returns a dict: path / meta / train / test / y_train / y_test / test_std / mu / sigma.
    """
    x, y, meta = load_series(path)
    train_len = meta.get("train_len")
    if train_len is None or train_len >= len(x):
        train_len = max(1, int(len(x) * 0.5))
    train, test = x[:train_len], x[train_len:]
    y_train, y_test = y[:train_len], y[train_len:]

    mu, sigma = float(train.mean()), float(train.std())
    if sigma < 1e-12:
        sigma = 1.0
    test_std = (test - mu) / sigma if standardize else test
    return {
        "path": str(path),
        "meta": meta,
        "train": train,
        "test": test,
        "y_train": y_train,
        "y_test": y_test,
        "test_std": test_std,
        "mu": mu,
        "sigma": sigma,
    }


def split_standardize_multi(path: str | Path, standardize: bool = True) -> Dict[str, Any]:
    """Load a TSB-AD-M multivariate series, split train/test by filename, and
    standardize column-wise with training-segment statistics.

    Returns a dict: path / meta / train / test (2-D) / y_train / y_test / test_std (2-D)
    / mu / sigma / is_multivariate=True.
    """
    df = pd.read_csv(path)
    if "Label" not in df.columns:
        raise ValueError(f"multivariate file is missing the Label column: {path}")
    y = df["Label"].to_numpy(dtype=float)
    feat_cols = [c for c in df.columns if c != "Label"]
    X = df[feat_cols].to_numpy(dtype=float)
    meta = parse_filename(Path(path).name)

    train_len = meta.get("train_len")
    if train_len is None or train_len >= len(X):
        train_len = max(1, int(len(X) * 0.5))
    Xtrain, Xtest = X[:train_len], X[train_len:]
    y_train, y_test = y[:train_len], y[train_len:]

    mu = Xtrain.mean(axis=0)
    sigma = Xtrain.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    Xtest_std = (Xtest - mu) / sigma if standardize else Xtest
    return {
        "path": str(path),
        "meta": meta,
        "is_multivariate": True,
        "train": Xtrain,
        "test": Xtest,
        "y_train": y_train,
        "y_test": y_test,
        "test_std": Xtest_std,
        "mu": mu,
        "sigma": sigma,
    }


def series_stats(x: np.ndarray) -> Dict[str, Any]:
    """Series summary statistics (univariate/multivariate), injected into the user prompt.

    Univariate series additionally carry morphology signals (kurt / skew / acf1) so
    the agent can judge the anomaly morphology: high kurt + low acf1 -> spike point
    anomalies; low/negative kurt + high acf1 -> smooth segment-level or stuck/flatline.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        return {
            "shape": list(x.shape),
            "n_samples": int(x.shape[0]),
            "n_channels": int(x.shape[1]),
            "mean_range": [round(float(x.mean(axis=0).min()), 4), round(float(x.mean(axis=0).max()), 4)],
            "n_abs_over_3": int((np.abs(x) > 3).sum()),
        }
    x = np.nan_to_num(x, nan=0.0)
    acf1 = 0.0
    if x.size > 1 and np.std(x) > 1e-12:
        acf1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    out = {
        "n": int(x.size),
        "mean": round(float(x.mean()), 4),
        "std": round(float(x.std()), 4),
        "min": round(float(x.min()), 4),
        "max": round(float(x.max()), 4),
        "n_abs_over_3": int((np.abs(x) > 3).sum()),
        "acf1": round(acf1, 2),
    }
    try:
        from scipy.stats import kurtosis as _kurt, skew as _skew

        out["kurt"] = round(float(_kurt(x)), 2)
        out["skew"] = round(float(_skew(x)), 2)
    except Exception:  # noqa: BLE001 -- if scipy is unavailable only the morphology signal is missing, not fatal
        out["kurt"] = None
        out["skew"] = None
    return out
