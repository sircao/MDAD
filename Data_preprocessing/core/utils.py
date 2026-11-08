"""Small helpers that normalize numpy arrays / DataFrames into a unified input format.

Accepted input format: a wide DataFrame (rows=time, columns=variables) or a 2-D numpy
array. Univariate tools (filtering, decomposition, etc.) accept a Series or a 1-D numpy
array.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def to_frame(data: Any) -> pd.DataFrame:
    """Normalize a 1-D/2-D numpy array or DataFrame into a (wide) DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data
    arr = np.asarray(data)
    if arr.ndim == 1:
        return pd.DataFrame(arr.reshape(-1, 1))
    if arr.ndim == 2:
        return pd.DataFrame(arr)
    raise ValueError(f"cannot convert input with shape={arr.shape} to a DataFrame")


def to_1d(data: Any) -> np.ndarray:
    """Normalize a Series / 1-D array into a float 1-D array."""
    if isinstance(data, pd.Series):
        return data.to_numpy(dtype=float)
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D input, got shape={arr.shape}")
    return arr
