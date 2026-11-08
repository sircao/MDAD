"""Merlion-style time-series anomaly detection (native implementation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from Data_preprocessing.core.base import BaseTool, ToolResult, require
from Data_preprocessing.core.utils import to_frame

_DETECTORS = ("default", "zscore", "iqr", "ewma", "isolation_forest")


def _ewma_scores(s: pd.Series, alpha: float) -> pd.Series:
    ewma = s.ewm(alpha=alpha, adjust=False).mean()
    resid = (s - ewma).abs()
    return resid > 3.0 * resid.std()


class MerlionTool(BaseTool):
    name = "merlion"
    category = "Algorithm Libraries & Frameworks"
    version = "2.0.0"
    description = "Time-series detection: default (z-score)/IQR/EWMA residual/isolation forest, outputs boolean anomaly flags."
    libraries = ("numpy", "pandas", "sklearn")
    origin = "https://github.com/salesforce/Merlion (native equivalent implementation)"

    def run(self, data, detector="default", **kwargs) -> ToolResult:
        params = {"detector": detector, **kwargs}
        try:
            require(["numpy", "pandas", "sklearn"])
            if detector not in _DETECTORS:
                return self.fail(ValueError(f"Supported detectors: {sorted(_DETECTORS)}"), params)
            df = to_frame(data)

            if detector in ("default", "zscore"):
                z = (df - df.mean()) / df.std().replace(0, 1e-8)
                anomalies = z.abs() > kwargs.get("threshold", 3.0)
            elif detector == "iqr":
                q1, q3 = df.quantile(0.25), df.quantile(0.75)
                iqr = q3 - q1
                anomalies = (df < q1 - 1.5 * iqr) | (df > q3 + 1.5 * iqr)
            elif detector == "ewma":
                alpha = kwargs.get("alpha", 0.3)
                anomalies = df.apply(lambda s: _ewma_scores(s, alpha))
            else:  # isolation_forest per column
                from sklearn.ensemble import IsolationForest

                def _col(s):
                    clf = IsolationForest(**kwargs).fit(s.to_numpy().reshape(-1, 1))
                    return clf.predict(s.to_numpy().reshape(-1, 1)) == -1

                anomalies = df.apply(_col)

            return self.ok(anomalies, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = MerlionTool()
