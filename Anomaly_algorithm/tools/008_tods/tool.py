"""TODS-style point-level/segment-level anomaly detection (native implementation)."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult, require
from Data_preprocessing.core.utils import to_frame

_DETECTORS = ("zscore", "iqr", "moving_average", "isolation_forest")


class TodsTool(BaseTool):
    name = "tods"
    category = "Algorithm Libraries & Frameworks"
    version = "2.0.0"
    description = "Point-level/segment-level detection: z-score/IQR/moving-average residual/isolation forest, outputs pointwise scores or flags."
    libraries = ("numpy", "pandas", "sklearn")
    origin = "https://github.com/datamllab/tods (native equivalent implementation)"

    def run(self, data, detector="zscore", return_scores=False, **kwargs) -> ToolResult:
        params = {"detector": detector, "return_scores": return_scores, **kwargs}
        try:
            require(["numpy", "pandas", "sklearn"])
            if detector not in _DETECTORS:
                return self.fail(ValueError(f"Supported detectors: {sorted(_DETECTORS)}"), params)
            df = to_frame(data)

            if detector == "zscore":
                z = (df - df.mean()) / df.std().replace(0, 1e-8)
                out = z.abs() if return_scores else z.abs() > kwargs.get("threshold", 3.0)
            elif detector == "iqr":
                q1, q3 = df.quantile(0.25), df.quantile(0.75)
                iqr = q3 - q1
                out = (df - q3) / iqr.replace(0, 1e-8) if return_scores else (df < q1 - 1.5 * iqr) | (df > q3 + 1.5 * iqr)
            elif detector == "moving_average":
                window = kwargs.get("window", 10)
                resid = (df - df.rolling(window, min_periods=1).mean()).abs()
                out = resid if return_scores else resid > kwargs.get("threshold", 3.0) * resid.std()
            else:  # isolation_forest per column
                from sklearn.ensemble import IsolationForest

                def _col(s):
                    clf = IsolationForest(**kwargs).fit(s.to_numpy().reshape(-1, 1))
                    pred = clf.predict(s.to_numpy().reshape(-1, 1))
                    return -clf.score_samples(s.to_numpy().reshape(-1, 1)) if return_scores else pred == -1

                out = df.apply(_col)

            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = TodsTool()
