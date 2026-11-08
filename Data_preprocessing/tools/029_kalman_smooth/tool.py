"""Kalman filtering and smoothing.

Estimate the latent state with a state-space model, usable for denoising, missing-value estimation,
and trend extraction. Filtering uses only current and past information; smoothing may use future
observations. Implemented here with a local-level state-space model.
"""

from __future__ import annotations

import numpy as np

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class KalmanSmoothTool(BaseTool):
    name = "kalman_smooth"
    category = "Filtering, Decomposition, and Time-Frequency Analysis"
    version = "1.0.0"
    description = "Kalman filtering (causal) or smoothing (including future observations) with a local-level state-space model."
    libraries = ("numpy",)

    def _run_1d(self, y: np.ndarray, method: str, q_ratio: float) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        missing = np.isnan(y)
        if missing.all():
            return y.copy()
        var = np.nanvar(y)
        R = max(var, 1e-6)
        Q = q_ratio * R
        x, P = float(np.nanmean(y)), R
        xf, Pf = np.zeros(len(y)), np.zeros(len(y))
        for t in range(len(y)):
            if not missing[t]:
                K = P / (P + R)
                x = x + K * (y[t] - x)
                P = (1.0 - K) * P
            xf[t], Pf[t] = x, P
            P = P + Q
        if method == "filter":
            return xf
        xs = xf.copy()
        for t in range(len(y) - 2, -1, -1):
            C = Pf[t] / (Pf[t] + Q)
            xs[t] = xf[t] + C * (xs[t + 1] - xf[t])
        return xs

    def run(self, data, method="smooth", q_ratio=0.01) -> ToolResult:
        params = {"method": method, "q_ratio": q_ratio}
        try:
            if method not in ("filter", "smooth"):
                return self.fail(ValueError(f"method only supports filter / smooth"), params)
            df = to_frame(data)
            out = df.apply(lambda s: self._run_1d(s.to_numpy(dtype=float), method, q_ratio))
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = KalmanSmoothTool()
