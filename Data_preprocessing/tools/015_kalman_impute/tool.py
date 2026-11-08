"""State-space and Kalman filtering imputation.

Use a state-space model to estimate the latent state and observation at missing moments,
suited to fairly smooth series with dynamic structure. Here a local-level state-space model
(random walk + observation noise) implements Kalman smoothing, treating missing values as
unobserved states; when the model is mis-specified, the imputation may deviate from the
true variation.
"""

from __future__ import annotations

import numpy as np

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class KalmanImputeTool(BaseTool):
    name = "kalman_impute"
    category = "Missing Value Handling"
    version = "1.0.0"
    description = "Kalman smoothing imputation with a local-level state-space model (forward filtering + backward smoothing)."
    libraries = ("numpy",)

    def _smooth_1d(self, y: np.ndarray, q_ratio: float) -> np.ndarray:
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
        xs = xf.copy()
        for t in range(len(y) - 2, -1, -1):
            C = Pf[t] / (Pf[t] + Q)
            xs[t] = xf[t] + C * (xs[t + 1] - xf[t])
        out = y.copy()
        out[missing] = xs[missing]
        return out

    def run(self, data, q_ratio=0.01) -> ToolResult:
        params = {"q_ratio": q_ratio}
        try:
            df = to_frame(data)
            out = df.apply(lambda s: self._smooth_1d(s.to_numpy(dtype=float), q_ratio))
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = KalmanImputeTool()
