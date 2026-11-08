"""RS-TOUTER sliding-window feature construction — sliding window view construction."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult, require
from Data_preprocessing.core.origins import RS_ROUTER_ROOT, ensure


class SlidingWindowTool(BaseTool):
    name = "rs_router.sliding_window"
    category = "RS-TOUTER"
    version = "1.0.0"
    description = "Convert a univariate series into windowed samples with context, serving as input features for PCA/one-class SVM."
    libraries = ("numpy",)
    origin = "TS-Router-D8FF/models/feature.py:11-14 Window.convert()"

    def run(self, data, window=100, stride=1) -> ToolResult:
        params = {"window": window, "stride": stride}
        try:
            require(["numpy"])
            ensure(RS_ROUTER_ROOT)
            from models.feature import Window

            X = np.asarray(data, dtype=float)
            if X.ndim == 1:
                X = X[:, None]
            return self.ok(Window(window=int(window), stride=int(stride)).convert(X), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = SlidingWindowTool()
