"""Changepoint-based segmentation.

Use changepoints to split the sequence into intervals with relatively consistent statistical
properties, then extract features or rescale each segment separately. ruptures provides offline
segmentation algorithms; it is used for preprocessing segmentation and is not equivalent to an
online anomaly detector.
"""

from __future__ import annotations

import numpy as np

from ...core.base import BaseTool, ToolResult, require


class ChangepointSegmentTool(BaseTool):
    name = "changepoint_segment"
    category = "Sampling, Windowing, and Data Volume Control"
    version = "1.0.0"
    description = "Use offline changepoint detection (ruptures) to split the sequence into intervals with relatively consistent statistics."
    libraries = ("ruptures",)

    def run(self, data, model="l2", n_bkps=2, min_size=2, jump=1) -> ToolResult:
        params = {"model": model, "n_bkps": n_bkps, "min_size": min_size, "jump": jump}
        try:
            require(["ruptures"])
            import ruptures as rpt

            signal = np.asarray(data, dtype=float)
            if signal.ndim == 1:
                signal = signal.reshape(-1, 1)
            algo = rpt.Binseg(model=model, min_size=min_size, jump=jump).fit(signal)
            breakpoints = algo.predict(n_bkps=n_bkps)
            return self.ok(breakpoints, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = ChangepointSegmentTool()
