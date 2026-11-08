"""RS-TOUTER score fusion — min-max normalization."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult


class MinmaxNormTool(BaseTool):
    name = "rs_router.minmax_norm"
    category = "RS-TOUTER"
    version = "1.0.0"
    description = "Min-max normalize a single score to a unified range (used when top_k=1)."
    libraries = ("numpy",)
    origin = "TS-Router-D8FF/ts_router.py:34-39 minmax()"

    def run(self, scores) -> ToolResult:
        params = {}
        try:
            scores = np.asarray(scores, dtype=float)
            if not np.isfinite(scores).all():
                return self.fail(ValueError("Scores contain non-finite values"), params)
            low, high = np.min(scores), np.max(scores)
            out = (scores - low) / (high - low) if high > low else np.zeros_like(scores)
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = MinmaxNormTool()
