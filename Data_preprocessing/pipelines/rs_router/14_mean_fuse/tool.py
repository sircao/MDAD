"""RS-TOUTER score fusion — mean fusion."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult, require


def _minmax(scores):
    scores = np.asarray(scores, dtype=float)
    low, high = np.min(scores), np.max(scores)
    return (scores - low) / (high - low) if high > low else np.zeros_like(scores)


class MeanFuseTool(BaseTool):
    name = "rs_router.mean_fuse"
    category = "RS-TOUTER"
    version = "1.0.0"
    description = "Unify the scales of scores from different experts/variables, then fuse them by averaging to obtain comparable final scores in [0,1]."
    libraries = ("sklearn", "numpy")
    origin = "TS-Router-D8FF/ts_router.py:42-49 mean_fuse()"

    def run(self, score_columns) -> ToolResult:
        params = {}
        try:
            require(["sklearn"])
            from sklearn.preprocessing import StandardScaler

            lengths = {len(np.asarray(c).ravel()) for c in score_columns}
            if len(lengths) != 1:
                return self.fail(ValueError(f"Expert score lengths are inconsistent: {sorted(lengths)}"), params)
            matrix = np.stack([np.asarray(c, dtype=float).ravel() for c in score_columns], axis=1)
            if not np.isfinite(matrix).all():
                return self.fail(ValueError("An expert returned non-finite scores"), params)
            fused = StandardScaler().fit_transform(matrix).mean(axis=1)
            return self.ok(_minmax(fused), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = MeanFuseTool()
