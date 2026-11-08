"""RS-TOUTER output validation — finiteness and length validation."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult


class ValidateScoresTool(BaseTool):
    name = "rs_router.validate_scores"
    category = "RS-TOUTER"
    version = "1.0.0"
    description = "Intercept non-numeric / infinite / length-inconsistent outputs, ensuring downstream evaluation remains usable."
    libraries = ("numpy",)
    origin = "TS-Router-D8FF/ts_router.py:64 run_specialist() — errors if len(score)!=len(test) or non-finite"

    def run(self, scores, expected_length=None) -> ToolResult:
        params = {"expected_length": expected_length}
        try:
            score = np.asarray(scores, dtype=float).ravel()
            if expected_length is not None and len(score) != int(expected_length):
                return self.fail(
                    ValueError(f"Score length {len(score)} != expected length {int(expected_length)}"), params
                )
            if not np.isfinite(score).all():
                return self.fail(ValueError("Scores contain non-finite values"), params)
            return self.ok(score, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = ValidateScoresTool()
