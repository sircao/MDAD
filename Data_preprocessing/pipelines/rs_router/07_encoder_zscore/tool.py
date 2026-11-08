"""RS-TOUTER normalization — encoder input standardization."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult


class EncoderZscoreTool(BaseTool):
    name = "rs_router.encoder_zscore"
    category = "RS-TOUTER"
    version = "1.0.0"
    description = "Normalize a single-channel series before feeding it into the time-series foundation model, aligning the representation with the pretraining distribution."
    libraries = ("numpy",)
    origin = "TS-Router-D8FF/ts_router.py:99 embed() — normalized = zscore(series, zero_std=1e-8)"

    def run(self, series, zero_std=1e-8) -> ToolResult:
        params = {"zero_std": zero_std}
        try:
            s = np.asarray(series, dtype=np.float32).reshape(-1, 1)
            std = np.std(s, axis=0, keepdims=True)
            out = ((s - np.mean(s, axis=0, keepdims=True)) / np.where(std == 0, zero_std, std)).ravel()
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = EncoderZscoreTool()
