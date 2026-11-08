"""RS-TOUTER normalization — split standardization."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult


def _zscore(data, zero_std=1.0):
    data = np.asarray(data, dtype=float)
    std = np.std(data, axis=0, keepdims=True)
    return (data - np.mean(data, axis=0, keepdims=True)) / np.where(std == 0, zero_std, std)


class SplitZscoreTool(BaseTool):
    name = "rs_router.split_zscore"
    category = "RS-TOUTER"
    version = "1.0.0"
    description = "Standardize the training and test sets independently, removing scale differences without mixing test statistics into the training fit."
    libraries = ("numpy",)
    origin = "TS-Router-D8FF/ts_router.py:28-31 zscore() + :146 predict() — train,test = zscore(train),zscore(test)"

    def run(self, train, test, zero_std=1.0) -> ToolResult:
        params = {"zero_std": zero_std}
        try:
            return self.ok(
                {"train_norm": _zscore(train, zero_std), "test_norm": _zscore(test, zero_std)},
                params,
            )
        except Exception as exc:
            return self.fail(exc, params)


TOOL = SplitZscoreTool()
