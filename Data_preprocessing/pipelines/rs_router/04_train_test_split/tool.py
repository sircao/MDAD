"""RS-TOUTER data splitting — train/test split."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult


class TrainTestSplitTool(BaseTool):
    name = "rs_router.train_test_split"
    category = "RS-TOUTER"
    version = "1.0.0"
    description = "Split into a reference prefix and a to-be-scored suffix, deciding which points participate in fitting and which produce scores."
    libraries = ("numpy",)
    origin = "TS-Router-D8FF/ts_router.py:140-141 predict() — train=data[:train_index]; test=data[train_index:]"

    def run(self, data, train_index) -> ToolResult:
        params = {"train_index": train_index}
        try:
            data = np.asarray(data, dtype=float)
            if not 0 < int(train_index) < len(data):
                return self.fail(ValueError("train_index must leave a non-empty prefix and suffix"), params)
            idx = int(train_index)
            return self.ok({"train": data[:idx], "test": data[idx:]}, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = TrainTestSplitTool()
