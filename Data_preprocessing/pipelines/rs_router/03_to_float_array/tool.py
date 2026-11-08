"""RS-TOUTER input reading — type conversion."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult


class ToFloatArrayTool(BaseTool):
    name = "rs_router.to_float_array"
    category = "RS-TOUTER"
    version = "1.0.0"
    description = "Convert the input uniformly to a float array for use by matrix computation libraries and deep learning frameworks."
    libraries = ("numpy",)
    origin = "TS-Router-D8FF/infer.py:19 — .to_numpy(dtype=float);ts_router.py:109 — np.asarray(data, dtype=float)"

    def run(self, data) -> ToolResult:
        params = {}
        try:
            return self.ok(np.asarray(data, dtype=float), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = ToFloatArrayTool()
