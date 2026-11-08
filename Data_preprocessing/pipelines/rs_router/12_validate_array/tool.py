"""RS-TOUTER model internals — input array validation."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult, require


class ValidateArrayTool(BaseTool):
    name = "rs_router.validate_array"
    category = "RS-TOUTER"
    version = "1.0.0"
    description = "Validate the input shape and numeric type, intercepting non-numeric / infinite values to expose invalid inputs as early as possible."
    libraries = ("sklearn", "numpy")
    origin = "TS-Router-D8FF/models/PCA.py:230 / utils/utility.py:273 — sklearn.utils.check_array"

    def run(self, data) -> ToolResult:
        params = {}
        try:
            require(["sklearn"])
            from sklearn.utils import check_array

            out = check_array(np.asarray(data, dtype=float))
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = ValidateArrayTool()
