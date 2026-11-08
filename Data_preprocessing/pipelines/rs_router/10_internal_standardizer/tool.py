"""RS-TOUTER model internals — built-in standardizer."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import FitApplyTool, ToolResult, require
from Data_preprocessing.core.origins import RS_ROUTER_ROOT, ensure


class InternalStandardizerTool(FitApplyTool):
    name = "rs_router.internal_standardizer"
    category = "RS-TOUTER"
    version = "1.0.0"
    description = "Standardize again inside the expert, satisfying the zero-mean unit-variance assumption of principal component analysis."
    libraries = ("sklearn", "numpy")
    origin = "TS-Router-D8FF/utils/utility.py:253-291 standardizer() (called at models/PCA.py:247)"

    def fit(self, data) -> ToolResult:
        params = {}
        try:
            require(["sklearn"])
            ensure(RS_ROUTER_ROOT)
            from utils.utility import standardizer

            _, self._fitted = standardizer(np.asarray(data, dtype=float), keep_scalar=True)
            self._fit_provenance = self._provenance(params)
            return self.ok("Fitted", params)
        except Exception as exc:
            return self.fail(exc, params)

    def apply(self, data) -> ToolResult:
        params = {}
        try:
            if self._fitted is None:
                return self.fail(RuntimeError("not fitted yet"), params)
            ensure(RS_ROUTER_ROOT)
            from sklearn.utils import check_array

            return self.ok(self._fitted.transform(check_array(np.asarray(data, dtype=float))), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = InternalStandardizerTool()
