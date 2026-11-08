"""RS-TOUTER model internals — prune zero columns."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import FitApplyTool, ToolResult


class PruneZeroColumnsTool(FitApplyTool):
    name = "rs_router.prune_zero_columns"
    category = "RS-TOUTER"
    version = "1.0.0"
    description = "Remove zero-variance / all-zero feature columns to prevent principal component decomposition from degrading or crashing."
    libraries = ("numpy",)
    origin = "TS-Router-D8FF/models/PCA.py:249-263 PCA.fit() zero_pruning"

    def fit(self, data) -> ToolResult:
        params = {}
        try:
            X = np.asarray(data, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            non_zero = np.any(X != 0, axis=0)
            if not np.any(non_zero):  # keep at least one column when all are zero
                non_zero = np.ones(X.shape[1], dtype=bool)
            self._fitted = {"non_zero_columns": non_zero}
            self._fit_provenance = self._provenance(params)
            return self.ok("Fitted", params)
        except Exception as exc:
            return self.fail(exc, params)

    def apply(self, data) -> ToolResult:
        params = {}
        try:
            if self._fitted is None:
                return self.fail(RuntimeError("not fitted yet"), params)
            X = np.asarray(data, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return self.ok(X[:, self._fitted["non_zero_columns"]], params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = PruneZeroColumnsTool()
