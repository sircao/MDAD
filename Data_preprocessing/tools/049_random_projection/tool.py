"""Random projection and feature compression.

Map high-dimensional data into a lower-dimensional space to reduce computation and
storage overhead. After compression, confirm that anomaly-relevant information is still
preserved.
"""

from __future__ import annotations

from ...core.base import FitApplyTool, ToolResult
from ...core.utils import to_frame


class RandomProjectionTool(FitApplyTool):
    name = "random_projection"
    category = "Multivariate Fusion, Encoding, and Dimensionality Reduction"
    version = "1.0.0"
    description = "Randomly project high-dimensional data into a low-dimensional space to reduce computation and storage overhead."
    libraries = ("sklearn",)

    def fit(self, data, n_components=8, method="gaussian", random_state=0) -> ToolResult:
        params = {"n_components": n_components, "method": method, "random_state": random_state}
        try:
            from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

            if method == "gaussian":
                model = GaussianRandomProjection(n_components=n_components, random_state=random_state)
            elif method == "sparse":
                model = SparseRandomProjection(n_components=n_components, random_state=random_state)
            else:
                return self.fail(ValueError(f"method only supports gaussian / sparse"), params)
            self._fitted = model.fit(to_frame(data))
            self._fit_provenance = self._provenance(params)
            return self.ok("Fitted", params)
        except Exception as exc:
            return self.fail(exc, params)

    def apply(self, data) -> ToolResult:
        params = {}
        try:
            if self._fitted is None:
                return self.fail(RuntimeError("Not fitted yet; call fit() on the training set first"), params)
            out = self._fitted.transform(to_frame(data))
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = RandomProjectionTool()
