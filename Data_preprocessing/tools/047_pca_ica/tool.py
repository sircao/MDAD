"""PCA, ICA, and low-rank representation.

Reduce dimensionality and redundancy via principal or independent components, and
optionally compute projections and reconstruction residuals. Low-variance variables
sometimes still contain important fault information, so evaluate before reducing
dimensionality.
"""

from __future__ import annotations

import numpy as np

from ...core.base import FitApplyTool, ToolResult
from ...core.utils import to_frame


class PcaIcaTool(FitApplyTool):
    name = "pca_ica"
    category = "Multivariate Fusion, Encoding, and Dimensionality Reduction"
    version = "1.0.0"
    description = "PCA/ICA dimensionality-reduction projection, optionally outputting reconstruction residuals."
    libraries = ("sklearn", "numpy")

    def fit(self, data, method="pca", n_components=2) -> ToolResult:
        params = {"method": method, "n_components": n_components}
        try:
            from sklearn.decomposition import PCA, FastICA

            if method == "pca":
                self._fitted = {"method": method, "model": PCA(n_components=n_components)}
            elif method == "ica":
                self._fitted = {"method": method, "model": FastICA(n_components=n_components, random_state=0)}
            else:
                return self.fail(ValueError(f"method only supports pca / ica"), params)
            self._fitted["model"].fit(to_frame(data))
            self._fit_provenance = self._provenance(params)
            return self.ok("Fitted", params)
        except Exception as exc:
            return self.fail(exc, params)

    def apply(self, data, with_residual=False) -> ToolResult:
        params = {"with_residual": with_residual}
        try:
            if self._fitted is None:
                return self.fail(RuntimeError("Not fitted yet; call fit() on the training set first"), params)
            df = to_frame(data)
            model = self._fitted["model"]
            proj = model.transform(df)
            if not with_residual:
                return self.ok(proj, params)
            recon = model.inverse_transform(proj)
            residual = df.to_numpy(dtype=float) - recon
            return self.ok({"projection": proj, "reconstruction_residual": residual}, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = PcaIcaTool()
