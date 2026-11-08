"""ELKI-style distance/density outlier detection (native implementation)."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult, require

_ALGOS = ("lof", "knn", "kth_neighbor")


def _knn_dist(X: np.ndarray, n_neighbors: int, kth_only: bool) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(X))).fit(X)
    dist, _ = nn.kneighbors(X)
    if kth_only:
        return dist[:, -1]  # distance to the k-th nearest neighbor (ELKI KNNOutlier style)
    return dist[:, 1:].mean(axis=1)  # average distance to k nearest neighbors


class ElkiTool(BaseTool):
    name = "elki"
    category = "Algorithm Libraries & Frameworks"
    version = "2.0.0"
    description = "Distance/density outlier detection: LOF, k-nearest-neighbor average distance, k-th nearest-neighbor distance, outputs pointwise anomaly scores."
    libraries = ("numpy", "sklearn")
    origin = "https://elki-project.github.io/algorithms/ (native equivalent implementation)"

    def run(self, data, algorithm="lof", **kwargs) -> ToolResult:
        params = {"algorithm": algorithm, **kwargs}
        try:
            require(["numpy", "sklearn"])
            if algorithm not in _ALGOS:
                return self.fail(ValueError(f"Supported algorithms: {sorted(_ALGOS)}"), params)
            X = np.asarray(data, dtype=float)
            if X.ndim == 1:
                X = X[:, None]

            if algorithm == "lof":
                from sklearn.neighbors import LocalOutlierFactor

                score = -LocalOutlierFactor(**kwargs).fit(X).negative_outlier_factor_
            elif algorithm == "knn":
                score = _knn_dist(X, kwargs.get("n_neighbors", 10), kth_only=False)
            else:  # kth_neighbor
                score = _knn_dist(X, kwargs.get("n_neighbors", 10), kth_only=True)

            return self.ok(np.asarray(score).ravel(), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = ElkiTool()
