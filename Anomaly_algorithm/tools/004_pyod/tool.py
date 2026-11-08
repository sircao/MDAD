"""PyOD-style unified anomaly detection interface (native implementation, with HBOS/KNN distance)."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult, require

_ALGOS = ("iforest", "lof", "ocsvm", "hbos", "knn", "pca")


def _hbos(X: np.ndarray, n_bins: int = 10, tol: float = 0.5) -> np.ndarray:
    """Histogram-based outlier score: accumulate log(1/bin density) across features; higher values are more anomalous."""
    X = np.asarray(X, dtype=float)
    scores = np.zeros(X.shape[0])
    for j in range(X.shape[1]):
        col = X[:, j]
        hist, edges = np.histogram(col, bins=n_bins)
        hist = hist.astype(float)
        idx = np.clip(np.digitize(col, edges) - 1, 0, n_bins - 1)
        scores += np.log(1.0 / (hist[idx] / X.shape[0] + tol))
    return scores


def _knn_distance(X: np.ndarray, n_neighbors: int = 10) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(X))).fit(X)
    dist, _ = nn.kneighbors(X)
    return dist[:, 1:].mean(axis=1)  # exclude self (distance 0)


class PyodTool(BaseTool):
    name = "pyod"
    category = "Algorithm Libraries & Frameworks"
    version = "2.0.0"
    description = "Unified anomaly detection interface: isolation forest/LOF/OCSVM/HBOS/KNN/PCA reconstruction, outputs pointwise anomaly scores."
    libraries = ("numpy", "scipy", "sklearn")
    origin = "https://github.com/yzhao062/pyod (native equivalent implementation, interface semantics aligned)"

    def run(self, data, algorithm="iforest", **kwargs) -> ToolResult:
        params = {"algorithm": algorithm, **kwargs}
        try:
            require(["numpy", "scipy", "sklearn"])
            if algorithm not in _ALGOS:
                return self.fail(ValueError(f"Supported algorithms: {sorted(_ALGOS)}"), params)
            X = np.asarray(data, dtype=float)
            if X.ndim == 1:
                X = X[:, None]

            if algorithm == "iforest":
                from sklearn.ensemble import IsolationForest

                clf = IsolationForest(**kwargs).fit(X)
                score = -clf.score_samples(X)
            elif algorithm == "lof":
                from sklearn.neighbors import LocalOutlierFactor

                clf = LocalOutlierFactor(**kwargs).fit(X)
                score = -clf.negative_outlier_factor_
            elif algorithm == "ocsvm":
                from sklearn.svm import OneClassSVM

                clf = OneClassSVM(**kwargs).fit(X)
                score = -clf.decision_function(X)
            elif algorithm == "hbos":
                score = _hbos(X, kwargs.get("n_bins", 10), kwargs.get("tol", 0.5))
            elif algorithm == "knn":
                score = _knn_distance(X, kwargs.get("n_neighbors", 10))
            else:  # pca reconstruction error
                from sklearn.decomposition import PCA

                pca = PCA(n_components=kwargs.get("n_components", None)).fit(X)
                recon = pca.inverse_transform(pca.transform(X))
                score = np.mean((X - recon) ** 2, axis=1)

            return self.ok(np.asarray(score).ravel(), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = PyodTool()
