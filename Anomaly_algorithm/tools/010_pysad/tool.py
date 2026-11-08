"""PySAD-style streaming anomaly detection (native implementation)."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult, require

_MODELS = ("exact_storm", "knnd", "sliding_zscore")


def _stream_knn_scores(X: np.ndarray, window: int, n_neighbors: int) -> np.ndarray:
    """Streaming KNN-distance scoring within a sliding buffer (ExactSTORM style)."""
    from sklearn.neighbors import NearestNeighbors

    scores = np.zeros(len(X))
    for t in range(len(X)):
        lo = max(0, t - window)
        buf = X[lo:t + 1]
        if len(buf) < n_neighbors + 1:
            continue
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(buf) - 1)).fit(buf)
        dist, _ = nn.kneighbors(X[t:t + 1])
        scores[t] = dist[0].mean()
    return scores


class PysadTool(BaseTool):
    name = "pysad"
    category = "Algorithm Libraries & Frameworks"
    version = "2.0.0"
    description = "Streaming/online anomaly detection: ExactSTORM/KNND/sliding z-score, ingests data incrementally and updates scores."
    libraries = ("numpy", "sklearn")
    origin = "https://github.com/selimfirat/pysad (native equivalent implementation)"

    def run(self, data, model="exact_storm", **kwargs) -> ToolResult:
        params = {"model": model, **kwargs}
        try:
            require(["numpy", "sklearn"])
            if model not in _MODELS:
                return self.fail(ValueError(f"Supported models: {sorted(_MODELS)}"), params)
            X = np.asarray(data, dtype=float)
            if X.ndim == 1:
                X = X[:, None]
            window = kwargs.get("window", 100)
            n_neighbors = kwargs.get("n_neighbors", 5)

            if model == "exact_storm":
                score = _stream_knn_scores(X, window, n_neighbors)
            elif model == "knnd":
                score = _stream_knn_scores(X, window, n_neighbors)
            else:  # sliding_zscore
                score = np.zeros(len(X))
                for t in range(len(X)):
                    lo = max(0, t - window)
                    buf = X[lo:t + 1]
                    std = buf.std(axis=0) + 1e-8
                    score[t] = np.mean(np.abs((X[t] - buf.mean(axis=0)) / std))
            return self.ok(score, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = PysadTool()
