"""Two-sided KNN distance: deviation from the median distance is anomalous, regardless of direction.

KNN distance implicitly assumes that anomalies are sparse/isolated, which inverts for "dense cluster
(collective)" anomalies -- anomalous points cluster together and their distances are actually small. Taking
``|knn distance - median distance|`` treats both the large-distance (isolated) and small-distance (dense) tails
as anomalous, fixing that inversion.
"""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult, require


class TwoSidedKnnTool(BaseTool):
    name = "two_sided_knn"
    category = "Algorithm Libraries & Frameworks"
    version = "1.0.0"
    description = "Two-sided KNN distance: |knn distance - median|, direction-agnostic, fixing the inversion caused by dense-cluster anomalies."
    libraries = ("numpy", "sklearn")
    origin = "Two-sided transform (direction-agnostic), native implementation"

    def run(self, data, n_neighbors=5, **kwargs) -> ToolResult:
        params = {"n_neighbors": n_neighbors, **kwargs}
        try:
            require(["numpy", "sklearn"])
            from sklearn.neighbors import NearestNeighbors

            X = np.asarray(data, dtype=float)
            if X.ndim == 1:
                X = X[:, None]
            nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(X))).fit(X)
            d, _ = nn.kneighbors(X)
            dist = d[:, 1:].mean(axis=1)
            med = np.median(dist)
            score = np.abs(dist - med)
            return self.ok(score, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = TwoSidedKnnTool()
