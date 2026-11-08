"""Matrix Profile subsequence features.

Compute the distance between each subsequence and its nearest-neighbor segment, which
can form segment-repetitiveness and novelty features.
(Native O(n^2) implementation, suitable for shorter series; stumpy provides a faster
version.)
"""

from __future__ import annotations

import numpy as np

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_1d


class MatrixProfileTool(BaseTool):
    name = "matrix_profile"
    category = "Feature Extraction and Feature Representation"
    version = "1.0.0"
    description = "Compute the Matrix Profile and profile index formed by subsequence nearest-neighbor distances."
    libraries = ("numpy",)

    def run(self, data, window=16, exclusion_zone=None) -> ToolResult:
        params = {"window": window, "exclusion_zone": exclusion_zone}
        try:
            s = to_1d(data)
            n = len(s)
            if window <= 0 or window > n:
                return self.fail(ValueError(f"Invalid window={window} (n={n})"), params)
            if n > 3000:
                return self.fail(
                    ValueError(f"Series length {n} is too large; the native O(n^2) implementation is slow; use stumpy instead"), params
                )
            excl = exclusion_zone or window // 2
            profile = np.full(n - window + 1, np.inf)
            pindex = np.zeros(n - window + 1, dtype=int)
            for i in range(n - window + 1):
                sub = s[i:i + window]
                for j in range(n - window + 1):
                    if abs(i - j) <= excl:
                        continue
                    dist = np.linalg.norm(sub - s[j:j + window])
                    if dist < profile[i]:
                        profile[i] = dist
                        pindex[i] = j
            return self.ok({"profile": profile, "profile_index": pindex}, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = MatrixProfileTool()
