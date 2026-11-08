"""Time-series compression.

Common methods include piecewise aggregate approximation (PAA), symbolic aggregate
approximation (SAX), piecewise linear approximation, wavelet compression, and low-rank
representation. Lossy compression may erase short-lived anomalies. (Native PAA/SAX
implementation; sktime provides a more complete version.)
"""

from __future__ import annotations

import numpy as np

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class SeriesCompressTool(BaseTool):
    name = "series_compress"
    category = "Multivariate Fusion, Encoding, and Dimensionality Reduction"
    version = "1.0.0"
    description = "Compress a series via PAA piecewise aggregation or SAX symbolic approximation."
    libraries = ("numpy", "scipy")

    def run(self, data, method="paa", n_segments=8, alphabet_size=4) -> ToolResult:
        params = {"method": method, "n_segments": n_segments, "alphabet_size": alphabet_size}
        try:
            df = to_frame(data)
            result = {}
            for col in df.columns:
                s = df[col].to_numpy(dtype=float)
                n = len(s)
                seg_len = n // n_segments
                if seg_len == 0:
                    return self.fail(ValueError(f"n_segments={n_segments} exceeds series length {n}"), params)
                paa = np.array([s[i * seg_len:(i + 1) * seg_len].mean() for i in range(n_segments)])
                if method == "paa":
                    result[col] = paa
                elif method == "sax":
                    from scipy.stats import norm

                    breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
                    result[col] = np.digitize(paa, breakpoints)
                else:
                    return self.fail(ValueError(f"method only supports paa / sax"), params)
            return self.ok(result, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = SeriesCompressTool()
