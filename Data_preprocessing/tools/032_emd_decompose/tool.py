"""Empirical mode decomposition and its variants (EMD / EEMD / CEEMDAN).

Decompose a non-stationary signal into multiple oscillatory modes, each of which can be used to
construct statistical or spectral features separately.
"""

from __future__ import annotations

import numpy as np

from ...core.base import BaseTool, ToolResult, require
from ...core.utils import to_frame


class EmdDecomposeTool(BaseTool):
    name = "emd_decompose"
    category = "Filtering, Decomposition, and Time-Frequency Analysis"
    version = "1.0.0"
    description = "EMD/EEMD/CEEMDAN decompose a non-stationary signal into several oscillatory modes (IMFs)."
    libraries = ("PyEMD",)

    def run(self, data, method="emd", trials=50) -> ToolResult:
        params = {"method": method, "trials": trials}
        try:
            require(["PyEMD"])
            from PyEMD import CEEMDAN, EEMD, EMD

            df = to_frame(data)
            result = {}
            for col in df.columns:
                s = df[col].to_numpy(dtype=float)
                if method == "emd":
                    imfs = EMD().emd(s)
                elif method == "eemd":
                    imfs = EEMD(trials=trials).eemd(s)
                elif method == "ceemdan":
                    imfs = CEEMDAN(trials=trials).ceemdan(s)
                else:
                    return self.fail(ValueError(f"method only supports emd / eemd / ceemdan"), params)
                result[col] = imfs
            return self.ok(result, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = EmdDecomposeTool()
