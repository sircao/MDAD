"""Variational mode decomposition (VMD).

Decompose the signal into narrowband modes with different center frequencies, usable for
constructing multi-band inputs. The number of modes and the penalty parameter need to be tuned
according to the data.
"""

from __future__ import annotations

import numpy as np

from ...core.base import BaseTool, ToolResult, require
from ...core.utils import to_frame


class VmdDecomposeTool(BaseTool):
    name = "vmd_decompose"
    category = "Filtering, Decomposition, and Time-Frequency Analysis"
    version = "1.0.0"
    description = "VMD decomposes the signal into narrowband modes with different center frequencies."
    libraries = ("vmdpy",)

    def run(self, data, K=3, alpha=2000, tau=0.0, DC=0, init=1, tol=1e-7) -> ToolResult:
        params = {"K": K, "alpha": alpha, "tau": tau, "DC": DC, "init": init, "tol": tol}
        try:
            require(["vmdpy"])
            from vmdpy import VMD

            df = to_frame(data)
            result = {}
            for col in df.columns:
                s = df[col].to_numpy(dtype=float)
                modes, _, _ = VMD(s, alpha, tau, K, DC, init, tol)
                result[col] = modes
            return self.ok(result, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = VmdDecomposeTool()
