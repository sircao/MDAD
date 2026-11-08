"""RS-TOUTER anti-degeneracy handling — add noise to constant columns."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult


class AddConstNoiseTool(BaseTool):
    name = "rs_router.add_const_noise"
    category = "RS-TOUTER"
    version = "1.0.0"
    description = "Inject tiny noise into constant columns to avoid division by zero during standardization, keeping every multivariate column computable."
    libraries = ("numpy",)
    origin = "TS-Router-D8FF/ts_router.py:142-145 predict() + models/PCA.py:252-258 PCA.fit()"

    def run(self, data, scale=1e-6, seed=0) -> ToolResult:
        params = {"scale": scale, "seed": seed}
        try:
            arr = np.asarray(data, dtype=float)
            if arr.ndim == 1:
                arr = arr[:, None]
            out = arr.copy()
            rng = np.random.default_rng(seed)
            for c in range(arr.shape[1]):
                if np.all(np.std(arr[:, c], keepdims=True) <= 1e-12):
                    out[:, c] = arr[:, c] + rng.normal(0.0, scale, size=arr.shape[0])
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = AddConstNoiseTool()
