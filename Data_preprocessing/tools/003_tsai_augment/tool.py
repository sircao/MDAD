"""TSAI: time series standardization, tensor conversion, and augmentation.

Geared toward deep learning data pipelines. Before using augmentation for anomaly
detection, check whether it changes the anomaly semantics.
"""

from __future__ import annotations

import numpy as np

from ...core.base import BaseTool, ToolResult, require


class TsaiAugmentTool(BaseTool):
    name = "tsai_augment"
    category = "Time Series Processing and Feature Extraction"
    version = "1.0.0"
    description = "Standardization and tensor conversion, with optional augmentation such as noise / scaling / time warping."
    libraries = ("tsai",)

    def run(self, data, standardize=True, augment=None, magnitude=0.1) -> ToolResult:
        params = {"standardize": standardize, "augment": augment, "magnitude": magnitude}
        try:
            require(list(self.libraries))
            import torch
            from tsai.data.preprocessing import TSStandardize
            from tsai.data.augmentation import TSRandomNoise

            x = np.asarray(data, dtype=np.float32)
            if x.ndim == 1:
                x = x[None, None, :]  # (1, C, T)
            elif x.ndim == 2:
                x = x[None, ...]      # (1, C, T)
            t = torch.from_numpy(x)

            if standardize:
                t = TSStandardize()(t)
            if augment == "noise":
                t = TSRandomNoise(magnitude=magnitude)(t)
            elif augment not in (None, "none"):
                return self.fail(
                    ValueError(f"Unsupported augmentation: {augment} (currently supports noise)"), params
                )
            return self.ok(t.numpy().squeeze(0), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = TsaiAugmentTool()
