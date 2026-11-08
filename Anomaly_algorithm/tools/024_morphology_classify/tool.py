"""Anomaly morphology classification: classifies the morphology of the raw time series and outputs rankings, confidence and matched detector directions.

Eight morphology classes: spike/level drift/trend/variance change/noise/persistence/stuck/seasonality. The
classification result routes downstream detection to direction-matched detector families, avoiding inversion by
single-direction detectors.
"""

from __future__ import annotations

import numpy as np

from Anomaly_algorithm.morphology import classify
from Anomaly_algorithm.morphology_evidence import extract_evidence
from Data_preprocessing.core.base import BaseTool, ToolResult, require


class MorphologyClassifyTool(BaseTool):
    name = "morphology_classify"
    category = "Anomaly Morphology Classification"
    version = "1.1.0"
    description = "Anomaly morphology classification: determines the anomaly morphology of the raw time series (spike/drift/trend/variance/noise/persistence/stuck/seasonality), returning rankings, confidence and matched detector directions."
    libraries = ("numpy", "scipy", "pandas")
    origin = "Native implementation (rule-based evidence + detector-direction routing)"

    def run(self, data, window=0, source="", method="legacy", reference=None, **kwargs) -> ToolResult:
        params = {"window": window, "source": source, "method": method, **kwargs}
        try:
            require(["numpy", "scipy"])
            x = np.asarray(data, dtype=float)
            if method == "evidence":
                params["reference_shape"] = None if reference is None else list(np.shape(reference))
                out = extract_evidence(x, reference=reference, source=str(source),
                                       scales=None if window == 0 else (window,))
                return self.ok(out.summary(), params)
            if method != "legacy":
                raise ValueError("method must be 'legacy' or 'evidence'")
            if reference is not None:
                raise ValueError("reference requires method='evidence'")
            if x.ndim == 2:  # multivariate: take cross-column mean as a single-series approximation
                x = np.nan_to_num(x.mean(axis=1), nan=0.0)
            out = classify(x.ravel(), window=int(window), source=str(source))
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = MorphologyClassifyTool()
