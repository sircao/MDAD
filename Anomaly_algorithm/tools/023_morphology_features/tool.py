"""Anomaly morphology feature extraction: extracts label-free signal features (statistical/time-domain/frequency-domain/morphology) from the raw time series.

Serves as the front layer of "classify morphology first, then match detectors". The features cover
discriminative signals for eight morphology classes: spike/drift/trend/variance/noise/persistence/stuck/seasonality.
"""

from __future__ import annotations

import numpy as np

from Anomaly_algorithm.morphology import extract_features
from Anomaly_algorithm.morphology_evidence import extract_evidence
from Data_preprocessing.core.base import BaseTool, ToolResult, require


class MorphologyFeaturesTool(BaseTool):
    name = "morphology_features"
    category = "Anomaly Morphology Classification"
    version = "1.1.0"
    description = "Anomaly morphology feature extraction: extracts label-free signal features such as kurtosis/acf1/cusum/slope/roll_std/constant run/high-frequency energy ratio/spectral peak from the raw time series."
    libraries = ("numpy", "scipy", "pandas")
    origin = "Native implementation (industrial TSAD morphology features)"

    def run(self, data, window=0, method="legacy", reference=None, **kwargs) -> ToolResult:
        params = {"window": window, "method": method, **kwargs}
        try:
            require(["numpy", "scipy"])
            x = np.asarray(data, dtype=float)
            if method == "evidence":
                params["reference_shape"] = None if reference is None else list(np.shape(reference))
                out = extract_evidence(x, reference=reference,
                                       scales=None if window == 0 else (window,))
                return self.ok(out.summary(), params)
            if method != "legacy":
                raise ValueError("method must be 'legacy' or 'evidence'")
            if reference is not None:
                raise ValueError("reference requires method='evidence'")
            if x.ndim == 2:  # multivariate: take cross-column mean as a single-series approximation
                x = np.nan_to_num(x.mean(axis=1), nan=0.0)
            feats = extract_features(x.ravel(), window=int(window))
            return self.ok(feats, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = MorphologyFeaturesTool()
