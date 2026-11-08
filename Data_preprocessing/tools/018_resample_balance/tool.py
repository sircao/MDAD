"""Over-sampling and under-sampling (class imbalance, supervised).

Adjust the training data distribution by under-sampling normal samples or over-sampling
anomalous samples. Unsupervised anomaly detection cannot treat this directly as a general
preprocessing method; it may only be applied to training data.
"""

from __future__ import annotations

import numpy as np

from ...core.base import BaseTool, ToolResult, require


class ResampleBalanceTool(BaseTool):
    name = "resample_balance"
    category = "Sampling, Windowing, and Data Volume Control"
    version = "1.0.0"
    description = "Adjust the class distribution via under-sampling / over-sampling / SMOTE; for labeled training data only."
    libraries = ("imblearn",)

    def run(self, data, labels, method="under", random_state=0) -> ToolResult:
        params = {"method": method, "random_state": random_state}
        try:
            require(["imblearn"])
            from imblearn.over_sampling import RandomOverSampler, SMOTE
            from imblearn.under_sampling import RandomUnderSampler

            X = np.asarray(data)
            y = np.asarray(labels).ravel()
            if method == "under":
                sampler = RandomUnderSampler(random_state=random_state)
            elif method == "over":
                sampler = RandomOverSampler(random_state=random_state)
            elif method == "smote":
                sampler = SMOTE(random_state=random_state)
            else:
                return self.fail(
                    ValueError(f"method only supports under / over / smote, got {method}"), params
                )
            X_res, y_res = sampler.fit_resample(X, y)
            warnings = ["For training data only; do not apply to validation/test sets"]
            return self.ok({"data": X_res, "labels": y_res}, params, warnings)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = ResampleBalanceTool()
