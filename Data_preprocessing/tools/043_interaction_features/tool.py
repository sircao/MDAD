"""Variable ratios, differences, and interaction features.

Combine multiple sensor variables to construct differences, ratios, products, and
interaction terms, converting domain relationships into model input features.
"""

from __future__ import annotations

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class InteractionFeaturesTool(BaseTool):
    name = "interaction_features"
    category = "Multivariate Fusion, Encoding, and Dimensionality Reduction"
    version = "1.0.0"
    description = "Construct combination features such as difference/ratio/product/interaction terms between variables."
    libraries = ("pandas",)

    def run(self, data, pairs=None, ops=("diff", "ratio", "product")) -> ToolResult:
        params = {"pairs": pairs, "ops": list(ops)}
        try:
            df = to_frame(data)
            cols = list(df.columns)
            if pairs is None:
                pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]
            out = df.copy()
            for a, b in pairs:
                if a not in df.columns or b not in df.columns:
                    continue
                if "diff" in ops:
                    out[f"{a}_diff_{b}"] = df[a] - df[b]
                if "ratio" in ops:
                    out[f"{a}_ratio_{b}"] = df[a] / df[b].replace(0, None)
                if "product" in ops:
                    out[f"{a}_prod_{b}"] = df[a] * df[b]
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = InteractionFeaturesTool()
