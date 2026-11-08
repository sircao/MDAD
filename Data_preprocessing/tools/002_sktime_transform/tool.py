"""SKTIME: unified time series transformation interface.

Provides time series transforms through a unified API, for scaling, differencing,
detrending, feature transformation, and sequence representation.
"""

from __future__ import annotations

from ...core.base import BaseTool, ToolResult, require
from ...core.utils import to_frame


class SktimeTransformTool(BaseTool):
    name = "sktime_transform"
    category = "Time Series Processing and Feature Extraction"
    version = "1.0.0"
    description = "Apply detrending, differencing, or scaling transforms through the sktime unified interface."
    libraries = ("sktime",)

    def run(self, data, transformer="detrend", **kwargs) -> ToolResult:
        params = {"transformer": transformer, **kwargs}
        try:
            require(list(self.libraries))
            import pandas as pd
            from sktime.transformations.series.detrend import Detrender
            from sktime.transformations.series.difference import Differencer

            df = to_frame(data)
            if transformer == "detrend":
                trans = Detrender()
            elif transformer == "difference":
                trans = Differencer(lags=kwargs.get("lags", 1))
            else:
                return self.fail(
                    ValueError(f"Unsupported transform: {transformer} (currently supports detrend / difference)"),
                    params,
                )

            out = df.apply(lambda col: trans.fit_transform(col.dropna()))
            return self.ok(pd.DataFrame(out, index=df.index, columns=df.columns), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = SktimeTransformTool()
