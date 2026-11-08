"""RS-TOUTER period estimation — autocorrelation analysis."""

from __future__ import annotations

from Data_preprocessing.core.base import BaseTool, ToolResult, require
from Data_preprocessing.core.origins import RS_ROUTER_ROOT, ensure


class AcfPeriodTool(BaseTool):
    name = "rs_router.acf_period"
    category = "RS-TOUTER"
    version = "1.0.0"
    description = "Automatically infer the series period based on the autocorrelation function (ACF), providing a window length for sliding-window features."
    libraries = ("statsmodels", "scipy", "numpy")
    origin = "TS-Router-D8FF/utils/slidingWindows.py:8-69 find_length_rank()"

    def run(self, data, rank=1) -> ToolResult:
        params = {"rank": rank}
        try:
            require(list(self.libraries))
            ensure(RS_ROUTER_ROOT)
            from utils.slidingWindows import find_length_rank

            return self.ok(int(find_length_rank(data, rank=rank)), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = AcfPeriodTool()
