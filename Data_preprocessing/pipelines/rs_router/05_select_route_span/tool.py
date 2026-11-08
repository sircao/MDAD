"""RS-TOUTER route span selection — route sequence selection."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult


class SelectRouteSpanTool(BaseTool):
    name = "rs_router.select_route_span"
    category = "RS-TOUTER"
    version = "1.0.0"
    description = "Select the range and size of the data fed into the encoder/router according to original/full/test."
    libraries = ("numpy",)
    origin = "TS-Router-D8FF/ts_router.py:124-130 predict() — route_span selection logic"

    def run(self, data, train_index, route_span="original", multi=None) -> ToolResult:
        params = {"train_index": train_index, "route_span": route_span, "multi": multi}
        try:
            data = np.asarray(data, dtype=float)
            if route_span not in {"original", "full", "test"}:
                return self.fail(ValueError("route_span supports only original / full / test"), params)
            if multi is None:
                multi = data.shape[1] > 1
            route = data
            if route_span == "test" or (route_span == "original" and multi):
                route = data[int(train_index):]
                if route_span == "original":
                    route = route[:10000]
            return self.ok(route, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = SelectRouteSpanTool()
