"""Calendar and holiday encoding.

Generate time-context features such as hour, weekday, month, season, and holiday.
Cyclical variables can be encoded with sine/cosine to avoid treating the start and end
of a cycle as far apart.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class CalendarEncodeTool(BaseTool):
    name = "calendar_encode"
    category = "Multivariate Fusion, Encoding, and Dimensionality Reduction"
    version = "1.0.0"
    description = "Generate time-context features such as hour/weekday/month/season/holiday (including sine/cosine encoding)."
    libraries = ("pandas", "numpy")

    def run(self, data, cyclical=("hour", "weekday", "month"), with_holiday=False, country=None) -> ToolResult:
        params = {"cyclical": list(cyclical), "with_holiday": with_holiday, "country": country}
        try:
            df = to_frame(data)
            if not isinstance(df.index, pd.DatetimeIndex):
                return self.fail(ValueError("Calendar encoding requires a DatetimeIndex"), params)
            idx = df.index
            out = pd.DataFrame(index=idx)
            out["hour"] = idx.hour
            out["weekday"] = idx.weekday
            out["month"] = idx.month
            out["season"] = idx.month % 12 // 3
            for name in cyclical:
                period = {"hour": 24, "weekday": 7, "month": 12, "season": 4}.get(name)
                if period is None:
                    continue
                value = out[name].to_numpy(dtype=float)
                out[f"{name}_sin"] = np.sin(2 * np.pi * value / period)
                out[f"{name}_cos"] = np.cos(2 * np.pi * value / period)
            if with_holiday:
                if importlib.util.find_spec("holidays") is None:
                    return self.fail(ImportError("The holidays package is required: pip install holidays"), params)
                import holidays

                country = country or "CN"
                hdays = holidays.country_holidays(country)
                out["is_holiday"] = idx.normalize().isin(hdays)
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = CalendarEncodeTool()
