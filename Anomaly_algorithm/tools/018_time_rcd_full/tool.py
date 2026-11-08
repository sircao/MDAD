"""Adapter for the complete pretrained Time-RCD anomaly scoring head.

The repository's encoder-only proxy lives in ``001_time_rcd``. This adapter
uses the separate public ``Time-RCD/time_rcd`` inference API and requires its
actual uni/multivariate anomaly-head checkpoint.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult, require
from Data_preprocessing.core.origins import DATA_ROOT

_TIME_RCD_ROOT = DATA_ROOT / "Time-RCD"
_DEFAULT_CHECKPOINTS = {
    "uni": _TIME_RCD_ROOT / "best_model" / "pretrain_checkpoint_best_uni.pth",
    "multi": _TIME_RCD_ROOT / "best_model" / "pretrain_checkpoint_best_multi.pth",
}


class FullTimeRcdTool(BaseTool):
    name = "time_rcd_full"
    category = "Zero-shot & General Models"
    version = "1.0.0"
    description = "Full Time-RCD pretrained anomaly detector; requires local anomaly-head weights and outputs per-time-step anomaly probabilities."
    libraries = ("torch", "numpy")
    origin = "https://github.com/thu-sail-lab/Time-RCD (time_rcd.TimeRCDDetector inference API)"

    def __init__(self) -> None:
        self._detectors: dict[tuple, object] = {}
        self._lock = threading.RLock()

    def run(self, data, checkpoint_path=None, variant=None, win_size=5000,
            batch_size=None, device=None, download=False, cache_dir=None,
            channelwise_uni=False, aggregate_channels="mean") -> ToolResult:
        variable_names = ([str(name) for name in data.columns] if hasattr(data, "columns")
                          else [str(index) for index in range(np.asarray(data).shape[-1]
                               if np.asarray(data).ndim > 1 else 1)])
        array = np.asarray(data, dtype=np.float64)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            return self.fail(ValueError(f"Expected non-empty (T,) or (T,C) data, got {array.shape}"), {})

        selected_variant = variant or ("multi" if array.shape[1] > 1 else "uni")
        if selected_variant not in {"uni", "multi"}:
            return self.fail(ValueError("variant must be 'uni' or 'multi'"), {"variant": selected_variant})
        if selected_variant == "uni" and array.shape[1] != 1:
            if not channelwise_uni:
                return self.fail(ValueError("variant='uni' requires one channel; set channelwise_uni=True to score channels separately"),
                                 {"variant": selected_variant})
        if selected_variant == "multi" and array.shape[1] < 2:
            return self.fail(ValueError("variant='multi' requires at least two channels"), {"variant": selected_variant})
        if aggregate_channels not in {"mean", "max"}:
            return self.fail(ValueError("aggregate_channels must be 'mean' or 'max'"),
                             {"aggregate_channels": aggregate_channels})

        env_name = "TIME_RCD_CHECKPOINT_UNI" if selected_variant == "uni" else "TIME_RCD_CHECKPOINT_MULTI"
        path_value = checkpoint_path or os.environ.get(env_name) or _DEFAULT_CHECKPOINTS[selected_variant]
        path = Path(path_value)
        params = {
            "variant": selected_variant,
            "checkpoint_path": str(path),
            "win_size": int(win_size),
            "batch_size": batch_size,
            "device": device,
            "download": bool(download),
            "channelwise_uni": bool(channelwise_uni),
            "aggregate_channels": aggregate_channels,
        }
        try:
            require(["torch", "numpy"])
            if not _TIME_RCD_ROOT.is_dir():
                raise FileNotFoundError(f"Time-RCD source directory not found: {_TIME_RCD_ROOT}")
            if str(_TIME_RCD_ROOT) not in sys.path:
                sys.path.insert(0, str(_TIME_RCD_ROOT))
            from time_rcd import TimeRCDDetector

            cache_key = (str(path.resolve()), selected_variant, int(win_size), batch_size, device)
            with self._lock:
                detector = self._detectors.get(cache_key)
                if detector is None:
                    options = {"win_size": int(win_size), "batch_size": batch_size, "device": device}
                    options = {key: value for key, value in options.items() if value is not None}
                    if path.is_file():
                        detector = TimeRCDDetector.from_local(path, variant=selected_variant, **options)
                    elif download:
                        detector = TimeRCDDetector.from_pretrained(
                            variant=selected_variant, cache_dir=cache_dir, **options
                        )
                    else:
                        raise FileNotFoundError(
                            f"Full Time-RCD checkpoint missing: {path}. Set {env_name}, pass checkpoint_path, "
                            "or explicitly pass download=True after arranging model-weight access."
                        )
                    self._detectors[cache_key] = detector
                if selected_variant == "uni" and array.shape[1] > 1:
                    per_channel = [np.asarray(detector.predict(array[:, column:column + 1]), dtype=float).reshape(-1)
                                   for column in range(array.shape[1])]
                    matrix = np.column_stack(per_channel)
                    scores = (np.nanmean(matrix, axis=1) if aggregate_channels == "mean"
                              else np.nanmax(matrix, axis=1))
                    output = {"scores": scores, "variable_scores": matrix,
                              "variable_names": variable_names, "aggregate_channels": aggregate_channels}
                else:
                    scores = np.asarray(detector.predict(array), dtype=float).reshape(-1)
                    output = scores
            if len(scores) != len(array) or not np.isfinite(scores).all():
                raise ValueError(f"Time-RCD returned invalid score shape/values: {scores.shape}")
            return self.ok(output, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = FullTimeRcdTool()
