"""TS-Router encoder-deviation proxy; this is not the full Time-RCD anomaly head."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult, require
from Data_preprocessing.core.origins import DATA_ROOT, RS_ROUTER_ROOT, ensure

_DEFAULT_CHECKPOINT = DATA_ROOT / "TS-Router-D8FF" / "checkpoints"


class TimeRcdTool(BaseTool):
    name = "time_rcd"
    category = "Zero-shot & General Models"
    version = "2.1.0"
    description = "Loads the Time-RCD encoder shared by TS-Router and outputs per-channel representation-deviation proxy scores; this is not the full Time-RCD anomaly head."
    libraries = ("torch", "numpy")
    origin = (
        "https://github.com/thu-sail-lab/Time-RCD + "
        "TS-Router-D8FF/checkpoints（encoder shards）+ "
        "TS-Router-D8FF/models/time_rcd/ts_encoder_bi_bias.py TimeSeriesEncoder"
    )

    def _load_encoder(self, checkpoint_dir: Path):
        ensure(RS_ROUTER_ROOT)
        import torch

        from models.time_rcd.ts_encoder_bi_bias import TimeSeriesEncoder

        manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
        cfg = manifest["encoder"]["config"]
        encoder = TimeSeriesEncoder(**cfg)
        state: dict = {}
        for name in manifest["encoder"]["shards"]:
            shard = torch.load(checkpoint_dir / name, map_location="cpu", weights_only=True)
            state.update(shard)
        encoder.load_state_dict(state, strict=True)
        encoder.eval().requires_grad_(False)
        return encoder

    def run(self, data, checkpoint_dir=None, score="deviation") -> ToolResult:
        params = {"checkpoint_dir": str(checkpoint_dir or _DEFAULT_CHECKPOINT), "score": score}
        try:
            require(["torch", "numpy"])
            import torch

            ckpt = Path(checkpoint_dir) if checkpoint_dir else _DEFAULT_CHECKPOINT
            encoder = self._load_encoder(ckpt)

            series = np.asarray(data, dtype=np.float32)
            if series.ndim == 1:
                series = series[:, None]
            T, C = series.shape

            per_channel = []
            with torch.no_grad():
                for c in range(C):
                    s = series[:, c].astype(np.float64)
                    std = float(s.std())
                    mean = float(s.mean())
                    norm = (s - mean) / (std if std > 1e-8 else 1e-8)
                    tensor = torch.from_numpy(norm.astype(np.float32)).reshape(1, T, 1)
                    mask = torch.ones((1, T), dtype=torch.bool)
                    local = encoder(tensor, mask)  # [1, T, 1, d_proj]
                    H = local[0, :, 0, :]           # [T, d_proj]
                    center = H.median(dim=0).values
                    dist = torch.norm(H - center, dim=1).numpy()  # [T]
                    per_channel.append(dist)

            scores = np.column_stack(per_channel) if C > 1 else per_channel[0]
            return self.ok(scores, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = TimeRcdTool()
