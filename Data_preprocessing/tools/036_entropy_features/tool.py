"""Entropy and complexity features.

Sample entropy, permutation entropy, approximate entropy, Lempel–Ziv complexity, spectral
entropy, and the Hurst exponent can describe sequence regularity, complexity, and long-range
dependence. (Native implementation; antropy provides a fuller set of similar features.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class EntropyFeaturesTool(BaseTool):
    name = "entropy_features"
    category = "Feature Extraction and Feature Representation"
    version = "1.0.0"
    description = "Extract sample entropy/approximate entropy/permutation entropy/spectral entropy/LZ complexity/Hurst exponent."
    libraries = ("numpy",)

    @staticmethod
    def _sampen(s: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        s = np.asarray(s, dtype=float)
        r *= np.std(s) + 1e-12
        def phi(dim):
            templates = np.array([s[i:i + dim] for i in range(len(s) - dim + 1)])
            if len(templates) < 2:
                return 0.0
            d = np.abs(templates[:, None, :] - templates[None, :, :]).max(axis=-1)
            matches = (d <= r).sum() - 1  # exclude self
            return float(np.maximum(matches, 0).sum())
        b = phi(m)
        a = phi(m + 1)
        return -np.log((a + 1e-12) / (b + 1e-12))

    @staticmethod
    def _apen(s: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        s = np.asarray(s, dtype=float)
        r *= np.std(s) + 1e-12
        def phi(dim):
            templates = np.array([s[i:i + dim] for i in range(len(s) - dim + 1)])
            if len(templates) < 2:
                return 0.0
            d = np.abs(templates[:, None, :] - templates[None, :, :]).max(axis=-1)
            matches = (d <= r).sum()
            return float(np.log(matches / len(templates) + 1e-12).sum() / len(templates))
        return phi(m) - phi(m + 1)

    @staticmethod
    def _pempen(s: np.ndarray, m: int = 3, tau: int = 1) -> float:
        from math import factorial
        from collections import Counter

        s = np.asarray(s, dtype=float)
        patterns = []
        for i in range(len(s) - (m - 1) * tau):
            patterns.append(tuple(np.argsort(s[i:i + m * tau:tau], kind="stable")))
        if not patterns:
            return 0.0
        counts = Counter(patterns)
        probs = np.array(list(counts.values())) / len(patterns)
        return float(-(probs * np.log(probs + 1e-12)).sum() / np.log(factorial(m) + 1e-12))

    @staticmethod
    def _lz(s: np.ndarray) -> float:
        b = "".join("1" if v > np.median(s) else "0" for v in np.asarray(s, dtype=float))
        i, c, n = 0, 1, len(b)
        while i + c <= n:
            sub = b[i:i + c]
            if sub in b[:i]:
                c += 1
            else:
                i += c
                c = 1
        return n / (i + c - 1) if (i + c - 1) > 0 else 0.0

    @staticmethod
    def _hurst(s: np.ndarray) -> float:
        s = np.asarray(s, dtype=float)
        n = len(s)
        if n < 64:
            return 0.5
        windows = [2 ** k for k in range(4, int(np.log2(n)) + 1)]
        rs = []
        for w in windows:
            chunks = s[: n - n % w].reshape(-1, w)
            mean = chunks.mean(axis=1, keepdims=True)
            dev = np.cumsum(chunks - mean, axis=1)
            r = dev.max(axis=1) - dev.min(axis=1)
            sd = chunks.std(axis=1) + 1e-12
            rs.append(np.mean(r / sd))
        log_w = np.log(windows)
        log_rs = np.log(rs)
        if len(windows) < 2 or np.std(log_w) == 0:
            return 0.5
        return float(np.polyfit(log_w, log_rs, 1)[0])

    def _features(self, s: np.ndarray) -> dict:
        s = np.asarray(s, dtype=float)
        spectrum = np.abs(np.fft.rfft(s))
        psd = spectrum ** 2
        psd = psd / (psd.sum() + 1e-12)
        spectral_entropy = float(-(psd * np.log(psd + 1e-12)).sum())
        return {
            "sample_entropy": self._sampen(s),
            "approx_entropy": self._apen(s),
            "permutation_entropy": self._pempen(s),
            "spectral_entropy": spectral_entropy,
            "lz_complexity": self._lz(s),
            "hurst_exponent": self._hurst(s),
        }

    def run(self, data) -> ToolResult:
        params = {}
        try:
            df = to_frame(data)
            rows = {col: self._features(df[col].to_numpy(dtype=float)) for col in df.columns}
            out = pd.DataFrame(rows).T
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = EntropyFeaturesTool()
