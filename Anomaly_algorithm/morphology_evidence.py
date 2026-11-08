"""Reference-relative, channel-preserving morphology evidence (offline v2).

Strengths are heuristic evidence, not calibrated anomaly probabilities. Labels
and source identities never determine the pointwise evidence. The v1 classifier
in morphology.py is retained for reproducible legacy experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.signal import find_peaks

from Anomaly_algorithm.morphology import SOURCE_PRIOR


VERSION = "evidence-v2"
KINDS = (
    "spike", "level_shift", "trend", "variance_change", "flatline",
    "seasonal", "relationship_change",
)


@dataclass
class Evidence:
    scores: dict[str, np.ndarray]
    channels: list[dict[str, Any]]
    regions: list[dict[str, Any]]
    scales: tuple[int, ...]
    reference_kind: str
    source_prior: str | None
    fire_threshold: float = 0.5

    @property
    def support(self) -> np.ndarray:
        return np.maximum.reduce(list(self.scores.values()))

    def summary(self, max_regions: int = 24) -> dict[str, Any]:
        if max_regions < 1:
            raise ValueError("max_regions must be positive")
        thr = self.fire_threshold
        ranked = [
            {"morphology": kind, "strength": float(score.max()),
             "supported_fraction": float(np.mean(score >= thr))}
            for kind, score in self.scores.items() if np.any(score >= thr)
        ]
        ranked.sort(key=lambda item: (-item["supported_fraction"], -item["strength"],
                                      item["morphology"]))
        morphology = ranked[0]["morphology"] if ranked else "unknown"
        regions = sorted(
            self.regions, key=lambda r: (-r["strength"], -(r["end"] - r["start"]),
                                         r["channel"], r["start"], r["morphology"])
        )
        return {
            "version": VERSION,
            "morphology": morphology,
            "confidence": "supported" if ranked else "insufficient",
            "ranked": ranked,
            "n_samples": len(self.support),
            "n_channels": len(self.channels),
            "scales": list(self.scales),
            "reference_kind": self.reference_kind,
            "channels": self.channels,
            "regions": regions[:max_regions],
            "region_count": len(regions),
            "regions_truncated": len(regions) > max_regions,
            "source_prior": self.source_prior,
            "prior_conflict": bool(self.source_prior and ranked
                                   and self.source_prior != morphology),
            "used_prior": False,
        }


def as_series(data: np.ndarray, name: str = "data") -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] < 8 or x.shape[1] < 1:
        raise ValueError(f"{name} must have shape (T, C), T >= 8 and C >= 1")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values; preserve alignment and repair upstream")
    return x


def _scale(x: np.ndarray) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    if mad > 1e-12:
        return mad
    sd = float(np.std(x))
    return sd if sd > 1e-12 else 1e-12


def _windows(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    # Center first to avoid cancellation in the cumulative sum of squares.
    shift = float(np.mean(x))
    z = x - shift
    c1 = np.concatenate(([0.0], np.cumsum(z)))
    c2 = np.concatenate(([0.0], np.cumsum(z * z)))
    mean = (c1[w:] - c1[:-w]) / w
    var = np.maximum((c2[w:] - c2[:-w]) / w - mean * mean, 0.0)
    return mean + shift, np.sqrt(var)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def _centered(values: np.ndarray, w: int, n: int) -> np.ndarray:
    out = np.zeros(n)
    start = w // 2
    out[start:start + len(values)] = values
    return out


def _excess(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.clip(values / max(threshold, 1e-12) - 1.0, 0.0, 1.0)


def _segment_view(x: np.ndarray, threshold: float, max_run: int) -> np.ndarray:
    out = x.copy()
    for start, end in _runs(_excess(np.abs(x), threshold) >= 0.5):
        if end - start <= max_run:
            left = x[start - 1] if start else 0.0
            right = x[end] if end < len(x) else 0.0
            out[start:end] = 0.5 * (left + right)
    return out


def _period(x: np.ndarray) -> int | None:
    z = x - np.mean(x)
    if np.std(z) < 1e-8 or len(z) < 32:
        return None
    size = 1 << (2 * len(z) - 1).bit_length()
    fft = np.fft.rfft(z, n=size)
    acf = np.fft.irfft(fft * fft.conj(), n=size)[:len(z) // 3 + 1]
    acf /= acf[0]
    peaks, _ = find_peaks(acf)
    good = peaks[(peaks >= 4) & (acf[peaks] >= 0.6)]
    return int(good[0]) if len(good) else None


def extract_evidence(
    data: np.ndarray,
    reference: np.ndarray | None = None,
    *,
    scales: Sequence[int] | None = None,
    source: str = "",
    thresholds: dict | None = None,
) -> Evidence:
    """Extract original-resolution evidence without flattening channels.

    A provided reference should be a protocol-approved normal segment. Without
    it, the series is a robust, possibly contaminated reference, not assumed
    normal. Regions are half-open intervals; no disjoint points are stitched.
    """
    x = as_series(data)
    ref = x if reference is None else as_series(reference, "reference")
    if x.shape[1] != ref.shape[1]:
        raise ValueError("data and reference must have the same number of channels")
    cap = min(len(x), len(ref)) // 2
    if scales is None:
        windows = tuple(w for w in (8, 16, 32, 64) if w <= cap) or (cap,)
    else:
        if not scales or any(isinstance(w, bool) or not isinstance(w, (int, np.integer))
                             or not 4 <= w <= cap for w in scales):
            raise ValueError(f"scales must contain integers between 4 and {cap}")
        windows = tuple(sorted(set(int(w) for w in scales)))
    n, n_channels = x.shape
    scores = {kind: np.zeros(n) for kind in KINDS}
    channels: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []

    # Threshold parameterization (defaults = original hard-coded values, preserving existing calls; experiments may override to lower thresholds).
    cfg = {
        "fire": 0.5,              # record trigger + summary inclusion threshold
        "spike_floor": 6.0, "spike_q": 0.999, "spike_mult": 1.5,
        "level_floor": 0.75, "level_q": 0.995, "level_add": 0.5,
        "var_floor": 1.5, "var_q": 0.995, "var_mult": 1.5,
        "flat_floor": 0.05, "flat_ratio": 0.2,
        "trend_floor": 1.0, "trend_q": 0.995, "trend_mult": 2.0,
        "seasonal_floor": 1.0, "seasonal_q": 0.995, "seasonal_mult": 2.0,
    }
    if thresholds:
        cfg.update(thresholds)
    fire = cfg["fire"]

    def record(kind: str, values: np.ndarray, channel: int, w: int,
               min_run: int = 1, max_run: int | None = None) -> None:
        for start, end in _runs(values >= fire):
            if end - start < min_run or (max_run is not None and end - start > max_run):
                continue
            scores[kind][start:end] = np.maximum(scores[kind][start:end], values[start:end])
            regions.append({"morphology": kind, "channel": channel, "start": start,
                            "end": end, "window": w,
                            "strength": float(np.max(values[start:end]))})

    for c in range(n_channels):
        center = float(np.median(ref[:, c]))
        scale = _scale(ref[:, c])
        z, r = (x[:, c] - center) / scale, (ref[:, c] - center) / scale
        period = _period(r)
        before = len(regions)
        threshold = max(cfg["spike_floor"], float(np.quantile(np.abs(r), cfg["spike_q"])) * cfg["spike_mult"])
        max_spike_run = max(3, windows[0] // 2)
        record("spike", _excess(np.abs(z), threshold), c, 1,
               max_run=max_spike_run)
        # Separate point and segment views so a single spike cannot masquerade
        # as a whole window of mean/variance change. Raw point evidence is kept.
        segment = _segment_view(z, threshold, max_spike_run)
        ref_segment = _segment_view(r, threshold, max_spike_run)
        for w in windows:
            mean, sd = _windows(segment, w)
            ref_mean, ref_sd = _windows(ref_segment, w)
            mean_threshold = max(cfg["level_floor"], float(np.quantile(np.abs(ref_mean), cfg["level_q"])) + cfg["level_add"])
            level = _centered(_excess(np.abs(mean), mean_threshold), w, n)
            level[np.abs(segment) < mean_threshold] = 0.0
            record("level_shift", level, c, w, min_run=w)

            high = max(cfg["var_floor"], float(np.quantile(ref_sd, cfg["var_q"])) * cfg["var_mult"])
            record("variance_change", _centered(_excess(sd, high), w, n),
                   c, w, min_run=w)
            low = float(np.quantile(ref_sd, 0.01))
            if low > cfg["flat_floor"]:
                # Low-variance evidence needs a sustained interval and a
                # non-flat reference; constant normal channels are not alarms.
                low_score = np.clip(1.0 - sd / (cfg["flat_ratio"] * low), 0.0, 1.0)
                record("flatline", _centered(low_score, w, n), c, w, min_run=w)

            half = w // 2
            left, _ = _windows(segment, half)
            delta = np.abs(left[half:half + len(mean)] - left[:len(mean)])
            rl, _ = _windows(ref_segment, half)
            rd = np.abs(rl[half:half + len(ref_mean)] - rl[:len(ref_mean)])
            slope_threshold = max(cfg["trend_floor"], float(np.quantile(rd, cfg["trend_q"])) * cfg["trend_mult"])
            record("trend", _centered(_excess(delta, slope_threshold), w, n),
                   c, w, min_run=2 * w)

        if period is not None and len(z) > period:
            residual = np.abs(z[period:] - z[:-period])
            rr = np.abs(r[period:] - r[:-period])
            threshold = max(cfg["seasonal_floor"], float(np.quantile(rr, cfg["seasonal_q"])) * cfg["seasonal_mult"])
            seasonal = np.zeros(n)
            seasonal[period:] = _excess(residual, threshold)
            record("seasonal", seasonal, c, period, min_run=max(2, period // 4))
        channels.append({
            "channel": c, "center": center, "scale": scale,
            "reference_constant": bool(np.std(ref[:, c]) < 1e-12),
            "period": period,
            "flat_ratio": float(np.mean(np.abs(np.diff(x[:, c])) <= scale * 1e-3)),
            "region_count": len(regions) - before,
        })

    # One strong reference partner per channel bounds scoring at O(T*C),
    # rather than materializing a time-by-channel-by-channel tensor.
    active = np.flatnonzero(np.std(ref, axis=0) > 1e-12)
    if len(active) > 1:
        corr = np.corrcoef(ref[:, active], rowvar=False)
        np.fill_diagonal(corr, 0.0)
        pairs: set[tuple[int, int]] = set()
        for i, c in enumerate(active):
            j = int(np.argmax(np.abs(corr[i])))
            if abs(corr[i, j]) >= 0.8:
                pairs.add(tuple(sorted((int(c), int(active[j])))))
        for a, b in sorted(pairs):
            ra = ref[:, a] - np.mean(ref[:, a])
            rb = ref[:, b] - np.mean(ref[:, b])
            beta = float(np.dot(ra, rb) / np.dot(rb, rb))
            rr = ra - beta * rb
            residual = (x[:, a] - np.mean(ref[:, a])) - beta * (x[:, b] - np.mean(ref[:, b]))
            # Avoid interpreting roundoff in a deterministic relation as an event.
            scale = max(_scale(rr), _scale(ref[:, a]) * 0.01)
            threshold = max(6 * scale, float(np.quantile(np.abs(rr), 0.999)) * 1.5)
            record("relationship_change", _excess(np.abs(residual), threshold),
                   a, 1, min_run=2)

    return Evidence(scores, channels, regions, windows,
                    "provided" if reference is not None else "robust_series",
                    SOURCE_PRIOR.get(source), fire)
