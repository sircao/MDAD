"""Two-stage routing using local morphology and real detector score evidence.

No labels, filenames, benchmark tables or model performance priors enter the
router. Numerical gates are fixed heuristics to validate on a separate tuning
split, not a guarantee of improvement over the explicitly chosen baseline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np
from scipy.stats import rankdata

from Anomaly_algorithm.morphology_evidence import Evidence, as_series, extract_evidence
from Architecture_design.client import OllamaError


LOGGER = logging.getLogger(__name__)
MIN_QUALITY = 0.6
SWITCH_MARGIN = 0.1


class Reviewer(Protocol):
    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]: ...


@dataclass
class RoutingResult:
    scores: np.ndarray
    decision: dict[str, Any]


def validate_scores(scores: np.ndarray, length: int, name: str) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or len(s) != length:
        raise ValueError(f"{name}: expected {length} point scores, got shape {s.shape}")
    if not np.isfinite(s).all():
        raise ValueError(f"{name}: non-finite scores; repair the producer instead of masking errors")
    return s


def _quality(scores: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    if np.ptp(scores) == 0:
        return {"constant": True, "quality": 0.0, "tail_support": 0.0,
                "evidence_rank": 0.0, "top_regions": []}
    supported = support >= 0.5
    ranks = (rankdata(scores, method="average") - 0.5) / len(scores)
    precision = []
    for fraction in (0.01, 0.05):
        count = max(1, int(np.ceil(len(scores) * fraction)))
        threshold = np.partition(scores, len(scores) - count)[len(scores) - count]
        # Include all boundary ties, but do not expand a sparse positive tail
        # into its tied minimum background just to fill the nominal budget.
        tail = (scores >= threshold) & (scores > scores.min())
        precision.append(float(np.mean(support[tail])))
    tail_support = float(np.mean(precision))
    evidence_rank = float(np.mean(ranks[supported])) if supported.any() else 0.0
    top = (scores >= np.quantile(scores, 0.99)) & (scores > scores.min())
    edges = np.diff(np.concatenate(([False], top, [False])).astype(int))
    intervals = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    top_regions = [{"start": int(a), "end": int(b)} for a, b in intervals[:12]]
    return {
        "constant": False,
        "quality": 0.5 * tail_support + 0.5 * evidence_rank,
        "tail_support": tail_support,
        "evidence_rank": evidence_rank,
        "top_regions": top_regions,
        "top_region_count": len(intervals),
    }


def route_scores(
    data: np.ndarray,
    candidate_scores: Mapping[str, np.ndarray],
    *,
    baseline: str,
    train_length: int,
    reviewer: Reviewer | None = None,
) -> RoutingResult:
    """Choose one detector per file, preserving its continuous scores exactly.

    Scores and data cover the full file. Only the declared training prefix is
    the reference; the gate examines the suffix, without reading any labels.
    An optional LLM may veto or choose among numerically qualified candidates.
    Invalid input fails; failed/invalid reviewer output explicitly reverts to
    the baseline and is recorded in the audit.
    """
    x = as_series(data)
    if isinstance(train_length, bool) or not isinstance(train_length, (int, np.integer)):
        raise ValueError("train_length must be an integer")
    if not 8 <= train_length <= len(x) - 8:
        raise ValueError("routing requires at least 8 reference and 8 evaluation samples")
    if baseline not in candidate_scores:
        raise ValueError(f"baseline {baseline!r} has no score array")
    if any(not isinstance(name, str) or not name for name in candidate_scores):
        raise ValueError("candidate names must be nonempty strings")
    candidates = {name: validate_scores(s, len(x), name)
                  for name, s in sorted(candidate_scores.items())}
    evidence: Evidence = extract_evidence(x, reference=x[:train_length])
    support = evidence.support[train_length:]
    diagnostics = {name: _quality(s[train_length:], support)
                   for name, s in candidates.items()}
    for item in diagnostics.values():
        for interval in item["top_regions"]:
            interval["start"] += train_length
            interval["end"] += train_length
    base_quality = diagnostics[baseline]["quality"]
    qualified = [
        name for name in candidates if name != baseline
        and not diagnostics[name]["constant"]
        and diagnostics[name]["quality"] >= MIN_QUALITY
        and diagnostics[name]["quality"] >= base_quality + SWITCH_MARGIN
    ]
    qualified.sort(key=lambda name: (-diagnostics[name]["quality"], name))
    selected = baseline
    reason = "insufficient_evidence" if not np.any(support >= 0.5) else "baseline_retained"
    reviewer_status = "not_requested"
    reviewer_reason: str | None = None
    if qualified:
        selected, reason = qualified[0], "numerical_evidence"
    if reviewer is not None and qualified:
        allowed = [baseline, *qualified[:2]]
        prompt = {
            "task": "Review detector evidence. Return exactly a JSON object with "
                    "'selected' (one allowed name) and 'reason' (a nonempty string). "
                    "Keep the baseline when evidence is ambiguous. Morphology evidence "
                    "is heuristic, not ground truth. Do not infer quality from model names.",
            "baseline": baseline,
            "allowed": allowed,
            "evidence": evidence.summary(),
            "detectors": {name: diagnostics[name] for name in allowed},
        }
        try:
            reply = reviewer.chat([{"role": "user", "content": json.dumps(prompt)}])
            if not isinstance(reply, dict):
                raise ValueError("reviewer response must be an object")
            content = reply.get("content")
            if not isinstance(content, str):
                raise ValueError("reviewer content must be a JSON string")
            parsed = json.loads(content)
            if not isinstance(parsed, dict) or parsed.get("selected") not in allowed:
                raise ValueError("reviewer selected a non-qualified detector")
            if not isinstance(parsed.get("reason"), str) or not parsed["reason"].strip():
                raise ValueError("reviewer must provide a reason")
            selected, reason = parsed["selected"], "reviewed_evidence"
            reviewer_status, reviewer_reason = "accepted", parsed["reason"]
        except (OllamaError, ValueError, TypeError) as exc:
            LOGGER.warning("Evidence reviewer failed; retaining baseline %s: %s", baseline, exc)
            selected, reason = baseline, "reviewer_failure"
            reviewer_status, reviewer_reason = "failed", str(exc)
    elif reviewer is not None:
        reviewer_status = "not_needed"
    decision = {
        "version": "evidence-v2",
        "baseline": baseline,
        "selected": selected,
        "reason": reason,
        "train_length": int(train_length),
        "evidence": evidence.summary(),
        "detectors": diagnostics,
        "qualified": qualified,
        "gate": {"minimum_quality": MIN_QUALITY, "switch_margin": SWITCH_MARGIN},
        "reviewer_status": reviewer_status,
        "reviewer_reason": reviewer_reason,
    }
    return RoutingResult(candidates[selected].copy(), decision)
