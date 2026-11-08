"""llm_verify_best.py — Local LLM verifies the best model selection (highest score + morphology/source-prior consistency).

Audit the best detector the routing picks on a "morphology + source" bucket, answering two questions:

  1. highest_score_ok —— whether best is the highest-scoring candidate detector (highest score);
  2. prior_conform_ok —— whether best matches the expectation of the morphology evidence and source prior (source->morphology->model).

The LLM only verifies (audits), **does not change the routing**; routing best is still decided by tuning's fast_vus argmax.
When Ollama is unavailable / parsing fails, fall back to the deterministic rule-based verdict (rule_based_verdict), ensuring
both the univariate (eval_loop_route) and multivariate (eval_loop_route_multi) pipelines produce a conclusion.

Products:
  runs/route_llm_verify.json          (univariate)
  runs/route_llm_verify_multi.json    (multivariate, path specified by the caller)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from Anomaly_algorithm.morphology import SOURCE_PRIOR  # noqa: E402

# consistent with the DETS of eval_loop_route / eval_loop_route_multi: all local, no remote.
DETS = ["Syntropic", "Time_RCD", "CHARM_stat_kNN", "OmniAnomaly"]

# detector profiles (fed to the LLM for morphology-fit reasoning).
DET_PROFILES = {
    "Syntropic": "Statistical synthesizer: weighted average of multiple robust-z normalized scores; strong on segment-level / level-drift / smooth collective anomalies",
    "Time_RCD": "Zero-shot foundation model for time-series root-cause detection; general-purpose, no clear morphology preference",
    "CHARM_stat_kNN": "Contrastive representation + statistical kNN outlier: based on local neighbor distances; strong on point / dense-cluster / local isolated anomalies",
    "OmniAnomaly": "VAE reconstruction family: strong on point anomalies / dense clusters (isolated extremes, dense range deviation); strong on point-anomaly sources such as Stock",
}

# morphology -> expected strongest detector (local prior of the 4-detector pool, used as rule fallback and LLM reference).
MORPH_TO_MODEL = {
    "level_shift": "Syntropic",
    "trend": "Syntropic",
    "persistent": "Syntropic",
    "seasonal": "Syntropic",
    "flatline": "Syntropic",
    "spike": "OmniAnomaly",
    "dense_cluster": "OmniAnomaly",
    "variance_change": "OmniAnomaly",
    "noise": "Time_RCD",
}


def _mode(vals) -> str:
    if not vals:
        return "unknown"
    return str(Counter(vals).most_common(1)[0][0])


def _aggregate_ranked(ranked_lists) -> list[dict]:
    """Aggregate the ranked evidence of multiple files in a bucket into each kind's mean strength / supported_fraction."""
    agg: dict[str, dict] = {}
    for rl in ranked_lists:
        for r in rl or []:
            kind = r.get("morphology")
            if not kind:
                continue
            a = agg.setdefault(str(kind), {"strength": [], "supported_fraction": []})
            try:
                a["strength"].append(float(r.get("strength", 0.0)))
                a["supported_fraction"].append(float(r.get("supported_fraction", 0.0)))
            except (TypeError, ValueError):
                pass
    out = [
        {"morphology": k,
         "strength": float(np.mean(v["strength"])) if v["strength"] else 0.0,
         "supported_fraction": float(np.mean(v["supported_fraction"])) if v["supported_fraction"] else 0.0}
        for k, v in agg.items()
    ]
    out.sort(key=lambda r: (-r["supported_fraction"], -r["strength"], r["morphology"]))
    return out


def build_bucket_payload(scope: str, key: str, rule: dict, files: list[str],
                         data: dict) -> dict:
    """Aggregate one bucket (scope=ms/m/cs/c/src/g) into the payload the LLM verification needs."""
    fs = [data[f] for f in files if f in data]
    morph = _mode([d["morph"] for d in fs])
    morph_c = _mode([d["morph_c"] for d in fs])
    source = _mode([d["source"] for d in fs])
    scores: dict[str, float] = {}
    for det in DETS:
        vs = [d["vus"][det] for d in fs if det in d.get("vus", {})]
        if vs:
            scores[det] = round(float(np.mean(vs)), 4)
    ranked = _aggregate_ranked([d.get("morph_ranked") for d in fs])
    return {
        "scope": scope, "key": key, "n_files": len(fs),
        "source": source, "morph": morph, "morph_c": morph_c,
        "morph_ranked": ranked, "scores": scores,
        "best": rule["best"] if isinstance(rule, dict) and rule.get("best") else None,
    }


def build_verify_prompt(p: dict) -> str:
    lines = [
        "You are the \"detector model-selection auditor\" for time-series anomaly detection. Below is the best "
        "detector the routing system picked on one data bucket. Verify whether this choice is reasonable using only the given scores and morphology/source-prior evidence; do not retrain.",
        "",
        f"Data bucket: {p.get('scope')}  key={p.get('key')}  ({p.get('n_files', 0)} files)",
        f"Source: {p.get('source')}   source-prior morphology: {SOURCE_PRIOR.get(p.get('source', '')) or 'none'}",
        f"Morphology signature: fine={p.get('morph')}  coarse={p.get('morph_c')}",
    ]
    ranked = p.get("morph_ranked") or []
    if ranked:
        lines.append("Morphology evidence (strength=evidence strength, supported_fraction=fraction of supporting points):")
        for r in ranked[:6]:
            lines.append(f"  - {r['morphology']}: strength={r.get('strength', 0.0):.2f}, "
                         f"supported_fraction={r.get('supported_fraction', 0.0):.3f}")
    lines += ["", "Candidate detector scores (VUS-PR, higher is better):"]
    for det in DETS:
        s = p.get("scores", {}).get(det)
        lines.append(f"  - {det}: {f'{s:.4f}' if s is not None else '—'}")
    lines += ["", "Detector profiles:"]
    for det in DETS:
        lines.append(f"  - {det}: {DET_PROFILES[det]}")
    lines += [
        "", f"Routing-picked best = {p.get('best')}",
        "", "Please verify two points:",
        "  1) highest_score_ok: whether best is the highest-scoring candidate (ties allowed);",
        "  2) prior_conform_ok: whether best matches the expectation of the morphology evidence and source prior.",
        "",
        'Output only one JSON: {"highest_score_ok": true/false, "prior_conform_ok": true/false, '
        '"verdict": "confirm|conflict|uncertain", '
        '"expected_model": "Syntropic|Time_RCD|CHARM_stat_kNN", "reasoning": "one sentence"}',
    ]
    return "\n".join(lines)


def _match_model(text, pool) -> str | None:
    t = (text or "").replace("-", "_").lower()
    if not t:
        return None
    for m in pool:
        if m.lower() in t or m.lower().replace("_", "") in t.replace("_", ""):
            return m
    return None


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "y")
    return bool(v)


def parse_verdict(content: str, pool=None) -> dict | None:
    """Parse the JSON verdict returned by the LLM; return None if it cannot be parsed (triggering the rule fallback)."""
    pool = pool or DETS
    if not content:
        return None
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    highest = data.get("highest_score_ok")
    prior = data.get("prior_conform_ok")
    if highest is None or prior is None:
        return None
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in ("confirm", "conflict", "uncertain"):
        verdict = "confirm" if (_as_bool(highest) and _as_bool(prior)) else "conflict"
    return {
        "highest_score_ok": _as_bool(highest),
        "prior_conform_ok": _as_bool(prior),
        "verdict": verdict,
        "expected_model": _match_model(data.get("expected_model"), pool),
        "reasoning": str(data.get("reasoning", ""))[:200],
    }


def rule_based_verdict(p: dict) -> dict:
    """Deterministic rule fallback (without LLM): highest score + whether the morphology/source-prior expectation matches."""
    scores = {k: v for k, v in (p.get("scores") or {}).items() if v is not None}
    best = p.get("best")
    mx = max(scores.values()) if scores else None
    highest_ok = bool(best and best in scores and mx is not None and scores[best] >= mx - 1e-4)
    prior_morph = SOURCE_PRIOR.get(p.get("source") or "")
    prim = (p.get("morph") or "unknown").split("+")[0]
    expect = MORPH_TO_MODEL.get(prior_morph) or MORPH_TO_MODEL.get(prim)
    prior_ok = True if expect is None else (best == expect)
    verdict = "confirm" if (highest_ok and prior_ok) else "conflict"
    return {
        "highest_score_ok": highest_ok,
        "prior_conform_ok": prior_ok,
        "verdict": verdict,
        "expected_model": expect,
        "reasoning": f"rule-based: highest score={'yes' if highest_ok else 'no'}, "
                     f"source-prior morphology={prior_morph or 'none'}, morphology expectation={expect or 'none'}, best={best}",
        "llm_used": False,
    }


def make_client(model: str | None = None, timeout: float | None = None):
    try:
        from Architecture_design.client import OllamaClient
    except Exception:  # noqa: BLE001
        return None
    kwargs = {}
    if model:
        kwargs["model"] = model
    if timeout is not None:
        kwargs["timeout"] = timeout
    try:
        return OllamaClient(**kwargs)
    except Exception:  # noqa: BLE001
        return None


def verify_best(payload: dict, client=None, model: str | None = None,
                timeout: float | None = None) -> dict:
    """Run LLM verification on one bucket's best selection; fall back to rule-based verdict on failure.

    Returns dict: {scope, key, best, highest_score_ok, prior_conform_ok, verdict,
    expected_model, reasoning, llm_used, error?}.
    """
    if client is None:
        client = make_client(model=model, timeout=timeout)
    error = None
    if client is not None:
        try:
            reply = client.chat([
                {"role": "system", "content": "You are the detector model-selection auditor for time-series anomaly detection; output only one JSON."},
                {"role": "user", "content": build_verify_prompt(payload)},
            ])
            v = parse_verdict(reply.get("content") or "")
            if v:
                v.update({"scope": payload.get("scope"), "key": payload.get("key"),
                          "best": payload.get("best"), "llm_used": True})
                return v
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {str(exc)[:120]}"
    base = rule_based_verdict(payload)
    base.update({"scope": payload.get("scope"), "key": payload.get("key"),
                 "best": payload.get("best")})
    if error:
        base["error"] = error
    return base


def verify_rules(specs: list[tuple[str, dict, dict]], data: dict,
                 model: str | None = None, timeout: float | None = None) -> dict:
    """Verify a set of bucket rules, returning {f"{scope}:{key}": verdict}.

    specs: [(scope, rules, bucket_files), ...]. rules and bucket_files share the same key
    space (e.g. rule_src and buckets_src are both keyed by source).
    """
    client = make_client(model=model, timeout=timeout)
    out: dict[str, dict] = {}
    for scope, rules, bucket_files in specs:
        for key, rule in rules.items():
            if not isinstance(rule, dict) or not rule.get("best"):
                continue
            files = bucket_files.get(key) or []
            payload = build_bucket_payload(scope, key, rule, files, data)
            out[f"{scope}:{key}"] = verify_best(payload, client=client, timeout=timeout)
    return out


def apply_corrections(specs: list[tuple[str, dict, dict]], verdicts: dict) -> int:
    """Override the best of buckets the LLM/prior judged as conflict with expected_model (prior correction).

    Called only under --llm-correct: best is replaced by the prior-expected model, and the complement loop is cleared (tp/fp/K=0),
    yielding a pure "prior-corrected base". The original best is stored in best_orig; both rule and verdict get the corrected flag.
    Returns the number of corrected buckets.
    """
    n = 0
    for scope, rules, _ in specs:
        for key, rule in rules.items():
            if not isinstance(rule, dict):
                continue
            v = verdicts.get(f"{scope}:{key}")
            if not v:
                continue
            if v.get("verdict") == "conflict" and not v.get("prior_conform_ok"):
                exp = v.get("expected_model")
                if exp in DETS and exp != rule.get("best"):
                    rule["best_orig"] = rule.get("best")
                    rule["best"] = exp
                    rule["tp_model"] = None
                    rule["fp_model"] = None
                    rule["K"] = 0
                    rule["corrected"] = True
                    v["corrected"] = True
                    v["best_orig"] = rule.get("best_orig")
                    n += 1
    return n


def summarize(verdicts: dict) -> dict:
    """Summarize verdict counts + the conflict-bucket list."""
    c = Counter((v.get("verdict") or "unknown") for v in verdicts.values())
    conflicts = [k for k, v in verdicts.items() if v.get("verdict") == "conflict"]
    llm_used = sum(1 for v in verdicts.values() if v.get("llm_used"))
    return {"counts": dict(c), "conflicts": conflicts, "llm_used": llm_used,
            "n": len(verdicts)}
