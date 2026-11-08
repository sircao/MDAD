"""Offline per-file evidence routing over saved, full-length detector scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from Architecture_design.evidence_routing import Reviewer, route_scores, validate_scores


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def _filename(name: str) -> str:
    if (not isinstance(name, str) or "/" in name or "\\" in name
            or Path(name).name != name or not name.endswith(".csv")):
        raise ValueError(f"expected a CSV basename, got {name!r}")
    return name


def run_benchmark(
    *,
    data_dir: Path,
    file_list: Path,
    score_dir: Path,
    output_dir: Path,
    models: Sequence[str],
    baseline: str,
    protocol_id: str,
    split: str,
    evaluate: bool = False,
    limit: int | None = None,
    reviewer: Reviewer | None = None,
    official_file_list: Path | None = None,
) -> dict[str, Any]:
    """Route without labels, then optionally evaluate frozen decisions.

    Input layout is the official runner's ``score_dir/model/file_stem.npy``.
    Missing/invalid candidates fail the run instead of silently changing the
    candidate pool or averaging over a reduced evaluation set.
    """
    if not protocol_id.strip():
        raise ValueError("protocol_id must identify the frozen candidate configuration")
    if split not in ("Tuning", "Eva"):
        raise ValueError("split must be Tuning or Eva")
    if (not models or len(set(models)) != len(models) or baseline not in models
            or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", m) or m in (".", "..") for m in models)):
        raise ValueError("models must be unique safe names and include the baseline")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    names = pd.read_csv(file_list)["file_name"].tolist()
    names = [_filename(name) for name in names]
    if not names or len(names) != len(set(names)):
        raise ValueError("file list must be nonempty and contain no duplicates")
    total = len(names)
    names = names[:limit] if limit is not None else names
    official_names = None
    if official_file_list is not None:
        official_names = [_filename(name) for name in
                          pd.read_csv(official_file_list)["file_name"].tolist()]
        if not official_names or len(official_names) != len(set(official_names)):
            raise ValueError("official file list must be nonempty without duplicates")
        if not set(names).issubset(official_names):
            raise ValueError("requested files do not belong to the declared official split")
    # Never replace an old experiment, including a partially completed one.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "scores").mkdir()
    manifest: dict[str, Any] = {
        "version": "evidence-v2", "status": "running", "split": split,
        "protocol_id": protocol_id, "models": list(models), "baseline": baseline,
        "file_list": str(file_list.resolve()), "file_list_sha256": _hash(file_list),
        "score_dir": str(score_dir.resolve()), "data_dir": str(data_dir.resolve()),
        "router_sha256": _hash(ROOT.parent / "Architecture_design" / "evidence_routing.py"),
        "evidence_sha256": _hash(ROOT.parent / "Anomaly_algorithm" / "morphology_evidence.py"),
        "versions": {"python": sys.version, "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": version("scipy")},
        "file_count": total, "requested_count": len(names), "completed_count": 0,
        "full_coverage": False, "evaluate": evaluate,
        "coverage_scope": "requested_file_list",
        "official_file_list": None if official_file_list is None else str(official_file_list.resolve()),
        "official_file_list_sha256": None if official_file_list is None else _hash(official_file_list),
        "official_count": None if official_names is None else len(official_names),
        "official_full_coverage": False,
        "reviewer": None if reviewer is None else {
            "class": type(reviewer).__name__, "model": getattr(reviewer, "model", None),
            "temperature": getattr(reviewer, "temperature", None),
        },
    }
    _write_json(output_dir / "run.json", manifest)
    inputs: list[dict[str, Any]] = []
    try:
        with (output_dir / "decisions.jsonl").open("w", encoding="utf-8") as stream:
            for name in names:
                path = data_dir / name
                data_hash = _hash(path)
                # The label column is not even loaded during routing.
                data = pd.read_csv(path, usecols=lambda column: column != "Label").to_numpy(dtype=float)
                match = re.search(r"_tr_(\d+)_", name)
                if match is None:
                    raise ValueError(f"missing training-prefix length in {name}")
                train_length = int(match.group(1))
                candidates: dict[str, np.ndarray] = {}
                hashes = {}
                for model in models:
                    score_path = score_dir / model / f"{Path(name).stem}.npy"
                    score_hash = _hash(score_path)
                    candidates[model] = validate_scores(
                        np.load(score_path, allow_pickle=False), len(data), model)
                    if _hash(score_path) != score_hash:
                        raise ValueError(f"score cache changed while loading: {score_path}")
                    hashes[model] = score_hash
                result = route_scores(data, candidates, baseline=baseline,
                                      train_length=train_length, reviewer=reviewer)
                if _hash(path) != data_hash:
                    raise ValueError(f"data changed while routing: {name}")
                output_score = output_dir / "scores" / f"{Path(name).stem}.npy"
                np.save(output_score, result.scores)
                record = {"file": name, "data_sha256": data_hash, "score_sha256": hashes,
                          "output_sha256": _hash(output_score),
                          **result.decision}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                inputs.append(record)
                manifest["completed_count"] += 1
                _write_json(output_dir / "run.json", manifest)
                print(f"{name}: {result.decision['selected']} ({result.decision['reason']})",
                      flush=True)

        # This phase cannot feed labels back into the now-persisted decisions.
        if evaluate:
            from TSB_AD.evaluation.metrics import get_metrics
            from TSB_AD.utils.slidingWindows import find_length_rank

            manifest["metrics_sha256"] = _hash(ROOT / "TSB_AD" / "evaluation" / "metrics.py")
            manifest["basic_metrics_sha256"] = _hash(ROOT / "TSB_AD" / "evaluation" / "basic_metrics.py")
            manifest["sliding_windows_sha256"] = _hash(ROOT / "TSB_AD" / "utils" / "slidingWindows.py")
            rows = []
            for record in inputs:
                name = record["file"]
                path = data_dir / name
                if _hash(path) != record["data_sha256"]:
                    raise ValueError(f"data changed between routing and evaluation: {name}")
                frame = pd.read_csv(path)
                labels = frame["Label"].to_numpy(dtype=float)
                if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
                    raise ValueError(f"{name}: metrics require finite binary labels with both classes")
                data = frame.drop(columns=["Label"]).to_numpy(dtype=float)
                window = int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))
                output_score = output_dir / "scores" / f"{Path(name).stem}.npy"
                if _hash(output_score) != record["output_sha256"]:
                    raise ValueError(f"routed scores changed before evaluation: {name}")
                selected = np.load(output_score, allow_pickle=False)
                base_path = score_dir / baseline / f"{Path(name).stem}.npy"
                if _hash(base_path) != record["score_sha256"][baseline]:
                    raise ValueError(f"baseline scores changed during evaluation: {name}")
                base = np.load(base_path, allow_pickle=False)
                metrics = get_metrics(selected, labels.astype(int), slidingWindow=window)
                base_metrics = (metrics if record["selected"] == baseline else
                                get_metrics(base, labels.astype(int), slidingWindow=window))
                vus, base_vus = float(metrics["VUS-PR"]), float(base_metrics["VUS-PR"])
                if not np.isfinite([vus, base_vus]).all():
                    raise ValueError(f"{name}: evaluation returned non-finite VUS-PR")
                rows.append({"file": name, "selected": record["selected"],
                             "VUS-PR": vus, "baseline_VUS-PR": base_vus,
                             "slidingWindow": window})
            pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
            manifest["mean_vus_pr"] = float(np.mean([r["VUS-PR"] for r in rows]))
            manifest["baseline_mean_vus_pr"] = float(np.mean([r["baseline_VUS-PR"] for r in rows]))
        manifest["status"] = "complete"
        manifest["full_coverage"] = len(names) == total
        manifest["official_full_coverage"] = (
            official_names is not None and set(names) == set(official_names))
        _write_json(output_dir / "run.json", manifest)
        return manifest
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(output_dir / "run.json", manifest)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["U", "M"], required=True)
    parser.add_argument("--split", choices=["Tuning", "Eva"], default="Tuning")
    parser.add_argument("--file-list", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--protocol-id", required=True,
                        help="Frozen model/checkpoint/preprocessing configuration identifier")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate saved decisions after routing; never used for selection")
    parser.add_argument("--review-model",
                        help="Optional local Ollama reviewer; disabled by default")
    args = parser.parse_args(argv)
    reviewer = None
    if args.review_model:
        from Architecture_design.client import OllamaClient
        reviewer = OllamaClient(model=args.review_model, host="http://127.0.0.1:11434",
                                temperature=0.0)
    manifest = run_benchmark(
        data_dir=args.data_dir or ROOT / "Datasets" / f"TSB-AD-{args.dataset}",
        file_list=args.file_list or ROOT / "Datasets" / "File_List" /
        f"TSB-AD-{args.dataset}-{args.split}.csv",
        score_dir=args.score_dir, output_dir=args.output_dir,
        models=args.models, baseline=args.baseline, protocol_id=args.protocol_id,
        split=args.split, evaluate=args.evaluate, limit=args.limit, reviewer=reviewer,
        official_file_list=ROOT / "Datasets" / "File_List" /
        f"TSB-AD-{args.dataset}-{args.split}.csv",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
