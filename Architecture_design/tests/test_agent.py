"""Smoke test: runs the full pipeline with a fixed decision script (MockClient), verifying that the executor actually runs the tools.

No network, no Ollama dependency. Usage: python Architecture_design/tests/test_agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from Architecture_design.agent import IndustrialAgent, _parse_final  # noqa: E402
from Architecture_design.dataset import series_stats, split_standardize  # noqa: E402
from Architecture_design.executor import ToolExecutor  # noqa: E402
from Architecture_design.main import MockClient  # noqa: E402
from Architecture_design.metrics import evaluate  # noqa: E402


def _first_tsbadu() -> Path:
    files = sorted((ROOT / "TSB-AD-U").glob("*.csv"))
    assert files, "TSB-AD-U data not found"
    return files[0]


def test_parse_final() -> None:
    assert _parse_final('prefix {"a": 1} suffix') == {"a": 1}
    assert _parse_final("no JSON") == {"raw": "no JSON"}


def test_end_to_end() -> None:
    path = _first_tsbadu()
    data = split_standardize(path)
    executor = ToolExecutor(data["test_std"], data["train"], data["y_test"], data["meta"])
    agent = IndustrialAgent(executor, client=MockClient(), max_iterations=5)

    outcome = agent.run("smoke test task", meta=data["meta"], stats=series_stats(data["test_std"]))

    # the mock script does a fixed 4 steps: search → rolling_volatility → dynamic_threshold → hysteresis_merge
    assert len(outcome["trace"]) == 4, f"Expected 4 steps, got {len(outcome['trace'])}"
    assert outcome["parsed"].get("toolchain"), "should parse the final JSON conclusion"

    search_result = outcome["trace"][0]["result"]
    assert search_result["ok"], f"search should actually run: {search_result.get('error')}"
    assert search_result.get("recommended"), "should give recommended detectors"

    detect_result = outcome["trace"][1]["result"]
    assert detect_result["ok"], f"detector should actually run: {detect_result.get('error')}"
    assert detect_result["kind"] == "scores", "rolling_volatility should output continuous scores"

    assert executor.score_history, "should record continuous scores for independent evaluation"
    metrics = evaluate(executor.score_history[0]["scores"], data["y_test"])
    assert "auc_roc" in metrics and "pa_f1" in metrics

    # the post-processing chain should also actually run
    assert outcome["trace"][2]["result"]["ok"], outcome["trace"][2]["result"].get("error")
    assert outcome["trace"][3]["result"]["ok"], outcome["trace"][3]["result"].get("error")


if __name__ == "__main__":
    test_parse_final()
    test_end_to_end()
    print("OK: smoke test passed (dry-run full pipeline actually ran)")
