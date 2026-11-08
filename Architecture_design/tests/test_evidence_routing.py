"""Offline regression tests: no LLM, GPU, checkpoint downloads or benchmark tuning."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TSB-AD"))

from Anomaly_algorithm.morphology_evidence import extract_evidence
from Anomaly_algorithm.registry import get
from Architecture_design.evidence_routing import _quality, route_scores
from evidence_route_bench import run_benchmark


def signal() -> np.ndarray:
    return np.random.default_rng(2024).normal(0, 0.05, 1000)


class EvidenceTests(unittest.TestCase):
    def test_normal_noise_periodicity_and_constant_are_not_anomalies(self):
        for x in (signal(), np.sin(2 * np.pi * np.arange(1000) / 50), np.ones(1000)):
            with self.subTest(kind=float(np.std(x))):
                evidence = extract_evidence(x, x[:300])
                self.assertEqual(evidence.summary()["morphology"], "unknown")
                self.assertEqual(evidence.summary()["confidence"], "insufficient")
                self.assertEqual(evidence.support.max(), 0)

    def test_spike_is_not_smeared_into_a_segment(self):
        x = signal()
        x[500] += 5
        evidence = extract_evidence(x, x[:300])
        self.assertEqual(evidence.summary()["morphology"], "spike")
        self.assertEqual(evidence.scores["spike"][500], 1)
        self.assertLessEqual(np.count_nonzero(evidence.support), 3)

    def test_level_shift_interior_and_boundary(self):
        x = signal()
        x[400:700] += 3
        evidence = extract_evidence(x, x[:300])
        self.assertEqual(evidence.summary()["morphology"], "level_shift")
        self.assertGreater(np.mean(evidence.scores["level_shift"][420:680] >= .5), .95)
        self.assertLess(np.mean(evidence.support[720:] >= .5), .02)
        self.assertEqual(evidence.scores["spike"].max(), 0)

    def test_flatline_at_background_center(self):
        x = signal()
        x[400:700] = 0
        evidence = extract_evidence(x, x[:300])
        self.assertEqual(evidence.summary()["morphology"], "flatline")
        self.assertGreater(np.mean(evidence.scores["flatline"][420:680] >= .5), .95)

    def test_disconnected_regions_are_not_stitched(self):
        x = signal()
        x[400:500] = 0
        x[700:800] = 0
        evidence = extract_evidence(x, x[:300])
        flat = [r for r in evidence.regions if r["morphology"] == "flatline"]
        self.assertTrue(flat)
        self.assertFalse(any(r["start"] < 500 and r["end"] > 700 for r in flat))
        self.assertEqual(evidence.scores["flatline"][550:650].max(), 0)

    def test_offset_and_channel_scale_invariance(self):
        x = signal()
        x[500] += 5
        a = extract_evidence(x, x[:300])
        b = extract_evidence(x * 100 + 500, x[:300] * 100 + 500)
        for kind in a.scores:
            np.testing.assert_allclose(a.scores[kind], b.scores[kind], atol=1e-6)

    def test_opposing_channels_and_source_prior_cannot_erase_spike(self):
        x = signal()
        x[500] += 5
        data = np.column_stack((x, -x))
        self.assertEqual(np.abs(data.mean(axis=1)).max(), 0)
        for source in ("", "NAB", "UCR"):
            evidence = extract_evidence(data, data[:300], source=source)
            summary = evidence.summary()
            self.assertEqual(summary["n_channels"], 2)
            self.assertEqual(summary["morphology"], "spike")
            self.assertFalse(summary["used_prior"])
            self.assertEqual(summary["prior_conflict"], bool(source))
            self.assertEqual(evidence.scores["spike"][500], 1)

    def test_short_spike_survives_long_input(self):
        x = np.zeros(100001)
        x[50001] = 10
        evidence = extract_evidence(x, x[:1000])
        self.assertEqual(evidence.scores["spike"][50001], 1)
        self.assertEqual(len(evidence.support), len(x))

    def test_relationship_break_without_marginal_shift(self):
        t = np.arange(1000)
        x = np.sin(2 * np.pi * t / 50)
        data = np.column_stack([x, x.copy()])
        data[400:700, 1] *= -1
        evidence = extract_evidence(data, data[:300])
        self.assertGreater(np.mean(evidence.scores["relationship_change"][400:700] >= .5), .8)

    def test_variance_change_and_seasonal_deviation(self):
        x = signal()
        x[400:700] *= 8
        e = extract_evidence(x, x[:300])
        self.assertGreater(np.mean(e.scores["variance_change"][450:650] >= .5), .8)
        t = np.arange(1000)
        periodic = np.sin(2 * np.pi * t / 50)
        periodic[400:700] *= 5
        e = extract_evidence(periodic, periodic[:300])
        self.assertGreater(e.scores["seasonal"][400:450].max(), .5)

    def test_trend_keeps_multiple_supported_morphologies(self):
        x = signal()
        x[400:800] += np.linspace(0, 10, 400)
        e = extract_evidence(x, x[:300])
        kinds = {item["morphology"] for item in e.summary()["ranked"]}
        self.assertIn("trend", kinds)
        self.assertIn("level_shift", kinds)

    def test_invalid_data_is_not_silently_repaired(self):
        for data in ([], np.ones(3), np.ones((10, 0)), np.ones((2, 3, 4)),
                     np.full(20, np.nan), np.full(20, np.inf)):
            with self.subTest(shape=np.shape(data)), self.assertRaises(ValueError):
                extract_evidence(data)
        with self.assertRaises(ValueError):
            extract_evidence(np.ones((20, 2)), np.ones((10, 3)))
        for scales in ([1], [2000], [8.5], [], [True]):
            with self.assertRaises(ValueError):
                extract_evidence(signal(), scales=scales)

    def test_tool_contracts_opt_in_and_legacy(self):
        x = signal()
        x[500] += 5
        for name in ("morphology_features", "morphology_classify"):
            tool = get(name)
            result = tool.run(np.column_stack([x, -x]), method="evidence",
                              reference=np.column_stack([x[:300], -x[:300]]))
            self.assertTrue(result.ok, result.error)
            self.assertEqual(result.data["n_channels"], 2)
            self.assertEqual(result.data["morphology"], "spike")
            legacy = tool.run(x)
            self.assertTrue(legacy.ok, legacy.error)
            self.assertNotIn("version", legacy.data)
            invalid = tool.run(x, method="typo")
            self.assertFalse(invalid.ok)


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.x = signal()
        self.x[400:700] += 3
        self.scores = {
            "baseline": np.arange(1000, dtype=float),
            "supported": np.where((np.arange(1000) >= 400) & (np.arange(1000) < 700), 10., 0.),
        }

    def route(self, **kwargs):
        return route_scores(self.x, self.scores, baseline="baseline", train_length=300, **kwargs)

    def test_real_score_evidence_switches_and_preserves_scores(self):
        result = self.route()
        self.assertEqual(result.decision["selected"], "supported")
        np.testing.assert_array_equal(result.scores, self.scores["supported"])
        self.assertFalse(np.shares_memory(result.scores, self.scores["supported"]))
        self.assertEqual(result.decision["reviewer_status"], "not_requested")

    def test_no_evidence_retains_baseline(self):
        result = route_scores(signal(), self.scores, baseline="baseline", train_length=300)
        self.assertEqual(result.decision["selected"], "baseline")
        self.assertEqual(result.decision["reason"], "insufficient_evidence")
        np.testing.assert_array_equal(result.scores, self.scores["baseline"])

    def test_sparse_tied_background_does_not_hide_a_perfect_spike_score(self):
        x = signal()
        x[500] += 5
        scores = {"baseline": np.arange(1000, dtype=float), "sparse": np.zeros(1000)}
        scores["sparse"][500] = 10
        result = route_scores(x, scores, baseline="baseline", train_length=300)
        self.assertEqual(result.decision["selected"], "sparse")
        self.assertEqual(result.decision["detectors"]["sparse"]["top_regions"],
                         [{"start": 500, "end": 501}])

    def test_inverted_detector_is_not_automatically_flipped(self):
        self.scores["inverted"] = -self.scores["supported"]
        result = self.route()
        self.assertNotIn("inverted", result.decision["qualified"])

    def test_ties_and_constant_candidates(self):
        self.scores["supported"] = self.scores["baseline"].copy()
        self.scores["constant"] = np.zeros(1000)
        result = self.route()
        self.assertEqual(result.decision["selected"], "baseline")
        self.assertTrue(result.decision["detectors"]["constant"]["constant"])
        support = np.array([1., 0., 0., 0.])
        scores = np.array([10., 10., 1., 1.])
        a = _quality(scores, support)["quality"]
        b = _quality(scores[::-1], support[::-1])["quality"]
        self.assertEqual(a, b)

    def test_strict_shape_nonfinite_baseline_and_train_contracts(self):
        for bad in (np.ones(999), np.ones((1000, 1)), np.full(1000, np.nan)):
            self.scores["supported"] = bad
            with self.assertRaises(ValueError):
                self.route()
        with self.assertRaises(ValueError):
            route_scores(self.x, {}, baseline="absent", train_length=300)
        for train in (0, 999, 300.5, True):
            with self.assertRaises(ValueError):
                route_scores(self.x, {"baseline": np.ones(1000)}, baseline="baseline",
                             train_length=train)

    def test_reviewer_cannot_bypass_gate(self):
        class BadReviewer:
            def chat(self, messages):
                return {"content": '{"selected":"unqualified","reason":"trust me"}'}
        with self.assertLogs("Architecture_design.evidence_routing", level="WARNING"):
            result = self.route(reviewer=BadReviewer())
        self.assertEqual(result.decision["selected"], "baseline")
        self.assertEqual(result.decision["reason"], "reviewer_failure")

    def test_invalid_reviewer_json_and_transport_errors_are_audited(self):
        from Architecture_design.client import OllamaError

        for value in ({"content": "not JSON"}, {"content": "[]"},
                      {"content": '{"selected":"supported"}'}, None):
            class BadReviewer:
                def chat(self, messages):
                    return value
            with self.subTest(value=value), self.assertLogs(
                    "Architecture_design.evidence_routing", level="WARNING"):
                result = self.route(reviewer=BadReviewer())
                self.assertEqual(result.decision["selected"], "baseline")
                self.assertEqual(result.decision["reviewer_status"], "failed")
        class FailedReviewer:
            def chat(self, messages):
                raise OllamaError("offline test")
        with self.assertLogs("Architecture_design.evidence_routing", level="WARNING"):
            self.assertEqual(self.route(reviewer=FailedReviewer()).decision["reason"],
                             "reviewer_failure")

    def test_reviewer_sees_actual_score_intervals_and_can_veto(self):
        class Reviewer:
            def chat(inner, messages):
                prompt = json.loads(messages[0]["content"])
                self.assertEqual(prompt["detectors"]["supported"]["top_regions"],
                                 [{"start": 400, "end": 700}])
                self.assertNotIn("labels", prompt)
                return {"content": '{"selected":"baseline","reason":"uncertain"}'}
        result = self.route(reviewer=Reviewer())
        self.assertEqual(result.decision["selected"], "baseline")
        self.assertEqual(result.decision["reviewer_status"], "accepted")


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.name = "001_demo_id_1_Sensor_tr_300_1st_400.csv"
        self.data = self.root / "data"
        self.data.mkdir()
        self.scores = self.root / "candidates"
        for model in ("baseline", "supported"):
            (self.scores / model).mkdir(parents=True)
        self.x = signal()
        self.x[400:700] += 3
        self.labels = np.zeros(1000)
        self.labels[400:700] = 1
        self.frame = pd.DataFrame({"value": self.x, "Label": self.labels})
        self.frame.to_csv(self.data / self.name, index=False)
        self.file_list = self.root / "list.csv"
        pd.DataFrame({"file_name": [self.name]}).to_csv(self.file_list, index=False)
        for model, values in (("baseline", np.arange(1000)),
                              ("supported", self.labels * 10)):
            np.save(self.scores / model / f"{Path(self.name).stem}.npy", values)

    def run_bench(self, output, **kwargs):
        return run_benchmark(data_dir=self.data, file_list=self.file_list,
                             score_dir=self.scores, output_dir=self.root / output,
                             models=["baseline", "supported"], baseline="baseline",
                             protocol_id="synthetic-regression-only", split="Tuning", **kwargs)

    def test_label_independence_persistence_and_no_overwrite(self):
        a = self.run_bench("a")
        self.frame["Label"] = ["not labels"] * 1000
        self.frame.to_csv(self.data / self.name, index=False)
        b = self.run_bench("b")
        self.assertEqual(a["status"], "complete")
        self.assertEqual(b["completed_count"], 1)
        self.assertTrue(a["full_coverage"])
        self.assertNotIn("mean_vus_pr", a)
        sa = np.load(self.root / "a" / "scores" / f"{Path(self.name).stem}.npy")
        sb = np.load(self.root / "b" / "scores" / f"{Path(self.name).stem}.npy")
        np.testing.assert_array_equal(sa, sb)
        decision = json.loads((self.root / "a" / "decisions.jsonl").read_text())
        self.assertEqual(decision["selected"], "supported")
        self.assertEqual(len(decision["score_sha256"]["baseline"]), 64)
        with self.assertRaises(FileExistsError):
            self.run_bench("a")

    def test_missing_cache_fails_instead_of_reducing_pool(self):
        path = self.scores / "supported" / f"{Path(self.name).stem}.npy"
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_bench("missing")
        status = json.loads((self.root / "missing" / "run.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertFalse(status["full_coverage"])
        self.assertNotIn("mean_vus_pr", status)

    def test_partial_run_is_not_reported_as_full_coverage(self):
        pd.DataFrame({"file_name": [self.name, "002_demo_tr_300_extra.csv"]}).to_csv(
            self.file_list, index=False)
        result = self.run_bench("partial", limit=1)
        self.assertEqual(result["file_count"], 2)
        self.assertEqual(result["completed_count"], 1)
        self.assertFalse(result["full_coverage"])

    def test_custom_list_is_not_official_full_coverage(self):
        official = self.root / "official.csv"
        pd.DataFrame({"file_name": [self.name, "002_demo_tr_300_extra.csv"]}).to_csv(
            official, index=False)
        result = self.run_bench("custom", official_file_list=official)
        self.assertTrue(result["full_coverage"])
        self.assertEqual(result["coverage_scope"], "requested_file_list")
        self.assertEqual(result["official_count"], 2)
        self.assertFalse(result["official_full_coverage"])

    def test_declared_official_split_rejects_foreign_files(self):
        official = self.root / "official.csv"
        pd.DataFrame({"file_name": ["002_demo_tr_300_extra.csv"]}).to_csv(official, index=False)
        with self.assertRaisesRegex(ValueError, "declared official split"):
            self.run_bench("foreign", official_file_list=official)
        self.assertFalse((self.root / "foreign").exists())

    def test_nan_data_preserves_alignment_by_failing(self):
        self.frame.loc[500, "value"] = np.nan
        self.frame.to_csv(self.data / self.name, index=False)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.run_bench("nan")

    def test_official_metric_evaluation_after_routing(self):
        result = self.run_bench("evaluated", evaluate=True)
        self.assertEqual(result["status"], "complete")
        self.assertGreater(result["mean_vus_pr"], result["baseline_mean_vus_pr"])
        rows = pd.read_csv(self.root / "evaluated" / "metrics.csv")
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows["VUS-PR"].iloc[0], .9)


if __name__ == "__main__":
    unittest.main()
