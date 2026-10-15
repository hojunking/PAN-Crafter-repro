"""Live tiny-model evaluation/retention tests, using injected numeric metrics.

All tensors and artifacts are disposable synthetic CPU fixtures. These tests
do not run benchmark metrics, launch a campaign or publish to Google Sheets.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from panda_method.test_stages import batch
from .evaluation import export_qualified, official_evaluation, retain_qualified
from .plan import qualify_evaluation, resolve_case
from .test_plan import baseline
from .test_training import config, model, take
from .training import Trainer, model_state_hash


class SyntheticSources:
    dataset, bands, peak = "QB", 4, 2047.
    mapping_sha256 = "c" * 64
    manifest = {"schema": "SYNTHETIC_TEST_ONLY", "no_real_source_claim": True}

    def __init__(self):
        self.observations = {"rr_eval": batch(split="test"),
                             "fr_eval": batch("FR", split="test", side=24)}

    def expected_ids(self, group):
        return tuple(f"{group}-synthetic-{i:02d}" for i in range(20))

    def batch(self, group, indices, *, device):
        if str(device) != "cpu":
            raise ValueError("CPU fixture only")
        return take(self.observations[group], [i % 2 for i in indices])

    def arrays(self, group, index):
        b = self.batch(group, [index], device="cpu")
        result = {name: ((getattr(b, field)[0].float() + 1) * (self.peak/2)).numpy().copy()
                  for name, field in (("pan", "pan"), ("ms", "ms"), ("lms", "native_lms"))}
        if b.gt is not None:
            result["gt"] = ((b.gt[0].float() + 1) * (self.peak/2)).numpy().copy()
        return result


def evaluation_config():
    recipe = baseline()
    recipe['teacher_architecture']['bands'] = 4
    recipe['qual_export'] = {"rgb_indices": [2, 1, 0], "display_range_DN": [0, 2047.],
                            "roi": "FULL_SCENE", "edge_gain": 1., "error_gain": 1.}
    return resolve_case('s2', baseline=recipe)


def engine():
    t = Trainer(model(), config())
    t.step([batch()], [batch("FR")])
    return t


def metrics(*, hqnr=.924, ergas=3.4):
    return {"rr": lambda prediction, reference: {
                "ergas": ergas, "scc": .97, "sam": 2., "psnr": 39., "ssim": .98,
                "q4": .95, "rmse": 11., "cc": .99},
            "fr": lambda prediction, reference: {
                "hqnr": hqnr, "d_s": .03, "d_lambda": .01, "jqm": .85,
                "jqm_weights": [.1, .2, .3, .4], "jqm_weight_source": "synthetic-fixture"}}


def evaluate(t, sources, functions=None):
    return official_evaluation(t, evaluation_config(), sources, device="cpu",
                               evaluator_revision="TEST", evaluator_pin={},
                               metric_functions=functions or metrics())


class OfficialEvaluationTests(unittest.TestCase):
    def test_complete_same_state_and_original_unmasked_native_references(self):
        t, sources = engine(), SyntheticSources()
        before = model_state_hash(t.stage)
        rng = torch.random.get_rng_state().clone()
        branch_rng = copy.deepcopy(t.objective.rng_state())
        calls = {"rr": 0, "fr": 0}
        checks = metrics()
        wrapped = {}
        for split in ("rr", "fr"):
            def metric(pred, arrays, branch=split):
                index = calls[branch]
                expected = sources.arrays(branch+"_eval", index)
                for name in ("pan", "ms", "lms"):
                    np.testing.assert_array_equal(arrays[name], expected[name])
                side = 16 if branch == "rr" else 24
                self.assertEqual(pred.shape, (4, side, side))
                self.assertEqual(arrays["pan"].shape, (1, side, side))
                self.assertFalse(torch.is_grad_enabled())
                self.assertFalse(t.stage.training)
                calls[branch] += 1
                torch.rand(1)  # Evaluation randomness cannot change training trajectory.
                return checks[branch](pred, arrays)
            wrapped[split] = metric
        record, outputs = evaluate(t, sources, wrapped)
        self.assertEqual(calls, {"rr": 20, "fr": 20})
        self.assertEqual(before, model_state_hash(t.stage))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(branch_rng["generators"]["rr"],
                                    t.objective.rng_state()["generators"]["rr"]))
        self.assertTrue(record["qualification"]["joint_pass"])
        self.assertTrue(record["test_aware"])
        for key in ("state_before", "rr_state", "fr_state", "state_after"):
            self.assertEqual(record[key]["model_state_sha256"], before)
            self.assertEqual(record[key]["completed_update"], 1)
        for split in ("rr", "fr"):
            self.assertEqual(len(outputs[split]), 20)
            self.assertEqual([r["scene_id"] for r in record[split]["per_scene"]],
                             list(sources.expected_ids(split+"_eval")))
            for row, raw in zip(record[split]["per_scene"], outputs[split]):
                self.assertEqual(row["raw_prediction_sha256"], hashlib.sha256(raw.tobytes()).hexdigest())

    def test_rounding_cannot_cross_either_target_gate(self):
        for h, e in ((.9229999999, 3.4), (.924, 3.5000001)):
            with self.subTest(h=h, e=e):
                record, _ = evaluate(engine(), SyntheticSources(), metrics(hqnr=h, ergas=e))
                self.assertFalse(record["qualification"]["joint_pass"])
                self.assertTrue(record["qualification"]["metrics_complete"])

    def test_model_mutation_during_metric_raises_and_poisons_attempt(self):
        t, sources = engine(), SyntheticSources()
        checks = metrics()
        original = metrics()["rr"]
        def mutate_once(pred, refs):
            next(t.stage.aligner.parameters()).add_(.001)
            return original(pred, refs)
        checks["rr"] = mutate_once
        with self.assertRaisesRegex(RuntimeError, "changed"):
            evaluate(t, sources, checks)
        self.assertTrue(t.poisoned)

    def test_nonfinite_metric_is_not_zero_filled(self):
        checks = metrics(ergas=float("nan"))
        with self.assertRaises(FloatingPointError):
            evaluate(engine(), SyntheticSources(), checks)

    def test_mixed_checkpoint_or_incomplete_scenes_cannot_qualify(self):
        sources = SyntheticSources()
        record, _ = evaluate(engine(), sources)
        for change in ("identity", "missing_scene"):
            item = copy.deepcopy(record)
            if change == "identity":
                item["fr_state"]["model_state_sha256"] = "f" * 64
            else:
                item["fr"]["per_scene"].pop()
            gate = qualify_evaluation(item, server="s2",
                    expected_rr_ids=sources.expected_ids("rr_eval"),
                    expected_fr_ids=sources.expected_ids("fr_eval"))
            self.assertFalse(gate["joint_pass"])


class StrictRetentionTests(unittest.TestCase):
    def test_failed_target_creates_no_weights_images_or_temporary_directories(self):
        t, sources = engine(), SyntheticSources()
        record, raw = evaluate(t, sources, metrics(hqnr=.9))
        with tempfile.TemporaryDirectory(prefix="teacher48-retention-test-") as temporary:
            target = Path(temporary)/"qualified"
            with patch("teacher48.evaluation.export_qualified") as exporter:
                with self.assertRaisesRegex(ValueError, "STRICT_TARGET_ONLY"):
                    retain_qualified(target, t, evaluation_config(), sources, record, raw,
                                     stream_states={}, provenance={"synthetic": True})
                exporter.assert_not_called()
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_stale_live_state_creates_no_artifact(self):
        t, sources = engine(), SyntheticSources()
        record, raw = evaluate(t, sources)
        t.step([batch()], [batch("FR")])
        with tempfile.TemporaryDirectory(prefix="teacher48-stale-test-") as temporary:
            with self.assertRaisesRegex(ValueError, "Live state moved"):
                retain_qualified(Path(temporary)/"qualified", t, evaluation_config(), sources, record, raw,
                                 stream_states={}, provenance={})
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_qualified_checkpoint_round_trip_preserves_exact_A_T_and_optimizer_state(self):
        t, sources = engine(), SyntheticSources()
        record, raw = evaluate(t, sources)
        state_sha = model_state_hash(t.stage)
        with tempfile.TemporaryDirectory(prefix="teacher48-pass-test-") as temporary:
            with patch("teacher48.evaluation.export_qualified", return_value={"unit-only": "abc"}) as exporter:
                result = retain_qualified(temporary, t, evaluation_config(), sources, record, raw,
                    stream_states={"rr": {"cursor": 2}, "fr": {"cursor": 2}},
                    provenance={"synthetic": True})
                exporter.assert_called_once()
            path = Path(result["artifact_path"])
            checkpoint = path/"teacher_aligner.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(result["status"], "QUAL_COMPLETE")
            self.assertEqual(result["model_state_sha256"], state_sha)
            self.assertEqual(result["checkpoint_file_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            for role, module in (("aligner", t.stage.aligner), ("teacher", t.stage.teacher)):
                for name, value in module.state_dict().items():
                    torch.testing.assert_close(value, payload[role][name], rtol=0, atol=0)
            restored = Trainer(model(), config())
            restored.restore_state(payload["training_state"])
            self.assertEqual(model_state_hash(restored.stage), state_sha)
            self.assertEqual(restored.updates, t.updates)
            self.assertEqual(payload["data_stream_states"]["rr"]["cursor"], 2)
            self.assertEqual(json.loads((path/"model_manifest.json").read_text())["status"], "QUAL_COMPLETE")
            self.assertEqual(list(path.glob(".qualified-*")), [])
            self.assertEqual(model_state_hash(t.stage), state_sha)

    def test_qualified_export_failure_keeps_checkpoint_and_explicit_pending_state(self):
        t, sources = engine(), SyntheticSources()
        record, raw = evaluate(t, sources)
        with tempfile.TemporaryDirectory(prefix="teacher48-export-test-") as temporary:
            with patch("teacher48.evaluation.export_qualified", side_effect=OSError("synthetic disk full")):
                result = retain_qualified(temporary, t, evaluation_config(), sources, record, raw,
                                         stream_states={}, provenance={})
            self.assertEqual(result["status"], "QUAL_PENDING")
            self.assertIn("synthetic disk full", result["export_error"])
            self.assertTrue((Path(result["artifact_path"])/"teacher_aligner.pt").is_file())

    def test_actual_exporter_all_scenes_and_unclipped_raw_identity(self):
        t, sources = engine(), SyntheticSources()
        record, raw = evaluate(t, sources)
        settings = evaluation_config()["baseline_resolved"]["qual_export"]
        with tempfile.TemporaryDirectory(prefix="teacher48-images-test-") as temporary:
            files = export_qualified(temporary, sources, record, raw, settings)
            self.assertEqual(len([p for p in files if p.endswith("prediction_unclipped_DN.npy")]), 40)
            for split in ("rr", "fr"):
                for i in range(20):
                    saved = np.load(Path(temporary)/f"{split}/scene_{i:02d}/prediction_unclipped_DN.npy")
                    np.testing.assert_array_equal(saved, raw[split][i])
            self.assertTrue((Path(temporary)/"rr/scene_00/GT.png").is_file())
            self.assertTrue((Path(temporary)/"rr/scene_00/absolute_error.png").is_file())
            self.assertTrue((Path(temporary)/"fr/scene_19/native_LMS.png").is_file())
            self.assertTrue((Path(temporary)/"fr/scene_19/corrected_edge_overlay.png").is_file())
        damaged = {**raw, "rr": [a.copy() for a in raw["rr"]]}
        damaged["rr"][0][0, 0, 0] += 1.
        with tempfile.TemporaryDirectory(prefix="teacher48-tamper-test-") as temporary:
            with self.assertRaisesRegex(ValueError, "differs"):
                export_qualified(temporary, sources, record, damaged, settings)


if __name__ == "__main__":
    unittest.main()
