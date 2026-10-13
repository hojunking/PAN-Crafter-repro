"""Tiny synthetic native inference with injected metrics, never benchmarks."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from panda_method.test_edge_guided import batch, stage
from teacher48.edge_training import EdgeStage1Trainer
from ta2.evaluation import digest
from .evaluation import (expected_ids, export_qualified, model_state_hash,
                         official_evaluation, paused_model, retain_qualified, retry_qualified)
from .plan import qualify_evaluation


torch.set_num_threads(1)


class SyntheticSources:
    sensor, bands, peak = "QB", 4, 2047.
    catalog_path, catalog_sha256 = "synthetic-registry", "a" * 64
    entry, lms_overrides = {"synthetic": True}, {}

    def manifest_sha(self, split):
        return "a" * 64

    def batch(self, split, rows, *, device="cpu"):
        resolution = "FR" if split == "fr" else "RR"
        role = {"train": "train", "val": "validation", "rr": "test", "fr": "test"}[split]
        out = batch(side=32, resolution=resolution, split=role, batch_size=len(rows))
        return replace(out, lineage=replace(out.lineage,
            sample_ids=tuple(f"QB:{split}:{i}" for i in rows)))

    def raw(self, split, rows):
        b = self.batch(split, rows)
        result = {name: ((getattr(b, field).float()+1)*(self.peak/2)).numpy().copy()
                  for name, field in (("pan", "pan"), ("ms", "ms"), ("lms", "native_lms"))}
        if b.gt is not None:
            result["gt"] = ((b.gt.float()+1)*(self.peak/2)).numpy().copy()
        return result


def engine():
    s = stage()
    optimizer = torch.optim.AdamW([{"params": s.aligner.parameters(), "lr": 1e-5},
                                   {"params": s.teacher.parameters(), "lr": 1e-4}])
    trainer = EdgeStage1Trainer(s, optimizer, gradient_log_interval=100)
    trainer.updates = 1
    return trainer


def config(trainer):
    return {"run_id": "EGR2_QB_S2_TEST", "server": "s2", "dataset": "QB", "seed": 261013201,
            "attempt": 1, "edge_protocol_digest": digest(trainer.stage.matcher.manifest()),
            "baseline_resolved": {"qual_export": {"rgb_indices": [2, 1, 0],
                "display_range_DN": [0, 2047.], "roi": "FULL_SCENE", "error_gain": 1.}}}


def metrics(hqnr=.924, ergas=3.4):
    return {"rr": lambda p, a: {"ergas": ergas, "scc": .97, "sam": 2., "psnr": 39.,
                                "ssim": .98, "q4": .95, "rmse": 11., "cc": .99},
            "fr": lambda p, a: {"hqnr": hqnr, "d_s": .03, "d_lambda": .01, "jqm": .85,
                                "jqm_weights": [.1, .2, .3, .4], "jqm_weight_source": "synthetic"}}


def evaluate(trainer, sources, functions=None, **kwargs):
    with patch("edgeres48.evaluation.validate_method_config"):
        return official_evaluation(trainer, config(trainer), sources, device="cpu", evaluator_pin={},
            evaluator_revision="TEST", metric_functions=functions or metrics(), **kwargs)


class OfficialEvaluationTests(unittest.TestCase):
    def test_complete_same_state_original_refs_all_metrics_without_predictions_retained(self):
        t, s = engine(), SyntheticSources()
        before, rng = model_state_hash(t.stage), torch.random.get_rng_state().clone()
        calls = {"rr": 0, "fr": 0}
        functions = metrics()
        for split, fn in list(functions.items()):
            def wrapped(raw, arrays, split=split, fn=fn):
                expected = s.raw(split, [calls[split]])
                for key in expected:
                    np.testing.assert_array_equal(arrays[key], expected[key][0])
                self.assertEqual(raw.shape, (4, 32, 32))
                self.assertFalse(torch.is_grad_enabled())
                self.assertFalse(t.stage.training)
                torch.rand(1)
                calls[split] += 1
                return fn(raw, arrays)
            functions[split] = wrapped
        record, raw = evaluate(t, s, functions)
        self.assertEqual(calls, {"rr": 20, "fr": 20})
        self.assertEqual(raw, {"rr": [], "fr": []})
        self.assertTrue(record["qualification"]["joint_pass"])
        self.assertEqual(model_state_hash(t.stage), before)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for split in ("rr", "fr"):
            self.assertEqual([r["sample_id"] for r in record[split]["per_scene"]], list(expected_ids(s, split)))
        self.assertIn("CC", record["rr"]["metrics"])
        self.assertIn("RMSE", record["rr"]["metrics"])
        self.assertIn("JQM", record["fr"]["metrics"])

    def test_rounding_incomplete_and_intervention_never_qualify(self):
        t, s = engine(), SyntheticSources()
        for h, e in ((.9229999999, 3.4), (.924, 3.500000001)):
            record, _ = evaluate(t, s, metrics(h, e))
            self.assertFalse(record["qualification"]["joint_pass"])
        good, _ = evaluate(t, s)
        for change in ("missing", "mixed", "diag"):
            record = copy.deepcopy(good)
            if change == "missing": record["rr"]["per_scene"].pop()
            if change == "mixed": record["fr_state"]["model_state_sha256"] = "b" * 64
            if change == "diag": record["full_method_eligible"] = False
            gate = qualify_evaluation(record, server="s2", expected_rr_ids=expected_ids(s, "rr"), expected_fr_ids=expected_ids(s, "fr"))
            self.assertFalse(gate["joint_pass"])

    def test_nonfinite_and_expired_evaluation_do_not_fake_completion(self):
        t, s = engine(), SyntheticSources()
        with self.assertRaises(FloatingPointError):
            evaluate(t, s, metrics(ergas=float("nan")))
        with self.assertRaisesRegex(TimeoutError, "INCOMPLETE"):
            evaluate(t, s, deadline_unix=0)

    def test_live_weight_mutation_detected_and_poisoned(self):
        t, s = engine(), SyntheticSources()
        functions = metrics()
        def mutate(p, a):
            next(t.stage.aligner.parameters()).add_(.01)
            return metrics()["rr"](p, a)
        functions["rr"] = mutate
        with self.assertRaisesRegex(RuntimeError, "changed"):
            evaluate(t, s, functions)
        self.assertTrue(t.poisoned)

    def test_modes_gradients_rng_and_call_counter_restore_after_exception(self):
        t = engine()
        next(t.stage.aligner.parameters()).grad = torch.ones_like(next(t.stage.aligner.parameters()))
        rng = t.stage._epsilon_rng.get_state().clone()
        mode = t.stage.training
        with self.assertRaisesRegex(ValueError, "test interrupt"):
            with paused_model(t, gradients=True):
                torch.rand(1, generator=t.stage._epsilon_rng)
                t.stage.forward_calls += 3
                next(t.stage.aligner.parameters()).grad.zero_()
                raise ValueError("test interrupt")
        self.assertTrue(torch.equal(rng, t.stage._epsilon_rng.get_state()))
        self.assertEqual(t.stage.forward_calls, 0)
        self.assertEqual(t.stage.training, mode)
        self.assertTrue(bool((next(t.stage.aligner.parameters()).grad == 1).all()))


class RetentionTests(unittest.TestCase):
    def test_failed_snapshot_creates_no_directory_or_weights(self):
        t, s = engine(), SyntheticSources()
        record, outputs = evaluate(t, s, metrics(hqnr=.9))
        with tempfile.TemporaryDirectory() as tmp, patch("edgeres48.evaluation.validate_method_config"):
            with self.assertRaisesRegex(ValueError, "STRICT_TARGET_ONLY"):
                retain_qualified(Path(tmp)/"artifacts", t, config(t), s, record, outputs, stream_states={}, provenance={})
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_qualified_save_reload_then_export_is_ordered_and_hash_bound(self):
        t, s = engine(), SyntheticSources()
        record, outputs = evaluate(t, s)
        with tempfile.TemporaryDirectory() as tmp, patch("edgeres48.evaluation.validate_method_config"):
            def exporter(path, trainer, sources, evaluation, settings, device, deadline_unix=None):
                self.assertTrue((Path(path).parent / "teacher_aligner.pt").is_file())
                return {"synthetic-only": "verified"}
            with patch("edgeres48.evaluation.export_qualified", side_effect=exporter):
                retained = retain_qualified(tmp, t, config(t), s, record, outputs, stream_states={}, provenance={})
            saved = torch.load(retained["checkpoint_path"], map_location="cpu", weights_only=False)
            self.assertEqual(saved["identity"]["model_state_sha256"], model_state_hash(t.stage))
            self.assertEqual(retained["status"], "QUAL_COMPLETE")

    def test_qual_export_failure_remains_pending_with_only_qualified_weight(self):
        t, s = engine(), SyntheticSources()
        record, outputs = evaluate(t, s)
        with tempfile.TemporaryDirectory() as tmp, patch("edgeres48.evaluation.validate_method_config"), \
                patch("edgeres48.evaluation.export_qualified", side_effect=ValueError("source unavailable")):
            retained = retain_qualified(tmp, t, config(t), s, record, outputs, stream_states={}, provenance={})
            self.assertEqual(retained["status"], "QUAL_PENDING")
            self.assertTrue(Path(retained["checkpoint_path"]).is_file())

    def test_stale_live_state_is_rejected_before_creating_artifacts(self):
        t, s = engine(), SyntheticSources()
        record, outputs = evaluate(t, s)
        t.updates += 1
        with tempfile.TemporaryDirectory() as tmp, patch("edgeres48.evaluation.validate_method_config"):
            with self.assertRaisesRegex(ValueError, "Live state moved"):
                retain_qualified(tmp, t, config(t), s, record, outputs, stream_states={}, provenance={})
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_pending_export_retries_only_saved_qualified_checkpoint(self):
        t, s = engine(), SyntheticSources()
        record, outputs = evaluate(t, s)
        with tempfile.TemporaryDirectory() as tmp, patch("edgeres48.evaluation.validate_method_config"):
            with patch("edgeres48.evaluation.export_qualified", side_effect=ValueError("temporary export failure")):
                saved = retain_qualified(tmp, t, config(t), s, record, outputs, stream_states={}, provenance={})
            # The live training run has moved; retry must reconstruct saved
            # state in an entirely fresh trainer, never roll the live run back.
            t.updates = 9
            with torch.no_grad():
                next(t.stage.aligner.parameters()).add_(.01)
            live = model_state_hash(t.stage)
            def verify_saved(path, trainer, sources, evaluation, settings, device, deadline_unix=None):
                self.assertEqual(trainer.updates, 1)
                self.assertEqual(model_state_hash(trainer.stage), record["state_after"]["model_state_sha256"])
                return {"unit-only": "ok"}
            with patch("edgeres48.evaluation.export_qualified", side_effect=verify_saved):
                result = retry_qualified(saved["artifact_path"], s, lambda cfg: engine(), device="cpu")
            self.assertEqual(result["status"], "QUAL_COMPLETE")
            self.assertNotIn("export_error", result)
            self.assertEqual(model_state_hash(t.stage), live)
            self.assertEqual(t.updates, 9)

    def test_real_tiny_qualified_export_all40_raw_and_display_is_idempotent(self):
        t, s = engine(), SyntheticSources()
        record, _ = evaluate(t, s)
        with tempfile.TemporaryDirectory() as tmp:
            settings = config(t)["baseline_resolved"]["qual_export"]
            first = export_qualified(tmp, t, s, record, settings, device="cpu")
            second = export_qualified(tmp, t, s, record, settings, device="cpu")
            self.assertEqual(first, second)
            self.assertEqual(len(list(Path(tmp).rglob("prediction_unclipped_DN.npy"))), 40)
            self.assertEqual(len(list(Path(tmp).rglob("prediction.png"))), 40)
            raw = np.load(Path(tmp)/"rr/image_00/prediction_unclipped_DN.npy", allow_pickle=False)
            self.assertEqual(raw.shape, (4, 32, 32))


if __name__ == "__main__":
    unittest.main()
