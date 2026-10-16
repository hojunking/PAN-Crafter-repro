"""Offline synthetic end-to-end runtime/CLI tests, never production calibration."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from donor24 import plan
from donor24.evaluation import model_state_hash
from donor30.artifacts import SCHEMA as DONOR_SCHEMA
from donor30.runtime import build_trainer
from ta2.evaluation import file_sha
from panda_stage2_condq import __main__ as cli, runtime
from panda_stage2_condq.calibration import CalibrationStats
from panda_stage2_condq.donor import load_donor
from panda_stage2_condq.method import frozen_donor_predictions
from panda_stage2_condq.spec import Stage2Spec, MASK2_POLICY
from pa.warp import warp_support_mask
from panda_stage2_condq.stream import StreamSpec, TrainStream
from panda_stage2_condq.training import TrainingSpec
from panda_stage2_condq.tests.test_donor_sources import synthetic_sources


class RuntimeCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.registered = synthetic_sources(cls.root, count=3)
        cfg = plan.resolve_case(plan.default_config("s1"), "A0")
        cfg["run_id"] = "SYNTHETIC_RUNTIME_TEST_NOT_AN_EXPERIMENT"
        trainer = build_trainer(cfg, cls.registered.sources)
        trainer.updates = 7
        identity = {"model_state_sha256": model_state_hash(trainer.stage),
            "completed_update": 7, "run_id": cfg["run_id"], "dataset": "WV3"}
        payload = {"schema": DONOR_SCHEMA, "config": cfg, "trainer": trainer.state_dict(),
                   "identity": identity, "state_hash": identity["model_state_sha256"]}
        cls.path = cls.root / "synthetic_donor.pt"
        torch.save(payload, cls.path)
        cls.sha = file_sha(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.old_threads)

    def donor(self, include_teacher=True):
        return load_donor(self.path, self.sha, self.registered, include_teacher=include_teacher)

    def test_complete_condq_calibration_counts_readback_and_no_overwrite(self):
        donor = self.donor()
        output = self.root / "complete_condq.npz"
        progress = []
        result = runtime.calibrate(donor, self.registered, output=output, k=5,
            batch_size=2, memory_budget_bytes=4 * 1024**2, include_condq=True,
            on_progress=progress.append)
        self.assertEqual(result["kind"], "exact_condq")
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["n_e"], 3 * 64 * 64)
        self.assertEqual(result["n_q"], 3 * 48 * 48)
        self.assertEqual([p["seen"] for p in progress], [2, 3])
        loaded = CalibrationStats.load(output, expected_binding=donor.binding,
                                       expected_file_sha256=result["file_sha256"])
        self.assertEqual(loaded.artifact_sha256, result["artifact_sha256"])
        errors = []
        for batch in self.registered.calibration_batches(2):
            out = frozen_donor_predictions(donor.aligner, donor.teacher, batch, sensor="WV3")
            errors.append((out["teacher_y"] - batch.gt).abs().mean(1).flatten().numpy())
        self.assertAlmostEqual(float(np.median(np.concatenate(errors))), loaded.tau_e, places=7)
        with self.assertRaises(FileExistsError):
            runtime.calibrate(donor, self.registered, output=output, k=5,
                batch_size=2, memory_budget_bytes=4 * 1024**2, include_condq=True)
        self.assertFalse(list(self.root.glob("*.partial")))

    def test_e_only_calibration_skips_scharr_and_conditional_tables(self):
        donor = self.donor()
        output = self.root / "tau_e_only.npz"
        with patch.object(runtime, "ScharrAbsolute", side_effect=AssertionError("Scharr should be skipped")), \
                patch.object(runtime, "reference_maps", side_effect=AssertionError("q should be skipped")):
            result = runtime.calibrate(donor, self.registered, output=output, k=5,
                batch_size=2, memory_budget_bytes=4 * 1024**2, include_condq=False)
        self.assertEqual(result["kind"], "tau_e_only")
        self.assertEqual(result["n_e"], 3 * 4096)
        self.assertEqual(result["n_q"], 0)
        loaded = CalibrationStats.load(output, expected_binding=donor.binding)
        self.assertEqual(loaded.memory_bytes, 0)

    def test_masked_calibration_and_condq_diagnostic_complete_same_bound_population(self):
        donor = self.donor()
        output = self.root / 'MASK2_condq.npz'
        result = runtime.calibrate(donor, self.registered, output=output, k=5,
            batch_size=2, memory_budget_bytes=4*1024**2, include_condq=True,
            mask_policy=MASK2_POLICY)
        loaded = CalibrationStats.load(output, expected_binding=donor.binding,
            expected_mask_policy=MASK2_POLICY, expected_file_sha256=result['file_sha256'])
        errors = []
        for rr in self.registered.calibration_batches(2):
            out = frozen_donor_predictions(donor.aligner, donor.teacher, rr, sensor='WV3')
            mask = warp_support_mask(64, 64, out['correction'])
            e = (out['teacher_y']-rr.gt).abs().mean(1, keepdim=True)
            errors.append(e[mask].cpu().numpy())
        values = np.concatenate(errors).astype(np.float64)
        self.assertEqual(loaded.tau_e, float(np.median(values)))
        self.assertEqual(result['n_e'], len(values))
        self.assertLess(result['n_e'], 3*4096)
        self.assertEqual(result['n_q'], 3*48*48)
        spec = Stage2Spec('WV3', 'panda_condq', 3, .5, .05, .5, 1e-6,
                          mask_policy=MASK2_POLICY)
        report = runtime.diagnostic(donor, self.registered, spec,
            TrainingSpec(microbatch_size=1, effective_batch_size=1), StreamSpec(10, 11),
            stats=loaded, output=self.root/'MASK2_condq_diagnostic', updates=1,
            calibration_device_budget_bytes=4*1024**2)
        self.assertEqual(report['status'], 'DIAGNOSTIC_COMPLETE')
        self.assertEqual(report['contract']['mask_policy'], MASK2_POLICY)
        self.assertEqual(report['completed_updates'], 1)
        donor.assert_frozen()

    def test_partial_failed_calibration_has_no_completed_artifact(self):
        donor = self.donor()
        output = self.root / "interrupted_calibration.npz"
        def failing(*args, **kwargs):
            yield self.registered.batch([0])
            raise RuntimeError("SYNTHETIC_INTERRUPT")
        with patch.object(self.registered, "calibration_batches", side_effect=failing):
            with self.assertRaisesRegex(RuntimeError, "SYNTHETIC_INTERRUPT"):
                runtime.calibrate(donor, self.registered, output=output, k=5,
                    batch_size=1, memory_budget_bytes=4 * 1024**2, include_condq=True)
        self.assertFalse(output.exists())
        with self.assertRaises(MemoryError):
            runtime.calibrate(donor, self.registered, output=output, k=5,
                batch_size=1, memory_budget_bytes=10, include_condq=True)
        self.assertFalse(output.exists())

    def test_diagnostic_single_update_isolated_manifest_and_checkpoint(self):
        donor = self.donor(include_teacher=False)
        spec = Stage2Spec("WV3", "gt_only", 3, 0, 0, 0, 1e-6)
        training = TrainingSpec(microbatch_size=1, effective_batch_size=1)
        stream_spec = StreamSpec(10, 11)
        output = self.root / "diagnostic_synthetic_only"
        report = runtime.diagnostic(donor, self.registered, spec, training, stream_spec,
            output=output, updates=1, calibration_device_budget_bytes=4 * 1024**2)
        self.assertEqual(report["status"], "DIAGNOSTIC_COMPLETE")
        self.assertFalse(report["official_experiment"])
        self.assertTrue(report["no_sheet_upload"])
        self.assertEqual(report["completed_updates"], 1)
        self.assertEqual(report["calibration_table_bytes"], 0)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["checkpoint"], report["checkpoint"])
        self.assertTrue((output / "diagnostic_checkpoint.pt").is_file())
        self.assertEqual(len((output / "train.jsonl").read_text().splitlines()), 1)
        self.assertFalse((output / "outbox").exists())
        donor.assert_frozen()
        with self.assertRaises(FileExistsError):
            runtime.diagnostic(donor, self.registered, spec, training, stream_spec, output=output, updates=1)
        with self.assertRaises(ValueError):
            runtime.diagnostic(donor, self.registered, spec, training, stream_spec,
                               output=self.root / "must_not_run", updates=3)

    def test_stream_paired_order_augmentation_resume_and_contract(self):
        spec = StreamSpec(100, 101)
        def new():
            return TrainStream(self.registered, spec, microbatch_size=1, effective_batch_size=2)
        def signature(batches):
            return [(b.lineage.sample_ids, b.lineage.augmentations) for b in batches]
        first, paired = new(), new()
        self.assertEqual(signature(first.next_update()), signature(paired.next_update()))
        saved = first.state_dict()
        expected = first.next_update()
        restored = new()
        restored.load_state_dict(saved)
        actual = restored.next_update()
        self.assertEqual(signature(expected), signature(actual))
        for left, right in zip(expected, actual):
            self.assertTrue(torch.equal(left.pan, right.pan))
            self.assertEqual(left.lineage.augmentations[0][:2], (True, True))
        self.assertEqual(first.state_dict(), restored.state_dict())
        changed = new()
        changed.spec = replace(spec, hflip_probability=.5)
        with self.assertRaisesRegex(ValueError, "contract"):
            changed.next_update()
        for field, value in (("cursor", 3), ("order", [0, 0, 2])):
            altered = deepcopy(saved)
            altered[field] = value
            with self.assertRaises(ValueError):
                new().load_state_dict(altered)
        changed = TrainStream(self.registered, StreamSpec(100, 102),
                              microbatch_size=1, effective_batch_size=2)
        with self.assertRaises(ValueError):
            changed.load_state_dict(saved)

    def test_cli_requires_explicit_donor_and_has_no_run_subcommand(self):
        with patch("panda_stage2_condq.sources.RegisteredRRTrain.from_catalog",
                   side_effect=AssertionError("CLI must not touch real data")):
            for args in ([], ["run"], ["calibrate"], ["diagnostic"], ["inspect-donor"]):
                with self.subTest(args=args), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        cli.main(args)
                    self.assertEqual(raised.exception.code, 2)

    def test_cli_smoke_only_dispatches_offline_suite_without_real_data(self):
        result = Mock()
        result.wasSuccessful.return_value = True
        runner = Mock()
        runner.run.return_value = result
        suite = unittest.TestSuite()
        with patch.dict(os.environ), \
                patch("unittest.defaultTestLoader.discover", return_value=suite) as discover, \
                patch("unittest.TextTestRunner", return_value=runner), \
                patch("panda_stage2_condq.sources.RegisteredRRTrain.from_catalog",
                      side_effect=AssertionError("smoke loaded production data")):
            self.assertEqual(cli.main(["smoke"]), 0)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            discover.assert_called_once()
            runner.run.assert_called_once_with(suite)

    def test_cli_inspect_donor_is_explicit_readonly_and_without_teacher(self):
        output = io.StringIO()
        with patch.object(cli, "ROOT", self.root), redirect_stdout(output), \
                patch("panda_stage2_condq.donor.build_reconstructor",
                      side_effect=AssertionError("inspect without Teacher constructed T")):
            code = cli.main(["inspect-donor", "--checkpoint", str(self.path),
                "--checkpoint-sha256", self.sha, "--sensor", "WV3",
                "--sources", str(self.registered.sources.catalog_path), "--without-teacher",
                "--cpu-threads", "1"])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertFalse(result["donor_selected"])
        self.assertFalse(result["training_started"])
        self.assertEqual(result["sample_count"], 3)
        self.assertFalse(result["checkpoint"]["teacher_constructed"])


if __name__ == "__main__":
    unittest.main()
