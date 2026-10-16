"""Synthetic CPU/H5 paired checkpoint and unchanged Stage 2 integration tests."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from directshift48.training import DirectStage1Trainer
from pa.warp import warp_pan
from panda_method.calibration import state_dict_sha256
from panda_method.reconstruction import ReconstructionSpec
from panda_stage2_condq.calibration import CalibrationBuilder, CalibrationStats
from panda_stage2_condq.donor import load_donor as load_legacy_donor
from panda_stage2_condq.losses import reference_maps
from panda_stage2_condq.method import ConditionalQStage2, frozen_donor_predictions
from panda_stage2_condq.spec import Stage2Spec
from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
from panda_stage2_condq.training import StudentTrainer, TrainingSpec
from directshift48.method import ScharrAbsolute
from ta2.evaluation import file_sha

from s2dshift48.aligner import METHOD_VERSION, S2DGlobalPANAligner
from s2dshift48.donor import SCHEMA, load_donor, save_donor_checkpoint
from s2dshift48.stage1 import build_s2d_stage1


def make_trainer(registered, *, policy="MASK2"):
    stage = build_s2d_stage1(registered.sources,
        teacher_spec=ReconstructionSpec(8, 112, (1, 2, 3), "P0", "BICUBIC_MS"),
        teacher_seed=123, aligner_seed=456, epsilon_seed=789,
        mask_policy=policy, lambda_abs=1., lambda_eps=.001)
    optimizer = torch.optim.AdamW([
        {"params": stage.aligner.parameters(), "lr": 1e-5},
        {"params": stage.teacher.parameters(), "lr": 1e-4}], weight_decay=.01)
    return DirectStage1Trainer(stage, optimizer, gradient_log_interval=100,
                               effective_batch_size=1, microbatch_size=1)


class S2DDonorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.registered = synthetic_sources(cls.root, count=3)
        cls.trainer = make_trainer(cls.registered)
        cls.trainer.step(cls.registered.batch([0]))
        cls.path = cls.root / "s2d_donor.pt"
        cls.receipt = save_donor_checkpoint(cls.path, cls.trainer, cls.registered,
                                            run_id="SYNTHETIC_NOT_QUALIFIED_DONOR")
        cls.sha = cls.receipt["sha256"]
        cls.payload = torch.load(cls.path, map_location="cpu", weights_only=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.old_threads)

    def load(self, **kwargs):
        return load_donor(self.path, self.sha, self.registered, **kwargs)

    def corrupt(self, name, mutate):
        payload = deepcopy(self.payload)
        mutate(payload)
        path = self.root / name
        torch.save(payload, path)
        return path, file_sha(path)

    def calibrate(self, donor):
        collector = CalibrationBuilder(self.registered.sample_ids, donor.binding,
            k=2, memory_budget_bytes=16 * 1024**2, include_condq=True)
        spatial = ScharrAbsolute("WV3")
        visited = []
        for batch in self.registered.calibration_batches(2):
            frozen = frozen_donor_predictions(donor.aligner, donor.teacher, batch, sensor="WV3")
            e_t, g, q = reference_maps(spatial, frozen["teacher_y"], batch.gt, frozen["pan_aligned"])
            collector.add(batch.lineage.sample_ids, e_t, g, q, sensor="WV3", split="train",
                resolution="RR", unaugmented=True, support_ok=bool(frozen["spatial_support_ok"].all()))
            visited.extend(batch.lineage.sample_ids)
        self.assertEqual(tuple(visited), self.registered.sample_ids)
        return collector.finalize()

    def test_new_identity_exact_same_pair_frozen_and_rng_preserved(self):
        before = torch.random.get_rng_state().clone()
        donor = self.load()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertIs(type(donor.aligner), S2DGlobalPANAligner)
        self.assertEqual(donor.binding["donor_method"], METHOD_VERSION)
        self.assertEqual(donor.binding["aligner_manifest"], donor.aligner.manifest())
        self.assertEqual(donor.binding["sample_count"], 3)
        self.assertEqual(donor.binding["donor_checkpoint_id"], self.sha)
        self.assertFalse(donor.checkpoint_metadata["qualified_donor"])
        self.assertFalse(donor.checkpoint_metadata["campaign_resume_complete"])
        self.assertEqual(state_dict_sha256(donor.aligner), state_dict_sha256(self.trainer.stage.aligner))
        self.assertEqual(state_dict_sha256(donor.teacher), state_dict_sha256(self.trainer.stage.teacher))
        donor.assert_frozen()
        batch = self.registered.batch([0])
        with torch.no_grad():
            old = self.trainer.stage.native_prediction(batch)
            new = frozen_donor_predictions(donor.aligner, donor.teacher, batch, sensor="WV3")
        self.assertTrue(torch.equal(old["u"], new["u"]))
        self.assertTrue(torch.equal(old["pan_aligned"], new["pan_aligned"]))
        self.assertTrue(torch.equal(old["y"], new["teacher_y"]))

    def test_gt_only_does_not_construct_teacher_and_single_original_pan_warp(self):
        with patch("s2dshift48.donor.build_reconstructor", side_effect=AssertionError("Teacher constructed")):
            donor = self.load(include_teacher=False)
        self.assertIsNone(donor.teacher)
        self.assertFalse(donor.checkpoint_metadata["teacher_constructed"])
        spec = Stage2Spec("WV3", "gt_only", 321, 0., 0., 0., 1e-6)
        stage = ConditionalQStage2(spec, donor.aligner, donor_binding=donor.binding)
        batch = self.registered.batch([1])
        with patch("panda_stage2_condq.method.warp_pan", wraps=warp_pan) as warp:
            out = stage(batch)
        self.assertEqual(warp.call_count, 1)
        self.assertIs(warp.call_args.args[0], batch.pan)
        self.assertTrue(torch.equal(warp.call_args.args[1], out["u"]))
        self.assertTrue(torch.equal(out["pan_aligned"], warp_pan(batch.pan, out["u"])))

    def test_whole_new_train_calibration_condq_update_frozen_and_stale_rejected(self):
        donor = self.load()
        stats = self.calibrate(donor)
        self.assertEqual(stats.metadata["n_e"], 3 * 64 * 64)
        self.assertEqual(stats.metadata["n_q"], 3 * 48 * 48)
        self.assertEqual(stats.binding["donor_method"], METHOD_VERSION)
        path = self.root / "new_s2d_stats.npz"
        receipt = stats.save(path)
        stats = CalibrationStats.load(path, expected_binding=donor.binding,
            expected_file_sha256=receipt["file_sha256"], memory_budget_bytes=16 * 1024**2)
        spec = Stage2Spec("WV3", "panda_condq", 321, .1, .01, .1, 1e-6)
        stage = ConditionalQStage2(spec, donor.aligner, donor.teacher,
                                  donor_binding=donor.binding, stats=stats)
        a_sha, t_sha = state_dict_sha256(stage.aligner), state_dict_sha256(stage.teacher)
        s_sha = state_dict_sha256(stage.student)
        engine = StudentTrainer(stage, TrainingSpec(effective_batch_size=1, microbatch_size=1,
            warmup_updates=0, total_updates=3, gradient_log_interval=100))
        result = engine.step_accumulated([self.registered.batch([2])])
        self.assertEqual(engine.updates, 1)
        self.assertGreater(result["loss"], 0)
        self.assertEqual(a_sha, state_dict_sha256(stage.aligner))
        self.assertEqual(t_sha, state_dict_sha256(stage.teacher))
        self.assertNotEqual(s_sha, state_dict_sha256(stage.student))
        self.assertTrue(all(p.grad is None and not p.requires_grad for model in
                            (stage.aligner, stage.teacher) for p in model.parameters()))
        for key, altered in (("donor_checkpoint_id", "legacy_donor"),
                             ("donor_method", "old_DirectShift"),
                             ("aligner_manifest_sha256", "0" * 64)):
            binding = deepcopy(donor.binding)
            binding[key] = altered
            with self.assertRaises(ValueError):
                stats.validate_binding(binding)
            with self.assertRaises(ValueError):
                ConditionalQStage2(spec, donor.aligner, donor.teacher,
                                  donor_binding=binding, stats=stats)

    def test_export_is_exclusive_and_requires_completed_update(self):
        with self.assertRaises(FileExistsError):
            save_donor_checkpoint(self.path, self.trainer, self.registered, run_id="different")
        self.assertEqual(file_sha(self.path), self.sha)
        trainer = make_trainer(self.registered)
        with self.assertRaisesRegex(ValueError, "completed"):
            save_donor_checkpoint(self.root / "zero.pt", trainer, self.registered, run_id="zero")
        with self.assertRaises(ValueError):
            save_donor_checkpoint(self.root / "bad-purpose.pt", self.trainer, self.registered,
                                  run_id="bad", purpose="QUALIFIED_BEST")
        self.assertFalse((self.root / "zero.pt").exists())

    def test_both_mask_policies_export_without_changing_stage2_loss(self):
        trainer = make_trainer(self.registered, policy="UNMASKED")
        trainer.step(self.registered.batch([0]))
        path = self.root / "unmasked.pt"
        receipt = save_donor_checkpoint(path, trainer, self.registered, run_id="unmasked")
        donor = load_donor(path, receipt["sha256"], self.registered)
        self.assertEqual(donor.binding["donor_source_mask_policy"], "all_pixels_bands_mean")
        self.assertEqual(self.load().checkpoint_metadata["source_pipeline"]["mask_policy"], "MASK2")

    def test_legacy_and_stage2_schema_and_legacy_loader_rejected(self):
        for index, schema in enumerate(("DONOR30_CHECKPOINT_V1", "PANDA_STAGE2_CONDQ_CHECKPOINT_V1")):
            path, sha = self.corrupt(f"wrong-schema-{index}.pt", lambda value: value.update(schema=schema))
            with self.assertRaisesRegex(ValueError, "schema"):
                load_donor(path, sha, self.registered)
        with self.assertRaisesRegex(ValueError, "schema"):
            load_legacy_donor(self.path, self.sha, self.registered)
        self.assertEqual(self.payload["schema"], SCHEMA)

    def test_sha_pair_geometry_architecture_source_and_implementation_tamper_fail(self):
        with self.assertRaisesRegex(ValueError, "SHA"):
            load_donor(self.path, "0" * 64, self.registered)
        alterations = [
            lambda v: v["trainer"]["stage"]["aligner.fc2.bias"].add_(.1),
            lambda v: v["trainer"]["stage"]["_extra_state"]["pipeline"].update(inference_units="x4_FR"),
            lambda v: v["trainer"]["stage"]["_extra_state"]["pipeline"]["teacher"].update(width=104),
            lambda v: v["source_inventory"].update(source_h5_sha256="0" * 64),
            lambda v: v["implementation_pin"].update({"aligner.py": "0" * 64}),
            lambda v: v.update(qualified_donor=True),
        ]
        for index, mutate in enumerate(alterations):
            path, sha = self.corrupt(f"tamper-{index}.pt", mutate)
            with self.assertRaises(ValueError):
                load_donor(path, sha, self.registered)


if __name__ == "__main__":
    unittest.main()
