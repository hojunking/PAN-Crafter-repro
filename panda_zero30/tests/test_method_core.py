"""CPU-only ZERO geometry, paired loss, frozen references and exact resume."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from directshift48.method import ScharrAbsolute
from donor30.artifacts import _same
from donor30.masking import masked_mean
from pa.warp import warp_pan, warp_support_mask
from panda_band30.tests.test_method import make_donor
from panda_band30.reconstruction import base_from_batch, reconstruct_with_base
from panda_method.calibration import state_dict_sha256
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.losses import reference_maps, compute_loss
from panda_stage2_condq.tests.test_method import batch
from panda_stage2_condq.training import TrainingSpec
from panda_zero30 import calibration as C
from panda_zero30.engine import MatureStudentTrainer
from panda_zero30.method import (
    MatureStage2, applied_correction, frozen_predictions, rotate_view,
    common_view_mask, paired_view_loss,
)
from panda_zero30.spec import MatureSpec, policy_manifest, view_manifest


def make_spec(policy="ZERO"):
    return MatureSpec("WV3", "panda_condq", 345, .25, 0., .5, 1e-6, policy_id=policy)


def make_fixture(policy="ZERO", shift=(1., -.5)):
    spec, rr = make_spec(policy), batch()
    donor = C.bind_donor(make_donor(shift=shift), spec)
    frozen = frozen_predictions(donor.aligner, donor.teacher, rr, geometry_key="s000")
    e, g, q = reference_maps(ScharrAbsolute("WV3"), frozen["teacher_y"], rr.gt, rr.pan)
    collector = CalibrationBuilder(["WV3:train:0"], C.binding_for_geometry(donor.binding, "s000"),
        k=5, memory_budget_bytes=8*1024**2, include_condq=True, mask_policy=spec.mask_policy)
    collector.add(["WV3:train:0"], e, g, q, sensor="WV3", loss_mask=frozen["loss_mask"])
    bank = {"s000": collector.finalize()}
    stage = MatureStage2(spec, donor.aligner, donor.teacher, donor_binding=donor.binding, stats_bank=bank)
    with torch.no_grad():
        stage.student.backbone.output[-1].weight.copy_(stage.teacher.backbone.output[-1].weight)
    return stage, donor, bank


def make_trainer(policy="ROTCONS"):
    stage, _, _ = make_fixture(policy)
    training = TrainingSpec(total_updates=50000, warmup_updates=0,
        microbatch_size=1, effective_batch_size=1, full_audit_interval=1)
    return MatureStudentTrainer(stage, training)


class MethodCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_registered_policies_share_only_s000_and_pin_view_contract(self):
        for policy, count, coefficient in (("ZERO", 1, 0.), ("PAIR0", 2, 0.), ("ROTCONS", 2, .05)):
            spec = make_spec(policy)
            self.assertEqual(spec.geometry_keys, ("s000",))
            self.assertEqual(spec.geometry_at_completed(50000), "s000")
            self.assertEqual(spec.geometry_at_next(0), "s000")
            self.assertEqual(spec.consistency_lambda, coefficient)
            self.assertEqual(view_manifest(spec)["student_views"], count)
            self.assertEqual(policy_manifest(spec)["view_loss"], view_manifest(spec))
            self.assertEqual(spec.beta, 0.)
            self.assertIn("INTERP23", spec.manifest()["student"]["base_reference"])
        with self.assertRaises(ValueError):
            replace(make_spec(), policy_id="FULL")
        with self.assertRaises(ValueError):
            replace(make_spec(), beta=.05)
        with self.assertRaises(ValueError):
            make_spec().geometry_at_next(50000)
        with self.assertRaises(ValueError):
            applied_correction(torch.zeros(1, 2), "s025")

    def test_zero_is_exact_original_pan_and_teacher_stays_full_u(self):
        rr, donor = batch(), make_donor(shift=(2., -1.))
        frozen = frozen_predictions(donor.aligner, donor.teacher, rr, geometry_key="s000")
        self.assertIs(frozen["pan_input"], rr.pan)
        self.assertIs(frozen["gt_loss"], rr.gt)
        self.assertEqual(int(torch.count_nonzero(frozen["pan_correction"])), 0)
        base = base_from_batch(rr, "interp23")
        expected = reconstruct_with_base(donor.teacher, warp_pan(rr.pan, frozen["u"]), base, rr.ms)
        torch.testing.assert_close(frozen["teacher_y"], expected["y"], rtol=0, atol=0)
        support = warp_support_mask(64, 64, torch.zeros_like(frozen["u"]))
        support &= warp_support_mask(64, 64, frozen["u"])
        self.assertTrue(torch.equal(frozen["loss_mask"], support))

    def test_masks_inverse_rotate_before_intersection_and_reject_bad_support(self):
        mask = torch.ones(2, 1, 64, 64, dtype=torch.bool)
        mask[..., :2, :] = False
        companion = rotate_view(mask)
        self.assertFalse(torch.equal(mask, companion))
        self.assertTrue(torch.equal(common_view_mask(mask, companion), mask))
        companion = companion.clone()
        companion[..., :3, :] = False
        expected = mask & rotate_view(companion, inverse=True)
        self.assertTrue(torch.equal(common_view_mask(mask, companion), expected))
        companion[..., 20, 20] = False
        with self.assertRaisesRegex(ValueError, "ROI8"):
            common_view_mask(mask, companion)

    def test_pair0_reuses_canonical_maps_and_has_both_view_gradients(self):
        stage, _, _ = make_fixture("PAIR0")
        rr = batch()
        with patch.object(stage.aligner, "forward", wraps=stage.aligner.forward) as aligner_call, \
             patch.object(stage.teacher.backbone, "forward", wraps=stage.teacher.backbone.forward) as teacher_call, \
             patch.object(stage.stats_bank["s000"], "lookup_exact", wraps=stage.stats_bank["s000"].lookup_exact) as lookup:
            out = stage(rr, return_maps=True)
        self.assertEqual(aligner_call.call_count, 1)
        self.assertEqual(teacher_call.call_count, 1)
        self.assertEqual(lookup.call_count, 1)
        out["student"]["y"].retain_grad()
        out["student_rotated"]["y"].retain_grad()
        expected0 = masked_mean(out["w_h"] * (out["student"]["y"] - rr.gt).abs().mean(1, keepdim=True),
                                out["loss_mask"])
        expectedr = masked_mean(rotate_view(out["w_h"]) *
            (out["student_rotated"]["y"] - rotate_view(rr.gt)).abs().mean(1, keepdim=True),
            rotate_view(out["loss_mask"]))
        torch.testing.assert_close(out["loss"], .5 * (expected0 + expectedr), rtol=0, atol=0)
        self.assertEqual(float(out["soft_weighted"]), 0.)
        self.assertEqual(float(out["consistency_weighted"]), 0.)
        torch.testing.assert_close(out["student_rotated"]["ms_base"],
                                   rotate_view(out["student"]["ms_base"]), rtol=0, atol=0)
        out["loss"].backward()
        for key in ("student", "student_rotated"):
            self.assertGreater(float(out[key]["y"].grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for module in (stage.aligner, stage.teacher) for p in module.parameters()))

    def test_rotcons_scalar_formula_units_and_two_way_gradient(self):
        spec = make_spec("ROTCONS")
        gt = torch.zeros(1, 8, 32, 32)
        first = torch.full_like(gt, .2, requires_grad=True)
        second = torch.full_like(gt, .6, requires_grad=True)
        mask = torch.ones(1, 1, 32, 32, dtype=torch.bool)
        z, one = torch.zeros_like(mask, dtype=torch.float32), torch.ones_like(mask, dtype=torch.float32)
        terms = {"loss_mask": mask, "maps": {"w_h": one, "d_e": z, "r_q_bar": z}}
        out = paired_view_loss(terms, first, second, gt, spec=spec)
        self.assertAlmostEqual(float(out["hard_weighted"]), .4, places=6)
        self.assertAlmostEqual(float(out["consistency_raw"]), .4, places=6)
        self.assertAlmostEqual(float(out["loss"]), .42, places=6)
        out["consistency_raw"].backward()
        self.assertLess(float(first.grad.sum()), 0.)
        self.assertGreater(float(second.grad.sum()), 0.)
        zero = paired_view_loss(terms, first, rotate_view(first), gt, spec=spec)
        self.assertEqual(float(zero["consistency_raw"]), 0.)

    def test_zero_matches_original_hard_loss_exactly(self):
        stage, _, _ = make_fixture("ZERO")
        rr = batch()
        out = stage(rr, return_maps=True)
        original = compute_loss(out["student"]["y"], out["teacher_y"], rr.gt, rr.pan,
            spec=stage.spec, scharr=stage.spatial, stats=stage.stats_bank["s000"],
            return_maps=True, loss_mask=out["loss_mask"])
        for key in ("loss", "hard_weighted", "gt_raw", "w_h", "r_q_bar"):
            torch.testing.assert_close(out[key], original[key], rtol=0, atol=0)
        self.assertIsNone(out["student_rotated"])

    def test_single_view_inference_unmasked_same_rr_fr_rule(self):
        stage, _, _ = make_fixture("ROTCONS")
        model = stage.inference_model()
        with patch.object(stage.teacher.backbone, "forward", side_effect=AssertionError("Teacher forbidden")), \
             patch.object(stage.student.backbone, "forward", wraps=stage.student.backbone.forward) as student:
            for resolution in ("RR", "FR"):
                rr = batch(side=64, resolution=resolution, split="test")
                out = model(replace(rr, gt=torch.tensor(float("nan"))))
                self.assertIs(out["pan_aligned"], rr.pan)
                self.assertEqual(int(torch.count_nonzero(out["pan_correction"])), 0)
                self.assertNotIn("loss_mask", out)
                self.assertFalse(out["y"].requires_grad)
            self.assertEqual(student.call_count, 2)

    def test_complete_resume_and_cross_policy_load_rejection(self):
        trainer = make_trainer("ROTCONS")
        first = trainer.step_accumulated([batch()])
        self.assertEqual(first["student_views"], 2)
        self.assertAlmostEqual(first["loss"], first["hard_weighted"] + first["consistency_weighted"], places=6)
        self.assertGreater(first["gradient"]["student"], 0.)
        self.assertGreater(first["gradient"]["lambda_consistency_norm"], 0.)
        self.assertIsNotNone(first["gradient"]["hard_consistency_cosine"])
        self.assertEqual(first["gradient"]["beta_soft_norm"], 0.)
        self.assertIsNone(trainer.stage._microbatch_diagnostics)
        state = deepcopy(trainer.state_dict())
        second = make_trainer("ROTCONS")
        second.load_state_dict(deepcopy(state))
        trainer.step_accumulated([batch()])
        second.step_accumulated([batch()])
        self.assertTrue(_same(trainer.state_dict(), second.state_dict()))
        wrong = make_trainer("PAIR0")
        before = state_dict_sha256(wrong.stage.student)
        with self.assertRaises(ValueError):
            wrong.load_state_dict(state)
        self.assertEqual(before, state_dict_sha256(wrong.stage.student))
        broken = deepcopy(state)
        broken["stage"]["_completed_update_state"].add_(1)
        with self.assertRaisesRegex(ValueError, "schedule state"):
            second.load_state_dict(broken)

    def test_calibration_shared_between_policies_but_wrong_geometry_rejected(self):
        stage, donor, bank = make_fixture("ZERO")
        for policy in ("PAIR0", "ROTCONS"):
            C.validate_stats_bank(bank, donor.binding, make_spec(policy), full_hash=True)
        with self.assertRaises(ValueError):
            C.validate_stats_bank({"s025": bank["s000"]}, donor.binding, stage.spec)


if __name__ == "__main__":
    unittest.main()
