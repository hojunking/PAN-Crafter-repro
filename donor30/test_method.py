"""Small synthetic CPU contracts; no real data, GPU, campaigns or network."""
from __future__ import annotations

import copy
from dataclasses import replace
from unittest.mock import patch
import unittest

import numpy as np
import torch

from directshift48.test_method import batch, model as unmasked_model
from pa.warp import warp_support_mask
from .masking import MaskedDonorStage1, masked_mean
from panda_method.calibration import canonical_sha256, state_dict_sha256
from panda_method.reconstruction import ReconstructionSpec, build_reconstructor
from .method import (DirectStage2, DirectCalibrationScales, ExactDirectCalibrator,
                     calibrate_training, calibration_binding, loss_terms, reference_maps)


torch.set_num_threads(1)


def model(sensor="GF2", **kwargs):
    return MaskedDonorStage1.from_fresh_stage(unmasked_model(sensor, **kwargs))


def setup(mode="eq", sensor="GF2"):
    donor, rr = model(sensor), batch(sensor=sensor)
    scales = calibrate_training(donor, [rr], expected_sample_ids=rr.lineage.sample_ids)
    student = build_reconstructor(ReconstructionSpec(4 if sensor == "GF2" else 8,
        8, (1, 1, 1), "P0", "BICUBIC_MS"), seed=42, role="S")
    stage = DirectStage2(donor, student, mode=mode, calibration=scales)
    return stage, donor, rr, scales


class CalibrationTests(unittest.TestCase):
    def test_exact_global_median_and_population_counts_both_sensors(self):
        for sensor in ("GF2", "WV3"):
            donor, rr = model(sensor), batch(sensor=sensor)
            with torch.no_grad():
                out = donor.native_prediction(rr)
                e, q = reference_maps(donor.spatial, out["y"], rr.gt, out["pan_aligned"])
            scales = calibrate_training(donor, [rr], expected_sample_ids=rr.lineage.sample_ids)
            mask = warp_support_mask(*rr.pan.shape[-2:], out["u"])
            self.assertAlmostEqual(scales.tau_e, float(np.median(e[mask].numpy().astype(np.float64))), places=15)
            self.assertAlmostEqual(scales.tau_q, float(np.median(q[..., 8:-8, 8:-8].numpy().astype(np.float64))), places=15)
            self.assertEqual((scales.n_e, scales.n_q, scales.n_samples), (2*29*29, 2*16*16, 2))
            self.assertEqual(scales, DirectCalibrationScales.from_dict(scales.to_dict()))
            self.assertTrue(donor.training)

    def test_calibration_rejects_duplicate_incomplete_foreign_and_test(self):
        donor, rr = model(), batch()
        with self.assertRaisesRegex(ValueError, "duplicated"):
            calibrate_training(donor, [rr, rr], expected_sample_ids=rr.lineage.sample_ids)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            calibrate_training(donor, [], expected_sample_ids=rr.lineage.sample_ids)
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            calibrate_training(donor, [rr], expected_sample_ids=("other1", "other2"))
        test = replace(rr, lineage=replace(rr.lineage, split="test"))
        with self.assertRaises(ValueError):
            calibrate_training(donor, [test], expected_sample_ids=rr.lineage.sample_ids)
        augmented = replace(rr, lineage=replace(rr.lineage, augmentations=((True, False, 0),)*2))
        with self.assertRaisesRegex(ValueError, "unaugmented"):
            calibrate_training(donor, [augmented], expected_sample_ids=rr.lineage.sample_ids)

    def test_exact_even_median_and_memory_budget(self):
        donor, rr = model(), batch()
        collector = ExactDirectCalibrator(calibration_binding(donor), donor.spatial,
            expected_sample_ids=rr.lineage.sample_ids, max_host_bytes=16)
        with self.assertRaises(MemoryError):
            collector.update(torch.zeros_like(rr.gt), rr.gt, rr.pan, split="train", sample_ids=rr.lineage.sample_ids,
                             valid_mask=warp_support_mask(32, 32, torch.zeros(2, 2)))
        self.assertEqual(collector.seen, set())
        chunks = [np.array([1., 4., 2., 3.], dtype=np.float32)]
        self.assertEqual(collector._median(chunks), (2.5, 4))
        self.assertEqual(chunks, [])

    def test_changed_donor_invalidates_calibration(self):
        donor, rr = model(), batch()
        scales = calibrate_training(donor, [rr], expected_sample_ids=rr.lineage.sample_ids)
        with torch.no_grad():
            donor.aligner.fc2.bias.add_(.01)
        student = build_reconstructor(donor.teacher.spec, seed=44, role="S")
        with self.assertRaisesRegex(ValueError, "Stale calibration"):
            DirectStage2(donor, student, calibration=scales)


class ObjectiveTests(unittest.TestCase):
    def test_exact_Scharr_fit_and_fixed_ROI_denominator(self):
        stage, donor, rr, scales = setup()
        t = torch.full_like(rr.gt, .6)
        y = torch.zeros_like(rr.gt)
        s = rr.gt.clone().requires_grad_()
        p = rr.pan
        mask = warp_support_mask(32, 32, torch.zeros(2, 2))
        out = loss_terms(stage.spatial, s, t, y, p, mode="eq", calibration=scales, valid_mask=mask)
        e, q = reference_maps(stage.spatial, t, y, p)
        we, wq = e/(e+scales.tau_e), q/(q+scales.tau_q)
        fit = masked_mean((1-we)*(s-t).abs().mean(1, keepdim=True)+we*s.abs().mean(1, keepdim=True), mask)
        edge = stage.spatial.descriptor(stage.spatial.rgb_luminance(s)).abs().mean(1, keepdim=True)
        spatial = (wq*edge)[..., 8:-8, 8:-8].mean()
        torch.testing.assert_close(out["fit"], fit)
        torch.testing.assert_close(out["spatial"], spatial)
        torch.testing.assert_close(out["loss"], fit+spatial)
        self.assertFalse(out["w_e"].requires_grad)
        self.assertFalse(out["q"].requires_grad)

    def test_uniform_control_changes_weight_only(self):
        stage, _, rr, scales = setup()
        s = rr.gt.clone().requires_grad_()
        t = rr.gt + .2
        args = (stage.spatial, s, t, rr.gt*.2, rr.pan)
        mask = warp_support_mask(32, 32, torch.zeros(2, 2))
        e = loss_terms(*args, mode="e", calibration=scales, valid_mask=mask)
        uniform = loss_terms(*args, mode="uniform", calibration=scales, valid_mask=mask)
        eq = loss_terms(*args, mode="eq", calibration=scales, valid_mask=mask)
        torch.testing.assert_close(e["fit"], uniform["fit"])
        torch.testing.assert_close(e["fit"], eq["fit"])
        self.assertEqual(float(e["spatial"]), 0.)
        torch.testing.assert_close(uniform["spatial"], uniform["edge_px"][..., 8:-8, 8:-8].mean())
        self.assertLessEqual(float(eq["spatial"]), float(uniform["spatial"]))

    def test_gt_mode_no_teacher_forward_or_calibration_requirement(self):
        donor, rr = model(), batch()
        student = build_reconstructor(donor.teacher.spec, seed=42, role="S")
        stage = DirectStage2(donor, student, mode="gt")
        with patch.object(stage.teacher, "forward", side_effect=AssertionError("Teacher called")):
            out = stage(rr)
        torch.testing.assert_close(out["loss"], masked_mean((out["student"]["y"]-rr.gt).abs(), out["loss_mask"]))

    def test_shared_frozen_A_T_student_only_gradient_and_train_mode(self):
        for mode in ("gt", "e", "eq", "uniform"):
            stage, donor, rr, _ = setup(mode)
            self.assertIs(stage.aligner, donor.aligner)
            self.assertIs(stage.teacher, donor.teacher)
            stage.train()
            self.assertFalse(stage.aligner.training)
            self.assertFalse(stage.teacher.training)
            self.assertTrue(stage.student.training)
            out = stage(rr)
            out["loss"].backward()
            self.assertTrue(all(p.grad is None for m in (stage.aligner, stage.teacher) for p in m.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in stage.student.parameters() if p.grad is not None), 0.)
            optimizer = torch.optim.AdamW(stage.student.parameters(), lr=1e-4)
            stage.assert_optimizer(optimizer)
            with self.assertRaisesRegex(ValueError, "Student"):
                stage.assert_optimizer(torch.optim.AdamW(stage.parameters(), lr=1e-4))

    def test_inference_graph_excludes_teacher_and_GT(self):
        stage, _, rr, _ = setup()
        with patch.object(stage.teacher, "forward", side_effect=AssertionError("Teacher used")):
            result = stage.native_prediction(rr)
            inference = stage.inference_model()
            self.assertFalse(hasattr(inference, "teacher"))
            obs = replace(rr, gt=None)
            deployed = inference(obs)
        torch.testing.assert_close(result["y"], deployed["y"], rtol=0, atol=0)
        self.assertFalse(result["y"].requires_grad)
        self.assertTrue(all(not name.startswith("teacher.") for name in inference.state_dict()))
        with self.assertRaisesRegex(ValueError, "no GT"):
            inference(rr)

    def test_frozen_mutation_source_support_and_checkpoint_guards(self):
        stage, _, rr, _ = setup()
        stage.to("cpu")
        state = copy.deepcopy(stage.state_dict())
        expected = stage.native_prediction(rr)["y"]
        stage.load_state_dict(state)
        torch.testing.assert_close(expected, stage.native_prediction(rr)["y"], rtol=0, atol=0)
        bad = copy.deepcopy(state)
        bad["_extra_state"]["pipeline"]["mode"] = "gt"
        with self.assertRaisesRegex(ValueError, "strict"):
            stage.load_state_dict(bad)
        foreign = replace(rr, lineage=replace(rr.lineage, source_manifest_sha256="b"*64))
        with self.assertRaises(ValueError):
            stage(foreign)
        with torch.no_grad():
            stage.aligner.fc2.bias.add_(1.)
        with self.assertRaisesRegex(ValueError, "Frozen donor"):
            stage(rr)
        donor = model()
        with torch.no_grad():
            donor.aligner.fc2.bias.fill_(20.)
        student = build_reconstructor(donor.teacher.spec, seed=42, role="S")
        unsafe = DirectStage2(donor, student, mode="gt")
        with self.assertRaisesRegex(ValueError, "SUPPORT_VIOLATION"):
            unsafe(rr)


if __name__ == "__main__":
    unittest.main()
