"""CPU synthetic tests only; never open train/test datasets or launch runs."""

import dataclasses
import unittest

import torch
from torch import nn

from panda_method.calibration import (
    CalibrationBinding, CalibrationScales, ExactTrainCalibrator,
    canonical_sha256, state_dict_sha256,
)
from panda_method.losses import (
    AbsoluteGradientLoss, SpatialLossSpec, reliability_weight, teacher_error, sipsa_spatial_spec,
)


def spatial(*, operator="central_difference", margin=1, padding="reflect", rgb=(2, 1, 0)):
    return AbsoluteGradientLoss(SpatialLossSpec(
        sensor="synthetic", rgb_indices=rgb, operator=operator, padding=padding,
        roi_margin=margin, radiometry="synthetic_float_same_units"))


def binding(loss):
    return CalibrationBinding.for_modules(
        nn.Linear(2, 2), nn.Linear(2, 3), pipeline={"native_lms": True, "base": "unchanged"},
        source_manifest_sha256=canonical_sha256({"split": "train"}),
        prior_mapping_sha256=canonical_sha256({"mapping": "synthetic"}), spatial=loss)


class TestSpatialLosses(unittest.TestCase):
    def test_configuration_has_no_implicit_policy(self):
        with self.assertRaises(TypeError):
            SpatialLossSpec(sensor="WV3", rgb_indices=(4, 2, 1))
        for change in ({"operator": "unknown"}, {"padding": "implicit"}, {"roi_margin": -1},
                       {"rgb_indices": (1, 1, 0)}, {"radiometry": ""}):
            with self.assertRaises(ValueError):
                dataclasses.replace(spatial().spec, **change)

    def test_rgb_uses_only_explicit_three_bands_and_arithmetic_mean(self):
        image = torch.arange(8.).view(1, 8, 1, 1).expand(1, 8, 4, 4)
        loss = spatial(rgb=(4, 2, 1))
        torch.testing.assert_close(loss.rgb_luminance(image), torch.full((1, 1, 4, 4), 7./3))
        with self.assertRaises(ValueError):
            loss.rgb_luminance(image[:, :4])

    def test_ramp_and_per_axis_not_l2_magnitude(self):
        yy, xx = torch.meshgrid(torch.arange(7.), torch.arange(9.), indexing="ij")
        image = (2 * yy + 3 * xx)[None, None]
        for operator in ("central_difference", "sobel_1_8", "forward_difference"):
            loss = spatial(operator=operator)
            desc = loss.descriptor(image)
            torch.testing.assert_close(desc[:, 0, 1:-1, 1:-1], torch.full((1, 5, 7), 2.))
            torch.testing.assert_close(desc[:, 1, 1:-1, 1:-1], torch.full((1, 5, 7), 3.))
            residual = loss.discrepancy(image, torch.zeros_like(image))
            self.assertAlmostEqual(loss.reduce(residual).item(), 2.5)
            torch.testing.assert_close(loss.descriptor(-image), desc)

    def test_canonical_sipsa_descriptor_matches_independent_difference_and_reflect_border(self):
        loss = AbsoluteGradientLoss(sipsa_spatial_spec(
            sensor="synthetic", rgb_indices=(2, 1, 0), roi_margin=0,
            radiometry="synthetic_float_same_units"))
        image = torch.tensor([[1., 4., -2., 5.], [7., 3., 9., -1.], [2., 8., 6., -3.]])[None, None]
        # Original SIPSA signed kernel is current minus next pixel. At the
        # final row/column, REFLECT uses the preceding row/column, not border.
        next_y = torch.cat((image[..., 1:, :], image[..., -2:-1, :]), dim=-2)
        next_x = torch.cat((image[..., 1:], image[..., -2:-1]), dim=-1)
        expected = torch.cat(((image - next_y).abs(), (image - next_x).abs()), dim=1)
        torch.testing.assert_close(loss.descriptor(image), expected, rtol=0, atol=0)
        self.assertEqual(loss.spec.operator, "forward_difference")
        self.assertEqual(loss.spec.padding, "reflect")

    def test_descriptor_gradient_and_reference_detach(self):
        torch.manual_seed(70)
        pan = torch.randn(2, 1, 6, 7, requires_grad=True)
        gt = torch.randn(2, 4, 6, 7, requires_grad=True)
        loss = spatial()
        objective = loss.stage1(pan, gt)
        objective.backward()
        self.assertGreater(pan.grad.abs().sum().item(), 0.)
        self.assertIsNone(gt.grad)
        self.assertEqual(list(loss.parameters()), [])

    def test_empty_roi_and_nonfinite_are_rejected(self):
        with self.assertRaises(ValueError):
            spatial(margin=2).reduce(torch.zeros(1, 1, 4, 4))
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                spatial().descriptor(torch.full((1, 1, 4, 4), value))

    def test_fixed_roi_excludes_border_but_is_not_weight_sum_normalized(self):
        loss = spatial(margin=1)
        values = torch.full((2, 1, 4, 6), 10.)
        values[..., 1:-1, 1:-1] = .25
        self.assertEqual(loss.pixel_mask(values).sum().item(), 16)
        self.assertEqual(loss.reduce(values).item(), .25)

    def test_teacher_error_is_spectral_mean_and_detached(self):
        teacher = torch.tensor([1., -3., 2., -2.]).view(1, 4, 1, 1).requires_grad_()
        gt = torch.zeros_like(teacher, requires_grad=True)
        e = teacher_error(teacher, gt)
        self.assertEqual(e.shape, (1, 1, 1, 1))
        self.assertEqual(e.item(), 2.)
        self.assertFalse(e.requires_grad)

    def test_blending_known_values_monotonicity_and_zero_median(self):
        x = torch.tensor([0., .5, 1., 2., 100.]).reshape(1, 1, 1, 5).requires_grad_()
        weight = reliability_weight(x, 1.)
        torch.testing.assert_close(weight.flatten(), torch.tensor([0., 1/3, .5, 2/3, 100/101]))
        self.assertTrue(bool((weight[..., 1:] >= weight[..., :-1]).all()))
        self.assertFalse(weight.requires_grad)
        torch.testing.assert_close(reliability_weight(x, 0.).flatten(), torch.tensor([0., 1., 1., 1., 1.]))
        for tau in (-1., float("inf"), float("nan"), torch.tensor([1., 2.])):
            with self.assertRaises(ValueError):
                reliability_weight(x, tau)

    def test_stage2_exact_formula_and_only_student_gradient(self):
        torch.manual_seed(80)
        student = torch.randn(2, 4, 7, 8, requires_grad=True)
        teacher = torch.randn_like(student, requires_grad=True)
        gt = torch.randn_like(student, requires_grad=True)
        pan = torch.randn(2, 1, 7, 8, requires_grad=True)
        loss = spatial()
        result = loss.stage2(student, teacher, gt, pan, .7, .9)
        expected_fit = ((1 - result["w_e"]) * result["soft_px"] + result["w_e"] * result["hard_px"]).mean()
        expected_spatial = (result["w_q"] * result["edge_px"])[..., 1:-1, 1:-1].mean()
        torch.testing.assert_close(result["loss"], expected_fit + expected_spatial)
        self.assertEqual(result["e"].shape, (2, 1, 7, 8))
        torch.testing.assert_close(result["q"], loss.discrepancy(pan, loss.rgb_luminance(gt)))
        for name in ("e", "q", "w_e", "w_q"):
            self.assertFalse(result[name].requires_grad)
        result["loss"].backward()
        self.assertGreater(student.grad.abs().sum().item(), 0.)
        for fixed in (teacher, gt, pan):
            self.assertIsNone(fixed.grad)

    def test_stage2_fit_uses_full_pixels_while_spatial_has_declared_roi(self):
        gt = torch.zeros(1, 4, 5, 5)
        teacher = torch.ones_like(gt)
        student = torch.zeros_like(gt)
        student[..., 0, :] = 4.
        pan = torch.zeros(1, 1, 5, 5)
        result = spatial(margin=2).stage2(student, teacher, gt, pan, 1., 1.)
        expected = .5 * (student - teacher).abs().mean() + .5 * student.abs().mean()
        torch.testing.assert_close(result["fit"], expected)
        self.assertEqual(result["spatial"].item(), 0.)

    def test_routing_is_independent_of_student(self):
        torch.manual_seed(50)
        gt, teacher = torch.randn(1, 4, 5, 5), torch.randn(1, 4, 5, 5)
        pan = torch.randn(1, 1, 5, 5)
        loss = spatial()
        first = loss.stage2(gt, teacher, gt, pan, .3, .4)
        second = loss.stage2(teacher * 5., teacher, gt, pan, .3, .4)
        for name in ("e", "q", "w_e", "w_q"):
            torch.testing.assert_close(first[name], second[name])

    def test_autocast_does_not_change_calibrated_gradient_dtype(self):
        torch.manual_seed(22)
        x = torch.randn(1, 1, 6, 6)
        loss = spatial()
        expected = loss.descriptor(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = loss.descriptor(x)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(loss.descriptor(x.double()).dtype, torch.float64)


class TestExactTrainCalibration(unittest.TestCase):
    def test_global_population_not_median_of_batch_medians(self):
        loss = spatial(margin=0)
        b = binding(loss)
        calibrator = ExactTrainCalibrator(b, loss, expected_sample_ids=["zero0", "zero1", "ten0", "ten1", "ten2"])
        for count, value, prefix in ((2, 0., "zero"), (3, 10., "ten")):
            gt = torch.zeros(count, 4, 2, 2)
            calibrator.update(gt + value, gt, torch.zeros(count, 1, 2, 2),
                              split="train", sample_ids=[f"{prefix}{i}" for i in range(count)])
        scales = calibrator.finalize()
        self.assertEqual(scales.tau_e, 10.)  # median([batch medians]) would incorrectly be 5.
        self.assertEqual(scales.tau_q, 0.)
        self.assertEqual((scales.n_e, scales.n_q, scales.n_samples), (20, 20, 5))
        self.assertEqual(CalibrationScales.from_dict(scales.to_dict()), scales)
        scales.assert_binding(b)
        with self.assertRaises(RuntimeError):
            calibrator.finalize()

    def test_even_median_midpoint_and_q_roi_pixel_population(self):
        loss = spatial(margin=1)
        cal = ExactTrainCalibrator(binding(loss), loss, expected_sample_ids=["one"])
        gt = torch.zeros(1, 4, 4, 4)
        teacher = torch.arange(16.).reshape(1, 1, 4, 4).expand_as(gt)
        pan = (torch.arange(16.).reshape(1, 1, 4, 4) ** 2)
        q = loss.discrepancy(pan, torch.zeros_like(pan))
        cal.update(teacher, gt, pan, split="train", sample_ids=["one"])
        result = cal.finalize()
        self.assertEqual(result.tau_e, 7.5)
        self.assertEqual(result.tau_q, torch.quantile(q[..., 1:-1, 1:-1].double().flatten(), .5).item())
        self.assertEqual((result.n_e, result.n_q), (16, 4))

    def test_split_duplicate_empty_and_policy_mismatch_guards(self):
        loss = spatial()
        cal = ExactTrainCalibrator(binding(loss), loss, expected_sample_ids=["one"])
        gt, pan = torch.zeros(1, 4, 4, 4), torch.zeros(1, 1, 4, 4)
        with self.assertRaises(ValueError):
            cal.finalize()
        for split in ("val", "test", "fr", ""):
            with self.assertRaises(ValueError):
                cal.update(gt, gt, pan, split=split, sample_ids=["one"])
        cal.update(gt, gt, pan, split="train", sample_ids=["one"])
        with self.assertRaises(ValueError):
            cal.update(gt, gt, pan, split="train", sample_ids=["one"])
        with self.assertRaises(ValueError):
            ExactTrainCalibrator(binding(loss), spatial(margin=0), expected_sample_ids=["one"])

    def test_full_population_required_and_unregistered_samples_rejected(self):
        loss = spatial()
        cal = ExactTrainCalibrator(binding(loss), loss, expected_sample_ids=["one", "two"])
        gt, pan = torch.zeros(1, 4, 4, 4), torch.zeros(1, 1, 4, 4)
        with self.assertRaises(ValueError):
            cal.update(gt, gt, pan, split="train", sample_ids=["outside"])
        cal.update(gt, gt, pan, split="train", sample_ids=["one"])
        with self.assertRaisesRegex(ValueError, "1 samples missing"):
            cal.finalize()
        cal.update(gt, gt, pan, split="train", sample_ids=["two"])
        self.assertEqual(cal.finalize().n_samples, 2)
        for bad in ([], ["one", "one"], "one"):
            with self.assertRaises(ValueError):
                ExactTrainCalibrator(binding(loss), loss, expected_sample_ids=bad)

    def test_checkpoint_and_pipeline_hashes_invalidate_calibration(self):
        loss = spatial()
        b = binding(loss)
        scales = CalibrationScales(1., 2., 100, 64, 1, b, canonical_sha256(["one"]))
        for field in dataclasses.fields(CalibrationBinding):
            changed = dataclasses.replace(b, **{field.name: canonical_sha256("changed")})
            with self.assertRaisesRegex(ValueError, field.name):
                scales.assert_binding(changed)

    def test_hashes_are_state_not_container_dependent(self):
        module = nn.Linear(2, 3)
        state = module.state_dict()
        self.assertEqual(state_dict_sha256(module), state_dict_sha256(dict(reversed(list(state.items())))))
        before = state_dict_sha256(module)
        with torch.no_grad():
            module.weight[0, 0] += 1.
        self.assertNotEqual(before, state_dict_sha256(module))
        # Includes dtype and shape; BF16 bytes are supported without numpy BF16.
        self.assertNotEqual(state_dict_sha256({"x": torch.ones(2)}),
                            state_dict_sha256({"x": torch.ones(2, 1)}))
        self.assertEqual(len(state_dict_sha256({"x": torch.ones(2, dtype=torch.bfloat16)})), 64)


if __name__ == "__main__":
    unittest.main()
