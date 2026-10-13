"""Synthetic edge contracts only; numeric choices here are NOT sensor policy."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from pa.warp import warp_pan, warp_support_mask
from panda_method.edge_match import EdgeMatcher, EdgeMatchSpec, ScharrEdges
from panda_method.geometry import transform_prior


def spec(**changes):
    settings = dict(sensor="QB", radius_rr=3, radius_fr=4,
                    tie_policy="min_norm_then_lexicographic",
                    low_texture_threshold=0., low_texture_policy="error",
                    boundary_policy="report")
    settings.update(changes)
    return EdgeMatchSpec(**settings)


def reference(dtype=torch.float32, height=32, width=36):
    rng = np.random.RandomState(137)
    raw = rng.normal(size=(height, width)).astype(np.float64)
    smooth = cv2.GaussianBlur(raw, (3, 3), .6)
    return torch.tensor(smooth, dtype=dtype)[None, None]


def lms_from(ref, sensor="QB"):
    return ref.repeat(1, 8 if sensor == "WV3" else 4, 1, 1)


def transform(image, hflip=False, vflip=False, rot90=0):
    if hflip:
        image = image.flip(-1)
    if vflip:
        image = image.flip(-2)
    return torch.rot90(image, rot90, (-2, -1))


class ScharrEdgesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_fixed_kernel_scale_and_axes_on_affine_ramp(self):
        yy, xx = torch.meshgrid(torch.arange(12.), torch.arange(14.), indexing="ij")
        gray = (2 * xx + 3 * yy)[None, None]
        result = ScharrEdges("QB").descriptor(gray)
        self.assertEqual(result.shape, (1, 2, 12, 14))
        torch.testing.assert_close(result[:, 0, 1:-1, 1:-1], torch.full((1, 10, 12), 2.))
        torch.testing.assert_close(result[:, 1, 1:-1, 1:-1], torch.full((1, 10, 12), 3.))

    def test_cv2_scharr_scale_reflect101_agrees_both_axes(self):
        gray = reference(torch.float64)
        actual = ScharrEdges("QB").descriptor(gray)[0].numpy()
        expected = np.stack([np.abs(cv2.Scharr(gray[0, 0].numpy(), cv2.CV_64F, 1, 0, scale=1 / 32)),
                             np.abs(cv2.Scharr(gray[0, 0].numpy(), cv2.CV_64F, 0, 1, scale=1 / 32))])
        np.testing.assert_allclose(actual, expected, atol=1e-15)

    def test_sensor_rgb_mapping_not_first_three_wv3(self):
        for sensor, count, rgb in (("WV3", 8, (4, 2, 1)), ("QB", 4, (2, 1, 0)), ("GF2", 4, (2, 1, 0))):
            edges = ScharrEdges(sensor)
            ms = torch.arange(float(count)).reshape(1, count, 1, 1).expand(1, count, 4, 5)
            expected = torch.full((1, 1, 4, 5), sum(rgb) / 3)
            torch.testing.assert_close(edges.rgb_luminance(ms), expected)
            self.assertEqual(edges.rgb_indices, rgb)

    def test_rgb_average_precedes_absolute_edges(self):
        gray = reference()
        ms = torch.zeros((1, 8, *gray.shape[-2:]))
        ms[:, 4] = gray[:, 0]
        ms[:, 2] = -gray[:, 0]
        ms[:, 0] = gray[:, 0] * 999  # coastal is irrelevant to RGB.
        edges = ScharrEdges("WV3")
        result = edges.descriptor(edges.rgb_luminance(ms))
        self.assertEqual(float(result.abs().max()), 0.)
        self.assertGreater(float(edges.descriptor(gray).mean()), 0.)

    def test_no_adaptive_edge_normalization(self):
        gray = reference()
        edges = ScharrEdges("QB")
        torch.testing.assert_close(edges.descriptor(gray * 7), edges.descriptor(gray) * 7)

    def test_fp32_fp64_preserved_even_inside_autocast(self):
        edges = ScharrEdges("QB")
        with torch.autocast("cpu", dtype=torch.bfloat16):
            for dtype in (torch.float32, torch.float64):
                gray = reference(dtype)
                self.assertEqual(edges.descriptor(gray).dtype, dtype)
        for dtype in (torch.float16, torch.bfloat16, torch.int64):
            with self.assertRaisesRegex(ValueError, "FP32/FP64"):
                edges.descriptor(reference().to(dtype))

    def test_absolute_loss_shares_descriptor_roi_and_gradient(self):
        pan = reference().requires_grad_()
        gt = lms_from(torch.roll(pan.detach(), 1, -1))
        edges = ScharrEdges("QB")
        loss = edges.absolute_loss(pan, gt, roi_margin=3)
        expected = (edges.descriptor(pan) - edges.descriptor(edges.rgb_luminance(gt))).abs()[..., 3:-3, 3:-3].mean()
        torch.testing.assert_close(loss, expected)
        gradient = torch.autograd.grad(loss, pan)[0]
        self.assertGreater(float(gradient.abs().sum()), 0.)
        self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_roi_and_invalid_shape_finite_checks(self):
        edges = ScharrEdges("QB")
        gray = reference()
        for margin in (-1, True, 16, .5):
            with self.assertRaises(ValueError):
                edges.absolute_loss(gray, lms_from(gray), roi_margin=margin)
        with self.assertRaises(TypeError):
            edges.absolute_loss(gray, lms_from(gray))
        with self.assertRaises(ValueError):
            edges.descriptor(torch.full_like(gray, float("nan")))
        with self.assertRaises(ValueError):
            edges.absolute_loss(gray, lms_from(gray).double(), roi_margin=1)

    def test_fixed_operator_has_no_mutable_checkpoint_kernel(self):
        edges = ScharrEdges("WV3")
        self.assertEqual(list(edges.parameters()), [])
        self.assertEqual(list(edges.state_dict()), [])
        self.assertEqual(edges.fingerprint(), ScharrEdges("WV3").fingerprint())
        self.assertNotEqual(edges.fingerprint(), ScharrEdges("QB").fingerprint())


class EdgeMatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_signed_integer_translation_matches_actual_warp(self):
        ref = reference()
        matcher = EdgeMatcher(spec())
        for dy, dx in ((0, 0), (1, 0), (-1, 0), (0, 2), (0, -2), (1, -2), (-2, 1)):
            with self.subTest(shift=(dy, dx)):
                pan = torch.roll(ref, (dy, dx), (-2, -1))
                out = matcher(pan, lms_from(ref), resolution="RR")
                torch.testing.assert_close(out.b, torch.tensor([[float(dy), float(dx)]]))
                corrected = warp_pan(pan, out.b)
                torch.testing.assert_close(corrected[..., 5:-5, 5:-5], ref[..., 5:-5, 5:-5], atol=2e-6, rtol=1e-5)
                self.assertEqual(int(out.tie_count[0]), 1)

    def test_unique_minimum_transforms_with_all_d4_operations(self):
        ref = reference(height=36, width=36)
        pan = torch.roll(ref, (1, -2), (-2, -1))
        matcher = EdgeMatcher(spec())
        canonical = torch.tensor([[1., -2.]])
        for hflip in (False, True):
            for rotation in range(4):
                kwargs = dict(hflip=hflip, vflip=False, rot90=rotation)
                output = matcher(transform(pan, **kwargs), lms_from(transform(ref, **kwargs)), resolution="RR")
                expected = transform_prior(canonical, **kwargs)
                torch.testing.assert_close(output.b, expected)
                self.assertEqual(int(output.tie_count[0]), 1)

    def test_scores_use_one_common_interior_and_two_axes(self):
        ref = reference()
        pan = torch.roll(ref, (1, -1), (-2, -1))
        edges = ScharrEdges("QB")
        matcher = EdgeMatcher(spec(radius_rr=2), edges)
        output = matcher(pan, lms_from(ref), resolution="RR")
        self.assertIs(matcher.edges, edges)
        ep, em = edges.descriptor(pan), edges.descriptor(ref)
        margin = 3
        self.assertEqual(output.roi_margin, margin)
        for index, (dy, dx) in enumerate(output.candidates.tolist()):
            expected = (ep[..., margin + dy:pan.shape[-2] - margin + dy,
                           margin + dx:pan.shape[-1] - margin + dx] - em[..., margin:-margin, margin:-margin]).abs().mean()
            torch.testing.assert_close(output.scores[0, index], expected)
        expected_count = (pan.shape[-2] - 6) * (pan.shape[-1] - 6)
        self.assertEqual(int(output.support_count[0]), expected_count)
        self.assertTrue(bool((output.diagnostics["candidate_support_count"] == expected_count).all()))

    def test_no_gradient_or_gt_input_to_matcher(self):
        ref = reference().requires_grad_()
        pan = torch.roll(ref, 1, -2)
        out = EdgeMatcher(spec())(pan, lms_from(ref), resolution="RR")
        self.assertFalse(out.b.requires_grad)
        self.assertFalse(out.scores.requires_grad)
        self.assertIsNone(out.b.grad_fn)
        with self.assertRaises(TypeError):
            EdgeMatcher(spec())(pan, lms_from(ref), resolution="RR", gt=lms_from(ref))

    def test_fractional_shift_is_not_fitted_by_matcher(self):
        ref = reference()
        pan = warp_pan(ref, torch.tensor([[-1.25, .2]]))
        out = EdgeMatcher(spec())(pan, lms_from(ref), resolution="RR")
        torch.testing.assert_close(out.b, torch.tensor([[1., 0.]]))
        # Known quantization remainder is left to a continuous residual CNN.
        self.assertGreater(float((out.b - torch.tensor([[1.25, -.2]])).abs().sum()), .4)

    def test_rr_fr_current_grid_radii_not_x4_output(self):
        ref = reference()
        pan = torch.roll(ref, (3, 0), (-2, -1))
        matcher = EdgeMatcher(spec(radius_rr=1, radius_fr=4))
        fr = matcher(pan, lms_from(ref), resolution="FR")
        torch.testing.assert_close(fr.b, torch.tensor([[3., 0.]]))
        self.assertEqual(fr.roi_margin, 5)
        self.assertEqual(fr.scores.shape[1], 81)
        rr = matcher(pan, lms_from(ref), resolution="RR")
        self.assertEqual(rr.scores.shape[1], 9)
        self.assertTrue(bool((rr.b.abs() <= 1).all()))
        with self.assertRaises(ValueError):
            matcher(pan, lms_from(ref), resolution="auto")
        with self.assertRaises(TypeError):
            matcher(pan, lms_from(ref))

    def test_input_specific_batched_outputs_not_one_global_prior(self):
        ref = reference()
        pans = torch.cat([torch.roll(ref, (1, 0), (-2, -1)), torch.roll(ref, (-1, 2), (-2, -1))])
        lms = lms_from(ref).repeat(2, 1, 1, 1)
        result = EdgeMatcher(spec())(pans, lms, resolution="RR")
        torch.testing.assert_close(result.b, torch.tensor([[1., 0.], [-1., 2.]]))

    def test_low_texture_error_or_explicit_zero_is_visible(self):
        pan = torch.zeros((1, 1, 16, 16))
        lms = lms_from(pan)
        with self.assertRaisesRegex(ValueError, "Low-texture"):
            EdgeMatcher(spec())(pan, lms, resolution="RR")
        out = EdgeMatcher(spec(low_texture_policy="zero", tie_policy="error"))(pan, lms, resolution="RR")
        torch.testing.assert_close(out.b, torch.zeros((1, 2)))
        self.assertTrue(bool(out.low_texture[0]))
        self.assertEqual(int(out.tie_count[0]), 49)
        self.assertFalse(bool(out.boundary_hit[0]))

    def test_exact_tie_policy_error_or_min_norm_lexicographic(self):
        # Constant nonzero gradients create ties while not being low texture.
        xx = torch.arange(24.).reshape(1, 1, 1, 24).expand(1, 1, 24, 24)
        with self.assertRaisesRegex(ValueError, "Nonunique"):
            EdgeMatcher(spec(tie_policy="error"))(xx, lms_from(xx), resolution="RR")
        result = EdgeMatcher(spec())(xx, lms_from(xx), resolution="RR")
        self.assertGreater(int(result.tie_count[0]), 1)
        torch.testing.assert_close(result.b, torch.zeros((1, 2)))
        self.assertFalse(bool(result.low_texture[0]))

    def test_boundary_report_or_error(self):
        ref = reference()
        pan = torch.roll(ref, (3, 0), (-2, -1))
        result = EdgeMatcher(spec())(pan, lms_from(ref), resolution="RR")
        self.assertTrue(bool(result.boundary_hit[0]))
        with self.assertRaisesRegex(ValueError, "boundary"):
            EdgeMatcher(spec(boundary_policy="error"))(pan, lms_from(ref), resolution="RR")

    def test_radius_zero_is_explicit_no_search(self):
        ref = reference()
        result = EdgeMatcher(spec(radius_rr=0, boundary_policy="error"))(ref, lms_from(ref), resolution="RR")
        torch.testing.assert_close(result.b, torch.zeros((1, 2)))
        self.assertEqual(result.scores.shape, (1, 1))
        self.assertFalse(bool(result.boundary_hit.any()))

    def test_fp32_fp64_and_autocast_match_integer_result(self):
        matcher = EdgeMatcher(spec())
        for dtype in (torch.float32, torch.float64):
            ref = reference(dtype)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                result = matcher(torch.roll(ref, 1, -1), lms_from(ref), resolution="RR")
            self.assertEqual(result.b.dtype, dtype)
            self.assertEqual(result.scores.dtype, dtype)
            torch.testing.assert_close(result.b, torch.tensor([[0., 1.]], dtype=dtype))

    def test_specs_shapes_invalid_and_empty_support_fail_explicitly(self):
        for settings in ({"radius_rr": -1}, {"radius_fr": True}, {"low_texture_threshold": float("nan")},
                         {"low_texture_threshold": -1}, {"tie_policy": "first"},
                         {"boundary_policy": "ignore"}, {"low_texture_policy": "silent"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                spec(**settings)
        with self.assertRaises(TypeError):
            EdgeMatchSpec(sensor="QB")
        with self.assertRaises(ValueError):
            EdgeMatcher(spec(), ScharrEdges("WV3"))
        ref = reference()
        with self.assertRaisesRegex(ValueError, "No valid"):
            EdgeMatcher(spec(radius_rr=16))(ref, lms_from(ref), resolution="RR")
        with self.assertRaises(ValueError):
            EdgeMatcher(spec())(ref, lms_from(ref).double(), resolution="RR")

    def test_manifest_settings_identity_and_no_hidden_defaults(self):
        first = EdgeMatcher(spec())
        self.assertEqual(first.fingerprint(), EdgeMatcher(spec()).fingerprint())
        self.assertNotEqual(first.fingerprint(), EdgeMatcher(spec(radius_fr=5)).fingerprint())
        self.assertEqual(first.manifest()["shared_edges"], first.edges.manifest())
        self.assertIn("not_D4_equivariant", first.spec.manifest()["tie_symmetry_note"])


class KnownSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_all_valid_mask_matches_unmasked_common_interior(self):
        ref = reference()
        pan = torch.roll(ref, 1, -1)
        matcher = EdgeMatcher(spec())
        unmasked = matcher(pan, lms_from(ref), resolution="RR")
        masked = matcher(pan, lms_from(ref), resolution="RR", pan_valid_mask=torch.ones_like(pan, dtype=torch.bool))
        torch.testing.assert_close(unmasked.scores, masked.scores)
        torch.testing.assert_close(unmasked.b, masked.b)
        self.assertTrue(bool(masked.common_mask.all()))

    def test_large_invalid_border_values_never_change_scores(self):
        ref = reference(height=40, width=44)
        pan = torch.roll(ref, (1, -1), (-2, -1))
        valid = torch.ones_like(pan, dtype=torch.bool)
        valid[..., :7, :] = False
        valid[..., -6:, :] = False
        valid[..., :, :5] = False
        valid[..., :, -4:] = False
        matcher = EdgeMatcher(spec())
        before = matcher(pan, lms_from(ref), resolution="RR", pan_valid_mask=valid)
        contaminated = pan.clone()
        contaminated[~valid] = 1e6
        after = matcher(contaminated, lms_from(ref), resolution="RR", pan_valid_mask=valid)
        torch.testing.assert_close(before.scores, after.scores, atol=0, rtol=0)
        torch.testing.assert_close(before.b, after.b)
        self.assertFalse(bool(before.common_mask.all()))
        self.assertTrue(bool((before.diagnostics["candidate_support_count"] == before.support_count[:, None]).all()))

    def test_interior_hole_scharr_erosion_and_all_candidate_intersection(self):
        ref = reference(height=24, width=24)
        valid = torch.ones_like(ref, dtype=torch.bool)
        valid[..., 12, 12] = False
        matcher = EdgeMatcher(spec(radius_rr=1))
        result = matcher(ref, lms_from(ref), resolution="RR", pan_valid_mask=valid)
        # Scharr expands the hole by1; ±1 candidate coverage expands again.
        full = torch.ones_like(valid)
        full[..., 10:15, 10:15] = False
        torch.testing.assert_close(result.common_mask, full[..., 2:-2, 2:-2])
        self.assertEqual(int(result.support_count[0]), 20 * 20 - 25)

    def test_batch_support_counts_can_differ_but_not_across_candidates(self):
        ref = reference()
        pan = ref.repeat(2, 1, 1, 1)
        lms = lms_from(ref).repeat(2, 1, 1, 1)
        valid = torch.ones_like(pan, dtype=torch.bool)
        valid[1, :, :6] = False
        result = EdgeMatcher(spec())(pan, lms, resolution="RR", pan_valid_mask=valid)
        self.assertGreater(int(result.support_count[0]), int(result.support_count[1]))
        per_candidate = result.diagnostics["candidate_support_count"]
        self.assertTrue(bool((per_candidate == result.support_count[:, None]).all()))

    def test_invalid_mask_contract_or_empty_support_fails(self):
        ref = reference()
        matcher = EdgeMatcher(spec())
        masks = (torch.ones_like(ref), torch.ones((1, 32, 36), dtype=torch.bool), torch.zeros_like(ref, dtype=torch.bool))
        for mask in masks:
            with self.assertRaises(ValueError):
                matcher(ref, lms_from(ref), resolution="RR", pan_valid_mask=mask)


class CampaignV2Tests(unittest.TestCase):
    def test_fixed_rplus5_identical_native_epsilon_support_all_offsets(self):
        ref = reference(height=64, width=64)
        matching = EdgeMatcher(EdgeMatchSpec.campaign_v2("QB"))
        native = matching(ref, lms_from(ref), resolution="RR")
        self.assertEqual(native.roi_margin, 9)
        self.assertEqual(int(native.support_count[0]), 46*46)
        for eps in ((0., 0.), (2., 0.), (-2., 0.), (0., 2.), (0., -2.),
                    (1.4142, -1.4142), (-.37, .91), (.001, -.001)):
            offset = torch.tensor([eps])
            shifted = matching(warp_pan(ref, offset), lms_from(ref), resolution="RR",
                               pan_valid_mask=warp_support_mask(64, 64, offset))
            torch.testing.assert_close(shifted.common_mask, native.common_mask, rtol=0, atol=0)
            torch.testing.assert_close(shifted.support_count, native.support_count, rtol=0, atol=0)
            self.assertTrue(bool(shifted.common_mask.all()))
        for resolution in ("RR", "FR"):
            result = matching(ref, lms_from(ref), resolution=resolution)
            self.assertEqual(result.roi_margin, 9)

    def test_known_mask_is_guarantee_check_never_shrinks_score_support(self):
        ref = reference(height=64, width=64)
        valid = torch.ones_like(ref, dtype=torch.bool)
        valid[..., 25, 25] = False
        matching = EdgeMatcher(EdgeMatchSpec.campaign_v2("QB"))
        with self.assertRaisesRegex(ValueError, "FAILED_INPUT_OR_SUPPORT"):
            matching(ref, lms_from(ref), resolution="RR", pan_valid_mask=valid)

    def test_low_texture_label_does_not_override_nonzero_argmin(self):
        ref = reference(height=64, width=64)
        pan = torch.roll(ref, (2, -1), (-2, -1))
        matching = EdgeMatcher(EdgeMatchSpec.campaign_v2("QB", low_texture_threshold=1e9))
        result = matching(pan, lms_from(ref), resolution="RR")
        self.assertTrue(bool(result.low_texture.all()))
        torch.testing.assert_close(result.b, torch.tensor([[2., -1.]]), rtol=0, atol=0)
        d = result.diagnostics
        self.assertGreater(float(d['zero_score'][0]), float(d['best_score'][0]))
        self.assertGreater(float(d['second_best_gap'][0]), 0.)
        self.assertGreater(float(d['score_range'][0]), 0.)

    def test_tie_l1_norm_precedes_lexicographic_dy(self):
        # Two exact descriptor matches have squared norm25: (-3,4) has L1=7,
        # (0,5) has L1=5. The old lexicographic-only rule would pick (-3,4).
        p = torch.zeros(1, 1, 22, 22)
        lms = lms_from(p)
        ep = torch.full((1, 2, 22, 22), 100.)
        em = torch.zeros_like(ep)
        target = torch.tensor([[[[1., 3.], [2., 7.]], [[4., 6.], [8., 9.]]]])
        em[..., 10:12, 10:12] = target
        ep[..., 7:9, 14:16] = target
        ep[..., 10:12, 15:17] = target
        matching = EdgeMatcher(EdgeMatchSpec.campaign_v2("QB", radius=5))
        with patch.object(matching.edges, "descriptor", side_effect=[ep, em]):
            result = matching(p, lms, resolution="RR")
        torch.testing.assert_close(result.b, torch.tensor([[0., 5.]]), rtol=0, atol=0)
        self.assertEqual(int(result.tie_count[0]), 2)
        self.assertEqual(float(result.diagnostics['second_best_gap'][0]), 0.)

    def test_v2_manifest_and_conflicting_policy_rejected(self):
        matching = EdgeMatchSpec.campaign_v2("WV3")
        self.assertEqual((matching.radius_rr, matching.radius_fr), (4, 4))
        self.assertIn('radius+5', matching.manifest()['common_interior_margin'])
        for change in ({'low_texture_policy': 'zero'}, {'boundary_policy': 'error'},
                       {'tie_policy': 'min_norm_then_lexicographic'}):
            with self.assertRaises(ValueError):
                replace(matching, **change)


if __name__ == "__main__":
    unittest.main()
