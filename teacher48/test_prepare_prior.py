"""Synthetic prior proposal tests; numerical policies below are test-only."""
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from teacher48.prepare_prior import (audited_hr_local_gates, propose_scene_prior,
                                    validate_prior_config)


def config():
    # Deliberately explicit synthetic-data policy, not a dataset recommendation.
    return dict(search_int=4, max_magnitude=4.0, min_peak_margin=0.005,
                max_primary_secondary_diff=0.3, edge_top_frac=0.3,
                window_size=32, stride=32, min_reference_std=0.001,
                min_pan_std=0.001, min_peak_zncc=0.3, min_valid_windows=3,
                min_valid_fraction=0.2, consensus_tolerance=0.3,
                min_inlier_fraction=0.6, min_direction_concentration=0.7,
                max_dispersion=0.2, min_confidence=0.1, min_prior_magnitude=0.05,
                rationale="Synthetic unit-test thresholds only; no real-data approval.")


def image():
    raw = np.random.RandomState(19).normal(size=(96, 96)).astype(np.float32)
    return cv2.GaussianBlur(raw, (3, 3), 0.5)


def propose(pan, lms, **overrides):
    args = dict(source_scene_id="synthetic-acquisition-19",
                prior_grid_name="synthetic-native-pan", pixel_scale=[1., 1.],
                origin_phase=[0., 0.], config=config(),
                provenance={"source": "synthetic unit test; not observed satellite data"})
    args.update(overrides)
    return propose_scene_prior(pan, lms, **args)


def raw_result(vector=(1., 0.), **overrides):
    dy, dx = vector
    result = dict(dy_lr_raw=dy, dx_lr_raw=dx, magnitude_raw=float(np.hypot(dy, dx)),
                  peak_zncc=0.95, peak_margin=0.3, secondary_dy=dy, secondary_dx=dx,
                  primary_secondary_diff=0., boundary_hit=False, accepted=True,
                  dy_lr_applied=dy, dx_lr_applied=dx)
    result.update(overrides)
    return result


class PriorProposalTests(unittest.TestCase):
    def test_integer_sampling_sign_both_axes_and_signs(self):
        reference = image()
        for shift in ((1, 0), (-1, 0), (0, 2), (0, -2), (1, -1)):
            with self.subTest(shift=shift):
                pan = np.roll(reference, shift, axis=(0, 1))
                out = propose(pan, reference[None])
                np.testing.assert_allclose(out["prior_shift_dy_dx"], shift, atol=.06)
                self.assertTrue(out["ready_for_review"], out["gate_reasons"])
                self.assertIs(out["accepted"], False)
                self.assertIn("no inversion", out["mapping_rule"])

    def test_flat_input_no_zero_fallback(self):
        out = propose(np.zeros((96, 96)), np.zeros((4, 96, 96)))
        self.assertIsNone(out["prior_shift_dy_dx"])
        self.assertFalse(out["ready_for_review"])
        self.assertFalse(out["accepted"])
        self.assertEqual(out["n_valid"], 0)
        self.assertTrue(all(w["estimator"] is None for w in out["windows"]))
        self.assertIn("pan_texture_below_minimum", out["windows"][0]["gate_reasons"])

    def test_identity_rejected_without_direction_fallback(self):
        ref = image()
        out = propose(ref, ref[None])
        self.assertLess(np.linalg.norm(out["prior_shift_dy_dx"]), .05)
        self.assertFalse(out["ready_for_review"])
        self.assertFalse(out["accepted"])
        self.assertIn("zero_or_near_zero_prior_no_fallback", out["gate_reasons"])

    def test_consensus_rejects_outliers_and_retains_every_window(self):
        vectors = [(1., .02)] * 3 + [(1.04, 0.)] * 3 + [(-2., -1.)] * 3
        with patch("teacher48.prepare_prior.estimator.estimate_shift",
                   side_effect=[raw_result(v) for v in vectors]):
            out = propose(image(), image()[None])
        self.assertEqual(len(out["windows"]), 9)
        self.assertEqual(out["n_valid"], 9)
        self.assertAlmostEqual(out["inlier_fraction"], 2 / 3)
        self.assertEqual(sum(w["consensus_inlier"] for w in out["windows"]), 6)
        np.testing.assert_allclose(out["prior_shift_dy_dx"], (1., 0.))
        self.assertTrue(out["ready_for_review"])
        self.assertFalse(out["accepted"])

    def test_local_gate_reasons_and_raw_rejected_shift_are_not_dropped(self):
        rejected = raw_result((3., 1.), accepted=False, boundary_hit=True,
                              primary_secondary_diff=1., peak_margin=.0001, peak_zncc=.1)
        with patch("teacher48.prepare_prior.estimator.estimate_shift", return_value=rejected):
            out = propose(image(), image()[None])
        self.assertEqual(out["n_valid"], 0)
        self.assertIsNone(out["prior_shift_dy_dx"])
        self.assertEqual(out["windows"][0]["estimator"]["dy_lr_raw"], 3.)
        self.assertEqual(out["windows"][0]["gate_reasons"], ["search_boundary_hit",
                          "primary_census_disagreement", "ambiguous_zncc_peak",
                          "peak_zncc_below_minimum"])

    def test_deterministic_json_and_input_hash_changes(self):
        ref = image()
        pan = np.roll(ref, 1, axis=0)
        first = propose(pan, ref[None])
        second = propose(pan, ref[None])
        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False)
        pan = pan.copy()
        pan[0, 0] += .1
        self.assertNotEqual(first["scene_input_hash"], propose(pan, ref[None])["scene_input_hash"])
        for key in ("estimator_source_sha256", "proposal_source_sha256", "config_sha256"):
            self.assertEqual(len(first[key]), 64)

    def test_shape_dtype_and_finiteness_validation(self):
        ref = image()
        for pan, lms in ((ref[None], ref[None]), (ref, ref),
                         (ref, ref[None, :32]), (ref[:16, :16], ref[None, :16, :16]),
                         (np.ones_like(ref, dtype=bool), ref[None]),
                         (ref.astype(complex), ref[None])):
            with self.subTest(pan_shape=pan.shape, lms_shape=lms.shape):
                with self.assertRaises(ValueError):
                    propose(pan, lms)
        broken = ref.copy()
        broken[2, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            propose(broken, ref[None])

    def test_complete_explicit_config_required_and_no_unknown_keys(self):
        for key in config():
            missing = config()
            missing.pop(key)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "missing"):
                validate_prior_config(missing)
        bad = config()
        bad["silently_tune"] = True
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_prior_config(bad)

    def test_invalid_configuration_values(self):
        for key, value in (("search_int", True), ("stride", 33), ("window_size", 8),
                           ("min_reference_std", 0), ("min_confidence", float("nan")),
                           ("min_peak_zncc", -1), ("consensus_tolerance", -1),
                           ("min_prior_magnitude", 5), ("rationale", "")):
            with self.subTest(key=key, value=value):
                bad = config()
                bad[key] = value
                with self.assertRaises(ValueError):
                    validate_prior_config(bad)

    def test_identity_grid_and_provenance_must_be_explicit(self):
        ref = image()
        for override in ({"source_scene_id": ""}, {"prior_grid_name": ""},
                         {"pixel_scale": [0, 1]}, {"origin_phase": [float("inf"), 0]},
                         {"provenance": {}}, {"provenance": {"unverified": float("nan")}}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                propose(ref, ref[None], **override)

    def test_no_input_mutation_and_reference_is_provided_lms_mean(self):
        ref = image()
        pan = np.roll(ref, 1, axis=0)
        lms = np.stack((ref - .5, ref + .5))
        original_pan, original_lms = pan.copy(), lms.copy()
        with patch("teacher48.prepare_prior.estimator.estimate_shift", return_value=raw_result()) as mock:
            propose(pan, lms)
        actual_reference, actual_moving, _ = mock.call_args_list[0][0]
        np.testing.assert_allclose(actual_reference, lms.astype(np.float64).mean(0)[:32, :32])
        np.testing.assert_array_equal(actual_moving, pan[:32, :32])
        np.testing.assert_array_equal(pan, original_pan)
        np.testing.assert_array_equal(lms, original_lms)

    def test_tail_windows_cover_native_scene_without_silent_crop(self):
        ref = image()[:85, :71]
        with patch("teacher48.prepare_prior.estimator.estimate_shift", return_value=raw_result()):
            out = propose(ref, ref[None])
        self.assertEqual(out["n_windows"], 9)
        self.assertEqual(out["windows"][-1]["crop_yxhw"], [53, 39, 32, 32])

    def test_historical_settings_are_partial_not_acceptance_policy(self):
        historical = audited_hr_local_gates()
        self.assertEqual(historical["gates"]["search_int"], 4)
        self.assertEqual(historical["gates"]["max_magnitude"], 4.)
        self.assertEqual(historical["gates"]["min_peak_margin"], .05)
        self.assertNotIn("min_confidence", historical["gates"])
        with self.assertRaises(ValueError):
            validate_prior_config(historical["gates"])


if __name__ == "__main__":
    unittest.main()
