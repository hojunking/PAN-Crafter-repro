"""Synthetic b_global contracts; numeric settings here are NOT sensor policy."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import h5py
import numpy as np
import torch

from pa.warp import warp_pan
from panda_method.global_prior import (
    GlobalPriorAsset, canonical_sha256, validate_global_settings,
)
from teacher48.prepare_global_prior import (
    PROPOSAL_SCHEMA, REVIEW_SCHEMA, SOURCE_SCHEMA, _file_hash, _write_new_json,
    aggregate_candidates, freeze_reviewed_proposal, main, match_patch,
    propose_train_prior, validate_train_source,
)


def config():
    return dict(search_int=4, max_magnitude=4., min_peak_margin=.005,
                max_primary_secondary_diff=.3, edge_top_frac=.3,
                match_window=64, min_reference_std=.001, min_pan_std=.001,
                min_peak_zncc=.3, min_valid_candidates=3, min_valid_fraction=.2,
                consensus_tolerance=.3, min_inlier_fraction=.6,
                min_direction_concentration=.7, max_dispersion=.2,
                min_confidence=.1, min_prior_magnitude=.05,
                rationale="Synthetic unit-test settings only; not a real-data policy.")


def reference():
    raw = np.random.RandomState(19).normal(200, 30, (64, 64)).astype(np.float32)
    return cv2.GaussianBlur(raw, (3, 3), .5)


def raw_result(vector=(1., 0.), **changes):
    dy, dx = vector
    result = dict(dy_lr_raw=dy, dx_lr_raw=dx, magnitude_raw=float(np.hypot(dy, dx)),
                  peak_zncc=.95, peak_margin=.3, secondary_dy=dy, secondary_dx=dx,
                  primary_secondary_diff=0., boundary_hit=False, accepted=True,
                  dy_lr_applied=dy, dx_lr_applied=dx)
    result.update(changes)
    return result


def candidates(vectors):
    ref = reference()
    with patch("teacher48.prepare_global_prior.estimator.estimate_shift",
               side_effect=[raw_result(v) for v in vectors]):
        rows = [dict(match_patch(ref, np.repeat(ref[None], 4, axis=0), config()), row=i)
                for i in range(len(vectors))]
    return rows


def source_fixture(directory, *, sensor="QB", rows=6):
    path = Path(directory) / ("train_" + sensor.lower() + ".h5")
    ref = reference()
    bands = 8 if sensor == "WV3" else 4
    with h5py.File(path, "w") as h5:
        h5.create_dataset("pan", data=np.repeat(np.roll(ref, 1, axis=0)[None, None], rows, axis=0))
        h5.create_dataset("lms", data=np.broadcast_to(ref, (rows, bands, 64, 64)))
        # Dereferencing either would fail: proving no GT/MS are needed/read.
        h5["gt"] = h5py.ExternalLink("missing-forbidden-gt.h5", "/gt")
        h5["ms"] = h5py.ExternalLink("missing-forbidden-ms.h5", "/ms")
    return dict(schema=SOURCE_SCHEMA, sensor=sensor, path=str(path), sha256=_file_hash(path),
                split="train", rows=rows, pan_key="pan", lms_key="lms",
                band_order=["synthetic-band-" + str(i) for i in range(bands)],
                band_order_evidence="Synthetic test fixture, not satellite evidence",
                lms_provenance="Explicit synthetic stored-LMS fixture",
                radiometry="stored_raw_dn", normalization="none",
                split_evidence="Synthetic train-only fixture")


def review_for(proposal):
    return dict(schema=REVIEW_SCHEMA, accepted=True, reviewed_by="synthetic-unit-test",
                rationale="Test-only review; does not approve real thresholds",
                settings_sha256=canonical_sha256(proposal["validation_policy"]),
                proposal_sha256=canonical_sha256(proposal))


def frozen_fixture(directory, sensor="QB"):
    source = source_fixture(directory, sensor=sensor)
    proposal = propose_train_prior(source, config())
    return freeze_reviewed_proposal(proposal, review_for(proposal)), source, proposal


class GlobalPatchMatcherTests(unittest.TestCase):
    def test_both_axes_and_signs_match_actual_warp(self):
        ref = reference()
        lms = np.repeat(ref[None], 4, axis=0)
        for shift in ((1, 0), (-1, 0), (0, 2), (0, -2), (1, -1)):
            with self.subTest(shift=shift):
                pan = np.roll(ref, shift, axis=(0, 1))
                row = match_patch(pan, lms, config())
                self.assertTrue(row["valid"], row["rejection_reasons"])
                np.testing.assert_allclose(row["candidate_dy_dx"], shift, atol=.06)
                corrected = warp_pan(torch.from_numpy(pan)[None, None],
                                     torch.tensor([shift], dtype=torch.float32))
                np.testing.assert_allclose(corrected[0, 0, 5:-5, 5:-5].numpy(), ref[5:-5, 5:-5], atol=.0001)

    def test_provided_lms_bandmean_is_reference_raw_pan_is_moving(self):
        ref = reference()
        lms = np.stack([ref, ref * 2, ref * 3, ref * 4])
        pan = np.roll(ref, 1, axis=0)
        with patch("teacher48.prepare_global_prior.estimator.estimate_shift", return_value=raw_result()) as matcher:
            match_patch(pan, lms, config())
        args = matcher.call_args.args
        np.testing.assert_array_equal(args[0], lms.astype(np.float64).mean(axis=0).astype(np.float32))
        np.testing.assert_array_equal(args[1], pan)

    def test_low_texture_is_invalid_not_zero_shift(self):
        row = match_patch(np.ones((64, 64)), np.ones((4, 64, 64)), config())
        self.assertFalse(row["valid"])
        self.assertIsNone(row["candidate_dy_dx"])
        self.assertEqual(row["confidence_weight"], 0.)
        self.assertIn("pan_texture_below_minimum", row["rejection_reasons"])

    def test_rejected_raw_shift_retained_not_applied_zero(self):
        ref = reference()
        raw = raw_result((3., 2.), accepted=False, boundary_hit=True,
                         dy_lr_applied=0., dx_lr_applied=0.)
        with patch("teacher48.prepare_global_prior.estimator.estimate_shift", return_value=raw):
            row = match_patch(ref, np.repeat(ref[None], 4, axis=0), config())
        self.assertEqual(row["candidate_dy_dx"], [3., 2.])
        self.assertFalse(row["valid"])
        self.assertEqual(row["confidence_weight"], 0.)

    def test_nonfinite_badshape_and_overflow_reject(self):
        ref = reference()
        for pan, lms in ((ref[:32], np.repeat(ref[None], 4, axis=0)),
                         (ref * np.nan, np.repeat(ref[None], 4, axis=0)),
                         (ref, np.ones((3, 64, 64))),
                         (np.full((64, 64), 1e100), np.ones((4, 64, 64)))):
            with self.subTest(shape=pan.shape), self.assertRaises(ValueError), np.errstate(over="ignore"):
                match_patch(pan, lms, config())

    def test_settings_must_be_complete_and_finite(self):
        for key in config():
            value = config()
            del value[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_global_settings(value)
        for update in ({"match_window": 32}, {"search_int": 31}, {"search_int": True},
                       {"min_prior_magnitude": 0}, {"min_prior_magnitude": float("nan")},
                       {"unknown": 3}, {"rationale": ""}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_global_settings(dict(config(), **update))


class GlobalAggregationTests(unittest.TestCase):
    def test_signed_consensus_not_magnitude_mean(self):
        rows = candidates([(1., -.1)] * 4 + [(-2., -1.)] * 2)
        result = aggregate_candidates(rows, config())
        np.testing.assert_allclose(result["prior_dy_dx"], (1., -.1))
        agg = result["aggregation"]
        self.assertEqual(agg["valid_count"], 6)
        self.assertEqual(agg["inlier_count"], 4)
        self.assertEqual(agg["outlier_count"], 2)
        self.assertLess(agg["post_consensus_weight_sum"], agg["pre_consensus_weight_sum"])
        self.assertEqual(len(result["candidates"]), 6)
        self.assertTrue(result["ready_for_review"], result["gate_reasons"])
        self.assertIn("component_weighted_median", agg["method"])

    def test_invalid_candidate_cannot_pollute_median(self):
        rows = candidates([(1., 0.)] * 3 + [(-3., -1.)] * 2)
        for row in rows[3:]:
            row.update(valid=False, rejection_reasons=["synthetic rejection"], confidence_weight=0.)
        result = aggregate_candidates(rows, config())
        self.assertEqual(result["prior_dy_dx"], [1., 0.])
        self.assertEqual(result["aggregation"]["rejected_count"], 2)
        self.assertEqual(result["aggregation"]["valid_count"], 3)

    def test_opposing_groups_fail_direction_consensus(self):
        result = aggregate_candidates(candidates([(1., 0.)] * 3 + [(-1., 0.)] * 3), config())
        self.assertFalse(result["ready_for_review"])
        self.assertIn("insufficient_translation_consensus", result["gate_reasons"])

    def test_zero_prior_has_no_invented_basis(self):
        result = aggregate_candidates(candidates([(0., 0.)] * 6), config())
        self.assertFalse(result["ready_for_review"])
        self.assertEqual(result["prior_dy_dx"], [0., 0.])
        self.assertIn("zero_or_near_zero_prior_no_fallback", result["gate_reasons"])

    def test_all_invalid_preserves_rows_and_no_prior(self):
        rows = [dict(match_patch(np.zeros((64, 64)), np.zeros((4, 64, 64)), config()), row=i)
                for i in range(6)]
        result = aggregate_candidates(rows, config())
        self.assertIsNone(result["prior_dy_dx"])
        self.assertFalse(result["ready_for_review"])
        self.assertEqual(result["aggregation"]["rejected_count"], 6)

    def test_missing_duplicate_and_reordered_rows_reject(self):
        rows = candidates([(1., 0.)] * 6)
        for changed in (rows[:2] + rows[3:], rows[::-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                aggregate_candidates(changed, config())

    def test_post_refit_weights_are_actual_final_inliers(self):
        rows = candidates([(1., -.1), (1.1, .1), (.9, -.1), (1.2, -.1), (-1., 0.)])
        result = aggregate_candidates(rows, config())
        expected = sum(r["confidence_weight"] for r in result["candidates"] if r["consensus_inlier"])
        self.assertAlmostEqual(result["aggregation"]["post_consensus_weight_sum"], expected)

    def test_raw_candidate_vector_weight_and_gate_tampering_reject(self):
        for key, value in (("candidate_dy_dx", [2., 1.]), ("confidence_weight", 999.)):
            rows = candidates([(1., 0.)] * 6)
            rows[0][key] = value
            with self.assertRaisesRegex(ValueError, "raw matcher"):
                aggregate_candidates(rows, config())
        rows = candidates([(1., 0.)] * 6)
        rows[0]["raw_estimator"]["accepted"] = False
        with self.assertRaisesRegex(ValueError, "raw matcher"):
            aggregate_candidates(rows, config())


class GlobalSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = source_fixture(self.temp.name)

    def test_all_rows_hashed_no_gt_ms_read_no_scene_id(self):
        proposal = propose_train_prior(self.source, config())
        self.assertEqual(proposal["schema"], PROPOSAL_SCHEMA)
        self.assertEqual([r["row"] for r in proposal["candidates"]], list(range(6)))
        self.assertFalse(proposal["accepted"])
        self.assertFalse(proposal["frozen"])
        self.assertFalse(proposal["source_scene_mapping_used"])
        self.assertNotIn("scene_id", json.dumps(proposal))
        self.assertEqual(proposal["created_from_splits"], ["train"])
        self.assertEqual(proposal["train_source_sha256"], self.source["sha256"])
        self.assertTrue(proposal["ready_for_review"], proposal["gate_reasons"])
        self.assertEqual(len(proposal["candidates"][0]["pan_sha256"]), 64)

    def test_split_source_shape_and_hash_failure(self):
        for change in ({"split": "validation"}, {"split": "test"}, {"path": "valid_qb.h5"},
                       {"path": "train_test_qb.h5"}, {"rows": 5}, {"sha256": "a" * 64},
                       {"sensor": "WV3"}, {"lms_key": "gt"}):
            with self.subTest(change=change), self.assertRaises((ValueError, FileNotFoundError)):
                propose_train_prior(dict(self.source, **change), config())

    def test_raw_provided_lms_changes_proposal_and_input_hash(self):
        first = propose_train_prior(self.source, config())
        with h5py.File(self.source["path"], "r+") as h5:
            h5["lms"][0, 0, 0, 0] += 1.
        self.source["sha256"] = _file_hash(self.source["path"])
        second = propose_train_prior(self.source, config())
        self.assertNotEqual(first["train_source_sha256"], second["train_source_sha256"])
        self.assertNotEqual(first["candidates"][0]["provided_lms_sha256"],
                            second["candidates"][0]["provided_lms_sha256"])

    def test_external_lms_dataset_rejected_even_with_source_hash(self):
        with h5py.File(self.source["path"], "r+") as h5:
            del h5["lms"]
            h5["lms"] = h5py.ExternalLink("missing.h5", "/lms")
        self.source["sha256"] = _file_hash(self.source["path"])
        with self.assertRaisesRegex(ValueError, "materialized"):
            propose_train_prior(self.source, config())

    def test_source_registration_has_no_implicit_numeric_defaults(self):
        for key in self.source:
            bad = dict(self.source)
            del bad[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_train_source(bad)


class GlobalAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.asset, self.source, self.proposal = frozen_fixture(self.temp.name)

    def test_roundtrip_and_deepcopy_fingerprint(self):
        manifest = self.asset.manifest
        old = self.asset.fingerprint()
        manifest["prior_dy_dx"][0] = 99.
        self.assertNotEqual(self.asset.vector[0], 99.)
        self.assertEqual(self.asset.fingerprint(), old)
        path = Path(self.temp.name) / "asset.json"
        _write_new_json(path, self.asset.manifest)
        loaded = GlobalPriorAsset.from_file(path, sensor="QB", expected_train_sha256=self.source["sha256"])
        self.assertEqual(loaded.fingerprint(), old)
        self.assertEqual(loaded.vector, self.asset.vector)
        self.assertEqual(loaded.min_magnitude, .05)
        self.assertEqual(loaded.max_magnitude, 4.)

    def test_sensor_and_training_hash_match_required(self):
        path = Path(self.temp.name) / "asset.json"
        _write_new_json(path, self.asset.manifest)
        for kw in ({"sensor": "WV3"}, {"expected_train_sha256": "b" * 64}):
            with self.assertRaises(ValueError):
                GlobalPriorAsset.from_file(path, **kw)

    def test_proposal_never_loadable_as_frozen_asset(self):
        with self.assertRaises(ValueError):
            GlobalPriorAsset.from_dict(self.proposal)

    def test_review_is_bound_to_exact_policy_and_proposal(self):
        for change in ({"proposal_sha256": "c" * 64}, {"settings_sha256": "d" * 64}, {"accepted": False}):
            with self.assertRaises(ValueError):
                freeze_reviewed_proposal(self.proposal, dict(review_for(self.proposal), **change))

    def test_consensus_modification_detected_even_with_new_review_hash(self):
        proposal = copy.deepcopy(self.proposal)
        proposal["prior_dy_dx"] = [2., 2.]
        with self.assertRaisesRegex(ValueError, "consensus"):
            freeze_reviewed_proposal(proposal, review_for(proposal))

    def test_core_asset_corruption_rejected(self):
        cases = (("frozen", False), ("accepted", False), ("created_from_splits", ["train", "test"]),
                 ("source_scene_mapping_used", True), ("prior_dy_dx", [0., 0.]),
                 ("prior_dy_dx", [float("nan"), 1.]), ("prior_dy_dx", [5., 0.]),
                 ("gate_reasons", ["unresolved"]), ("train_source_sha256", "unknown"))
        for key, value in cases:
            bad = self.asset.manifest
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                GlobalPriorAsset.from_dict(bad)

    def test_missing_threshold_or_inconsistent_counts_reject(self):
        bad = self.asset.manifest
        del bad["validation_policy"]["min_prior_magnitude"]
        with self.assertRaises(ValueError):
            GlobalPriorAsset.from_dict(bad)
        for key, value in (("valid_count", 99), ("inlier_count", 0), ("direction_concentration", 0),
                           ("post_consensus_weight_sum", 100)):
            bad = self.asset.manifest
            bad["aggregation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                GlobalPriorAsset.from_dict(bad)

    def test_source_registration_tampering_reject(self):
        bad = self.asset.manifest
        bad["source_registration"]["rows"] += 1
        with self.assertRaisesRegex(ValueError, "registration"):
            GlobalPriorAsset.from_dict(bad)

    def test_invalid_zero_proposal_cannot_freeze(self):
        proposal = copy.deepcopy(self.proposal)
        proposal.update(ready_for_review=False, gate_reasons=["zero_or_near_zero_prior_no_fallback"])
        with self.assertRaises(ValueError):
            freeze_reviewed_proposal(proposal, review_for(proposal))

    def test_atomic_json_never_overwrites(self):
        path = Path(self.temp.name) / "new.json"
        _write_new_json(path, {"old": True})
        with self.assertRaises(ValueError):
            _write_new_json(path, {"new": True})
        self.assertEqual(json.loads(path.read_text()), {"old": True})

    def test_cli_freezes_readonly_asset_with_explicit_review(self):
        directory = Path(self.temp.name)
        proposal, review, asset = (directory / name for name in ("proposal.json", "review.json", "asset.json"))
        _write_new_json(proposal, self.proposal)
        _write_new_json(review, review_for(self.proposal))
        main(["freeze", "--proposal", str(proposal), "--review", str(review), "--output", str(asset)])
        self.assertEqual(asset.stat().st_mode & 0o222, 0)
        self.assertEqual(GlobalPriorAsset.from_file(asset).fingerprint(), self.asset.fingerprint())


if __name__ == "__main__":
    unittest.main()
