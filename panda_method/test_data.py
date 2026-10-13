"""Independent CPU scene-binding/augmentation checks; no real data or runs."""

from dataclasses import replace
import itertools
import unittest

import torch

from pa.warp import warp_pan
from panda_method.data import (
    BatchLineage, PriorBinding, augment_batch, bind_scene_priors,
)
from panda_method.geometry import (
    ScenePriorRecord, ScenePriorRegistry, VerifiedGridTransform,
)


def record(scene, shift):
    return ScenePriorRecord(
        source_scene_id=scene, prior_shift_dy_dx=shift, prior_grid_name="native-pan",
        pixel_scale=(1., 1.), origin_phase=(0., 0.), n_windows=12, n_valid=10,
        inlier_fraction=10 / 12, direction_concentration=.8, dispersion=.2,
        confidence=.7, estimator_version="synthetic-not-an-estimator",
        reference_type="native_lms", scene_input_hash="1" * 64,
        mapping_rule="synthetic sampling dy/dx only", accepted=True,
        acceptance_evidence="Synthetic fixture; no real acquisition acceptance",
    )


def fixtures(*, transforms=True, reverse=False):
    records = [record("scene-a", (4., 8.)), record("scene-b", (-6., 2.))]
    pairs = [("patch-a0", "scene-a"), ("patch-b0", "scene-b"), ("patch-b1", "scene-b")]
    transform = VerifiedGridTransform(
        "native-pan", "rr-pan", ((.25, 0.), (0., .25)), (.25, -.125),
        "Synthetic paired-grid scale and nonzero differential phase evidence")
    registry = ScenePriorRegistry(records[::-1] if reverse else records)
    binding = PriorBinding(registry, sample_scene_pairs=pairs[::-1] if reverse else pairs,
                           grid_transforms=[transform] if transforms else [])
    return binding, transform


def lineage(binding, *, grid="rr-pan", resolution="RR", split="train"):
    return BatchLineage(
        split, resolution, "a" * 64, binding.fingerprint(),
        ("scene-b", "scene-a", "scene-b"), ("patch-b1", "patch-a0", "patch-b0"),
        grid, "Synthetic crop coordinates; no initial augmentation",
        "Synthetic training-only native source" if resolution == "FR" else "",
    )


def observations(*, bands=4, dtype=torch.float64, gt=True, lpan=True):
    # Non-square grids catch rotation/HW and LRMS/PAN correspondence mistakes.
    def values(channels, h, w, offset):
        return torch.arange(3 * channels * h * w, dtype=dtype).reshape(3, channels, h, w) + offset

    return dict(pan=values(1, 16, 20, 1), ms=values(bands, 4, 5, 2),
                native_lms=values(bands, 16, 20, 3),
                gt=values(bands, 16, 20, 4) if gt else None,
                lpan=values(1, 4, 5, 5) if lpan else None)


class DataBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mixed_scene_sample_order_is_exact(self):
        binding, _ = fixtures()
        batch = bind_scene_priors(**observations(), lineage=lineage(binding), binding=binding)
        # Convert canonical vectors first: .25*b + (.25,-.125).
        expected = torch.tensor([[-1.25, .375], [1.25, 1.875], [-1.25, .375]], dtype=torch.float64)
        torch.testing.assert_close(batch.prior_dy_dx, expected, rtol=0, atol=0)
        self.assertFalse(batch.prior_dy_dx.requires_grad)
        self.assertEqual(batch.lineage.sample_ids, ("patch-b1", "patch-a0", "patch-b0"))

    def test_same_grid_lookup_does_not_invent_a_conversion(self):
        binding, _ = fixtures(transforms=False)
        batch = bind_scene_priors(**observations(), lineage=lineage(binding, grid="native-pan"),
                                 binding=binding)
        expected = torch.tensor([[-6., 2.], [4., 8.], [-6., 2.]], dtype=torch.float64)
        torch.testing.assert_close(batch.prior_dy_dx, expected, rtol=0, atol=0)

    def test_no_shape_based_fourfold_or_identity_fallback(self):
        binding, _ = fixtures(transforms=False)
        with self.assertRaisesRegex(ValueError, "Missing explicit"):
            bind_scene_priors(**observations(), lineage=lineage(binding), binding=binding)
        with self.assertRaisesRegex(ValueError, "Missing explicit"):
            bind_scene_priors(**observations(), lineage=lineage(binding, grid="unknown-grid"),
                              binding=binding)

    def test_wrong_hash_wrong_scene_and_unknown_sample_are_rejected(self):
        binding, _ = fixtures()
        canonical = lineage(binding)
        wrong = [replace(canonical, prior_mapping_sha256="f" * 64),
                 replace(canonical, scene_ids=("scene-a", "scene-b", "scene-b")),
                 replace(canonical, scene_ids=("unknown-scene", "scene-a", "scene-b")),
                 replace(canonical, sample_ids=("unknown-sample", "patch-a0", "patch-b0"))]
        for item in wrong:
            with self.subTest(lineage=item), self.assertRaises(ValueError):
                bind_scene_priors(**observations(), lineage=item, binding=binding)

    def test_binding_fingerprint_covers_mapping_registry_and_transform_evidence(self):
        binding, transform = fixtures()
        reordered, _ = fixtures(reverse=True)
        self.assertEqual(binding.fingerprint(), reordered.fingerprint())
        pairs = [("patch-a0", "scene-a"), ("patch-b0", "scene-b"), ("patch-b1", "scene-b")]
        changed = [
            PriorBinding(binding.registry, sample_scene_pairs=pairs,
                         grid_transforms=[replace(transform, evidence="Different reviewed transform")]),
            PriorBinding(binding.registry, sample_scene_pairs=pairs,
                         grid_transforms=[replace(transform, phase_dy_dx=(0., 0.))]),
            PriorBinding(binding.registry,
                         sample_scene_pairs=[("patch-a0", "scene-b"), *pairs[1:]],
                         grid_transforms=[transform]),
            PriorBinding(ScenePriorRegistry([record("scene-a", (5., 8.)), record("scene-b", (-6., 2.))]),
                         sample_scene_pairs=pairs, grid_transforms=[transform]),
        ]
        for item in changed:
            self.assertNotEqual(binding.fingerprint(), item.fingerprint())
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                bind_scene_priors(**observations(), lineage=lineage(binding), binding=item)

    def test_invalid_source_maps_fail_without_guessing(self):
        binding, transform = fixtures()
        for pairs in ([], [("patch", "scene-a"), ("patch", "scene-b")]):
            with self.assertRaises(ValueError):
                PriorBinding(binding.registry, sample_scene_pairs=pairs, grid_transforms=[])
        with self.assertRaises(KeyError):
            PriorBinding(binding.registry, sample_scene_pairs=[("patch", "unknown")], grid_transforms=[])
        with self.assertRaises(ValueError):
            PriorBinding(binding.registry, sample_scene_pairs=[("patch", "scene-a")],
                         grid_transforms=[transform, transform])

    def test_all_observations_and_vectors_share_augmentation_order(self):
        binding, _ = fixtures()
        batch = bind_scene_priors(**observations(), lineage=lineage(binding), binding=binding)
        original = {key: getattr(batch, key).clone() for key in ("pan", "ms", "native_lms", "gt", "lpan")}
        for hf, vf, k in itertools.product((False, True), (False, True), range(4)):
            with self.subTest(hflip=hf, vflip=vf, rot90=k):
                actual = augment_batch(batch, hflip=hf, vflip=vf, rot90=k)
                for key, value in original.items():
                    expected = value.flip(-1) if hf else value
                    expected = expected.flip(-2) if vf else expected
                    expected = torch.rot90(expected, k, (-2, -1))
                    torch.testing.assert_close(getattr(actual, key), expected, rtol=0, atol=0)
                dy, dx = batch.prior_dy_dx[:, 0], batch.prior_dy_dx[:, 1]
                dy, dx = (-dy if vf else dy), (-dx if hf else dx)
                for _ in range(k):
                    dy, dx = -dx, dy
                torch.testing.assert_close(actual.prior_dy_dx, torch.stack((dy, dx), dim=-1),
                                           rtol=0, atol=0)
                self.assertEqual(actual.lineage.sample_ids, batch.lineage.sample_ids)
                self.assertEqual(actual.lineage.prior_mapping_sha256, binding.fingerprint())
                self.assertTrue(actual.lineage.geometry_evidence.startswith(batch.lineage.geometry_evidence))
                actual.validate(4, training=True, require_gt=True)
        for key, value in original.items():
            torch.testing.assert_close(getattr(batch, key), value, rtol=0, atol=0)

    def test_phase_is_converted_before_rotation_not_added_afterwards(self):
        binding, _ = fixtures()
        batch = bind_scene_priors(**observations(), lineage=lineage(binding), binding=binding)
        rotated = augment_batch(batch, hflip=False, vflip=False, rot90=1)
        expected = torch.tensor([[-.375, -1.25], [-1.875, 1.25], [-.375, -1.25]],
                                dtype=torch.float64)
        torch.testing.assert_close(rotated.prior_dy_dx, expected, rtol=0, atol=0)
        wrong_phase_order = torch.tensor([[-.25, -1.625], [-1.75, .875], [-.25, -1.625]],
                                        dtype=torch.float64)
        self.assertFalse(torch.equal(rotated.prior_dy_dx, wrong_phase_order))

    def test_batch_augmentation_commutes_with_pan_correction(self):
        binding, _ = fixtures()
        inputs = observations()
        inputs["pan"].zero_()
        inputs["pan"][:, :, 7, 10] = 1.
        batch = bind_scene_priors(**inputs, lineage=lineage(binding), binding=binding)
        corrected = warp_pan(batch.pan, batch.prior_dy_dx)
        for hf, vf, k in itertools.product((False, True), (False, True), range(4)):
            actual = augment_batch(batch, hflip=hf, vflip=vf, rot90=k)
            expected = corrected.flip(-1) if hf else corrected
            expected = expected.flip(-2) if vf else expected
            expected = torch.rot90(expected, k, (-2, -1))
            torch.testing.assert_close(warp_pan(actual.pan, actual.prior_dy_dx), expected,
                                       rtol=0, atol=2e-14)

    def test_fr_optional_fields_and_random_state_are_preserved(self):
        binding, _ = fixtures()
        batch = bind_scene_priors(**observations(gt=False, lpan=False), binding=binding,
                                 lineage=lineage(binding, resolution="FR", grid="native-pan"))
        before = torch.get_rng_state().clone()
        actual = augment_batch(batch, hflip=True, vflip=True, rot90=1)
        self.assertIsNone(actual.gt)
        self.assertIsNone(actual.lpan)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        actual.validate(4, training=True)


if __name__ == "__main__":
    unittest.main()
