"""CPU-only method contracts; synthetic observations, no real prior estimation."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import unittest

import torch
import torch.nn.functional as F

from pa.warp import warp_pan
from .calibration import state_dict_sha256
from .geometry import SceneAwarePANAligner, transform_prior
from .global_rr import (GLOBAL_RR_METHOD_VERSION, GlobalPANAligner, GlobalRRStage1,
                        GlobalRRStage2, RRTrainingSources)
from .losses import AbsoluteGradientLoss, SpatialLossSpec
from .reconstruction import ReconstructionSpec, build_reconstructor
from .stages import calibrate_training


torch.set_num_threads(1)
SOURCES = RRTrainingSources("a" * 64, "b" * 64, "QB", "c" * 64)
VECTOR = (.3, -.4)


@dataclass(frozen=True)
class _SyntheticLineage:
    split: str = "train"
    resolution: str = "RR"
    source_manifest_sha256: str = SOURCES.rr_manifest_sha256
    prior_mapping_sha256: str = SOURCES.prior_mapping_sha256
    prior_scope: str = "dataset_global"
    sensor: str = "QB"
    train_source_sha256: str = SOURCES.train_source_sha256
    canonical_prior_dy_dx: tuple = VECTOR
    grid_name: str = "PAN_RR"
    sample_ids: tuple = ("synthetic-0", "synthetic-1")

    def require_training(self):
        if self.split != "train" or self.resolution != "RR":
            raise ValueError("Synthetic benchmark/FR input cannot enter training")


@dataclass(frozen=True)
class _SyntheticBatch:
    pan: torch.Tensor
    ms: torch.Tensor
    native_lms: torch.Tensor
    prior_dy_dx: torch.Tensor
    lineage: _SyntheticLineage
    gt: torch.Tensor | None = None
    lpan: torch.Tensor | None = None

    def validate(self, bands, *, training=False, require_gt=False):
        b, _, h, w = self.pan.shape
        if self.native_lms.shape != (b, bands, h, w):
            raise ValueError("Synthetic shape mismatch")
        if training:
            self.lineage.require_training()
        if require_gt and self.gt is None:
            raise ValueError("RR GT required")
        return self


def batch(*, side=16):
    rng = torch.Generator().manual_seed(41)
    p = torch.rand(2, 1, side, side, generator=rng)
    ms = torch.rand(2, 4, side // 4, side // 4, generator=rng)
    return _SyntheticBatch(p, ms, F.interpolate(ms, scale_factor=4, mode="nearest"),
        torch.tensor([VECTOR, VECTOR]), _SyntheticLineage(),
        torch.rand(2, 4, side, side, generator=rng),
        torch.rand(2, 1, side // 4, side // 4, generator=rng))


def aligner(**changes):
    args = dict(view_margin=0, sensor="QB", train_source_sha256=SOURCES.train_source_sha256,
                prior_asset_sha256=SOURCES.prior_mapping_sha256,
                canonical_prior_dy_dx=VECTOR, min_prior_magnitude=.01, max_prior_magnitude=4.)
    args.update(changes)
    return GlobalPANAligner(4, **args)


def recon(role="T", layout="P0"):
    return build_reconstructor(ReconstructionSpec(4, 8, (1, 1, 1), layout, "BICUBIC_MS"),
                               seed=42, role=role)


def stage1(**changes):
    args = dict(sources=SOURCES, epsilon_seed=123, boundary_policy="report",
                calibration_protocol="synthetic RR training population; no augmentation")
    args.update(changes)
    return GlobalRRStage1(aligner(), recon(), AbsoluteGradientLoss(SpatialLossSpec(
        "QB", (2, 1, 0), "forward_difference", "reflect", 3, "synthetic_common_0_1")), **args)


class GlobalAlignerTests(unittest.TestCase):
    def test_same_architecture_zero_init_raw_magnitude_and_detachment(self):
        model, rr = aligner(), batch()
        old = SceneAwarePANAligner(4, view_margin=0)
        self.assertEqual([(n, tuple(p.shape)) for n, p in model.named_parameters()],
                         [(n, tuple(p.shape)) for n, p in old.named_parameters()])
        prior = rr.prior_dy_dx.clone().requires_grad_()
        head = []
        hook = model.fc1.register_forward_pre_hook(lambda _, args: head.append(args[0]))
        c, details = model(rr.pan, rr.native_lms, prior, return_details=True)
        hook.remove()
        torch.testing.assert_close(c, prior, rtol=0, atol=0)
        self.assertEqual(tuple(head[0].shape), (2, 66))
        torch.testing.assert_close(head[0][:, -2:], prior, rtol=0, atol=0)
        torch.testing.assert_close(details["alpha"], torch.zeros(2, 1), rtol=0, atol=0)
        c.sum().backward()
        self.assertIsNone(prior.grad)
        self.assertEqual(float(model.fc1.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.fc2.weight.grad.abs().sum()), 0.)
        self.assertEqual(model.conditioning_manifest()["prior_scope"], "dataset_global")

    def test_augmentation_preserves_sampling_geometry(self):
        rr, model = batch(), aligner()
        for hflip in (False, True):
            for vflip in (False, True):
                for k in range(4):
                    def apply(x):
                        if hflip:
                            x = x.flip(-1)
                        if vflip:
                            x = x.flip(-2)
                        return torch.rot90(x, k, (-2, -1))
                    b = transform_prior(rr.prior_dy_dx, hflip=hflip, vflip=vflip, rot90=k)
                    c = model(apply(rr.pan), apply(rr.native_lms), b)
                    torch.testing.assert_close(c, b, rtol=0, atol=0)
                    torch.testing.assert_close(warp_pan(apply(rr.pan), b),
                                               apply(warp_pan(rr.pan, rr.prior_dy_dx)),
                                               rtol=1e-5, atol=2e-6)

    def test_zero_near_zero_nonfinite_large_and_limits_rejected(self):
        rr, model = batch(), aligner()
        for v in ((0., 0.), (1e-9, 0.), (5., 0.), (float("nan"), .3), (float("inf"), 0.)):
            with self.subTest(vector=v), self.assertRaises(ValueError):
                model(rr.pan, rr.native_lms, torch.tensor([v, v]))
        for limits in ((0., 4.), (-1., 4.), (5., 4.), (.01, float("inf"))):
            with self.assertRaises(ValueError):
                aligner(min_prior_magnitude=limits[0], max_prior_magnitude=limits[1])
        with self.assertRaises(ValueError):
            aligner(canonical_prior_dy_dx=(0., 0.))

    def test_old_wrong_asset_and_unmarked_checkpoints_rejected_before_mutation(self):
        model = aligner()
        before = state_dict_sha256(model)
        legacy = SceneAwarePANAligner(4, view_margin=0).state_dict()
        other = aligner(prior_asset_sha256="e" * 64).state_dict()
        for saved in (legacy, other):
            for strict in (True, False):
                with self.assertRaisesRegex(RuntimeError, "identity"):
                    model.load_state_dict(saved, strict=strict)
                self.assertEqual(before, state_dict_sha256(model))
        model.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertEqual(before, state_dict_sha256(model))

    def test_inclusive_magnitude_bound_is_canonical_not_rounded_tensor_norm(self):
        rr = batch()
        model = aligner(canonical_prior_dy_dx=(.3, .4), min_prior_magnitude=.5,
                        max_prior_magnitude=.5)
        for dtype in (torch.float32, torch.float64):
            raw = torch.tensor([[.3, .4], [.3, .4]], dtype=dtype)
            for k in range(4):
                b = transform_prior(raw, rot90=k)
                out = model(rr.pan, rr.native_lms, b)
                torch.testing.assert_close(out, b.to(out.dtype), rtol=0, atol=0)

    def test_same_magnitude_non_orbit_and_patch_specific_b_rejected(self):
        rr, model = batch(), aligner()
        for values in (((.5, 0.), (.3, -.4)), ((.3, -.4), (.299, -.4)),
                       ((.4, -.3), (.2, -.4))):
            with self.assertRaisesRegex(ValueError, "D4"):
                model(rr.pan, rr.native_lms, torch.tensor(values))


class GlobalStage1Tests(unittest.TestCase):
    def test_rr_only_exact_loss_one_u_two_a_and_same_provided_lms_prior(self):
        rr, model = batch(), stage1()
        acalls, ucalls = [], []
        ha = model.aligner.register_forward_pre_hook(lambda _, args: acalls.append(args))
        hu = model.teacher.register_forward_pre_hook(lambda _, args: ucalls.append(args))
        out = model(rr)
        ha.remove(); hu.remove()
        self.assertEqual(len(acalls), 2)
        self.assertEqual(len(ucalls), 1)
        self.assertIs(acalls[0][2], acalls[1][2])
        torch.testing.assert_close(acalls[0][1], rr.native_lms, rtol=0, atol=0)
        torch.testing.assert_close(acalls[1][0], warp_pan(rr.pan, out["epsilon_rr"]))
        self.assertFalse(acalls[1][0].requires_grad)
        torch.testing.assert_close(out["loss"], out["rec"] + out["abs_rr"] + .001 * out["eps_rr"])
        self.assertFalse(any(key.endswith("_fr") or key.startswith("epsilon_fr_") for key in out))
        self.assertEqual(model.pipeline_manifest()["training_resolutions"], ["RR"])
        self.assertLessEqual(float(out["epsilon_rr"].norm(dim=-1).max()), 2.)
        self.assertFalse(torch.equal(model(rr)["epsilon_rr"], out["epsilon_rr"]))
        with self.assertRaises(TypeError):
            model(rr, rr)

    def test_gradient_routing_and_detached_native_target(self):
        for term in ("rec", "abs_rr", "eps_rr"):
            model = stage1()
            with torch.no_grad():
                model.teacher.backbone.output[-1].weight.normal_(std=.01)
            out = model(batch())
            c0 = out["native"]["correction"]
            c0.retain_grad()
            out[term].backward()
            ag = sum(float(p.grad.abs().sum()) for p in model.aligner.parameters() if p.grad is not None)
            tg = [p.grad for p in model.teacher.parameters() if p.grad is not None]
            self.assertGreater(ag, 0., term)
            if term == "rec":
                self.assertGreater(sum(float(g.abs().sum()) for g in tg), 0.)
            else:
                self.assertEqual(len(tg), 0)
            if term == "eps_rr":
                self.assertIsNone(c0.grad)
            else:
                self.assertGreater(float(c0.grad.abs().sum()), 0.)

    def test_resume_and_rng_provenance(self):
        rr, model = batch(), stage1()
        rng = torch.random.get_rng_state().clone()
        model(rr)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        saved = copy.deepcopy(model.state_dict())
        self.assertEqual(saved["_extra_state"]["method"], GLOBAL_RR_METHOD_VERSION)
        expected = model(rr)
        restored = stage1()
        restored.load_state_dict(saved)
        actual = restored(rr)
        for key in ("loss", "epsilon_rr", "c_eps_rr"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(actual["forward_index"], expected["forward_index"])
        with self.assertRaises(ValueError):
            restored.load_state_dict(saved, strict=False)
        saved["_extra_state"]["method"] = "PANDA_MethodPlan_v1.3_2026-10-13"
        with self.assertRaises(ValueError):
            restored.load_state_dict(saved)

    def test_no_scene_ids_required_and_split_identity_guards(self):
        rr, model = batch(), stage1()
        self.assertFalse(hasattr(rr.lineage, "scene_ids"))
        model(rr)
        for changes in ({"split": "test"}, {"split": "validation"}, {"resolution": "FR"},
                        {"sensor": "GF2"}, {"train_source_sha256": "e" * 64},
                        {"prior_scope": "scene"}, {"prior_mapping_sha256": "e" * 64},
                        {"canonical_prior_dy_dx": (.4, -.3)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                model(replace(rr, lineage=replace(rr.lineage, **changes)))

    def test_rr_whole_image_inference_and_fr_unresolved_failure(self):
        model, rr = stage1(), batch(side=32)
        evaluated = replace(rr, lineage=replace(rr.lineage, split="test",
                                               source_manifest_sha256="f" * 64))
        out = model.native_prediction(evaluated)
        self.assertEqual(tuple(out["correction"].shape), (2, 2))
        self.assertEqual(tuple(out["y"].shape), (2, 4, 32, 32))
        fr = replace(evaluated, gt=None, lineage=replace(evaluated.lineage,
                     resolution="FR", grid_name="PAN_FR"))
        with self.assertRaisesRegex(NotImplementedError, "FR prior"):
            model.native_prediction(fr)

    def test_support_guard_rolls_back_counter_rng(self):
        model, rr = stage1(boundary_policy="require_spatial_support"), batch()
        saved_rng = model._epsilon_rng.get_state().clone()
        with torch.no_grad():
            model.aligner.fc2.bias[:] = 20.
        with self.assertRaisesRegex(ValueError, "support"):
            model(rr)
        self.assertEqual(model.forward_calls, 0)
        self.assertTrue(torch.equal(saved_rng, model._epsilon_rng.get_state()))


class GlobalStage2Tests(unittest.TestCase):
    def calibrated(self):
        first, rr = stage1(), batch()
        scales = calibrate_training(first, [rr], expected_sample_ids=rr.lineage.sample_ids)
        return first, rr, scales

    def test_frozen_shared_pan_e_q_calibration_and_student_gradients(self):
        first, rr, scales = self.calibrated()
        second = GlobalRRStage2(first, recon("S", "PLH"), scales)
        self.assertIs(first.aligner, second.aligner)
        second.train()
        self.assertFalse(second.aligner.training)
        self.assertFalse(second.teacher.training)
        before = (state_dict_sha256(second.aligner), state_dict_sha256(second.teacher))
        optimizer = torch.optim.SGD(second.trainable_parameters(), lr=.01)
        second.assert_optimizer(optimizer)
        out = second(rr)
        self.assertIs(out["teacher"]["pan_aligned"], out["student"]["pan_aligned"])
        out["loss"].backward()
        optimizer.step()
        self.assertEqual(before, (state_dict_sha256(second.aligner), state_dict_sha256(second.teacher)))
        self.assertTrue(all(p.grad is None for p in second.aligner.parameters()))
        self.assertTrue(all(p.grad is None for p in second.teacher.parameters()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in second.student.parameters()))
        for name in ("e", "q", "w_e", "w_q"):
            self.assertFalse(out[name].requires_grad)
        torch.testing.assert_close(out["e"], (out["teacher"]["y"] - rr.gt).abs().mean(1, keepdim=True))
        self.assertEqual(scales.n_samples, 2)
        self.assertEqual(scales.n_e, 2 * 16 * 16)
        with self.assertRaisesRegex(ValueError, "trainable"):
            first(rr)

    def test_checkpoint_resume_inference_and_calibration_invalidation(self):
        first, rr, scales = self.calibrated()
        second = GlobalRRStage2(first, recon("S"), scales)
        saved = copy.deepcopy(second.state_dict())
        second.load_state_dict(saved)
        self.assertEqual(saved["_extra_state"]["method"], GLOBAL_RR_METHOD_VERSION)
        deploy = second.inference_model()
        obs = replace(rr, gt=None, lineage=replace(rr.lineage, split="inference"))
        self.assertEqual(deploy(obs)["y"].shape, rr.gt.shape)
        with self.assertRaises(NotImplementedError):
            deploy(replace(obs, lineage=replace(obs.lineage, resolution="FR", grid_name="PAN_FR")))
        with self.assertRaises(ValueError):
            second(replace(rr, lineage=replace(rr.lineage, split="test")))
        first2, _, scales2 = self.calibrated()
        with torch.no_grad():
            first2.aligner.fc2.bias.add_(.1)
        with self.assertRaisesRegex(ValueError, "stale"):
            GlobalRRStage2(first2, recon("S"), scales2)

    def test_stage2_rejects_forged_canonical_prior_even_with_matching_sha(self):
        first, rr, scales = self.calibrated()
        second = GlobalRRStage2(first, recon("S"), scales)
        # This forged vector is in the model's D4 orbit. Only lineage pinning,
        # not the Aligner's shape/magnitude/orbit guard, catches this forgery.
        forged = replace(rr, prior_dy_dx=torch.tensor([[.4, -.3], [.4, -.3]]),
                         lineage=replace(rr.lineage, canonical_prior_dy_dx=(.4, -.3)))
        with self.assertRaisesRegex(ValueError, "canonical prior"):
            second(forged)

    def test_real_global_observation_adapter_stage1_calibration_stage2(self):
        from .global_data import GlobalBatchLineage, GlobalObservationBatch, augment_global_batch
        fixture = batch()
        lineage = GlobalBatchLineage(split="train", resolution="RR",
            source_manifest_sha256=SOURCES.rr_manifest_sha256,
            prior_mapping_sha256=SOURCES.prior_mapping_sha256,
            sample_ids=fixture.lineage.sample_ids, grid_name="PAN_RR",
            geometry_evidence="Synthetic stored LMS, exact H/V then CCW augmentation",
            sensor="QB", train_source_sha256=SOURCES.train_source_sha256,
            canonical_prior_dy_dx=VECTOR, augmentations=((False, False, 0),) * 2)
        rr = GlobalObservationBatch(fixture.pan, fixture.ms, fixture.native_lms,
                                    fixture.prior_dy_dx, lineage, fixture.gt, fixture.lpan)
        rr = augment_global_batch(rr, hflip=True, vflip=False, rot90=1)
        first = stage1()
        out = first(rr)
        torch.testing.assert_close(out["native"]["correction"], rr.prior_dy_dx, rtol=0, atol=0)
        scales = calibrate_training(first, [rr], expected_sample_ids=rr.lineage.sample_ids)
        second = GlobalRRStage2(first, recon("S", "PLH"), scales)
        self.assertTrue(bool(torch.isfinite(second(rr)["loss"])))
        with self.assertRaisesRegex(ValueError, "recorded augmentation"):
            first.native_prediction(replace(rr, prior_dy_dx=fixture.prior_dy_dx))


if __name__ == "__main__":
    unittest.main()
