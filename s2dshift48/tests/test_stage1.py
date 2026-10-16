"""Synthetic CPU Stage1 objective/identity checks; no experiment is launched."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from directshift48.method import DirectShiftStage1
from directshift48.training import DirectStage1Trainer, build_direct_stage1
from donor30.masking import MaskedDonorStage1, masked_mean
from pa.warp import warp_pan, warp_support_mask
from panda_method.edge_data import EdgeBatchLineage, EdgeObservationBatch
from panda_method.reconstruction import ReconstructionSpec

from s2dshift48.aligner import METHOD_VERSION, boundary_manifest
from s2dshift48.stage1 import S2DMaskedStage1, S2DShiftStage1, build_s2d_stage1


def sources(sensor="GF2"):
    return SimpleNamespace(sensor=sensor, bands=8 if sensor == "WV3" else 4,
                           manifest_sha=lambda split: "a" * 64)


def build(mask="MASK2", sensor="GF2", baseline=False, **kwargs):
    src = sources(sensor)
    args = dict(teacher_spec=ReconstructionSpec(src.bands, 112, (1, 2, 3), "P0", "BICUBIC_MS"),
                teacher_seed=17, aligner_seed=29, epsilon_seed=31,
                lambda_abs=.25, lambda_eps=.003)
    args.update(kwargs)
    if baseline:
        stage = build_direct_stage1(src, **args)
        return MaskedDonorStage1.from_fresh_stage(stage) if mask == "MASK2" else stage
    return build_s2d_stage1(src, mask_policy=mask, **args)


def batch(sensor="GF2", side=64, split="train", resolution="RR", count=1):
    bands = sources(sensor).bands
    generator = torch.Generator().manual_seed(45 + side)
    shape = (count, bands, side, side)
    return EdgeObservationBatch(
        torch.rand(count, 1, side, side, generator=generator),
        torch.rand(count, bands, side // 4, side // 4, generator=generator),
        torch.rand(*shape, generator=generator),
        EdgeBatchLineage(split=split, resolution=resolution, sensor=sensor,
            source_manifest_sha256="a" * 64, sample_ids=tuple(f"x{i}" for i in range(count)),
            grid_name="PAN_" + resolution, augmentations=((False, False, 0),) * count),
        None if resolution == "FR" else torch.rand(*shape, generator=generator))


class Stage1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_existing_forward_reused_exactly_and_new_identity(self):
        self.assertIs(S2DShiftStage1.forward, DirectShiftStage1.forward)
        self.assertIs(S2DMaskedStage1.forward, MaskedDonorStage1.forward)
        for mask in ("UNMASKED", "MASK2"):
            stage = build(mask)
            manifest = stage.pipeline_manifest()
            self.assertEqual(manifest["method"], METHOD_VERSION)
            self.assertTrue(manifest["revision"].startswith("S2D"))
            self.assertEqual(manifest["mask_policy"], mask)
            self.assertEqual(stage.get_extra_state()["method"], METHOD_VERSION)
            self.assertEqual(manifest["boundary"]["local_convolution_receptive_field"], 44)
            self.assertEqual(stage.alignment_manifest()["boundary"], boundary_manifest())
            self.assertEqual(manifest["boundary"]["prediction_dependent_mask"], mask == "MASK2")

    def test_paired_teacher_initialization_epsilon_and_caller_rng(self):
        rng = torch.random.get_rng_state().clone()
        stage, old, rr = build(), build(baseline=True), batch()
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for key, value in stage.teacher.state_dict().items():
            self.assertTrue(torch.equal(value, old.teacher.state_dict()[key]), key)
        current, baseline = stage(rr), old(rr)
        self.assertTrue(torch.equal(current["epsilon_rr"], baseline["epsilon_rr"]))
        self.assertTrue(torch.equal(stage._epsilon_rng.get_state(), old._epsilon_rng.get_state()))
        # Both fresh heads predict zero: reconstruction and fixed spatial terms
        # are precisely equal despite the different feature architecture.
        for key in ("loss", "rec", "abs_rr", "eps_rr"):
            torch.testing.assert_close(current[key], baseline[key], rtol=0, atol=0)

    def test_mask_policy_changes_only_reconstruction_reduction(self):
        rr = batch()
        masked, unmasked = build(), build("UNMASKED")
        with torch.no_grad():
            for stage in (masked, unmasked):
                stage.aligner.fc2.bias.copy_(torch.tensor([.7, -.3]))
        left, right = masked(rr), unmasked(rr)
        support = warp_support_mask(64, 64, left["native"]["u"].detach())
        self.assertTrue(torch.equal(left["loss_mask"], support))
        self.assertFalse(left["loss_mask"].requires_grad)
        error = (left["native"]["y"] - rr.gt).abs()
        torch.testing.assert_close(left["rec"], masked_mean(error, support), rtol=0, atol=0)
        torch.testing.assert_close(right["rec"], error.mean(), rtol=0, atol=0)
        self.assertNotIn("loss_mask", right)
        for key in ("abs_rr", "eps_rr", "epsilon_rr"):
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        torch.testing.assert_close(left["loss"], left["rec"] + .25*left["abs_rr"] + .003*left["eps_rr"],
                                   rtol=0, atol=0)

    def test_gradient_routes_native_epsilon_target_detached(self):
        for mask in ("MASK2", "UNMASKED"):
            stage, rr = build(mask), batch()
            # Existing named-fresh Teacher zero-initializes its final residual
            # convolution. L_rec -> A is consequently zero on the very first
            # update, not disconnected. Move that layer off zero to test the
            # nondegenerate reconstruction route without changing production.
            with torch.no_grad():
                stage.teacher.backbone.output[-1].weight.normal_(
                    0., .001, generator=torch.Generator().manual_seed(98))
            out = stage(rr)
            params = list(stage.aligner.parameters()) + list(stage.teacher.parameters())
            count_a = len(list(stage.aligner.parameters()))
            for term in ("rec", "abs_rr", "eps_rr"):
                gradients = torch.autograd.grad(out[term], params, retain_graph=True, allow_unused=True)
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients[:count_a] if g is not None), 0)
                teacher_norm = sum(float(g.abs().sum()) for g in gradients[count_a:] if g is not None)
                self.assertGreater(teacher_norm, 0) if term == "rec" else self.assertEqual(teacher_norm, 0)
            native_grad = torch.autograd.grad(out["eps_rr"], out["native"]["u"],
                                              retain_graph=True, allow_unused=True)[0]
            self.assertIsNone(native_grad)
            self.assertFalse(out["epsilon_rr"].requires_grad)

    def test_original_pan_one_native_warp_one_teacher_lms_not_shifted(self):
        for mask in ("UNMASKED", "MASK2"):
            stage, rr, seen = build(mask), batch(), []
            hook = stage.aligner.register_forward_pre_hook(lambda module, args: seen.append(args))
            with patch("directshift48.method.warp_pan", wraps=warp_pan) as native_warp, \
                    patch("donor30.masking.warp_pan", wraps=warp_pan) as masked_warp, \
                    patch.object(stage.teacher, "forward", wraps=stage.teacher.forward) as teacher:
                result = stage(rr)
            hook.remove()
            self.assertEqual(teacher.call_count, 1)
            self.assertEqual(native_warp.call_count + masked_warp.call_count, 2)
            self.assertIs(native_warp.call_args_list[0].args[0], rr.pan)
            self.assertEqual(len(seen), 2)
            self.assertIs(seen[0][0], rr.pan)
            self.assertIs(seen[0][1], rr.native_lms)
            self.assertIs(seen[1][1], rr.native_lms)
            self.assertIs(teacher.call_args.args[2], rr.native_lms)
            self.assertEqual(result["native"]["pan_aligned"].shape, rr.pan.shape)

    def test_prediction_gt_independent_and_no_fr_shift_rescale(self):
        stage, rr = build(), batch()
        with torch.no_grad():
            stage.aligner.fc2.bias.copy_(torch.tensor([.5, -.75]))
            original = stage.native_prediction(rr)
            changed = stage.native_prediction(replace(rr, gt=rr.gt + 100))
        self.assertTrue(torch.equal(original["y"], changed["y"]))
        for resolution, side in (("RR", 64), ("FR", 512)):
            inputs = batch(side=side, resolution=resolution, split="test")
            out = stage.predict_shift(inputs.pan, inputs.native_lms, resolution=resolution)
            torch.testing.assert_close(out["u"], torch.tensor([[.5, -.75]]), rtol=0, atol=0)

    def test_old_new_and_mask_checkpoint_cross_load_rejected_before_mutation(self):
        fresh, old, unmasked = build(), build(baseline=True), build("UNMASKED")
        for target, source in ((fresh, old), (old, fresh), (fresh, unmasked), (unmasked, fresh)):
            before = target.aligner.fc2.bias.detach().clone()
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                target.load_state_dict(source.state_dict(), strict=True)
            self.assertTrue(torch.equal(before, target.aligner.fc2.bias))
        restored = build()
        restored.load_state_dict(fresh.state_dict(), strict=True)
        with self.assertRaisesRegex(ValueError, "strict"):
            restored.load_state_dict(fresh.state_dict(), strict=False)

    def test_wrap_only_fresh_s2d_and_explicit_architecture_policy(self):
        with self.assertRaises(ValueError):
            S2DMaskedStage1.from_fresh_stage(build(baseline=True))
        with self.assertRaises(ValueError):
            S2DMaskedStage1.from_fresh_stage(build())
        stage = build("UNMASKED")
        self.assertIs(S2DMaskedStage1.from_fresh_stage(stage).aligner, stage.aligner)
        stage.forward_calls = 1
        with self.assertRaises(ValueError):
            S2DMaskedStage1.from_fresh_stage(stage)
        for mask in ("", None, "AUTO"):
            with self.assertRaises(ValueError):
                build(mask)
        with self.assertRaises(ValueError):
            build(teacher_spec=ReconstructionSpec(4, 104, (1, 2, 2), "P0", "BICUBIC_MS"))
        old = build(baseline=True)
        with self.assertRaises(TypeError):
            S2DShiftStage1(old.aligner, old.teacher, sources=old.sources, epsilon_seed=31)

    def test_source_dtype_and_support_fail_without_advancing_epsilon(self):
        stage, rr = build(), batch()
        rng = stage._epsilon_rng.get_state().clone()
        invalid = [replace(rr, lineage=replace(rr.lineage, split="test")),
                   replace(rr, lineage=replace(rr.lineage, source_manifest_sha256="b"*64)),
                   replace(rr, pan=rr.pan.double(), ms=rr.ms.double(),
                           native_lms=rr.native_lms.double(), gt=rr.gt.double())]
        for candidate in invalid:
            with self.assertRaises(ValueError):
                stage(candidate)
            self.assertTrue(torch.equal(rng, stage._epsilon_rng.get_state()))
            self.assertEqual(stage.forward_calls, 0)
        with torch.no_grad():
            stage.aligner.fc2.bias.copy_(torch.tensor([10., 0.]))
        with self.assertRaisesRegex(ValueError, "SUPPORT_VIOLATION"):
            stage(rr)
        self.assertTrue(torch.equal(rng, stage._epsilon_rng.get_state()))
        self.assertEqual(stage.forward_calls, 0)
        invalid_support = torch.ones_like(rr.pan, dtype=torch.bool)
        invalid_support[..., 8, 8] = False
        with self.assertRaisesRegex(ValueError, "Synthetic PAN support"):
            stage.predict_shift(rr.pan, rr.native_lms, resolution="RR", pan_valid_mask=invalid_support)

    def test_existing_optimizer_engine_update_and_strict_resume(self):
        stage, rr = build(), batch(side=32)
        optimizer = torch.optim.SGD(stage.parameters(), lr=1e-4, momentum=.9)
        engine = DirectStage1Trainer(stage, optimizer, gradient_log_interval=100)
        result = engine.step(rr)
        self.assertEqual(engine.updates, 1)
        self.assertGreater(result["optimizer_update_norm"]["aligner"], 0)
        self.assertGreater(result["optimizer_update_norm"]["teacher"], 0)
        self.assertEqual(result["term_gradient_norm"]["weighted_abs"]["teacher"], 0)
        self.assertEqual(result["term_gradient_norm"]["weighted_eps_rr"]["teacher"], 0)
        state = engine.state_dict()
        new_stage = build()
        resumed = DirectStage1Trainer(new_stage, torch.optim.SGD(new_stage.parameters(), lr=1e-4, momentum=.9),
                                      gradient_log_interval=100)
        resumed.load_state_dict(state)
        self.assertEqual(engine.step(rr), resumed.step(rr))
        for p, q in zip(stage.parameters(), new_stage.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
