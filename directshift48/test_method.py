"""Synthetic CPU numerical contracts; never runs a campaign or downloads data."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from model.pancrafter_paper import ChannelLayerNorm, ResBlock
from pa.warp import warp_pan, warp_support_mask
from panda_method.edge_data import EdgeBatchLineage, EdgeObservationBatch
from panda_method.reconstruction import ReconstructionSpec, build_reconstructor
from .method import (DirectPANAligner, DirectShiftStage1, DirectTrainingSources,
                     ScharrAbsolute, boundary_manifest, epsilon_loss, sample_epsilon)

torch.set_num_threads(1)


def batch(*, side=32, sensor="GF2", batch_size=2, resolution="RR", split="train"):
    bands = 8 if sensor == "WV3" else 4
    rng = torch.Generator().manual_seed(110 + side)
    pan = torch.rand(batch_size, 1, side, side, generator=rng)
    ms = torch.rand(batch_size, bands, side//4, side//4, generator=rng)
    lms = torch.rand(batch_size, bands, side, side, generator=rng)
    gt = None if resolution == "FR" else torch.rand(batch_size, bands, side, side, generator=rng)
    lineage = EdgeBatchLineage(split=split, resolution=resolution, sensor=sensor,
        source_manifest_sha256="a"*64, sample_ids=tuple("s%d" % i for i in range(batch_size)),
        grid_name="PAN_"+resolution, augmentations=((False, False, 0),)*batch_size)
    return EdgeObservationBatch(pan, ms, lms, lineage, gt)


def model(sensor="GF2", **kwargs):
    bands = 8 if sensor == "WV3" else 4
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(971)
        aligner = DirectPANAligner(bands)
    teacher = build_reconstructor(ReconstructionSpec(bands, 8, (1, 1, 1), "P0", "BICUBIC_MS"),
                                  seed=123, role="T")
    return DirectShiftStage1(aligner, teacher, sources=DirectTrainingSources(sensor, "a"*64),
                            epsilon_seed=234, **kwargs)


class DirectAlignerTests(unittest.TestCase):
    def test_exact_architecture_zero_total_and_common_ln_block(self):
        stage, rr = model(), batch()
        a = stage.aligner
        self.assertIs(type(a.res), ResBlock)
        self.assertIsInstance(a.res.in_layers[0], ChannelLayerNorm)
        self.assertIsInstance(a.res.out_layers[0], ChannelLayerNorm)
        self.assertIsNone(a.res.mod)
        self.assertIsNone(a.res.se)
        self.assertTrue(torch.equal(a.res.out_layers[-1].weight, torch.zeros_like(a.res.out_layers[-1].weight)))
        self.assertEqual(a.pan_stem[2].num_groups, 4)
        self.assertEqual(a.joint2[2].num_groups, 8)
        self.assertIsNone(a.pan_stem[1].bias)
        self.assertEqual(tuple(a.fc1.weight.shape), (32, 64))
        u, details = a(rr.pan, rr.native_lms, return_details=True)
        torch.testing.assert_close(u, torch.zeros(2, 2), rtol=0, atol=0)
        self.assertEqual(details["head_input"].shape, (2, 64))
        self.assertFalse(hasattr(stage, "matcher"))
        self.assertFalse(any("prior" in key or "residual_revision" in key for key in a.state_dict()))

    def test_input_crop_precedes_spatial_normalization_and_GN(self):
        a, rr = model().aligner, batch()
        with torch.no_grad():
            a.fc2.weight.fill_(.03)
            p, m = rr.pan.clone(), rr.native_lms.clone()
            p[..., :8, :] = 1e6
            p[..., -8:, :] = -1e6
            p[..., :, :8] = 2e6
            p[..., :, -8:] = -2e6
            m[..., :8, :] = -5e6
            expected, original = a(rr.pan, rr.native_lms, return_details=True)
            actual, altered = a(p, m, return_details=True)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(original["features"], altered["features"], rtol=0, atol=0)
        p = rr.pan.clone().requires_grad_()
        a(p, rr.native_lms).sum().backward()
        self.assertEqual(float(p.grad[..., :8, :].abs().sum()), 0.)
        self.assertGreater(float(p.grad[..., 8:-8, 8:-8].abs().sum()), 0.)
        self.assertEqual(boundary_manifest()["local_convolution_receptive_field"], 25)

    def test_current_grid_output_unbounded_and_no_four_times_FR(self):
        stage = model()
        with torch.no_grad():
            stage.aligner.fc2.bias.copy_(torch.tensor([4.5, -5.25]))
        for side, resolution in ((64, "RR"), (256, "RR"), (512, "FR")):
            rr = batch(side=side, batch_size=1, resolution=resolution, split="test")
            result = stage.predict_shift(rr.pan, rr.native_lms, resolution=resolution)
            torch.testing.assert_close(result["u"], torch.tensor([[4.5, -5.25]]), rtol=0, atol=0)

    def test_legacy_or_different_semantics_checkpoint_rejected(self):
        a = model().aligner
        state = copy.deepcopy(a.state_dict())
        del state["_direct_shift_identity"]
        with self.assertRaisesRegex(RuntimeError, "checkpoint identity"):
            a.load_state_dict(state, strict=False)
        with self.assertRaises(ValueError):
            DirectPANAligner(4, view_margin=4)

    def test_actual_W112_d123_lane_config_build_train64_native256_512(self):
        from . import plan
        from .evaluation import configured_pipeline
        from .runtime import build_trainer
        root = Path(__file__).resolve().parents[1]
        for server, sensor in (("s1", "WV3"), ("s3", "GF2")):
            baseline = json.loads((root/"config"/"directshift48"/(server+".baseline.json")).read_text())
            config = plan.resolve_case(server, baseline=baseline)
            sources = SimpleNamespace(sensor=sensor, bands=8 if sensor == "WV3" else 4,
                                      manifest_sha=lambda split: "a"*64)
            engine = build_trainer(config, sources, "cpu")
            pipeline = configured_pipeline(engine, config, sources)
            self.assertEqual(pipeline["teacher"]["width"], 112)
            self.assertEqual(tuple(pipeline["teacher"]["depth"]), (1, 2, 3))
            self.assertEqual(engine.stage.alignment_manifest(), plan.alignment_protocol_manifest(baseline))
            self.assertEqual(engine.effective_batch_size, 48)
            self.assertEqual(engine.microbatch_size, 12)
            rr = batch(side=64, sensor=sensor, batch_size=1)
            out = engine.stage(rr)
            self.assertTrue(bool(torch.isfinite(out["loss"])))
            out["loss"].backward()
            self.assertGreater(float(engine.stage.aligner.fc2.weight.grad.abs().sum()), 0.)
            engine.stage.zero_grad(set_to_none=True)
            del out, rr
            for side, resolution in ((256, "RR"), (512, "FR")):
                test = batch(side=side, sensor=sensor, batch_size=1, resolution=resolution, split="test")
                prediction = engine.stage.native_prediction(test)
                self.assertEqual(tuple(prediction["y"].shape), (1, sources.bands, side, side))
                self.assertFalse(prediction["y"].requires_grad)
                self.assertTrue(bool(torch.isfinite(prediction["y"]).all()))
                self.assertEqual(tuple(prediction["u"].shape), (1, 2))
                del test, prediction
            del engine


class ScharrAndWarpTests(unittest.TestCase):
    def test_fixed_scharr_scale_axes_RGB_order(self):
        yy, xx = torch.meshgrid(torch.arange(32.), torch.arange(32.), indexing="ij")
        plane = (2*xx + 3*yy)[None, None]
        spatial = ScharrAbsolute("WV3")
        edges = spatial.descriptor(plane)
        torch.testing.assert_close(edges[:, 0, 1:-1, 1:-1], torch.full((1, 30, 30), 2.), rtol=0, atol=0)
        torch.testing.assert_close(edges[:, 1, 1:-1, 1:-1], torch.full((1, 30, 30), 3.), rtol=0, atol=0)
        self.assertEqual(spatial.rgb_indices, (4, 2, 1))
        self.assertEqual(ScharrAbsolute("GF2").rgb_indices, (2, 1, 0))
        self.assertEqual(ScharrAbsolute("QB").rgb_indices, (2, 1, 0))
        ms = torch.zeros(1, 8, 32, 32)
        ms[:, 4] = xx
        ms[:, 2] = -xx
        ms[:, 0] = 1e5 * yy  # Coastal must never masquerade as RGB.
        self.assertEqual(float(spatial.descriptor(spatial.rgb_luminance(ms)).sum()), 0.)

    def test_absolute_reduction_is_mean_of_two_axes_not_sum(self):
        yy, xx = torch.meshgrid(torch.arange(32.), torch.arange(32.), indexing="ij")
        loss = ScharrAbsolute("GF2").absolute_loss((2*xx+4*yy)[None, None], torch.zeros(1, 4, 32, 32))
        self.assertEqual(float(loss), 3.)

    def test_warp_sign_and_subpixel_gradient_finite_difference(self):
        yy, xx = torch.meshgrid(torch.arange(32., dtype=torch.float64), torch.arange(32., dtype=torch.float64), indexing="ij")
        pan = (yy.square()*.001+xx*.03)[None, None]
        offset = torch.tensor([[.37, -.23]], dtype=torch.float64, requires_grad=True)
        value = warp_pan(pan, offset)[..., 8:-8, 8:-8].mean()
        gradient = torch.autograd.grad(value, offset)[0]
        for axis in (0, 1):
            step = torch.zeros_like(offset)
            step[0, axis] = 1e-5
            plus = warp_pan(pan, offset.detach()+step)[..., 8:-8, 8:-8].mean()
            minus = warp_pan(pan, offset.detach()-step)[..., 8:-8, 8:-8].mean()
            self.assertAlmostEqual(float(gradient[0, axis]), float((plus-minus)/2e-5), places=7)
        shifted = warp_pan(pan, torch.tensor([[0., 1.]], dtype=torch.float64))
        torch.testing.assert_close(shifted[..., 8:-8, 8:-8], pan[..., 8:-8, 9:-7], rtol=0, atol=1e-12)


class EpsilonAndStageTests(unittest.TestCase):
    def test_area_uniform_R3_rng_partitioning_and_probabilities(self):
        before = torch.random.get_rng_state().clone()
        a, b = torch.Generator().manual_seed(82), torch.Generator().manual_seed(82)
        all_eps = sample_epsilon(60000, a)
        parts = torch.cat([sample_epsilon(123, b), sample_epsilon(59877, b)])
        torch.testing.assert_close(all_eps, parts, rtol=0, atol=0)
        torch.testing.assert_close(before, torch.random.get_rng_state(), rtol=0, atol=0)
        radius = all_eps.norm(dim=1)
        self.assertLessEqual(float(radius.max()), 3.000001)
        self.assertAlmostEqual(float((radius <= 1).float().mean()), 1/9, delta=.005)
        self.assertAlmostEqual(float((radius <= 2).float().mean()), 4/9, delta=.005)
        self.assertAlmostEqual(float(radius.square().mean()), 4.5, delta=.04)
        self.assertLess(float(all_eps.mean(0).abs().max()), .025)

    def test_R3_extremes_support_and_branch_crop_equal(self):
        stage, rr = model(), batch(side=64, batch_size=4)
        eps = torch.tensor([[3., 0.], [-3., 0.], [0., 3.], [0., -3.]])
        mask = warp_support_mask(64, 64, eps)
        self.assertTrue(bool(mask[..., 8:-8, 8:-8].all()))
        stage.predict_shift(warp_pan(rr.pan, eps), rr.native_lms, resolution="RR", pan_valid_mask=mask)
        mask[..., 8, 8] = False
        with self.assertRaisesRegex(ValueError, "fixed clean"):
            stage.predict_shift(rr.pan, rr.native_lms, resolution="RR", pan_valid_mask=mask)

    def test_epsilon_anchor_stopgrad_and_exact_sign(self):
        native = torch.tensor([[.3, -.2]], requires_grad=True)
        eps = torch.tensor([[1., -2.]])
        shifted = (native.detach()-eps).requires_grad_()
        self.assertAlmostEqual(float(epsilon_loss(shifted, native, eps)), 0., places=7)
        loss = epsilon_loss(shifted+.1, native, eps)
        gs, gn = torch.autograd.grad(loss, (shifted, native), allow_unused=True)
        self.assertIsNone(gn)
        torch.testing.assert_close(gs, torch.full_like(gs, .5), rtol=0, atol=0)

    def test_one_teacher_two_aligners_two_warps_no_prewarp(self):
        stage, rr = model(), batch()
        with patch.object(stage.teacher, "forward", wraps=stage.teacher.forward) as teacher, \
             patch.object(stage.aligner, "forward", wraps=stage.aligner.forward) as aligner, \
             patch("directshift48.method.warp_pan", wraps=warp_pan) as warp:
            out = stage(rr)
        self.assertEqual(teacher.call_count, 1)
        self.assertEqual(aligner.call_count, 2)
        self.assertEqual(warp.call_count, 2)
        self.assertIs(aligner.call_args_list[0].args[0], rr.pan)
        self.assertIs(aligner.call_args_list[0].args[1], rr.native_lms)
        self.assertIs(aligner.call_args_list[1].args[1], rr.native_lms)
        self.assertFalse({"b", "delta", "pan_pre", "match_diagnostics"} & set(out["native"]))
        torch.testing.assert_close(out["loss"], out["rec"]+out["abs_rr"]+.001*out["eps_rr"])

    def test_exact_parameter_routes_and_two_zero_init_gradient_stages(self):
        stage, rr = model(), batch()
        out = stage(rr)
        a, t = list(stage.aligner.parameters()), list(stage.teacher.parameters())
        for term in ("abs_rr", "eps_rr"):
            grads = torch.autograd.grad(out[term], a+t, retain_graph=True, allow_unused=True)
            self.assertTrue(all(g is None for g in grads[len(a):]))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads[:len(a)] if g is not None), 0.)
        grads = torch.autograd.grad(out["rec"], a+t, retain_graph=True, allow_unused=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in grads[len(a):] if g is not None), 0.)
        # The preserved Teacher has a zero-init output conv, so rec cannot yet
        # reach PAN/Aligner at update0. The route must open after its first step.
        self.assertEqual(sum(float(g.abs().sum()) for g in grads[:len(a)] if g is not None), 0.)
        out["loss"].backward()
        self.assertGreater(float(stage.aligner.fc2.weight.grad.abs().sum()), 0.)
        self.assertEqual(float(stage.aligner.fc1.weight.grad.abs().sum()), 0.)
        self.assertEqual(float(stage.aligner.pan_stem[1].weight.grad.abs().sum()), 0.)
        with torch.no_grad():
            stage.aligner.fc2.weight.add_(-.01*stage.aligner.fc2.weight.grad)
            for p in stage.teacher.parameters():
                if p.grad is not None:
                    p.add_(-.01*p.grad)
        stage.zero_grad(set_to_none=True)
        out2 = stage(rr)
        rec_a = torch.autograd.grad(out2["rec"], a, retain_graph=True, allow_unused=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in rec_a if g is not None), 0.)
        out2["loss"].backward()
        self.assertGreater(float(stage.aligner.fc1.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(stage.aligner.pan_stem[1].weight.grad.abs().sum()), 0.)

    def test_FR_and_heldout_inference_never_build_training_graph(self):
        stage = model()
        for resolution, split in (("FR", "test"), ("RR", "validation")):
            heldout = batch(resolution=resolution, split=split)
            output = stage.native_prediction(heldout)
            self.assertFalse(output["y"].requires_grad)
            self.assertFalse(output["u"].requires_grad)
            with self.assertRaises(ValueError):
                stage(heldout)

    def test_fixed_support_violation_no_clamp_or_rng_advance(self):
        stage, rr = model(), batch()
        with torch.no_grad():
            stage.aligner.fc2.bias.fill_(15.)
        before = stage._epsilon_rng.get_state().clone()
        with self.assertRaisesRegex(ValueError, "SUPPORT_VIOLATION"):
            stage(rr)
        self.assertTrue(torch.equal(before, stage._epsilon_rng.get_state()))
        self.assertEqual(stage.forward_calls, 0)
        self.assertEqual(float(stage.predict_shift(rr.pan, rr.native_lms, resolution="RR")["u"].min()), 15.)

    def test_checkpoint_rng_exact_resume_and_semantics_rejection(self):
        stage, rr = model(), batch()
        stage(rr)
        state = copy.deepcopy(stage.state_dict())
        restored = model()
        restored.load_state_dict(state)
        torch.testing.assert_close(stage(rr)["epsilon_rr"], restored(rr)["epsilon_rr"], rtol=0, atol=0)
        changed = copy.deepcopy(state)
        changed["_extra_state"]["pipeline"]["eps"]["radius"] = 2.
        with self.assertRaisesRegex(ValueError, "pipeline"):
            restored.load_state_dict(changed)
        with self.assertRaises(ValueError):
            restored.load_state_dict(state, strict=False)
        json.dumps(stage.pipeline_manifest(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
