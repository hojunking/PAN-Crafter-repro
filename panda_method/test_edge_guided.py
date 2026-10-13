"""Synthetic CPU Stage1 contracts, not experiments or real-data radius selection."""
from __future__ import annotations

import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from model.pancrafter_paper import ChannelLayerNorm, ResBlock
from pa.aligner import PANGlobalAligner
from pa.warp import warp_pan, warp_support_mask
from .calibration import state_dict_sha256
from .edge_data import EdgeBatchLineage, EdgeObservationBatch, augment_edge_batch
from .edge_guided import (METHOD_VERSION, EdgeGuidedStage1, EdgeTrainingSources,
                          ResidualPANAligner)
from .edge_match import EdgeMatchSpec, EdgeMatcher
from .geometry import SceneAwarePANAligner
from .global_rr import GlobalPANAligner
from .reconstruction import ReconstructionSpec, build_reconstructor


torch.set_num_threads(1)
SOURCES = EdgeTrainingSources("QB", "a" * 64)


def batch(*, side=32, resolution="RR", split="train", aligned=False, batch_size=2):
    g = torch.Generator().manual_seed(190 + side)
    pan = torch.rand(batch_size, 1, side, side, generator=g)
    ms = torch.rand(batch_size, 4, side//4, side//4, generator=g)
    if aligned:
        lms = warp_pan(pan, torch.tensor([[2., -1.]]).expand(batch_size, 2)).repeat(1, 4, 1, 1)
    else:
        lms = torch.rand(batch_size, 4, side, side, generator=g)
    lineage = EdgeBatchLineage(split=split, resolution=resolution,
        source_manifest_sha256=SOURCES.rr_manifest_sha256,
        sample_ids=tuple(f"synthetic-{i}" for i in range(batch_size)), sensor="QB",
        grid_name="PAN_" + resolution, augmentations=((False, False, 0),) * batch_size)
    gt = torch.rand(batch_size, 4, side, side, generator=g) if resolution == "RR" else None
    return EdgeObservationBatch(pan, ms, lms, lineage, gt)


def matcher(**changes):
    kwargs = dict(sensor="QB", radius_rr=2, radius_fr=3,
                  tie_policy="min_norm_then_lexicographic", low_texture_threshold=0.,
                  low_texture_policy="zero", boundary_policy="report")
    kwargs.update(changes)
    return EdgeMatcher(EdgeMatchSpec(**kwargs))


def teacher(*, layout="P0", base="BICUBIC_MS"):
    return build_reconstructor(ReconstructionSpec(4, 8, (1, 1, 1), layout, base), seed=41, role="T")


def stage(**changes):
    args = dict(sources=SOURCES, roi_margin=3, boundary_policy="report", epsilon_seed=123)
    args.update(changes)
    return EdgeGuidedStage1(ResidualPANAligner(4, view_margin=0), teacher(), matcher(), **args)


class ResidualCNNTests(unittest.TestCase):
    def test_structure_fc64_zero_residual_no_prior_and_common_ln(self):
        a, rr = ResidualPANAligner(4, view_margin=0), batch()
        self.assertIs(type(a.res), ResBlock)
        self.assertIsInstance(a.res.in_layers[0], ChannelLayerNorm)
        self.assertIsInstance(a.res.out_layers[0], ChannelLayerNorm)
        self.assertTrue(any(isinstance(m, nn.GroupNorm) for m in a.pan_stem.modules()))
        self.assertEqual(tuple(a.fc1.weight.shape), (32, 64))
        self.assertFalse(hasattr(a, "canonical_prior_dy_dx"))
        seen = []
        hook = a.fc1.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
        delta, detail = a(rr.pan, rr.native_lms, return_details=True)
        hook.remove()
        torch.testing.assert_close(delta, torch.zeros(2, 2), rtol=0, atol=0)
        self.assertEqual(tuple(seen[0].shape), (2, 64))
        self.assertIs(detail["head_input"], detail["features"])
        # Double zero initialization creates staged gradients, not absent routes.
        delta.sum().backward()
        self.assertGreater(float(a.fc2.weight.grad.abs().sum()), 0.)
        self.assertEqual(float(a.fc1.weight.grad.abs().sum()), 0.)
        with torch.no_grad():
            a.fc2.weight.normal_(std=.02)
        a.zero_grad(set_to_none=True)
        a(rr.pan, rr.native_lms).square().mean().backward()
        self.assertGreater(float(a.fc1.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(a.res.out_layers[-1].weight.grad.abs().sum()), 0.)
        self.assertEqual(float(a.res.in_layers[2].weight.grad.abs().sum()), 0.)
        with torch.no_grad():
            a.res.out_layers[-1].weight.normal_(std=.01)
        a.zero_grad(set_to_none=True)
        a(rr.pan, rr.native_lms).square().mean().backward()
        self.assertGreater(float(a.res.in_layers[2].weight.grad.abs().sum()), 0.)

    def test_legacy_scene_global_and_different_view_checkpoint_rejected_before_mutation(self):
        a = ResidualPANAligner(4, view_margin=0)
        legacy = [PANGlobalAligner(4), SceneAwarePANAligner(4, view_margin=0),
                  GlobalPANAligner(4, view_margin=0, sensor="QB", train_source_sha256="b"*64,
                    prior_asset_sha256="c"*64, canonical_prior_dy_dx=(.3, -.4),
                    min_prior_magnitude=.01, max_prior_magnitude=4.),
                  ResidualPANAligner(4, view_margin=1)]
        before = state_dict_sha256(a)
        for old in legacy:
            for strict in (True, False):
                with self.assertRaisesRegex(RuntimeError, "checkpoint identity"):
                    a.load_state_dict(old.state_dict(), strict=strict)
                self.assertEqual(before, state_dict_sha256(a))
        a.load_state_dict(copy.deepcopy(a.state_dict()))
        self.assertEqual(before, state_dict_sha256(a))

    def test_cartesian_residual_unclamped_and_zero_b_legal(self):
        model, rr = stage(), batch()
        constant = replace(rr, pan=torch.ones_like(rr.pan), native_lms=torch.ones_like(rr.native_lms))
        with torch.no_grad():
            model.aligner.fc2.bias.copy_(torch.tensor([-.25, .75]))
        out = model.native_prediction(constant)
        torch.testing.assert_close(out["b"], torch.zeros(2, 2), rtol=0, atol=0)
        torch.testing.assert_close(out["delta"], torch.tensor([[-.25, .75]]).expand(2, 2))
        torch.testing.assert_close(out["u"], out["delta"], rtol=0, atol=0)


class EdgeStage1Tests(unittest.TestCase):
    def test_positive_scalar_sweep_and_checkpoint_binding(self):
        model = stage(lambda_abs=.3, lambda_eps=.0001)
        out = model(batch())
        torch.testing.assert_close(out['weighted_abs'], .3*out['abs_rr'])
        torch.testing.assert_close(out['weighted_eps_rr'], .0001*out['eps_rr'])
        torch.testing.assert_close(out['loss'], out['rec']+out['weighted_abs']+out['weighted_eps_rr'])
        with self.assertRaisesRegex(ValueError, 'pipeline'):
            stage().load_state_dict(model.state_dict())
        for value in (0., -1., float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                stage(lambda_abs=value)
            with self.assertRaises(ValueError):
                stage(lambda_eps=value)

    def test_exact_rr_objective_two_matches_two_cnns_one_u(self):
        model, rr = stage(), batch(aligned=True)
        mcalls, acalls, tcalls = [], [], []
        hm = model.matcher.register_forward_pre_hook(lambda _, args, kw: mcalls.append((args, kw)),
                                                      with_kwargs=True)
        ha = model.aligner.register_forward_pre_hook(lambda _, args: acalls.append(args))
        ht = model.teacher.register_forward_pre_hook(lambda _, args: tcalls.append(args))
        out = model(rr)
        hm.remove(); ha.remove(); ht.remove()
        self.assertEqual((len(mcalls), len(acalls), len(tcalls)), (2, 2, 1))
        torch.testing.assert_close(out["loss"], out["rec"] + out["abs_rr"] + .001*out["eps_rr"])
        torch.testing.assert_close(out["native"]["u"], out["native"]["b"], rtol=0, atol=0)
        torch.testing.assert_close(out["shifted"]["u"], out["shifted"]["b"], rtol=0, atol=0)
        torch.testing.assert_close(mcalls[0][0][0], rr.pan, rtol=0, atol=0)
        torch.testing.assert_close(mcalls[1][0][0], warp_pan(rr.pan, out["epsilon_rr"]), rtol=0, atol=0)
        self.assertIsNone(mcalls[0][1]["pan_valid_mask"])
        torch.testing.assert_close(mcalls[1][1]["pan_valid_mask"],
                                   warp_support_mask(*rr.pan.shape[-2:], out["epsilon_rr"]))
        for args, kw in mcalls:
            torch.testing.assert_close(args[1], rr.native_lms, rtol=0, atol=0)
            self.assertEqual(kw["resolution"], "RR")
            self.assertEqual(len(args), 2)  # Matcher sees no GT/epsilon/prior label.
        for i, branch in enumerate((out["native"], out["shifted"])):
            torch.testing.assert_close(acalls[i][0], branch["pan_pre"])
            self.assertFalse(branch["b"].requires_grad)
            self.assertFalse(branch["pan_pre"].requires_grad)
        direct = model.matcher(warp_pan(rr.pan, out["epsilon_rr"]), rr.native_lms, resolution="RR",
                               pan_valid_mask=warp_support_mask(*rr.pan.shape[-2:], out["epsilon_rr"]))
        torch.testing.assert_close(out["shifted"]["b"], direct.b, rtol=0, atol=0)
        self.assertFalse(torch.equal(out["shifted"]["b"], out["native"]["b"]-out["epsilon_rr"]))
        self.assertIs(tcalls[0][0], out["native"]["pan_aligned"])
        self.assertLessEqual(float(out["epsilon_rr"].norm(dim=1).max()), 2.)
        self.assertNotIn("eps_fr", out)
        with self.assertRaises(TypeError):
            model(rr, rr)

    def test_final_warp_is_single_original_not_prewarped_pan(self):
        model, rr = stage(), batch(aligned=True)
        with torch.no_grad():
            model.aligner.fc2.bias.copy_(torch.tensor([-.45, .35]))
        with patch("panda_method.edge_guided.warp_pan", wraps=warp_pan) as warper:
            out = model.native_prediction(rr)
        self.assertEqual(warper.call_count, 2)
        self.assertIs(warper.call_args_list[0].args[0], rr.pan)
        self.assertIs(warper.call_args_list[1].args[0], rr.pan)
        torch.testing.assert_close(out["b"], torch.tensor([[2., -1.]]).expand(2, 2))
        expected = warp_pan(rr.pan, out["b"]+out["delta"])
        double = warp_pan(out["pan_pre"], out["delta"])
        torch.testing.assert_close(out["pan_aligned"], expected, rtol=0, atol=0)
        self.assertGreater(float((expected-double).abs().max()), .01)
        bicubic = F.interpolate(rr.ms, scale_factor=4, mode="bicubic", align_corners=False)
        torch.testing.assert_close(out["ms_base"], bicubic, rtol=0, atol=0)
        self.assertFalse(torch.equal(out["ms_base"], rr.native_lms))

    def test_loss_routes_and_native_epsilon_target_stop_only(self):
        for term in ("rec", "abs_rr", "eps_rr"):
            model = stage()
            with torch.no_grad():
                model.teacher.backbone.output[-1].weight.normal_(std=.01)
            out = model(batch())
            out["native"]["u"].retain_grad()
            out["shifted"]["u"].retain_grad()
            out[term].backward()
            ag = sum(float(p.grad.abs().sum()) for p in model.aligner.parameters() if p.grad is not None)
            tg = [p.grad for p in model.teacher.parameters() if p.grad is not None]
            self.assertGreater(ag, 0., term)
            self.assertEqual(len(list(model.matcher.parameters())), 0)
            if term == "rec":
                self.assertGreater(sum(float(g.abs().sum()) for g in tg), 0.)
            else:
                self.assertFalse(tg)
            if term == "eps_rr":
                self.assertIsNone(out["native"]["u"].grad)
                self.assertGreater(float(out["shifted"]["u"].grad.abs().sum()), 0.)
            else:
                self.assertGreater(float(out["native"]["u"].grad.abs().sum()), 0.)
                self.assertIsNone(out["shifted"]["u"].grad)

    def test_same_scharr_absolute_loss_no_teacher_and_gt_does_not_change_b(self):
        rr, model = batch(), stage()
        native = model.native_prediction(rr)
        alternate = model.native_prediction(replace(rr, gt=1.-rr.gt))
        torch.testing.assert_close(native["b"], alternate["b"], rtol=0, atol=0)
        torch.testing.assert_close(native["u"], alternate["u"], rtol=0, atol=0)
        out = model(rr)
        direct = model.matcher.edges.absolute_loss(out["native"]["pan_aligned"], rr.gt, roi_margin=3)
        torch.testing.assert_close(out["abs_rr"], direct, rtol=0, atol=0)

    def test_fr_and_rr_whole_input_inference_current_grid_no_fourfold(self):
        model = stage()
        with torch.no_grad():
            model.aligner.fc2.bias.copy_(torch.tensor([.25, -.5]))
        for resolution, side in (("RR", 256), ("FR", 512)):
            observation = batch(side=side, resolution=resolution, split="test", aligned=True, batch_size=1)
            calls = []
            hook = model.matcher.register_forward_pre_hook(lambda _, a, kw: calls.append((a, kw)),
                                                           with_kwargs=True)
            out = model.native_prediction(observation)
            hook.remove()
            self.assertEqual(len(calls), 1)  # No internal 64x64 tiling.
            self.assertEqual(calls[0][1]["resolution"], resolution)
            self.assertEqual(tuple(out["y"].shape), (1, 4, side, side))
            self.assertFalse(out["y"].requires_grad)
            self.assertFalse(out["u"].requires_grad)
            torch.testing.assert_close(out["b"], torch.tensor([[2., -1.]]), rtol=0, atol=0)
            torch.testing.assert_close(out["u"], torch.tensor([[2.25, -1.5]]), rtol=0, atol=0)
            if resolution == "FR":
                self.assertIsNone(observation.gt)
            with self.assertRaises(ValueError):
                model(observation)

    def test_reject_validation_wrong_sensor_source_fr_training_and_fake_fr_gt(self):
        rr, model = batch(), stage()
        for change in ({"split": "validation"}, {"split": "test"},
                       {"source_manifest_sha256": "b"*64}, {"sensor": "GF2"}):
            with self.assertRaises(ValueError):
                model(replace(rr, lineage=replace(rr.lineage, **change)))
        fr = batch(resolution="FR", split="train")
        with self.assertRaises(ValueError):
            model(fr)
        with self.assertRaises(ValueError):
            model.native_prediction(replace(fr, gt=rr.gt))
        self.assertEqual(model.forward_calls, 0)

    def test_online_augmentation_recomputes_matcher_no_stored_prior(self):
        rr, model = batch(aligned=True), stage()
        transformed = augment_edge_batch(rr, hflip=True, vflip=False, rot90=1)
        self.assertFalse(hasattr(transformed, "prior_dy_dx"))
        self.assertFalse(hasattr(transformed.lineage, "scene_ids"))
        out = model.native_prediction(transformed)
        # (2,-1) -> Hflip(2,1) -> CCW(-1,2).
        torch.testing.assert_close(out["b"], torch.tensor([[-1., 2.]]).expand(2, 2))

    def test_rng_replay_global_rng_unchanged_and_failed_forward_rollback(self):
        rr, model = batch(), stage()
        global_rng = torch.random.get_rng_state().clone()
        model(rr)
        self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))
        saved = copy.deepcopy(model.state_dict())
        expected = model(rr)
        restored = stage()
        restored.load_state_dict(saved)
        actual = restored(rr)
        for key in ("loss", "epsilon_rr", "eps_rr"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(actual["forward_index"], expected["forward_index"])
        before = restored._epsilon_rng.get_state().clone()
        count = restored.forward_calls
        # Inject failure only after epsilon draw to check exact rewind.
        with patch.object(restored, "predict_shift", side_effect=[
                restored.predict_shift(rr.pan, rr.native_lms, resolution="RR"), RuntimeError("synthetic")]):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                restored(rr)
        self.assertEqual(restored.forward_calls, count)
        self.assertTrue(torch.equal(before, restored._epsilon_rng.get_state()))

    def test_semantic_checkpoint_and_nested_strict_false_rejected_before_weights(self):
        model = stage()
        saved = copy.deepcopy(model.state_dict())
        self.assertEqual(saved["_extra_state"]["method"], METHOD_VERSION)
        before = state_dict_sha256(model.aligner)
        for mutate in (lambda d: d["_extra_state"].update(method="legacy"),
                       lambda d: d["_extra_state"]["pipeline"].update(roi_margin=4),
                       lambda d: d["_extra_state"]["pipeline"]["matcher"]["matcher"].update(radius_fr=8),
                       lambda d: d["_extra_state"]["sources"].update(rr_manifest_sha256="b"*64)):
            bad = copy.deepcopy(saved)
            mutate(bad)
            with self.assertRaises(ValueError):
                model.load_state_dict(bad)
            self.assertEqual(before, state_dict_sha256(model.aligner))
        with self.assertRaises(ValueError):
            model.load_state_dict(saved, strict=False)
        parent = nn.ModuleDict({"edge": model})
        nested = copy.deepcopy(parent.state_dict())
        nested["edge._extra_state"]["method"] = "legacy"
        with self.assertRaises(ValueError):
            parent.load_state_dict(nested, strict=False)
        self.assertEqual(before, state_dict_sha256(model.aligner))

    def test_explicit_recipe_and_fixed_support_guard(self):
        for t in (teacher(layout="PLH"), teacher(base="NATIVE_LMS")):
            with self.assertRaisesRegex(ValueError, "P0/BICUBIC"):
                EdgeGuidedStage1(ResidualPANAligner(4, view_margin=0), t, matcher(),
                    sources=SOURCES, roi_margin=3, boundary_policy="report", epsilon_seed=1)
        model = stage(boundary_policy="require_spatial_support")
        with torch.no_grad():
            model.aligner.fc2.bias.fill_(20.)
        with self.assertRaisesRegex(ValueError, "support"):
            model(batch())
        self.assertEqual(model.forward_calls, 0)


if __name__ == "__main__":
    unittest.main()
