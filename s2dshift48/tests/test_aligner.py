"""CPU architecture/identity tests; no dataset, GPU, training job or network."""
import copy
import inspect
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from directshift48.method import DirectPANAligner
from pa.aligner import znorm
from pa.warp import warp_pan
from s2dshift48.aligner import S2DGlobalPANAligner, boundary_manifest


class S2DAlignerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91016)

    def test_shapes_all_sensors_rr_fr_and_zero_head(self):
        for bands in (4, 8):
            model = S2DGlobalPANAligner(bands)
            for size in (24, 64, 256, 512):
                with self.subTest(bands=bands, size=size), torch.no_grad():
                    shapes = {}
                    hooks = [getattr(model, name).register_forward_hook(
                        lambda _m, _a, out, key=name: shapes.__setitem__(key, tuple(out.shape)))
                        for name in ("pan_stem", "ms_stem", "joint1", "joint2", "res")]
                    pan, lms = torch.randn(1, 1, size, size), torch.randn(1, bands, size, size)
                    u, details = model(pan, lms, return_details=True)
                    for hook in hooks:
                        hook.remove()
                    spatial = (size - 16) // 4
                    self.assertEqual(shapes, {
                        "pan_stem": (1, 16, spatial, spatial),
                        "ms_stem": (1, 16, spatial, spatial),
                        "joint1": (1, 32, spatial, spatial),
                        "joint2": (1, 64, spatial, spatial),
                        "res": (1, 64, spatial, spatial),
                    })
                    self.assertTrue(torch.equal(u, torch.zeros(1, 2)))
                    self.assertEqual(u.dtype, torch.float32)
                    self.assertEqual(set(details), {"features", "head_input", "head_hidden"})
                    self.assertEqual(details["features"].shape, (1, 64))
                    self.assertIs(details["head_input"], details["features"])
                    self.assertEqual(details["head_hidden"].shape, (1, 32))
                    self.assertEqual(warp_pan(pan, u).shape, pan.shape)

    def test_crop_znorm_s2d_order_and_phase_are_exact(self):
        model = S2DGlobalPANAligner(8)
        pan = torch.arange(64 * 64, dtype=torch.float32).reshape(1, 1, 64, 64)
        lms = torch.randn(1, 8, 64, 64)
        captured = {}
        hooks = [getattr(model, name).register_forward_pre_hook(
            lambda _m, args, key=name: captured.__setitem__(key, args[0].detach().clone()))
            for name in ("pan_stem", "ms_stem")]
        with torch.no_grad():
            model(pan, lms)
        for hook in hooks:
            hook.remove()
        for source, key in ((pan, "pan_stem"), (lms, "ms_stem")):
            cropped = source[..., 8:-8, 8:-8]
            normalized = znorm(cropped)
            expected = F.pixel_unshuffle(normalized, 4)
            self.assertTrue(torch.equal(captured[key], expected))
            self.assertTrue(torch.equal(F.pixel_shuffle(expected, 4), normalized))
            for channel in range(source.shape[1]):
                for row in range(4):
                    for col in range(4):
                        self.assertTrue(torch.equal(expected[:, channel * 16 + row * 4 + col],
                                                    normalized[:, channel, row::4, col::4]))
        wrong = znorm(F.pixel_unshuffle(pan[..., 8:-8, 8:-8], 4))
        self.assertFalse(torch.equal(captured["pan_stem"], wrong))

    def test_detached_statistics_preserve_expected_input_gradient(self):
        model = S2DGlobalPANAligner(4)
        pan = torch.randn(1, 1, 24, 28, requires_grad=True)
        lms = torch.randn(1, 4, 24, 28)
        captured = []
        hook = model.pan_stem.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
        model(pan, lms)
        hook.remove()
        gradient, = torch.autograd.grad(captured[0].sum(), pan)
        crop = pan.detach()[..., 8:-8, 8:-8]
        expected = torch.zeros_like(pan)
        expected[..., 8:-8, 8:-8] = 1 / torch.sqrt(crop.var((2, 3), keepdim=True, unbiased=False) + 1e-6)
        self.assertTrue(torch.equal(gradient, expected))

    def test_outer_crop_cannot_influence_features_or_shift(self):
        model = S2DGlobalPANAligner(4)
        with torch.no_grad():
            model.fc2.weight.normal_()
            model.fc2.bias.normal_()
        pan, lms = torch.randn(2, 1, 64, 64), torch.randn(2, 4, 64, 64)
        changed_pan, changed_lms = pan.clone(), lms.clone()
        for image in (changed_pan, changed_lms):
            image[..., :8, :] = 500
            image[..., -8:, :] = -700
            image[..., :, :8] = 400
            image[..., :, -8:] = -900
        with torch.no_grad():
            u1, d1 = model(pan, lms, return_details=True)
            u2, d2 = model(changed_pan, changed_lms, return_details=True)
        self.assertTrue(torch.equal(u1, u2))
        self.assertTrue(torch.equal(d1["features"], d2["features"]))

    def test_nonzero_head_unclamped_current_pan_units(self):
        model = S2DGlobalPANAligner(4)
        with torch.no_grad():
            model.fc2.bias.copy_(torch.tensor([5., -7.]))
        for size in (64, 256, 512):
            yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
            pan = (yy.float() * 10 + xx.float())[None, None]
            lms = pan.repeat(1, 4, 1, 1)
            with torch.no_grad():
                correction = model(pan, lms)
                warped = warp_pan(pan, correction)
            self.assertTrue(torch.equal(correction, torch.tensor([[5., -7.]])))
            self.assertAlmostEqual(warped[0, 0, 20, 20].item(), pan[0, 0, 25, 13].item(), places=3)

    def test_head_gradient_exists_at_identity_and_trunk_after_head_moves(self):
        model = S2DGlobalPANAligner(4)
        pan, lms = torch.randn(2, 1, 64, 64), torch.randn(2, 4, 64, 64)
        model(pan, lms).square().sum().backward()
        self.assertTrue(torch.equal(model.fc2.weight.grad, torch.zeros_like(model.fc2.weight)))
        model.zero_grad(set_to_none=True)
        (model(pan, lms) - 1).square().sum().backward()
        self.assertGreater(model.fc2.weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.pan_stem[1].weight.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.fc2.weight.normal_(std=.05)
        model(pan, lms).square().sum().backward()
        self.assertGreater(model.pan_stem[1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.ms_stem[1].weight.grad.abs().sum().item(), 0)

    def test_invalid_inputs_and_nonfinite_intermediate_fail(self):
        model = S2DGlobalPANAligner(4)
        bad_pairs = [
            (torch.randn(0, 1, 64, 64), torch.randn(0, 4, 64, 64)),
            (torch.randn(1, 2, 64, 64), torch.randn(1, 4, 64, 64)),
            (torch.randn(1, 1, 64, 64), torch.randn(1, 8, 64, 64)),
            (torch.randn(1, 1, 64, 64).double(), torch.randn(1, 4, 64, 64).double()),
            (torch.randn(1, 1, 20, 24), torch.randn(1, 4, 20, 24)),
            (torch.randn(1, 1, 64, 66), torch.randn(1, 4, 64, 66)),
            (torch.full((1, 1, 64, 64), float("nan")), torch.randn(1, 4, 64, 64)),
            (torch.randn(1, 1, 64, 64), torch.full((1, 4, 64, 64), float("inf"))),
        ]
        for pan, lms in bad_pairs:
            with self.subTest(shape=tuple(pan.shape), dtype=pan.dtype), self.assertRaises(ValueError):
                model(pan, lms)
        hook = model.joint1.register_forward_hook(lambda _m, _a, out: out * float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "joint1"):
            model(torch.randn(1, 1, 64, 64), torch.randn(1, 4, 64, 64))
        hook.remove()
        model.half()
        with self.assertRaisesRegex(ValueError, "parameters"):
            model(torch.randn(1, 1, 64, 64), torch.randn(1, 4, 64, 64))

    def test_old_new_crossload_rejected_before_mutation(self):
        old, new = DirectPANAligner(8), S2DGlobalPANAligner(8)
        for source, destination in ((old, new), (new, old)):
            for strict in (True, False):
                before = copy.deepcopy(destination.state_dict())
                with self.subTest(destination=type(destination).__name__, strict=strict):
                    with self.assertRaises((RuntimeError, ValueError)):
                        destination.load_state_dict(source.state_dict(), strict=strict)
                    self.assertTrue(all(torch.equal(before[k], v) for k, v in destination.state_dict().items()))
        restored = S2DGlobalPANAligner(8)
        restored.load_state_dict(new.state_dict())
        self.assertTrue(all(torch.equal(new.state_dict()[k], v) for k, v in restored.state_dict().items()))

    def test_invalid_matching_identity_state_rejected_before_mutation(self):
        model = S2DGlobalPANAligner(4)
        for corruption in ("missing", "extra", "shape", "dtype", "nonfinite"):
            state = copy.deepcopy(model.state_dict())
            state["pan_stem.1.weight"].add_(1)
            if corruption == "missing":
                del state["fc2.weight"]
            elif corruption == "extra":
                state["extra"] = torch.zeros(1)
            elif corruption == "shape":
                state["fc2.weight"] = torch.zeros(3, 32)
            elif corruption == "dtype":
                state["fc2.weight"] = state["fc2.weight"].double()
            else:
                state["fc2.weight"].fill_(float("nan"))
            before = copy.deepcopy(model.state_dict())
            with self.subTest(corruption=corruption), self.assertRaises(RuntimeError):
                model.load_state_dict(state)
            self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))

    def test_nested_identity_guard_rejects_even_non_strict(self):
        source = nn.ModuleDict({"aligner": DirectPANAligner(4)})
        destination = nn.ModuleDict({"aligner": S2DGlobalPANAligner(4)})
        before = copy.deepcopy(destination.state_dict())
        with self.assertRaisesRegex(RuntimeError, "identity"):
            destination.load_state_dict(source.state_dict(), strict=False)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in destination.state_dict().items()))

    def test_manifest_params_and_no_gt_interface(self):
        with mock.patch.object(DirectPANAligner, "__init__", side_effect=AssertionError("old constructed")):
            model = S2DGlobalPANAligner(8)
        self.assertIsInstance(model, DirectPANAligner)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 124898)
        self.assertEqual(sum(p.numel() for p in S2DGlobalPANAligner(4).parameters()), 115682)
        manifest = model.manifest()
        self.assertEqual(manifest["s2d"]["factor"], 4)
        self.assertEqual(manifest["joint_stride"], [1, 1])
        self.assertTrue(manifest["s2d"]["normalized_before_S2D"])
        self.assertEqual(boundary_manifest()["local_rf_progression"], [4, 12, 20, 28, 36, 44])
        self.assertIn("global", boundary_manifest()["rf_limitation"])
        self.assertEqual(set(inspect.signature(model.forward).parameters),
                         {"pan", "native_lms", "return_details"})
        self.assertNotEqual(model.manifest(), DirectPANAligner(8).manifest())


if __name__ == "__main__":
    unittest.main()
