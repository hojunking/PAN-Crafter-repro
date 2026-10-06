"""CPU numerical contracts; does not claim source-data or actual CUDA checks."""
import copy
import unittest

import torch
from torch.nn import functional as F

from pa.warp import warp_pan
from ta2.model import TeacherModel, state_hash
from ta2.losses import (StructSupportViolation, assert_struct_support,
                        compute_objective, gaussian_scharr, structure_loss)


def configuration(**changes):
    cfg = dict(case_id="TEST_ONLY", aligner_policy="LEARNED", a_reference="NATIVE_LMS",
               u_reference="BICUBIC_MS", lambda_epsilon=.001, epsilon_every=1,
               lambda_struct=.01, struct_target="GT", struct_descriptor="NCC",
               struct_scales=[.8, 1.6], struct_branch="NATIVE", band_weights="UNIFORM",
               a_receives_rec=True, epsilon_detach_anchor=True, struct_ramp_updates=0,
               auxiliary="NONE", lambda_scale=0, lr_u=1e-4, lr_a=1e-5,
               total_updates=50000)
    cfg.update(changes)
    return cfg


def sample(bands=4, count=1, size=64, dtype=torch.float32):
    generator = torch.Generator().manual_seed(761)
    p = torch.randn(count, 1, size, size, dtype=dtype, generator=generator) * .35
    ms = torch.randn(count, bands, size // 4, size // 4, dtype=dtype, generator=generator) * .2
    gt = warp_pan(p, p.new_tensor([[.65, -.45]]).expand(count, -1)).expand(-1, bands, -1, -1).clone()
    lms = F.interpolate(ms, scale_factor=4, mode="bicubic", align_corners=False)
    return dict(pan=p, ms=ms, lms=lms, gt=gt)


class ModelContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_identical_u_across_removed_aligner_and_rng_restore(self):
        before = torch.random.get_rng_state().clone()
        a = TeacherModel(4, 42, configuration())
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))
        b = TeacherModel(4, 42, configuration(aligner_policy="BYPASS", a_reference="NA"))
        g = TeacherModel(4, 42, configuration(aligner_policy="GLOBAL_LEARNED", a_reference="NA"))
        self.assertEqual(state_hash(a.backbone), state_hash(b.backbone))
        self.assertEqual(state_hash(a.backbone), state_hash(g.backbone))
        self.assertEqual(sum(p.numel() for p in g.a_parameters()), 2)
        c = TeacherModel(4, 42, configuration(a_reference="BICUBIC_MS"))
        self.assertEqual(state_hash(a.aligner), state_hash(c.aligner))
        self.assertEqual(a.backbone.attn_locations, ())
        self.assertEqual(a.backbone.depth, (1, 2, 3))
        self.assertTrue(all("mod." not in key for key in a.backbone.state_dict()))

    def test_native_lms_changes_base_only_not_architecture(self):
        data = sample()
        data["lms"] = data["lms"] + .2
        u = TeacherModel(4, 71, configuration(u_reference="NATIVE_LMS"))
        b = TeacherModel(4, 71, configuration())
        with torch.no_grad():
            self.assertTrue(torch.equal(u(**{key: data[key] for key in ("pan", "ms", "lms")})["prediction"], data["lms"]))
            out = b(data["pan"], data["ms"], data["lms"])
            expected = F.interpolate(data["ms"], scale_factor=4, mode="bicubic", align_corners=False)
            self.assertTrue(torch.equal(out["prediction"], expected))
        self.assertEqual(state_hash(u.backbone), state_hash(b.backbone))
        with self.assertRaisesRegex(ValueError, "BLOCKED_LMS"):
            u(data["pan"], data["ms"], None)

    def test_fixed_shift_required_and_no_gt_inference_input(self):
        data = sample()
        net = TeacherModel(4, 1, configuration(aligner_policy="PER_IMAGE_GRAD"))
        with self.assertRaisesRegex(ValueError, "explicit verified"):
            net(data["pan"], data["ms"], data["lms"])
        with self.assertRaises(TypeError):
            net(**data)
        shift = torch.tensor([[.25, -.5]], requires_grad=True)
        out = net(data["pan"], data["ms"], data["lms"], shift)
        self.assertIsNotNone(torch.autograd.grad(out["warped_pan"].square().mean(), shift)[0])

    def test_zero_sampler_and_bypass_match_and_units_are_pixels(self):
        for size in (64, 256, 512):
            ramp = torch.arange(size).float()[None, None, None, :].expand(1, 1, size, size)
            shift = torch.tensor([[0., 1.]])
            warped = warp_pan(ramp, shift)
            self.assertTrue(torch.allclose(warped[..., 4:-4, 4:-4], ramp[..., 4:-4, 5:-3], atol=1e-4))
            self.assertTrue(torch.equal(warp_pan(ramp, torch.zeros_like(shift)), ramp))

    def test_eight_band_head(self):
        net = TeacherModel(8, 1, configuration())
        data = sample(bands=8)
        with torch.no_grad():
            self.assertEqual(net(data["pan"], data["ms"], data["lms"])["prediction"].shape,
                             (1, 8, 64, 64))


class StructureContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_polarity_and_low_texture_fixed_batch_denominator(self):
        p = sample(dtype=torch.float64)["pan"]
        matching = p.expand(-1, 4, -1, -1)
        matched, diagnostic = structure_loss(p, matching)
        reversed_loss, _ = structure_loss(p, -matching)
        self.assertLess(float(matched), 1e-8)
        self.assertLess(float(reversed_loss), 1e-8)
        self.assertTrue(bool(diagnostic["valid"].all()))
        wrong = torch.roll(matching, 2, -1)
        wrong_loss, _ = structure_loss(p, wrong)
        mixed, diag = structure_loss(torch.cat((p, p)), torch.cat((wrong, torch.zeros_like(wrong))))
        self.assertAlmostEqual(float(mixed), float(wrong_loss) / 2, places=12)
        self.assertEqual(float(diag["low_texture_sample_fraction"]), .5)

    def test_exact_structure_gradient_matches_finite_difference(self):
        batch = sample(dtype=torch.float64)
        c = torch.tensor([[.27, -.19]], dtype=torch.float64, requires_grad=True)
        def loss(value):
            return structure_loss(warp_pan(batch["pan"], value), batch["gt"])[0]
        analytical = torch.autograd.grad(loss(c), c)[0]
        for component in range(2):
            delta = torch.zeros_like(c); delta[0, component] = 1e-5
            numerical = (loss(c.detach() + delta) - loss(c.detach() - delta)) / 2e-5
            self.assertAlmostEqual(float(analytical[0, component]), float(numerical), places=6)
        self.assertGreater(float(analytical.norm()), 1e-3)

    def test_support_includes_filter_and_sampler_not_just_output(self):
        assert_struct_support(64, 64, torch.tensor([[8., -8.]]))
        # Output-only bicubic support would accept this. The filter halo does not.
        with self.assertRaisesRegex(StructSupportViolation, "STRUCT_SUPPORT_VIOLATION"):
            assert_struct_support(64, 64, torch.tensor([[10., 0.]]))
        assert_struct_support(48, 48, torch.zeros(1, 2))
        with self.assertRaises(StructSupportViolation):
            assert_struct_support(32, 32, torch.zeros(1, 2))
        assert_struct_support(512, 512, torch.zeros(1, 2), margin=128, resize_scale=8)
        with self.assertRaises(StructSupportViolation):
            assert_struct_support(512, 512, torch.tensor([[-112., 0.]]), margin=128, resize_scale=8)

    def test_cal_weights_and_ngf_are_finite_without_target_gradients(self):
        data = sample()
        p = data["pan"].requires_grad_()
        target = data["gt"].requires_grad_()
        for kind in ("NCC", "NGF"):
            loss, diag = structure_loss(p, target, descriptor=kind, band_weights=[1., 2., 3., 4.])
            gp, gy = torch.autograd.grad(loss, (p, target), allow_unused=True)
            self.assertIsNone(gy)
            self.assertTrue(bool(torch.isfinite(gp).all()))
            self.assertGreater(float(gp.norm()), 0)
        with self.assertRaises(ValueError):
            structure_loss(p, target, band_weights=[0., 0., 0., 0.])

    def test_scharr_scale_is_fixed_01_no_clipping(self):
        ramp = torch.arange(64).double()[None, None, None, :].expand(1, 1, 64, 64) / 64
        grad = gaussian_scharr(ramp, .8)
        self.assertTrue(torch.allclose(grad[:, :, 0, 16:-16, 16:-16], torch.zeros(1, 1, 32, 32).double(), atol=1e-12))
        self.assertTrue(torch.allclose(grad[:, :, 1, 16:-16, 16:-16], torch.full((1, 1, 32, 32), 1/64).double(), atol=1e-12))


class ObjectiveContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_structure_has_no_u_gradient_and_zero_head_body_expected(self):
        config = configuration()
        net = TeacherModel(4, 101, config)
        result = compute_objective(net, sample(), config, 1, torch.tensor([[.3, -.4]]))
        grads = torch.autograd.grad(result["struct"], tuple(net.parameters()), allow_unused=True)
        pairs = dict(zip((name for name, _ in net.named_parameters()), grads))
        self.assertTrue(all(value is None for key, value in pairs.items() if key.startswith("backbone.")))
        self.assertGreater(float(pairs["aligner.fc2.bias"].abs().sum()), 0)
        self.assertEqual(float(pairs["aligner.pan_stem.1.weight"].abs().sum()), 0)
        with torch.no_grad():
            net.aligner.fc2.bias -= .05 * pairs["aligner.fc2.bias"]
            net.aligner.fc2.weight -= .05 * pairs["aligner.fc2.weight"]
        subsequent = compute_objective(net, sample(), config, 2, torch.tensor([[.3, -.4]]))
        body_grad = torch.autograd.grad(subsequent["struct"], net.aligner.pan_stem[1].weight)[0]
        self.assertGreater(float(body_grad.abs().sum()), 0)

    def test_s06_detaches_rec_only_and_preserves_struct(self):
        config = configuration(a_receives_rec=False)
        net = TeacherModel(4, 4, config)
        with torch.no_grad():
            net.backbone.output[-1].weight.normal_(0, .005)
        result = compute_objective(net, sample(), config, 1, torch.tensor([[.3, -.4]]))
        ga = torch.autograd.grad(result["rec"], tuple(net.a_parameters()), allow_unused=True, retain_graph=True)
        self.assertTrue(all(value is None for value in ga))
        gs = torch.autograd.grad(result["struct"], tuple(net.a_parameters()), allow_unused=True, retain_graph=True)
        self.assertGreater(sum(float(value.norm()) for value in gs if value is not None), 0)
        gu = torch.autograd.grad(result["rec"], tuple(net.u_parameters()), allow_unused=True)
        self.assertGreater(sum(float(value.norm()) for value in gu if value is not None), 0)

    def test_every2_ramp_and_scalar_coefficients_applied_once(self):
        cfg = configuration(epsilon_every=2, struct_ramp_updates=5000)
        net = TeacherModel(4, 12, cfg)
        data = sample()
        a = compute_objective(net, data, cfg, 1)
        self.assertEqual(float(a["epsilon"]), 0)
        self.assertFalse(a["diagnostics"]["epsilon_active"])
        b = compute_objective(net, data, cfg, 2, torch.tensor([[.3, -.4]]))
        self.assertAlmostEqual(float(b["epsilon"]), .35, places=6)
        self.assertTrue(torch.equal(b["weighted_epsilon"], .001 * b["epsilon"]))
        self.assertTrue(torch.equal(b["weighted_struct"], .01 * 2/5000 * b["struct"]))
        self.assertTrue(torch.equal(b["total"], b["rec"] + b["weighted_epsilon"] + b["weighted_struct"] + b["weighted_scale"]))

    def test_o04_removes_only_native_anchor_stop_gradient(self):
        net = TeacherModel(4, 52, configuration())
        data, epsilon = sample(), torch.tensor([[.3, -.4]])
        one_way = compute_objective(net, data, configuration(lambda_struct=0), 1, epsilon)
        symmetric = compute_objective(net, data, configuration(lambda_struct=0, epsilon_detach_anchor=False), 1, epsilon)
        self.assertEqual(float(one_way["epsilon"]), float(symmetric["epsilon"]))
        g_one = torch.autograd.grad(one_way["epsilon"], net.aligner.fc2.bias)[0]
        g_sym = torch.autograd.grad(symmetric["epsilon"], net.aligner.fc2.bias)[0]
        self.assertTrue(torch.equal(g_one, torch.tensor([.5, -.5])))
        self.assertTrue(torch.equal(g_sym, torch.zeros(2)))

    def test_s05_direct_single_warp_corrected_branch(self):
        cfg = configuration(struct_branch="NATIVE_CORRECTED")
        net = TeacherModel(4, 18, cfg)
        data, epsilon = sample(), torch.tensor([[.3, -.4]])
        result = compute_objective(net, data, cfg, 1, epsilon)
        a = structure_loss(data["pan"], data["gt"])[0]
        b = structure_loss(warp_pan(data["pan"], epsilon), data["gt"])[0]
        self.assertTrue(torch.allclose(result["struct"], .5 * (a + b)))

    def test_s07_uses_huber_pixel_labels_not_gt_in_forward(self):
        cfg = configuration(struct_descriptor="PSEUDO_HUBER")
        net = TeacherModel(4, 19, cfg)
        pseudo = torch.tensor([[.5, -1.]])
        result = compute_objective(net, sample(), cfg, 1, torch.tensor([[.3, -.4]]), pseudo_shift=pseudo)
        expected = F.huber_loss(torch.zeros_like(pseudo), pseudo, delta=.25)
        self.assertTrue(torch.equal(result["struct"], expected))

    def test_x_identity_mean_and_backward_checkpoint(self):
        base = configuration()
        model = TeacherModel(4, 82, base)
        data, epsilon = sample(count=2), torch.tensor([[.3, -.4], [-.3, .4]])
        expected = compute_objective(model, data, base, 1, epsilon)
        expected_gradient = torch.autograd.grad(expected["total"], model.aligner.fc2.bias)[0]
        for auxiliary, choice in (("CROP", 64), ("SCALE", 1), ("SCALE_CONSISTENCY", 1)):
            cfg = configuration(auxiliary=auxiliary, lambda_scale=.001 if auxiliary == "SCALE_CONSISTENCY" else 0)
            actual = compute_objective(model, data, cfg, 1, epsilon, auxiliary_choice=choice)
            self.assertTrue(torch.allclose(actual["total"], expected["total"], atol=1e-7, rtol=1e-6))
            gradient = torch.autograd.grad(actual["total"], model.aligner.fc2.bias)[0]
            self.assertTrue(torch.allclose(gradient, expected_gradient, atol=1e-7, rtol=1e-5))

    def test_x_crop_and_scale_are_real_auxiliary_geometry(self):
        data, epsilon = sample(), torch.tensor([[.3, -.4]])
        model = TeacherModel(4, 20, configuration())
        for auxiliary, choice in (("CROP", 48), ("CROP", 56), ("SCALE", 2), ("SCALE", 4), ("SCALE", 8)):
            cfg = configuration(auxiliary=auxiliary)
            result = compute_objective(model, data, cfg, 1, epsilon, auxiliary_choice=choice)
            self.assertEqual(result["output"]["prediction"].shape[-2:], (64, 64))
            self.assertEqual(result["diagnostics"]["auxiliary"]["margin"], 16 if auxiliary == "CROP" else 16 * choice)
            self.assertAlmostEqual(float(result["epsilon"]), .35, places=6)
            self.assertTrue(bool(torch.isfinite(result["total"])))
            gradient = torch.autograd.grad(result["total"], model.aligner.fc2.bias)[0]
            self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_unknown_branch_or_missing_cal_is_not_silently_accepted(self):
        data, epsilon = sample(), torch.tensor([[.3, -.4]])
        model = TeacherModel(4, 11, configuration())
        with self.assertRaises(ValueError):
            compute_objective(model, data, configuration(auxiliary="UNKNOWN"), 1, epsilon)
        with self.assertRaisesRegex(ValueError, "CAL_FROZEN"):
            compute_objective(model, data, configuration(band_weights="CAL_FROZEN"), 1, epsilon)
        with self.assertRaisesRegex(ValueError, "radius2"):
            compute_objective(model, data, configuration(), 1, torch.tensor([[3., 0.]]))


if __name__ == "__main__":
    unittest.main()
