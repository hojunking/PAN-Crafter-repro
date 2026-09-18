#!/usr/bin/env python3
"""CPU-only FH12 numerical/model/routing regression tests (no dataset access)."""
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fh12.losses import routed_student_backward, student_losses, teacher_loss
from fh12.model import LAYOUTS, build_model, frontend_self_test, state_hash, sync_frontend
from pa.losses import output_edge_loss_per_sample
from pa.offset import sample_offsets


class FH12ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.gen = torch.Generator().manual_seed(91)
        self.pan = torch.randn(2, 1, 32, 32, generator=self.gen)
        self.ms = torch.randn(2, 8, 8, 8, generator=self.gen)
        self.lp = torch.randn(2, 1, 8, 8, generator=self.gen)
        self.gt = torch.randn(2, 8, 32, 32, generator=self.gen)

    def build(self, layout="PLH", depth=(1, 1, 1), seed=71001, **kw):
        return build_model(layout, 8, list(depth), seed, **kw)

    def unlock_head(self, model):
        with torch.no_grad():
            model.backbone.output[-1].weight.copy_(
                .03 * torch.randn(model.backbone.output[-1].weight.shape, generator=self.gen))

    def test_frontend_shared_grid_all_production_shapes_and_signs(self):
        report = frontend_self_test()
        self.assertTrue(report["passed"])
        self.assertEqual([r["size"] for r in report["records"]], [64, 256, 512])

    def test_exact_layout_and_zero_shift(self):
        p, lp, ms = torch.ones_like(self.pan), torch.ones_like(self.lp) * 3, self.ms
        base = F.interpolate(ms, scale_factor=4, mode="bicubic", align_corners=False)
        expected = {"P": p, "L": torch.ones_like(p) * 3,
                    "H": torch.ones_like(p) * -2, "MS": base}
        for layout, names in LAYOUTS.items():
            out = sync_frontend(p, ms, lp, torch.zeros(2, 2), layout)
            self.assertTrue(torch.equal(out["x_in"], torch.cat([expected[n] for n in names], 1)))
            self.assertLess(float(out["H"].min()), -1., "signed HP must not be clamped")

    def test_frontend_rejects_silent_fallback(self):
        for lp in (None, self.pan, self.lp[:, :, :-1]):
            with self.assertRaises(ValueError):
                sync_frontend(self.pan, self.ms, lp, torch.zeros(2, 2), "P0")

    def test_construction_preserves_all_cpu_rng(self):
        random.seed(918)
        np.random.seed(918)
        torch.manual_seed(918)
        py_before = random.getstate()
        np_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        self.build()
        self.assertEqual(py_before, random.getstate())
        np_after = np.random.get_state()
        self.assertEqual(np_before[0], np_after[0])
        np.testing.assert_array_equal(np_before[1], np_after[1])
        self.assertEqual(np_before[2:], np_after[2:])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))

    def test_coupled_initial_weights_and_nontrivial_function(self):
        records = [self.build(layout) for layout in LAYOUTS]
        reference, manifest = records[0]
        self.unlock_head(reference)
        ref_out = reference(self.pan, self.ms, self.lp)
        for (model, man), layout in zip(records, LAYOUTS):
            for key in ("P", "MS", "input_bias", "body", "A"):
                self.assertEqual(man["hashes"][key], manifest["hashes"][key], (layout, key))
            self.assertTrue(man["extra_kernel_zero"])
            with torch.no_grad():
                model.backbone.output[-1].weight.copy_(reference.backbone.output[-1].weight)
            out = model(self.pan, self.ms, self.lp)
            self.assertTrue(torch.allclose(out["y"], ref_out["y"], atol=2e-6, rtol=2e-5))
            self.assertTrue(torch.equal(out["delta"], ref_out["delta"]))
            self.assertTrue(torch.equal(out["ms_base"], ref_out["ms_base"]))

    def test_same_name_shape_coupling_across_depth_and_width(self):
        shallow, _ = self.build("P0", depth=(1, 2, 1))
        deep, _ = self.build("P0", depth=(1, 2, 3))
        wide, _ = build_model("P0", 12, [1, 2, 1], 71001)
        for model in (deep, wide):
            sd = model.state_dict()
            for name, tensor in shallow.state_dict().items():
                if name in sd and sd[name].shape == tensor.shape:
                    self.assertTrue(torch.equal(tensor, sd[name]), name)

    def test_teacher_scratch_and_rejected_donor(self):
        model, manifest = self.build()
        self.assertTrue(manifest["from_scratch"])
        self.assertEqual(manifest["pretrained_aligner_loads"], 0)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual(int(model.aligner.fc2.weight.count_nonzero()), 0)
        self.assertEqual(int(model.aligner.fc2.bias.count_nonzero()), 0)
        self.assertGreater(int(model.aligner.fc1.weight.count_nonzero()), 0)
        with self.assertRaises(ValueError):
            self.build(teacher_aligner_state=model.aligner.state_dict())
        with self.assertRaises(ValueError):
            self.build(role="S")

    def test_margin4_aligner_ignores_lp_and_single_base(self):
        model, _ = self.build()
        sizes = []
        handle = model.aligner.register_forward_pre_hook(lambda m, args: sizes.append((args[0].shape, args[1].shape)))
        out = model(self.pan, self.ms, self.lp)
        other = model(self.pan, self.ms, self.lp * 19)
        handle.remove()
        self.assertEqual(sizes[0], (torch.Size([2, 1, 24, 24]), torch.Size([2, 8, 24, 24])))
        self.assertTrue(torch.equal(out["delta"], other["delta"]))
        self.assertTrue(torch.equal(out["y"], out["ms_base"]))

    def test_full_model_64_256_512_and_native_aligner_sizes(self):
        model, _ = self.build()
        model.eval()
        sizes = []
        hook = model.aligner.register_forward_pre_hook(lambda m, args: sizes.append(args[0].shape[-1]))
        with torch.no_grad():
            for size in (64, 256, 512):
                pan = torch.randn(1, 1, size, size, generator=self.gen)
                ms = torch.randn(1, 8, size // 4, size // 4, generator=self.gen)
                lp = torch.randn(1, 1, size // 4, size // 4, generator=self.gen)
                out = model(pan, ms, lp)
                self.assertEqual(out["y"].shape, (1, 8, size, size))
                self.assertTrue(bool(torch.isfinite(out["y"]).all()))
        hook.remove()
        self.assertEqual(sizes, [56, 248, 504])

    def test_autocast_keeps_frontend_aligner_and_final_output_fp32(self):
        model, _ = self.build()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = model(self.pan, self.ms, self.lp)
        for key in ("y", "delta", "ms_base", "L", "H", "pan_aligned", "x_in"):
            self.assertEqual(out[key].dtype, torch.float32, key)

    def test_teacher_odd_offset_is_a_only_and_no_unet_call(self):
        model, _ = self.build()
        self.unlock_head(model)
        calls = []
        handle = model.backbone.register_forward_hook(lambda *args: calls.append(1))
        out = model(self.pan, self.ms, self.lp)
        rng = torch.Generator().manual_seed(918)
        before = rng.get_state().clone()
        even = teacher_loss(model, out, self.gt, self.pan, self.ms, 0, rng)
        self.assertFalse(even["offset_active"])
        self.assertTrue(torch.equal(before, rng.get_state()))
        odd = teacher_loss(model, out, self.gt, self.pan, self.ms, 1, rng)
        self.assertTrue(odd["offset_active"])
        self.assertEqual(len(calls), 1)
        handle.remove()
        self.assertTrue(torch.allclose(odd["total"], odd["rec"] + 1e-4 * odd["off"]))
        grad_u = torch.autograd.grad(odd["off"], list(model.backbone.parameters()), retain_graph=True, allow_unused=True)
        self.assertTrue(all(g is None for g in grad_u))
        grad_target = torch.autograd.grad(odd["off"], out["delta"], retain_graph=True, allow_unused=True)[0]
        self.assertIsNone(grad_target)
        grad_live = torch.autograd.grad(odd["rec"], out["delta"], retain_graph=True)[0]
        self.assertGreater(float(grad_live.abs().sum()), 0.)
        grad_a = torch.autograd.grad(odd["off"], list(model.aligner.parameters()))
        self.assertGreater(sum(float(g.abs().sum()) for g in grad_a), 0.)

    def test_offset_disk_distribution(self):
        eps = sample_offsets(30000, 2., torch.Generator().manual_seed(918))
        radii_squared = (eps**2).sum(1)
        self.assertLessEqual(float(radii_squared.max()), 4.)
        self.assertAlmostEqual(float(radii_squared.mean()), 2., delta=.03)
        self.assertLess(float(eps.mean(0).abs().max()), .02)

    def test_student_formula_detaches_and_fixed_denominators(self):
        y = (self.gt + .3).requires_grad_()
        teacher = (self.gt + .1).requires_grad_()
        gt = self.gt.clone().requires_grad_()
        weights = torch.tensor([.2, .8], requires_grad=True)
        losses = student_losses(dict(y=y), dict(y=teacher), gt, .1, weights)
        difficulty = .5
        advantage = .2 / (.3 + 1e-6)
        expected_hard = (1 + difficulty) * .3
        expected_soft = .1 * (1 - difficulty) * advantage * .2
        self.assertAlmostEqual(float(losses["hard"]), expected_hard, places=6)
        self.assertAlmostEqual(float(losses["soft"]), expected_soft, places=6)
        self.assertAlmostEqual(float(losses["L_A"]), .5 * expected_hard, places=6)
        self.assertAlmostEqual(float(losses["L_U"]), expected_hard + expected_soft, places=6)
        losses["L_U"].backward()
        self.assertIsNotNone(y.grad)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(gt.grad)
        self.assertIsNone(weights.grad)
        self.assertFalse(losses["difficulty"].requires_grad)
        self.assertFalse(losses["advantage"].requires_grad)

    def test_edge_is_signed_scharr_two_direction_interior(self):
        teacher = torch.zeros_like(self.gt)
        y = self.gt.clone().requires_grad_()
        losses = student_losses(dict(y=y), dict(y=teacher), -self.gt, .1, torch.ones(2))
        gold = output_edge_loss_per_sample(y, -self.gt)
        self.assertTrue(torch.equal(losses["edge_i"], gold))
        self.assertGreater(float(gold.mean()), .1)

    def test_separate_teacher_student_layout_and_gradient_routing(self):
        teacher, _ = self.build("PLH")
        self.unlock_head(teacher)
        teacher.eval().requires_grad_(False)
        t_before = state_hash(teacher.state_dict())
        student, manifest = self.build("P0", seed=72001, role="S", teacher_aligner_state=teacher.aligner.state_dict())
        self.unlock_head(student)
        self.assertEqual(teacher.input_layout, "PLH")
        self.assertEqual(student.input_layout, "P0")
        self.assertTrue(manifest["student_aligner_independent_clone"])
        for t, s in zip(teacher.aligner.parameters(), student.aligner.parameters()):
            self.assertNotEqual(t.data_ptr(), s.data_ptr())
            self.assertTrue(torch.equal(t, s))
        with torch.no_grad():
            t_out = teacher(self.pan, self.ms, self.lp)
        out = student(self.pan, self.ms, self.lp)
        losses = student_losses(out, t_out, self.gt, .1, torch.tensor([.3, .7]))
        u, a = list(student.backbone.parameters()), list(student.aligner.parameters())
        gold_u = torch.autograd.grad(losses["L_U"], u, retain_graph=True)
        gold_a = torch.autograd.grad(losses["L_A"], a, retain_graph=True)
        routed_student_backward(student, losses, scale=7.)
        for param, gold in zip(u + a, gold_u + gold_a):
            self.assertTrue(torch.allclose(param.grad / 7., gold, atol=2e-7, rtol=2e-5))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in a), 0.)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertEqual(t_before, state_hash(teacher.state_dict()))

    def test_extra_frequency_paths_c_gradient_remains_live(self):
        # Explicit independently isolated channels demonstrate both LP/HP branches
        # differentiate through c (not just the always-present P channel).
        delta = torch.tensor([[.4, -.7], [-.2, 1.3]], requires_grad=True)
        out = sync_frontend(self.pan, self.ms, self.lp, delta, "PLH")
        for branch in ("L", "H"):
            grad = torch.autograd.grad(out[branch].square().mean(), delta, retain_graph=True)[0]
            self.assertGreater(float(grad.abs().sum()), 0., branch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
