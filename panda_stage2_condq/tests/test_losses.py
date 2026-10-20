"""Synthetic CPU checks of the exact new equations, never campaign execution."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from directshift48.method import ScharrAbsolute
from panda_stage2_condq.losses import compute_loss, reference_maps
from panda_stage2_condq.spec import Stage2Spec

torch.set_num_threads(1)


class FixedRanks:
    tau_e = .25

    def lookup_exact(self, g, q):
        return torch.ones_like(q)


def config(variant="panda_condq", sensor="WV3", **kwargs):
    args = dict(sensor=sensor, variant=variant, seed=345, alpha=1., beta=.1,
                gamma=.5 if variant == "panda_condq" else 0., delta=1e-6)
    if variant == "gt_only":
        args.update(alpha=0., beta=0., gamma=0.)
    args.update(kwargs)
    return Stage2Spec(**args)


def images(sensor="WV3"):
    rng = torch.Generator().manual_seed(3)
    bands = 8 if sensor == "WV3" else 4
    gt = torch.rand(2, bands, 64, 64, generator=rng)
    teacher = gt + .25
    student = (gt + .5).detach().requires_grad_()
    pan = torch.rand(2, 1, 64, 64, generator=rng)
    return student, teacher, gt, pan


class LossTests(unittest.TestCase):
    def test_explicit_coefficients_and_new_variant_only(self):
        for kwargs in ({"delta": 0}, {"alpha": -1}, {"beta": float("nan")},
                       {"gamma": float("inf")}, {"seed": True}, {"variant": "eq"}):
            with self.assertRaises(ValueError):
                config(**kwargs)
        with self.assertRaises(ValueError):
            config("gt_only", alpha=1.)
        with self.assertRaises(ValueError):
            config("panda_e", gamma=.2)
        with self.assertRaises(ValueError):
            config(gamma=0)

    def test_gt_only_matches_all_bands_full_image_without_teacher_or_q(self):
        for sensor in ("WV3", "GF2"):
            s, t, gt, pan = images(sensor)
            out = compute_loss(s, None, gt, pan, spec=config("gt_only", sensor))
            torch.testing.assert_close(out["loss"], (s - gt).abs().mean(), rtol=1e-7, atol=1e-7)
            with self.assertRaisesRegex(ValueError, "skip Teacher"):
                compute_loss(s, t, gt, pan, spec=config("gt_only", sensor))

    def test_zero_coefficients_recover_gt_L1_and_e_variant_skips_q(self):
        class NeverScharr:
            def descriptor(self, _):
                raise AssertionError("panda_e must not use Scharr")
        s, t, gt, pan = images()
        out = compute_loss(s, t, gt, pan, spec=config("panda_e", alpha=0, beta=0),
                           scharr=NeverScharr(), stats=SimpleNamespace(tau_e=.25))
        torch.testing.assert_close(out["loss"], (s - gt).abs().mean())

    def test_exact_panda_e_formula_and_detached_advantage_gradient(self):
        s, t, gt, pan = images()
        t.requires_grad_(); gt.requires_grad_(); pan.requires_grad_()
        spec = config("panda_e")
        out = compute_loss(s, t, gt, pan, spec=spec, stats=FixedRanks(), return_maps=True)
        maps = out["maps"]
        e_s = (s - gt.detach()).abs().mean(1, keepdim=True)
        e_t = (t.detach() - gt.detach()).abs().mean(1, keepdim=True)
        d = e_t / (e_t + .25)
        a = ((e_s.detach() - e_t).clamp_min(0) / (e_s.detach() + spec.delta)).detach()
        expected = ((1 + spec.alpha*d)*e_s).mean() + spec.beta*((1-d)*a*(s-t.detach()).abs().mean(1, keepdim=True)).mean()
        torch.testing.assert_close(out["loss"], expected)
        gradient = torch.autograd.grad(out["loss"], s, retain_graph=True)[0]
        reference = torch.autograd.grad(expected, s)[0]
        torch.testing.assert_close(gradient, reference)
        out["loss"].backward()
        self.assertIsNone(t.grad); self.assertIsNone(gt.grad); self.assertIsNone(pan.grad)
        for name in ("d_e", "a_T", "r_q_bar", "w_h"):
            self.assertFalse(maps[name].requires_grad)
        self.assertTrue(maps["e_s"].requires_grad)
        self.assertTrue(maps["delta_ts"].requires_grad)

    def test_q_only_weights_roi8_hard_soft_borders_stay_supervised(self):
        s, t, gt, pan = images()
        out = compute_loss(s, t, gt, pan, spec=config(), scharr=ScharrAbsolute("WV3"),
                           stats=FixedRanks(), return_maps=True)
        m = out["maps"]
        for value in m.values():
            self.assertEqual(tuple(value.shape), (2, 1, 64, 64))
        self.assertEqual(int(torch.count_nonzero(m["r_q_bar"])), 2*48*48)
        self.assertEqual(float(m["r_q_bar"][..., :8, :].sum()), 0.)
        self.assertTrue(bool((m["w_h"] >= 1).all()))
        expected_hard = (m["w_h"] * m["e_s"]).mean()
        expected_soft = ((1-m["d_e"])*m["a_T"]*m["delta_ts"]).mean()
        torch.testing.assert_close(out["hard_weighted"], expected_hard)
        torch.testing.assert_close(out["soft_unscaled"], expected_soft)
        out["loss"].backward()
        self.assertGreater(float(s.grad[..., :8, :].abs().sum()), 0.)
        with self.assertRaises(TypeError):
            compute_loss(s, t, gt, pan, spec=config(), stats=FixedRanks(), valid_mask=torch.zeros_like(pan).bool())

    def test_hard_only_has_no_soft_graph_gate_or_teacher_gradient(self):
        s, t, gt, pan = images()
        t.requires_grad_()
        out = compute_loss(s, t, gt, pan, spec=config(beta=0.),
                           scharr=ScharrAbsolute("WV3"), stats=FixedRanks(), return_maps=True)
        self.assertIs(out["loss"], out["hard_weighted"])
        for field in ("soft_unscaled", "soft_weighted"):
            self.assertEqual(float(out[field]), 0.)
            self.assertFalse(out[field].requires_grad)
        for field in ("a_T", "delta_ts"):
            self.assertEqual(int(torch.count_nonzero(out["maps"][field])), 0)
            self.assertFalse(out["maps"][field].requires_grad)
        # Both e and q hard weighting remain active and unchanged.
        maps = out["maps"]
        expected = ((1 + maps["d_e"] + .5*maps["d_e"]*maps["r_q_bar"]) * maps["e_s"]).mean()
        torch.testing.assert_close(out["loss"], expected)
        actual_grad = torch.autograd.grad(out["loss"], s, retain_graph=True)[0]
        reference_grad = torch.autograd.grad(expected, s)[0]
        torch.testing.assert_close(actual_grad, reference_grad)
        self.assertIsNone(t.grad)

    def test_gate_zero_ties_and_zero_median(self):
        s, t, gt, pan = images()
        t = (gt + 1.).detach()
        out = compute_loss(s, t, gt, pan, spec=config("panda_e"), stats=FixedRanks(), return_maps=True)
        self.assertEqual(float(out["soft_unscaled"]), 0.)
        self.assertEqual(float(out["maps"]["a_T"].sum()), 0.)
        out = compute_loss(s, gt, gt, pan, spec=config("panda_e"),
                           stats=SimpleNamespace(tau_e=0.), return_maps=True)
        self.assertEqual(float(out["maps"]["d_e"].sum()), 0.)
        self.assertGreater(float(out["soft_unscaled"]), 0.)
        for key in ("d_e", "a_T"):
            self.assertTrue(bool(((out["maps"][key] >= 0)&(out["maps"][key] <= 1)).all()))
        perfect = compute_loss(gt.requires_grad_(), gt, gt, pan, spec=config("panda_e"),
                               stats=SimpleNamespace(tau_e=0.))
        self.assertEqual(float(perfect["loss"]), 0.)

    def test_scharr_absolute_axis_not_signed_difference_and_nonRGB_hard(self):
        yy, xx = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")
        gt = torch.zeros(1, 8, 64, 64)
        for c in (4, 2, 1):
            gt[:, c] = xx
        pan = -xx[None, None]
        e, g, q = reference_maps(ScharrAbsolute("WV3"), gt.clone(), gt, pan)
        torch.testing.assert_close(q, torch.zeros_like(q), atol=0, rtol=0)
        changed = gt.clone(); changed[:, 7] = 100.
        _, _, q_changed = reference_maps(ScharrAbsolute("WV3"), gt, changed, pan)
        torch.testing.assert_close(q, q_changed, atol=0, rtol=0)
        loss = compute_loss(gt.requires_grad_(), None, changed, pan, spec=config("gt_only"))["loss"]
        self.assertEqual(float(loss), 12.5)

    def test_bad_dtype_nonfinite_rank_and_shapes_fail_closed(self):
        s, t, gt, pan = images()
        with self.assertRaises(ValueError):
            compute_loss(s.double(), t, gt, pan, spec=config(), stats=FixedRanks())
        class BadRanks(FixedRanks):
            def lookup_exact(self, g, q):
                return torch.full_like(q, 1.1)
        with self.assertRaisesRegex(ValueError, "ranks"):
            compute_loss(s, t, gt, pan, spec=config(), stats=BadRanks(), scharr=ScharrAbsolute("WV3"))

    def test_full48_equals_micro4_times12_student_gradient(self):
        rng = torch.Generator().manual_seed(73)
        gt = torch.rand(48, 8, 64, 64, generator=rng)
        base = torch.rand(48, 8, 64, 64, generator=rng)
        teacher = gt + .1
        pan = torch.rand(48, 1, 64, 64, generator=rng)
        spec, scharr = config(), ScharrAbsolute("WV3")
        full_parameter = torch.tensor(.2, requires_grad=True)
        full = compute_loss(base + full_parameter, teacher, gt, pan,
                            spec=spec, scharr=scharr, stats=FixedRanks())
        full["loss"].backward()
        micro_parameter = torch.tensor(.2, requires_grad=True)
        for first in range(0, 48, 4):
            sl = slice(first, first+4)
            micro = compute_loss(base[sl] + micro_parameter, teacher[sl], gt[sl], pan[sl],
                                 spec=spec, scharr=scharr, stats=FixedRanks())
            (micro["loss"] * (4/48)).backward()
        torch.testing.assert_close(full_parameter.grad, micro_parameter.grad, rtol=2e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
