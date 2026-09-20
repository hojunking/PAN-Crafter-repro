"""Synthetic CPU checks for distribution/gradient and A-only size diagnostics."""
from __future__ import annotations

from unittest.mock import patch
import random
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qg40.alignment_diagnostics import (alignment_response, border_sampling,
    capture_alignment_sizes, fit_axis16_response, normalization_context, native_pan_ms_proxy)
from qg40.calibration import AXIS16, axis16_q
from qg40.diagnostics import capture_diagnostics
from qg40.losses import student_losses
from qg40.model import build_model, state_hash


class ConstantAligner(nn.Module):
    aligner_margin = 4

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(.1))
        self.child = nn.Dropout()
        self.calls = []

    def predict_delta(self, pan, ms_base):
        self.calls.append(tuple(pan.shape[-2:]))
        return torch.zeros(len(pan), 2, device=pan.device) + self.anchor

    def forward(self, *args, **kwargs):
        raise AssertionError("Size diagnostics must not execute the reconstruction U")


class NativeDataset:
    base_count = 1
    max_pixel = 2047

    def __init__(self, size, constant=False):
        gen = np.random.default_rng(51 + size)
        if constant:
            pan, ms = np.ones((1, 1, size, size), np.float32), np.ones((1, 4, size // 4, size // 4), np.float32)
        else:
            pan = gen.random((1, 1, size, size), dtype=np.float32) * 2047
            ms = gen.random((1, 4, size // 4, size // 4), dtype=np.float32) * 2047
        self.arrays = {"pan": pan, "ms": ms}

    def __len__(self):
        return self.base_count

    def base(self, _index):
        raise AssertionError("A-only diagnostics must not read a GT/LP dataset row")


class AlignmentDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_axis_slopes_ideal_constant_and_cross_axis(self):
        epsilon = torch.tensor(AXIS16)
        native = torch.tensor([[3., -4.], [-2., 5.]])
        ideal = fit_axis16_response(native, native[:, None] - epsilon[None])
        self.assertTrue(torch.equal(ideal["response_matrix"], -torch.eye(2).expand(2, -1, -1)))
        self.assertTrue(torch.equal(ideal["q"], torch.zeros(2)))
        self.assertTrue(torch.equal(ideal["response_gain"], torch.ones(2)))
        constant = fit_axis16_response(native, native[:, None].expand(-1, 16, -1))
        self.assertTrue(torch.equal(constant["response_matrix"], torch.zeros(2, 2, 2)))
        self.assertTrue(torch.equal(constant["q"], torch.full((2,), .46875)))
        self.assertTrue(torch.equal(constant["response_gain"], torch.zeros(2)))
        matrix = torch.tensor([[-.5, .25], [-.75, -1.]])
        shifted = native[:, None] + torch.einsum("oi,ki->ko", matrix, epsilon)[None]
        mixed = fit_axis16_response(native, shifted)
        self.assertTrue(torch.equal(mixed["response_matrix"], matrix.expand(2, -1, -1)))

    def test_response_preserves_axis16_q_contract_and_border_coordinate_definition(self):
        gen = torch.Generator().manual_seed(3)
        pan = torch.randn(2, 1, 32, 32, generator=gen)
        ms = torch.randn(2, 4, 8, 8, generator=gen)
        base = F.interpolate(ms, scale_factor=4, mode="bicubic", align_corners=False)
        model = ConstantAligner()
        old_q, old_radius, old_native = axis16_q(model, pan, ms)
        measured = alignment_response(model, pan, base)
        self.assertTrue(torch.equal(old_q, measured["q"]))
        self.assertTrue(torch.equal(old_radius, measured["per_radius_q"]))
        self.assertTrue(torch.equal(old_native, measured["native_delta"]))
        fraction = border_sampling(torch.tensor([[0., 0.], [0., 1.], [.5, -1.]]), 64, 64)
        expected = torch.tensor([0., 1 / 64, 1 - 63 * 63 / (64 * 64)])
        self.assertTrue(torch.equal(fraction["border_sampling_fraction"], expected))
        self.assertEqual(float(fraction["bicubic_border_support_fraction"][0]), 1 - 61 * 61 / 4096)

    def test_constant_context_is_explicitly_low_confidence(self):
        context = normalization_context(torch.ones(1, 1, 64, 64), torch.ones(1, 4, 64, 64))
        self.assertTrue(torch.equal(context["zscore_variance"], torch.zeros(1, 5)))
        self.assertTrue(torch.equal(context["zscore_epsilon_fraction"], torch.ones(1, 5)))
        self.assertTrue(bool(context["low_texture_channels"].all()))
        self.assertEqual(float(context["low_texture_confidence"][0]), 0.)

    def test_native_proxy_recovers_known_sampling_sign_on_fixed_support(self):
        gen = torch.Generator().manual_seed(991)
        low = torch.randn(1, 1, 16, 16, generator=gen)
        pan = low.repeat_interleave(4, 2).repeat_interleave(4, 3)
        ms = torch.roll(pan, shifts=(4, -4), dims=(2, 3)).expand(-1, 4, -1, -1).clone()
        result = native_pan_ms_proxy(pan, ms)
        self.assertEqual(result["method"]["support_lr"], [14, 14])
        self.assertEqual(len(result["method"]["grid_dydx_lr"]), 9)
        self.assertFalse(result["alignment_ground_truth"])
        for band in result["scenes"][0]["bands"]:
            self.assertEqual(band["best_signal_offset_lr"], [1, -1])
            self.assertEqual(band["best_signal_offset_hr"], [4, -4])
            self.assertAlmostEqual(band["best_correlation"], 1., places=12)
            self.assertTrue(band["best_on_search_boundary"])
            self.assertGreater(band["peak_margin"], .5)

    def test_native_proxy_constant_and_periodic_ambiguity_are_explicit(self):
        constant = native_pan_ms_proxy(torch.ones(1, 1, 64, 64), torch.ones(1, 4, 64, 64))
        self.assertTrue(constant["scenes"][0]["low_pan_texture"])
        for band in constant["scenes"][0]["bands"]:
            self.assertTrue(band["low_texture"])
            self.assertEqual(band["correlations"], [None] * 9)
            self.assertIsNone(band["best_signal_offset_lr"])
        y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
        low = ((x + y) % 2).float()[None, None] * 2 - 1
        pan = low.repeat_interleave(4, 2).repeat_interleave(4, 3)
        periodic = native_pan_ms_proxy(pan, pan.expand(-1, 4, -1, -1))
        self.assertTrue(periodic["scenes"][0]["repeated_pattern_or_aperture_warning"])
        self.assertGreater(periodic["scenes"][0]["bands"][0]["tied_peak_count"], 1)
        self.assertAlmostEqual(periodic["scenes"][0]["bands"][0]["peak_margin"], 0.)

    def test_native_proxy_spectral_polarity_is_not_interpreted_as_shift_ground_truth(self):
        low = torch.randn(1, 1, 16, 16, generator=torch.Generator().manual_seed(992))
        pan = low.repeat_interleave(4, 2).repeat_interleave(4, 3)
        ms = torch.cat((pan, -pan, pan * .7, -pan * .8), 1)
        result = native_pan_ms_proxy(pan, ms)
        scene = result["scenes"][0]
        self.assertTrue(scene["spectral_polarity_warning"])
        self.assertTrue(scene["per_band_preferred_offsets_disagree"])
        negative = scene["bands"][1]
        self.assertAlmostEqual(negative["zero_offset_correlation"], -1.)
        self.assertEqual(negative["strongest_absolute_signal_offset_lr"], [0, 0])
        self.assertFalse(result["alignment_ground_truth"])
        self.assertFalse(result["method"]["spectral_response_fitted"])
        self.assertIn("never physical misalignment", result["caveat"])

    def test_native_proxy_preserves_sources_gradients_rng_and_deadline(self):
        pan = torch.randn(1, 1, 64, 64, generator=torch.Generator().manual_seed(993), requires_grad=True)
        ms = pan.detach().expand(-1, 4, -1, -1).clone().requires_grad_(True)
        pan.grad, ms.grad = torch.ones_like(pan), torch.ones_like(ms)
        before_pan, before_ms = pan.detach().clone(), ms.detach().clone()
        before = torch.get_rng_state().clone()
        native_pan_ms_proxy(pan, ms)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(pan, before_pan) and torch.equal(ms, before_ms))
        self.assertTrue(torch.equal(pan.grad, torch.ones_like(pan)))
        self.assertTrue(torch.equal(ms.grad, torch.ones_like(ms)))
        with patch("qg40.alignment_diagnostics.check_deadline", side_effect=TimeoutError("deadline")):
            with self.assertRaises(TimeoutError):
                native_pan_ms_proxy(pan, ms)

    def test_size_probes_are_bounded_a_only_and_preserve_sources_modes_rng_gradients(self):
        datasets = dict(rr=NativeDataset(256), fr=NativeDataset(512, constant=True))
        originals = {(split, key): value.copy() for split, dataset in datasets.items() for key, value in dataset.arrays.items()}
        model = ConstantAligner().train()
        model.child.eval()
        model.anchor.grad = torch.tensor(7.)
        before_rng = torch.get_rng_state().clone()
        before_python, before_numpy = random.getstate(), np.random.get_state()
        before_state = state_hash(model.state_dict())
        original_predict = model.predict_delta
        def consuming_predict(*args):
            random.random(); np.random.random(); torch.rand(())
            return original_predict(*args)
        with patch.object(model, "predict_delta", side_effect=consuming_predict):
            result = capture_alignment_sizes(model, datasets, device="cpu", max_scenes=1)
        self.assertEqual(result["n_views"], 14)
        self.assertEqual(result["reconstruction_forward_calls"], 0)
        self.assertFalse(result["alignment_ground_truth"])
        self.assertEqual(len(model.calls), 14 * 17)
        self.assertEqual(set(model.calls), {(64, 64), (128, 128), (256, 256), (512, 512)})
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(before_python, random.getstate())
        after_numpy = np.random.get_state()
        self.assertEqual(before_numpy[0], after_numpy[0])
        np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
        self.assertEqual(before_numpy[2:], after_numpy[2:])
        self.assertEqual(before_state, state_hash(model.state_dict()))
        self.assertEqual(float(model.anchor.grad), 7.)
        for split, dataset in datasets.items():
            for key, value in dataset.arrays.items():
                np.testing.assert_array_equal(value, originals[(split, key)])
        for row in result["records"]:
            self.assertIn("native_pan_ms_proxy", row)
            self.assertFalse(row["native_pan_ms_proxy"]["alignment_ground_truth"])
            if row["split"] == "fr":
                self.assertEqual(row["context"]["low_texture_confidence"], [0.])
        with self.assertRaises(ValueError):
            capture_alignment_sizes(model, datasets, device="cpu", max_scenes=3)

    def test_size_deadline_preserves_modes_rng_and_gradients(self):
        model = ConstantAligner().train()
        model.child.eval()
        model.anchor.grad = torch.tensor(9.)
        before = torch.get_rng_state().clone()
        with patch("qg40.alignment_diagnostics.check_deadline", side_effect=TimeoutError("deadline")):
            with self.assertRaises(TimeoutError):
                capture_alignment_sizes(model, dict(rr=NativeDataset(256), fr=NativeDataset(512)), device="cpu", max_scenes=1)
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(float(model.anchor.grad), 9.)

    def test_extended_student_diagnostics_match_loss_and_actual_a_gradient(self):
        gen = torch.Generator().manual_seed(99)
        pan = torch.randn(2, 1, 32, 32, generator=gen)
        ms = torch.randn(2, 4, 8, 8, generator=gen)
        lp = torch.randn(2, 1, 8, 8, generator=gen)
        gt = torch.randn(2, 4, 32, 32, generator=gen)
        gt[0].zero_()  # Fixed low-texture stratum, never a sampling/loss edit.
        class Dataset:
            base_count, has_gt = 2, True
            def __len__(self):
                return 2
            def get_view(self, index, rot, augment):
                return gt[index], gt[index], ms[index], lp[index], pan[index], torch.tensor([index, rot])
        teacher, _ = build_model("P0", 8, [1, 1, 1], 81001)
        student, _ = build_model("PLH", 8, [1, 1, 1], 82001, role="S", teacher_aligner_state=teacher.aligner.state_dict())
        with torch.no_grad():
            for model in (teacher, student):
                model.backbone.output[-1].weight.copy_(.04 * torch.randn(model.backbone.output[-1].weight.shape, generator=gen))
                model.aligner.fc2.weight.copy_(.02 * torch.randn(model.aligner.fc2.weight.shape, generator=gen))
        teacher.eval().requires_grad_(False)
        student.train()
        student.backbone.encoder1[0].eval()
        for parameter in student.parameters():
            parameter.grad = torch.full_like(parameter, .375)
        frozen = state_hash(teacher.state_dict())
        q, _, _ = axis16_q(teacher, pan, ms)
        cache = q[:, None].expand(-1, 4).numpy().copy()
        summary, arrays = capture_diagnostics(student, teacher, Dataset(), device="cpu", tau_R=.2,
            q_ref=.4, q_cache=cache, indices=[0, 1], synthetic_test=True, batch_size=2)
        self.assertTrue(student.training)
        self.assertFalse(student.backbone.encoder1[0].training)
        self.assertEqual(summary["frozen_teacher_state_hash"], frozen)
        self.assertTrue(summary["cache_online_q_matches"])
        self.assertEqual(summary["texture_strata"]["low"]["count"], 1)
        self.assertEqual(summary["texture_strata"]["high"]["count"], 1)
        self.assertEqual(arrays["band_l1"].shape, (2, 4))
        self.assertEqual(arrays["difficulty"].shape, (2, 32, 32))
        self.assertEqual(arrays["response_matrix"].shape, (2, 2, 2))
        np.testing.assert_allclose(arrays["difficulty"], arrays["e_teacher"] / (arrays["e_teacher"] + .2))
        np.testing.assert_allclose(arrays["soft_weight"], .1 * (1. - arrays["difficulty"]) * arrays["advantage"])
        self.assertEqual(summary["epsilon_dominated_error_fraction"], float((arrays["e_student"] <= 1e-6).mean()))
        self.assertEqual(summary["soft_weight_mean"], float(arrays["soft_weight"].mean()))
        self.assertTrue(all(torch.equal(p.grad, torch.full_like(p, .375)) for p in student.parameters()))
        out = student(pan[:1], ms[:1], lp[:1])
        with torch.no_grad():
            tout = teacher(pan[:1], ms[:1], lp[:1])
        losses = student_losses(out, tout, gt[:1], .2, [.4 / (.4 + float(q[0]))])
        gradients = torch.autograd.grad(losses["L_A"], list(student.aligner.parameters()))
        expected = sum(float(g.double().square().sum()) for g in gradients) ** .5
        self.assertAlmostEqual(summary["gradient_rows"][0]["a_la_gradient_norm"], expected, places=7)
        self.assertAlmostEqual(float(arrays["hard_loss"][0]), float(losses["hard"]), places=7)
        self.assertAlmostEqual(float(arrays["soft_loss"][0]), float(losses["soft"]), places=7)
        self.assertAlmostEqual(float(arrays["weighted_edge_loss"][0]), float(losses["edge_weighted"]), places=7)


if __name__ == "__main__":
    unittest.main()
