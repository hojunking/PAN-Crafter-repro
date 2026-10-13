"""Disposable CPU tensors only: no campaign, real dataset, GPU or evaluation writes."""
import copy
from dataclasses import replace
import unittest

import torch
from torch.nn import functional as F

from qg40.model import build_model as legacy_build
from pa.warp import warp_pan
from .calibration import state_dict_sha256
from .data import BatchLineage, ObservationBatch
from .geometry import SceneAwarePANAligner
from .losses import AbsoluteGradientLoss, SpatialLossSpec
from .reconstruction import ReconstructionSpec, build_reconstructor, aligned_observations
from .stages import Stage1, Stage2, TrainingSources, calibrate_training, epsilon_loss


torch.set_num_threads(1)
SOURCES = TrainingSources("a" * 64, "b" * 64, "c" * 64)


def spatial():
    return AbsoluteGradientLoss(SpatialLossSpec("QB", (2, 1, 0),
                              "forward_difference", "reflect", 3, "unit_test_common_0_1"))


def recon(*, role="T", bands=4, layout="P0"):
    return build_reconstructor(ReconstructionSpec(bands, 8, (1, 1, 1), layout, "BICUBIC_MS"),
                               seed=42, role=role)


def batch(resolution="RR", *, split="train", bands=4, tag="one", side=16):
    rng = torch.Generator().manual_seed(21 + side)
    p = torch.rand(2, 1, side, side, generator=rng)
    m = torch.rand(2, bands, side // 4, side // 4, generator=rng)
    lms = F.interpolate(m, scale_factor=4, mode="nearest")
    lineage = BatchLineage(split, resolution,
        SOURCES.rr_manifest_sha256 if resolution == "RR" else SOURCES.fr_manifest_sha256,
        SOURCES.prior_mapping_sha256, ("synthetic-A", "synthetic-B"),
        (f"{tag}-0", f"{tag}-1"), resolution, "synthetic identity crop/augmentation",
        "synthetic training-only native observations" if resolution == "FR" else "")
    return ObservationBatch(p, m, lms, torch.tensor([[.3, -.4], [-.2, .1]]), lineage,
        gt=torch.rand(2, bands, side, side, generator=rng) if resolution == "RR" else None,
        lpan=torch.rand(2, 1, side // 4, side // 4, generator=rng))


def stage1():
    return Stage1(SceneAwarePANAligner(4, view_margin=0), recon(), spatial(),
                  sources=SOURCES, epsilon_seed=123, boundary_policy="report",
                  calibration_protocol="synthetic fixed observations; no augmentation")


def calibrated():
    model = stage1()
    rr = batch()
    scales = calibrate_training(model, [rr], expected_sample_ids=rr.lineage.sample_ids)
    return model, rr, scales


class ReconstructionTests(unittest.TestCase):
    def test_legacy_state_and_outputs_preserved_all_layouts_and_bands(self):
        for bands in (4, 8):
            for layout in ("P0", "PL", "PH", "PLH"):
                with self.subTest(bands=bands, layout=layout):
                    old, _ = legacy_build(layout, 8, (1, 1, 1), 42, role="T", bands=bands)
                    new = recon(bands=bands, layout=layout)
                    self.assertEqual(state_dict_sha256(old.backbone), state_dict_sha256(new.backbone))
                    # Nonzero output makes the comparison sensitive to every input, not just the base.
                    with torch.no_grad():
                        new.backbone.output[-1].weight.fill_(.01)
                    old.backbone.load_state_dict(new.backbone.state_dict(), strict=True)
                    rr = batch(bands=bands)
                    c = rr.prior_dy_dx
                    p, low = aligned_observations(rr.pan, c, lpan=rr.lpan, need_lpan=layout != "P0")
                    actual = new(p, rr.ms, rr.native_lms, aligned_lpan=low)
                    expected = old(rr.pan, rr.ms, rr.lpan, delta_override=c)
                    torch.testing.assert_close(actual["x_in"], expected["x_in"], rtol=0, atol=0)
                    torch.testing.assert_close(actual["y"], expected["y"], rtol=0, atol=0)

    def test_reference_is_not_reconstruction_base(self):
        rr = batch()
        out = recon()(rr.pan, rr.ms, rr.native_lms)
        bicubic = F.interpolate(rr.ms, scale_factor=4, mode="bicubic", align_corners=False)
        self.assertFalse(torch.equal(rr.native_lms, bicubic))
        torch.testing.assert_close(out["ms_base"], bicubic)
        model = stage1()
        seen = []
        handle = model.aligner.register_forward_pre_hook(lambda _, args: seen.append(args[1]))
        model.native_prediction(rr)
        handle.remove()
        torch.testing.assert_close(seen[0], rr.native_lms)


class Stage1Tests(unittest.TestCase):
    def test_exact_objective_one_teacher_four_aligner_calls(self):
        model, rr, fr = stage1(), batch(), batch("FR", side=24)
        calls = []
        teacher_calls = []
        h1 = model.aligner.register_forward_pre_hook(lambda _, args: calls.append(args))
        h2 = model.teacher.register_forward_pre_hook(lambda _, args: teacher_calls.append(args))
        out = model(rr, fr)
        h1.remove(); h2.remove()
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(teacher_calls), 1)
        self.assertIs(teacher_calls[0][0], out["native"]["pan_aligned"])
        torch.testing.assert_close(out["loss"], out["rec"] + out["abs_rr"]
                                   + .001 * (out["eps_rr"] + out["eps_fr"]))
        torch.testing.assert_close(calls[0][0], rr.pan)
        torch.testing.assert_close(calls[1][0], warp_pan(rr.pan, out["epsilon_rr"]))
        torch.testing.assert_close(calls[2][0], fr.pan)
        torch.testing.assert_close(calls[3][0], warp_pan(fr.pan, out["epsilon_fr"]))
        self.assertFalse(calls[1][0].requires_grad)
        self.assertFalse(calls[3][0].requires_grad)
        self.assertIs(calls[0][2], calls[1][2])
        self.assertIs(calls[2][2], calls[3][2])
        self.assertLessEqual(float(out["epsilon_rr"].norm(dim=-1).max()), 2.)
        out2 = model(rr, fr)
        self.assertFalse(torch.equal(out["epsilon_rr"], out2["epsilon_rr"]))
        self.assertEqual(out2["forward_index"], 1)

    def test_stop_gradient_target_only(self):
        c0 = torch.tensor([[2., 1.]], requires_grad=True)
        ce = torch.tensor([[1., 2.]], requires_grad=True)
        eps = torch.tensor([[.2, -.3]], requires_grad=True)
        epsilon_loss(ce, c0, eps).backward()
        self.assertIsNone(c0.grad)
        self.assertIsNone(eps.grad)
        self.assertGreater(float(ce.grad.abs().sum()), 0.)
        ideal = c0.detach() - eps.detach()
        # Subtract/add of non-binary fractions accumulates float32 roundoff.
        self.assertLess(float(epsilon_loss(ideal, c0, eps)), 1e-7)

    def test_loss_routes(self):
        for term in ("rec", "abs_rr", "eps_rr", "eps_fr"):
            model = stage1()
            # Fresh zero output blocks rec->A by construction. Activate the
            # disposable projection to verify the true downstream graph.
            with torch.no_grad():
                model.teacher.backbone.output[-1].weight.normal_(std=.01)
            out = model(batch(), batch("FR", side=24))
            out[term].backward()
            a_grad = sum(float(p.grad.abs().sum()) for p in model.aligner.parameters() if p.grad is not None)
            self.assertGreater(a_grad, 0, term)
            t_grads = [p.grad for p in model.teacher.parameters() if p.grad is not None]
            if term == "rec":
                self.assertGreater(sum(float(g.abs().sum()) for g in t_grads), 0.)
            else:
                self.assertEqual(len(t_grads), 0)

    def test_resume_rng_and_no_global_rng_consumption(self):
        model = stage1()
        rr, fr = batch(), batch("FR")
        global_before = torch.random.get_rng_state().clone()
        model(rr, fr)
        self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))
        saved = copy.deepcopy(model.state_dict())
        expected = model(rr, fr)
        restored = stage1()
        restored.load_state_dict(saved, strict=True)
        actual = restored(rr, fr)
        for key in ("epsilon_rr", "epsilon_fr", "loss"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(actual["forward_index"], expected["forward_index"])

    def test_reject_test_or_missing_fr_lineage_and_no_fake_gt(self):
        model = stage1()
        for fr in (batch("FR", split="test"),
                   replace(batch("FR"), gt=batch().gt),
                   replace(batch("FR"), lineage=replace(batch("FR").lineage, native_training_evidence=""))):
            with self.assertRaises(ValueError):
                model(batch(), fr)
        with self.assertRaises(ValueError):
            model(batch(split="validation"), batch("FR"))
        self.assertEqual(model.forward_calls, 0)

    def test_strict_checkpoint_policy_and_support_guard(self):
        model = stage1()
        saved = copy.deepcopy(model.state_dict())
        with self.assertRaises(ValueError):
            model.load_state_dict(saved, strict=False)
        bad = copy.deepcopy(saved)
        bad["_extra_state"]["pipeline"]["aligner_view_margin"] = 9
        before = state_dict_sha256(model.aligner)
        with self.assertRaises(ValueError):
            model.load_state_dict(bad)
        self.assertEqual(before, state_dict_sha256(model.aligner))
        model.boundary_policy = "require_spatial_support"
        far = replace(batch(), prior_dy_dx=torch.full((2, 2), 20.))
        with self.assertRaisesRegex(ValueError, "support"):
            model(far, batch("FR"))
        self.assertEqual(model.forward_calls, 0)


class Stage2Tests(unittest.TestCase):
    def test_calibration_population_and_freeze_shared_pan(self):
        first, rr, scales = calibrated()
        self.assertEqual(scales.n_samples, 2)
        self.assertEqual(scales.n_e, 2 * 16 * 16)
        second = Stage2(first, recon(role="S", layout="PLH"), scales)
        self.assertIs(second.aligner, first.aligner)
        second.train()
        self.assertFalse(second.aligner.training)
        self.assertFalse(second.teacher.training)
        self.assertTrue(second.student.training)
        before = (state_dict_sha256(second.aligner), state_dict_sha256(second.teacher))
        opt = torch.optim.SGD(second.trainable_parameters(), lr=.01)
        second.assert_optimizer(opt)
        out = second(rr)
        self.assertIs(out["teacher"]["pan_aligned"], out["student"]["pan_aligned"])
        out["loss"].backward()
        opt.step()  # Synthetic disposable one-step unit check, not a campaign.
        self.assertEqual(before, (state_dict_sha256(second.aligner), state_dict_sha256(second.teacher)))
        self.assertTrue(all(p.grad is None for p in second.aligner.parameters()))
        self.assertTrue(all(p.grad is None for p in second.teacher.parameters()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in second.student.parameters()))
        for key in ("e", "q", "w_e", "w_q"):
            self.assertFalse(out[key].requires_grad)
        with self.assertRaises(ValueError):
            second.assert_optimizer(torch.optim.SGD(second.parameters(), lr=.01))

    def test_stale_calibration_and_parameter_alias_rejected(self):
        first, rr, scales = calibrated()
        with self.assertRaises(ValueError):
            Stage2(first, first.teacher, scales)
        with torch.no_grad():
            first.aligner.fc2.bias.add_(.1)
        with self.assertRaisesRegex(ValueError, "stale"):
            Stage2(first, recon(role="S"), scales)

    def test_student_only_fr_inference_and_stage2_checkpoint(self):
        first, rr, scales = calibrated()
        second = Stage2(first, recon(role="S", layout="PLH"), scales)
        second.load_state_dict(copy.deepcopy(second.state_dict()))
        deploy = second.inference_model().eval()
        self.assertFalse(hasattr(deploy, "teacher"))
        fr = batch("FR", split="test", side=24)
        out = deploy(fr)
        self.assertEqual(out["y"].shape, (2, 4, 24, 24))
        with self.assertRaises(ValueError):
            deploy(rr)
        with self.assertRaises(ValueError):
            second(batch(split="test"))

    def test_frozen_ownership_and_changed_pipeline_rejected(self):
        first, rr, scales = calibrated()
        second = Stage2(first, recon(role="S"), scales)
        with self.assertRaisesRegex(ValueError, "trainable"):
            first(rr, batch("FR"))
        second.aligner.view_margin = 1
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            second(rr)

    def test_incomplete_training_calibration_is_not_published(self):
        first, rr = stage1(), batch()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            calibrate_training(first, [rr], expected_sample_ids=[*rr.lineage.sample_ids, "missing"])


if __name__ == "__main__":
    unittest.main()
