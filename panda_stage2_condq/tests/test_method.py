"""Actual W112-D123 CPU model contracts and frozen-gradient isolation."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from directshift48.method import DirectPANAligner, RGB_INDICES
from pa.warp import warp_pan
from panda_method.calibration import canonical_sha256, state_dict_sha256
from panda_method.edge_data import EdgeBatchLineage, EdgeObservationBatch
from panda_method.reconstruction import build_reconstructor
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.method import ConditionalQStage2, native_support
from panda_stage2_condq.spec import Stage2Spec

torch.set_num_threads(1)


def batch(*, side=64, sensor="WV3", resolution="RR", split="train", count=1):
    generator = torch.Generator().manual_seed(20261016 + side)
    bands = 8 if sensor == "WV3" else 4
    pan = torch.rand(count, 1, side, side, generator=generator)
    ms = torch.rand(count, bands, side//4, side//4, generator=generator)
    lms = torch.rand(count, bands, side, side, generator=generator)
    gt = None if resolution == "FR" else torch.rand(count, bands, side, side, generator=generator)
    lineage = EdgeBatchLineage(split, resolution, "a"*64,
        tuple(f"{sensor}:{split}:{i}" for i in range(count)), sensor, "PAN_"+resolution,
        ((False, False, 0),)*count)
    return EdgeObservationBatch(pan, ms, lms, lineage, gt)


def fixture(variant="panda_condq", *, sensor="WV3", shift=(1., -.5), seed=345):
    spec = Stage2Spec(sensor, variant, seed, 0. if variant == "gt_only" else 1.,
                      0. if variant == "gt_only" else .1,
                      .5 if variant == "panda_condq" else 0., 1e-6)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        aligner = DirectPANAligner(spec.bands)
    with torch.no_grad():
        aligner.fc2.bias.copy_(torch.tensor(shift))
    teacher = None if variant == "gt_only" else build_reconstructor(spec.reconstruction_spec, seed=12, role="T")
    binding = {"sensor": sensor, "source_split": "RR_train_only",
        "sample_ids_sha256": canonical_sha256([sensor+":train:0"]),
        "source_manifest_sha256": "a"*64,
        "aligner_state_sha256": state_dict_sha256(aligner),
        "teacher_state_sha256": "b"*64 if teacher is None else state_dict_sha256(teacher),
        "donor_checkpoint_id": "c"*64,
        "reconstruction_spec": {"bands": spec.bands, "width": 112, "depth": [1, 2, 3],
                                "input_layout": "P0", "base_reference": "BICUBIC_MS"},
        "scharr": {"kernel": "absolute_xy_/32", "padding": "reflect1",
                   "rgb_indices": list(RGB_INDICES[sensor]), "roi_margin": 8},
        "warp": "bicubic_border_FP32_align_corners_false_current_pan_px"}
    stats = None
    if variant != "gt_only":
        collector = CalibrationBuilder([sensor+":train:0"], binding, k=5,
            memory_budget_bytes=4*1024**2, include_condq=variant == "panda_condq")
        e = torch.full((1, 1, 64, 64), .25)
        collector.add([sensor+":train:0"], e,
            torch.zeros_like(e) if variant == "panda_condq" else None,
            torch.zeros_like(e) if variant == "panda_condq" else None, sensor=sensor)
        stats = collector.finalize()
    return ConditionalQStage2(spec, aligner, teacher, donor_binding=binding, stats=stats)


class MethodTests(unittest.TestCase):
    def test_actual_capacity_fresh_paired_student_hash_for_all_variants(self):
        hashes = []
        for variant in ("gt_only", "panda_e", "panda_condq"):
            stage = fixture(variant)
            hashes.append(stage.student_initial_state_sha256)
            self.assertEqual(stage.student.spec.width, 112)
            self.assertEqual(stage.student.spec.depth, (1, 2, 3))
            self.assertEqual(stage.student.spec.input_layout, "P0")
            self.assertEqual(stage.student.spec.base_reference, "BICUBIC_MS")
            self.assertNotEqual(hashes[-1], stage.donor_binding["teacher_state_sha256"])
            if variant == "gt_only":
                self.assertIsNone(stage.teacher); self.assertIsNone(stage.stats)
                self.assertFalse(any(k.startswith("teacher.") for k in stage.state_dict()))
            else:
                self.assertFalse({id(p) for p in stage.student.parameters()} & {id(p) for p in stage.teacher.parameters()})
            del stage
        self.assertEqual(len(set(hashes)), 1)

    def test_one_warp_shared_inputs_frozen_eval_and_student_only_gradient(self):
        stage, rr = fixture(), batch()
        stage.train()
        self.assertFalse(stage.aligner.training); self.assertFalse(stage.teacher.training)
        teacher_args, student_args = [], []
        hook_t = stage.teacher.register_forward_pre_hook(lambda m, args: teacher_args.append(args))
        hook_s = stage.student.register_forward_pre_hook(lambda m, args: student_args.append(args))
        optimizer = torch.optim.AdamW(stage.student.parameters(), lr=1e-4)
        stage.assert_optimizer(optimizer)
        with patch("panda_stage2_condq.method.warp_pan", wraps=warp_pan) as warp:
            out = stage(rr, return_maps=True)
            self.assertEqual(warp.call_count, 1)
            torch.testing.assert_close(warp.call_args[0][1], torch.tensor([[1., -.5]]))
        for t_arg, s_arg in zip(teacher_args[0], student_args[0]):
            self.assertIs(t_arg, s_arg)
        self.assertIs(teacher_args[0][1], rr.ms)
        self.assertIs(teacher_args[0][2], rr.native_lms)
        self.assertLess(float(out["warp_support_fraction"]), 1.)
        self.assertTrue(bool(out["spatial_support_ok"].all()))
        self.assertNotIn("loss_mask", out)
        out["loss"].backward()
        self.assertTrue(all(p.grad is None for m in (stage.aligner, stage.teacher) for p in m.parameters()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in stage.student.parameters()))
        optimizer.step()
        stage.assert_frozen_binding()
        hook_t.remove(); hook_s.remove()
        with self.assertRaises(ValueError):
            stage.assert_optimizer(torch.optim.AdamW(stage.parameters(), lr=1e-4))

    def test_guard_applies_even_gt_only_without_adaptive_mask(self):
        stage = fixture("gt_only", shift=(9., 0.))
        with self.assertRaisesRegex(ValueError, "SUPPORT_VIOLATION"):
            stage(batch())
        self.assertTrue(all(p.grad is None for p in stage.student.parameters()))
        with self.assertRaisesRegex(ValueError, "SUPPORT_VIOLATION"):
            native_support(torch.zeros(1, 1, 64, 64), torch.tensor([[9., 0.]]))

    def test_train_source_lineage_and_float32_are_enforced(self):
        stage, rr = fixture("gt_only"), batch()
        for changed in (replace(rr, lineage=replace(rr.lineage, split="test")),
                        replace(rr, lineage=replace(rr.lineage, source_manifest_sha256="d"*64)),
                        replace(rr, pan=rr.pan.double(), ms=rr.ms.double(),
                                native_lms=rr.native_lms.double(), gt=rr.gt.double())):
            with self.assertRaises(ValueError):
                stage(changed)

    def test_inference_contains_only_A_S_and_ignores_teacher_GT_and_CDF(self):
        stage = fixture()
        inference = stage.inference_model().eval()
        self.assertEqual(set(inference._modules), {"aligner", "student"})
        self.assertFalse(hasattr(inference, "stats"))
        with patch.object(stage.teacher, "forward", side_effect=AssertionError("Teacher used at inference")), \
             patch.object(stage.stats, "lookup_exact", side_effect=AssertionError("CDF used at inference")):
            for side, resolution in ((256, "RR"), (512, "FR")):
                observation = batch(side=side, resolution=resolution, split="test")
                out = inference(observation)
                self.assertEqual(tuple(out["y"].shape), (1, 8, side, side))
                self.assertFalse(out["y"].requires_grad)
                torch.testing.assert_close(out["u"], torch.tensor([[1., -.5]]))
                if resolution == "RR":
                    # A deliberately invalid attached GT cannot influence inference.
                    changed = replace(observation, gt=torch.tensor(float("nan")))
                    torch.testing.assert_close(inference(changed)["y"], out["y"], atol=0, rtol=0)

    def test_strict_checkpoint_old_loss_changed_calibration_and_frozen_mutation_rejected(self):
        stage = fixture()
        state = copy.deepcopy(stage.state_dict())
        stage.load_state_dict(state)
        old = copy.deepcopy(state)
        old["_extra_state"]["method_version"] = "DONOR30_DIRECT_SCHARR_STAGE2_MASKED_V2"
        with self.assertRaisesRegex(ValueError, "Strict same-contract"):
            stage.load_state_dict(old)
        changed = copy.deepcopy(state)
        changed["_extra_state"]["calibration_artifact_sha256"] = "d"*64
        with self.assertRaises(ValueError):
            stage.load_state_dict(changed)
        changed = copy.deepcopy(state)
        changed["aligner.fc2.bias"].add_(1)
        with self.assertRaisesRegex(ValueError, "before load"):
            stage.load_state_dict(changed)
        stage.assert_frozen_binding()
        with torch.no_grad():
            stage.aligner.fc2.bias.add_(1)
        with self.assertRaisesRegex(ValueError, "version changed"):
            stage(batch())


if __name__ == "__main__":
    unittest.main()
