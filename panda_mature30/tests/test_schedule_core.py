"""CPU-only real W112/D123 schedule, calibration and resume regression tests."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from directshift48.method import ScharrAbsolute
from donor30.artifacts import _same
from pa.warp import warp_pan, warp_support_mask
from panda_band30.reconstruction import base_from_batch, reconstruct_with_base
from panda_band30.tests.test_method import make_donor, fixture as old_fixture
from panda_band30.tests.test_calibration import SyntheticInventory
from panda_method.calibration import canonical_sha256, state_dict_sha256
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.losses import reference_maps, compute_loss
from panda_stage2_condq.tests.test_method import batch
from panda_stage2_condq.training import TrainingSpec

from panda_mature30 import calibration as C
from panda_mature30.spec import MatureSpec, geometry_manifest, policy_manifest
from panda_mature30.method import (
    MatureStage2, applied_correction, frozen_predictions, warp_support,
)
from panda_mature30.engine import (
    MatureStudentTrainer, build_fresh_stage2, advance, save_checkpoint, load_checkpoint,
)


def make_spec(policy="DOWN"):
    return MatureSpec("WV3", "panda_condq", 345, .25, .05, .5, 1e-6, policy_id=policy)


def make_fixture(policy="DOWN", shift=(1., -.5)):
    spec, donor, rr = make_spec(policy), make_donor(shift=shift), batch()
    donor = C.bind_donor(donor, spec)
    bank = {}
    for key in spec.geometry_keys:
        f = frozen_predictions(donor.aligner, donor.teacher, rr, geometry_key=key)
        e, g, q = reference_maps(ScharrAbsolute("WV3"), f["teacher_y"], f["gt_loss"], f["pan_loss"])
        collector = CalibrationBuilder(["WV3:train:0"], C.binding_for_geometry(donor.binding, key),
            k=5, memory_budget_bytes=8*1024**2, include_condq=True, mask_policy=spec.mask_policy)
        collector.add(["WV3:train:0"], e, g, q, sensor="WV3", loss_mask=f["loss_mask"])
        bank[key] = collector.finalize()
    stage = MatureStage2(spec, donor.aligner, donor.teacher,
                        donor_binding=donor.binding, stats_bank=bank)
    with torch.no_grad():
        stage.student.backbone.output[-1].weight.copy_(stage.teacher.backbone.output[-1].weight)
    return stage, donor, bank


def make_trainer(policy="DOWN"):
    stage, _, _ = make_fixture(policy)
    training = TrainingSpec(total_updates=50000, warmup_updates=0,
        microbatch_size=1, effective_batch_size=1, full_audit_interval=1)
    return MatureStudentTrainer(stage, training)


def synthetic_boundary_state(trainer, update):
    """Consistent synthetic Adam state to test counters, not claim 10K training."""
    state = deepcopy(trainer.state_dict())
    state["updates"] = update
    state["scheduler"]["completed_updates"] = update
    state["stage"]["_completed_update_state"].fill_(update)
    for group in state["optimizer"]["param_groups"]:
        group["lr"] = trainer.training.rate(update)
    for moments in state["optimizer"]["state"].values():
        moments["step"].fill_(update)
    return state


class SyntheticStream:
    count = 1

    def __init__(self):
        self.epoch, self.cursor = 0, 0

    def next_update(self):
        self.epoch += 1
        return [batch()]

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state):
        self.epoch, self.cursor = state["epoch"], state["cursor"]


class ScheduleCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_registered_policy_boundaries_and_manifest_are_explicit(self):
        spec = make_spec()
        for completed, evaluation_key, next_key in (
                (0, "s050", "s050"), (9999, "s050", "s050"),
                (10000, "s050", "s0375"), (10001, "s0375", "s0375"),
                (30000, "s0375", "s025"), (30001, "s025", "s025"),
                (49999, "s025", "s025")):
            self.assertEqual(spec.geometry_at_completed(completed), evaluation_key)
            self.assertEqual(spec.geometry_at_next(completed), next_key)
        self.assertEqual(spec.geometry_at_completed(50000), "s025")
        with self.assertRaises(ValueError):
            spec.geometry_at_next(50000)
        self.assertEqual(spec.geometry_keys, ("s050", "s0375", "s025"))
        self.assertEqual(policy_manifest(spec)["schedule"], "PIECEWISE_CONSTANT")
        self.assertIn("INTERP23", spec.manifest()["student"]["base_reference"])
        for changed in ({"policy_id": "ramp"}, {"upsampler": "bicubic"},
                        {"alpha": .5}, {"frame": "mid"}, {"aligner_outputs": 16}):
            with self.assertRaises(ValueError):
                replace(spec, **changed)

    def test_cap_is_radial_direction_preserving_finite_and_current_pixel(self):
        u = torch.tensor([[0., 0.], [.3, .4], [3., 4.], [-4., 3.]])
        result = applied_correction(u, "cap085")
        expected = torch.tensor([[0., 0.], [.3, .4], [.51, .68], [-.68, .51]])
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(torch.linalg.vector_norm(result, dim=1),
                                   torch.tensor([0., .5, .85, .85]))
        with self.assertRaises(ValueError):
            applied_correction(u.double(), "cap085")

    def test_teacher_gt_base_stay_fixed_and_mask_tracks_actual_geometry(self):
        rr, donor = batch(), make_donor(shift=(2., -1.))
        base = base_from_batch(rr, "interp23")
        u = donor.aligner(rr.pan, rr.native_lms)
        teacher = reconstruct_with_base(donor.teacher, warp_pan(rr.pan, u), base, rr.ms)["y"]
        for key in ("s100", "s050", "s0375", "s025", "cap085"):
            f = frozen_predictions(donor.aligner, donor.teacher, rr, geometry_key=key)
            applied = applied_correction(u, key)
            torch.testing.assert_close(f["pan_input"], warp_pan(rr.pan, applied), atol=0, rtol=0)
            torch.testing.assert_close(f["teacher_y"], teacher, atol=0, rtol=0)
            self.assertIs(f["gt_loss"], rr.gt)
            torch.testing.assert_close(f["ms_base_frame"], base, atol=0, rtol=0)
            expected = warp_support_mask(64, 64, applied) & warp_support_mask(64, 64, u)
            self.assertTrue(torch.equal(f["loss_mask"], expected))
            torch.testing.assert_close(f["ms_correction"], torch.zeros_like(u))
            torch.testing.assert_close(f["output_correction"], torch.zeros_like(u))
        with self.assertRaisesRegex(ValueError, "SUPPORT_VIOLATION"):
            warp_support(64, 64, torch.tensor([[20., 0.]]), "cap085")

    def test_fixed_half_numerically_matches_existing_band30_loss(self):
        new, _, _ = make_fixture("HALF")
        old = old_fixture("pan", .5)
        rr = batch()
        first, second = new(rr, return_maps=True), old(rr, return_maps=True)
        for name in ("loss", "loss_mask", "teacher_y", "gt_loss", "pan_loss"):
            torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)
        torch.testing.assert_close(first["student"]["y"], second["student"]["y"], atol=0, rtol=0)
        for name in first["maps"]:
            torch.testing.assert_close(first["maps"][name], second["maps"][name], atol=0, rtol=0)

    def test_forward_uses_next_phase_but_inference_uses_completed_phase(self):
        stage, _, _ = make_fixture()
        stage.set_completed_update(10000)
        contract = deepcopy(stage.get_extra_state())
        out = stage(batch(), return_maps=True)
        self.assertEqual(out["geometry_key"], "s0375")
        self.assertEqual(stage.deployment_manifest()["geometry_key"], "s050")
        torch.testing.assert_close(out["pan_correction"], out["u"] * .375)
        terms = compute_loss(out["student"]["y"], out["teacher_y"], out["gt_loss"], out["pan_loss"],
            spec=stage.spec, scharr=stage.spatial, stats=stage.stats_bank["s0375"],
            loss_mask=out["loss_mask"], return_maps=True)
        torch.testing.assert_close(out["loss"], terms["loss"], atol=0, rtol=0)
        out["loss"].backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum())
                            for p in stage.student.parameters()))
        self.assertTrue(all(p.grad is None for m in (stage.aligner, stage.teacher)
                            for p in m.parameters()))
        stage.set_completed_update(10001)
        self.assertEqual(stage.deployment_manifest()["geometry_key"], "s0375")
        self.assertEqual(stage.get_extra_state(), contract)
        self.assertEqual(int(stage.state_dict()["_completed_update_state"]), 10001)

    def test_inference_has_no_teacher_gt_calibration_and_same_rr_fr_rule(self):
        for policy in ("DOWN", "CAP085"):
            stage, _, _ = make_fixture(policy, shift=(1., -1.))
            stage.set_completed_update(30001)
            inference = stage.inference_model()
            self.assertEqual(set(inference._modules), {"aligner", "student"})
            with patch.object(stage.teacher, "forward", side_effect=AssertionError("Teacher forbidden")):
                for resolution in ("RR", "FR"):
                    rr = batch(side=64, resolution=resolution, split="test")
                    out = inference(replace(rr, gt=torch.tensor(float("nan"))))
                    key = stage.spec.geometry_at_completed(30001)
                    torch.testing.assert_close(out["pan_correction"], applied_correction(out["u"], key))
                    self.assertNotIn("loss_mask", out)
                    self.assertFalse(out["y"].requires_grad)

    def test_missing_wrong_phase_and_mutated_tables_fail_closed(self):
        stage, donor, bank = make_fixture()
        with self.assertRaisesRegex(ValueError, "every and only"):
            MatureStage2(stage.spec, donor.aligner, donor.teacher,
                         donor_binding=donor.binding, stats_bank={"s050": bank["s050"]})
        wrong = dict(bank, s0375=bank["s050"])
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            C.validate_stats_bank(wrong, donor.binding, stage.spec)
        before = state_dict_sha256(stage.student)
        saved = deepcopy(stage.state_dict())
        saved["_extra_state"]["pipeline"]["mature30_policy"]["phases"][0]["last_update"] = 9999
        with self.assertRaisesRegex(ValueError, "same-contract"):
            stage.load_state_dict(saved)
        self.assertEqual(state_dict_sha256(stage.student), before)
        with self.assertRaises(MemoryError):
            stage.move_calibration("cpu", memory_budget_bytes=1)
        stage.stats_bank["s050"]._q[0].add_(1.)
        with self.assertRaisesRegex(ValueError, "immutable calibration"):
            stage._assert_contract()

    def test_complete_trainer_resume_crosses_each_phase_without_optimizer_reset(self):
        initial = make_trainer()
        initial.step_accumulated([batch()])
        for before in (9999, 10000, 29999, 30000):
            state = synthetic_boundary_state(initial, before)
            first, second = make_trainer(), make_trainer()
            # PyTorch optimizer loading may share tensor storage with the
            # supplied state. Independent deserializations are independent.
            first.load_state_dict(deepcopy(state))
            second.load_state_dict(deepcopy(state))
            previous_parameters = tuple(id(p) for p in first.stage.student.parameters())
            previous_optimizer = id(first.optimizer)
            expected = first.stage.spec.geometry_at_next(before)
            one = first.step_accumulated([batch()])
            two = second.step_accumulated([batch()])
            self.assertEqual(one["mature30_geometry_key"], expected)
            self.assertEqual(one["mature30_deployment"]["geometry_key"], expected)
            self.assertEqual(one["update"], before + 1)
            self.assertEqual(first.stage.completed_update, before + 1)
            self.assertEqual(id(first.optimizer), previous_optimizer)
            self.assertEqual(tuple(id(p) for p in first.stage.student.parameters()), previous_parameters)
            self.assertTrue(_same(first.state_dict(), second.state_dict()))
        broken = deepcopy(state)
        broken["stage"]["_completed_update_state"].sub_(1)
        pristine = make_trainer()
        before_hash = state_dict_sha256(pristine.stage.student)
        with self.assertRaisesRegex(ValueError, "schedule state"):
            pristine.load_state_dict(broken)
        self.assertEqual(state_dict_sha256(pristine.stage.student), before_hash)

    def test_phase_counter_drift_rejected(self):
        trainer = make_trainer()
        trainer.stage.set_completed_update(1)
        with self.assertRaisesRegex(ValueError, "counter"):
            trainer.step_accumulated([batch()])

    def test_engine_complete_checkpoint_with_stream_and_next_update(self):
        from panda_mature30 import plan
        cfg = plan.resolve_case("s1", "DOWN")
        cfg["stage2_spec"]["seed"] = 345
        cfg["stream_seeds"]["S_init"] = 345
        cfg["training_spec"] = asdict(TrainingSpec(total_updates=50000, warmup_updates=0,
            microbatch_size=1, effective_batch_size=1, full_audit_interval=1))
        _, donor, bank = make_fixture()
        cfg["donor_binding"] = deepcopy(donor.binding)
        pin = {"protocol": "SYNTHETIC_ONLY"}
        first = build_fresh_stage2(cfg, donor, bank, evaluator_pin=pin)
        stream = SyntheticStream()
        advance(first, stream)
        pins = {"code_pin": {"synthetic.py": "d"*64}, "evaluator_pin": pin}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            saved = save_checkpoint(path, first, stream, **pins)
            _, second_donor, second_bank = make_fixture()
            second = build_fresh_stage2(cfg, second_donor, second_bank, evaluator_pin=pin)
            second_stream = SyntheticStream()
            load_checkpoint(path, saved["file_sha256"], second, second_stream, **pins)
            self.assertTrue(_same(first.state_dict(), second.state_dict()))
            self.assertEqual(stream.state_dict(), second_stream.state_dict())
            advance(first, stream)
            advance(second, second_stream)
            self.assertTrue(_same(first.state_dict(), second.state_dict()))
            self.assertEqual(stream.state_dict(), second_stream.state_dict())
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, first, stream, **pins)
            with self.assertRaisesRegex(ValueError, "deployment"):
                save_checkpoint(Path(directory)/"wrong_eval.pt", first, stream,
                                evaluation={}, **pins)

    def test_full_train_phase_calibration_artifact_load_and_reuse(self):
        spec, donor, inventory = make_spec(), make_donor(), SyntheticInventory()
        donor.binding.update(sample_count=3, sample_ids_sha256=canonical_sha256(list(inventory.sample_ids)))
        with tempfile.TemporaryDirectory() as directory:
            artifacts = {}
            for key in spec.geometry_keys:
                path = Path(directory) / (key + ".npz")
                with patch.object(C, "code_pin", return_value={"synthetic": "fixed"}):
                    artifacts[key] = C.calibrate_geometry(
                        donor, inventory, spec=spec, geometry_key=key, output=path, k=5,
                        batch_size=2, memory_budget_bytes=8*1024**2)
                self.assertEqual(artifacts[key]["sample_count"], 3)
                self.assertEqual(artifacts[key]["n_q"], 3*48*48)
                self.assertEqual(artifacts[key]["geometry_key"], key)
            bank = C.load_stats_bank(donor, spec, artifacts, memory_budget_bytes=8*1024**2)
            C.validate_stats_bank(bank, C.bind_donor(donor, spec).binding, spec, full_hash=True)
            # Same applied geometry and donor give exactly reusable HALF stats,
            # without relabeling a .5 table as .375 or altering its source ID.
            half = make_spec("HALF")
            reused = C.load_stats_bank(donor, half, {"s050": artifacts["s050"]},
                                      memory_budget_bytes=8*1024**2)
            self.assertEqual(reused["s050"].artifact_sha256, bank["s050"].artifact_sha256)
            wrong = {**artifacts, "s0375": artifacts["s050"]}
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                C.load_stats_bank(donor, spec, wrong, memory_budget_bytes=8*1024**2)
            with self.assertRaises(MemoryError):
                C.load_stats_bank(donor, spec, artifacts, memory_budget_bytes=10)


if __name__ == "__main__":
    unittest.main()
