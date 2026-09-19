#!/usr/bin/env python3
"""CPU/tmp-only tests of A10, donor isolation, diagnostics and exact new resume."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from diffusers.optimization import get_scheduler

import fh20r1.training as T
import fh20r1.references as R
from fh20r1.plan import active_cases, build_config, N2_TEACHER_RUN, case_for
from fh12.model import build_model, state_hash
from tools.fh12_training_tests import TinyData, TinyModel, fake_build, fake_result


def optimizer():
    u = torch.nn.Parameter(torch.ones(())); a = torch.nn.Parameter(torch.ones(()))
    return torch.optim.AdamW([dict(params=[u], lr=1e-4, name="U"),
                              dict(params=[a], lr=3e-6, name="A")])


class SyntheticRun:
    def __init__(self, root):
        self.root = Path(root)
        self.case = active_cases("s1", include_reserve=False)[0]
        self.cfg = build_config(self.case)
        self.cfg.update(batch_size=2, num_worker=0, num_iter=3, num_warmup=0, log_iter=1)
        self.path = self.root / "config.yaml"; self.path.write_text(yaml.safe_dump(self.cfg))
        self.wd = self.root / self.cfg["work_dir"]
        data_path = self.root / self.cfg["fh20r1"]["dataset_manifest"]
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(json.dumps({"test": True}))
        self.data = TinyData()

    def reference(self, *args, **kwargs):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(918)
            model = TinyModel().eval().requires_grad_(False)
        return model, dict(tau_R=.1, q_ref=.5), np.full((8, 4), .5, np.float32), {"synthetic": "reference"}

    def run(self, evaluator, resume=False, inject_pause=False):
        with patch.object(T, "validate_config", return_value=self.case), \
             patch.object(T, "runtime_context", return_value={}), \
             patch.object(T, "source_identity", return_value={"test": True}), \
             patch.object(T, "build_dataset", side_effect=lambda *a, **kw: self.data), \
             patch.object(T, "build_model", side_effect=fake_build), \
             patch.object(T, "FRMetrics", return_value=object()), \
             patch.object(T, "evaluate_model", side_effect=evaluator), \
             patch.object(T, "GRID_STEPS", (1, 3)), \
             patch.object(T, "FULLSTATE_STEPS", (2, 3)), \
             patch.object(T, "DIAGNOSTIC_EVERY", 2), \
             patch.object(R, "load_training_reference", side_effect=self.reference), \
             patch.object(R, "calibration_indices", return_value=np.arange(8)):
            return T.train_run(self.path, "cpu", resume=resume, root=self.root)

    def state(self):
        return torch.load(self.wd / "last/training_state.pt", map_location="cpu", weights_only=False)


class FH20R1TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_base_cosine_is_exact_fh12_scheduler(self):
        a, b = optimizer(), optimizer()
        new = T.make_scheduler(a, "BASE")
        old = get_scheduler("cosine", optimizer=b, num_warmup_steps=100, num_training_steps=50000)
        for step in range(50001):
            self.assertEqual([g["lr"] for g in a.param_groups], [g["lr"] for g in b.param_groups], step)
            a.step(); b.step(); new.step(); old.step()

    def test_a10_boundary_and_nonaccumulating_multiplier(self):
        base, a10 = optimizer(), optimizer()
        sb, sa = T.make_scheduler(base, "BASE"), T.make_scheduler(a10, "A10")
        for step in range(10005):
            self.assertEqual(base.param_groups[0]["lr"], a10.param_groups[0]["lr"])
            multiplier = 1. if step < 10000 else 1. / 3.
            self.assertAlmostEqual(a10.param_groups[1]["lr"], base.param_groups[1]["lr"] * multiplier, places=19)
            base.step(); a10.step(); sb.step(); sa.step()
        self.assertEqual(T.aligner_factor(9999, "A10"), 1.)
        self.assertEqual(T.aligner_factor(10000, "A10"), 1. / 3.)
        self.assertEqual(T.aligner_factor(25000, "A10"), 1. / 3.)

    def test_a10_schedule_resume_on_switch_is_exact(self):
        original = optimizer(); schedule = T.make_scheduler(original, "A10")
        for _ in range(10000): original.step(); schedule.step()
        op_state, sch_state = copy.deepcopy(original.state_dict()), copy.deepcopy(schedule.state_dict())
        resumed = optimizer(); resumed_sch = T.make_scheduler(resumed, "A10")
        resumed.load_state_dict(op_state); resumed_sch.load_state_dict(sch_state)
        self.assertEqual(resumed_sch.last_epoch, 10000)
        for _ in range(5):
            self.assertEqual([g["lr"] for g in original.param_groups], [g["lr"] for g in resumed.param_groups])
            original.step(); schedule.step(); resumed.step(); resumed_sch.step()

    def test_10000_is_fullstate_only_not_official_grid(self):
        self.assertIn(10000, T.FULLSTATE_STEPS)
        self.assertNotIn(10000, T.GRID_STEPS)
        self.assertEqual(len(T.GRID_STEPS), 50)

    def test_n2_only_loads_a_preserves_head_and_identical_fresh_u(self):
        case = replace(case_for(N2_TEACHER_RUN), width=8, depth=(1, 1, 1))
        original, initial = build_model("PL", 8, [1, 1, 1], 71001)
        donor = copy.deepcopy(original.aligner.state_dict())
        donor["fc2.weight"].fill_(.17); donor["fc2.bias"].fill_(.23)
        changed, manifest = T.build_n2_teacher(case, donor, {"status": "VERIFIED"}, initial)
        self.assertEqual(state_hash(changed.backbone.state_dict()), state_hash(original.backbone.state_dict()))
        self.assertTrue(torch.equal(changed.aligner.fc2.weight, donor["fc2.weight"]))
        self.assertTrue(torch.equal(changed.aligner.fc2.bias, donor["fc2.bias"]))
        self.assertFalse(manifest["from_scratch"])
        self.assertTrue(manifest["fresh_F2_U_verified"])
        self.assertEqual(manifest["pretrained_aligner_loads"], 1)
        self.assertNotEqual(changed.aligner.fc2.weight.data_ptr(), donor["fc2.weight"].data_ptr())
        with self.assertRaisesRegex(ValueError, "BLOCKED_DONOR"):
            T.build_n2_teacher(case, None, {"status": "BLOCKED_DONOR"}, initial)
        altered = copy.deepcopy(initial); altered["hashes"]["body"] = "wrong"
        with self.assertRaisesRegex(ValueError, "differs from F2"):
            T.build_n2_teacher(case, donor, {"status": "VERIFIED"}, altered)

    def test_diagnostics_leave_model_actual_gradients_and_rng_unchanged(self):
        torch.manual_seed(51); model = TinyModel().train()
        teacher = TinyModel().eval().requires_grad_(False)
        for parameter in model.parameters(): parameter.grad = torch.full_like(parameter, .731)
        before = {name: p.grad.clone() for name, p in model.named_parameters()}
        model_hash, teacher_hash = state_hash(model.state_dict()), state_hash(teacher.state_dict())
        batch = torch.utils.data.default_collate([TinyData()[i] for i in range(2)])
        random.seed(22); np.random.seed(23); torch.manual_seed(24)
        saved = T.rng_state()
        result = T.gradient_diagnostic(model, teacher, batch, .1, torch.tensor([.4, .6]))
        observed = (random.random(), np.random.rand(4), torch.rand(4))
        T.restore_rng(saved)
        expected = (random.random(), np.random.rand(4), torch.rand(4))
        self.assertEqual(observed[0], expected[0])
        np.testing.assert_array_equal(observed[1], expected[1])
        torch.testing.assert_close(observed[2], expected[2], rtol=0, atol=0)
        self.assertTrue(model.training)
        self.assertEqual(model_hash, state_hash(model.state_dict()))
        self.assertEqual(teacher_hash, state_hash(teacher.state_dict()))
        for name, p in model.named_parameters(): self.assertTrue(torch.equal(p.grad, before[name]))
        self.assertEqual(result["applied_soft_A"], 0.)
        self.assertGreater(result["raw_hard_U"], 0.)

    def test_atomic_timing_and_fullstate_bundle_failure_preserves_old_commit(self):
        with tempfile.TemporaryDirectory(prefix="fh20r1-bundle-") as name:
            wd = Path(name)
            T.save_committed_resume(wd, dict(update=1, timing_segments=[]), "run", "cfg")
            old = (wd / "last").resolve()
            with patch.object(T, "atomic_json", side_effect=OSError("disk failed")):
                with self.assertRaises(OSError):
                    T.save_committed_resume(wd, dict(update=2, timing_segments=[]), "run", "cfg")
            self.assertEqual((wd / "last").resolve(), old)
            self.assertEqual(json.loads((wd / "meta/timing_committed.json").read_text())["committed_update"], 1)
            T.save_committed_resume(wd, dict(update=2, timing_segments=[]), "run", "cfg")
            self.assertEqual(json.loads((wd / "meta/timing_committed.json").read_text())["committed_update"], 2)
            self.assertFalse(old.exists())

    def test_complete_synthetic_training_saves_three_types_of_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="fh20r1-run-") as name:
            run = SyntheticRun(Path(name))
            self.assertEqual(run.run(lambda *a, **kw: fake_result()), 0)
            state = run.state()
            self.assertEqual(state["update"], 3)
            self.assertTrue((run.wd / "restart_fullstates/2/training_state.pt").is_file())
            self.assertFalse((run.wd / "candidates/2").exists())
            self.assertTrue((run.wd / "candidates/3/model.safetensors").is_file())
            self.assertEqual(state["diagnostics_done"], [2])
            self.assertTrue((run.wd / "diagnostics/gradient_norms.csv").is_file())
            self.assertEqual([r["update"] for r in json.loads((run.wd / "official/raw_grid.json").read_text())["records"]], [1, 3])
            doc = json.loads((run.wd / "meta/timing_committed.json").read_text())
            self.assertEqual(doc["full_state_identity_path"], "last/identity.json")
            self.assertEqual(doc["segments"], state["timing_segments"])
            self.assertTrue(all(row["update_to"] <= 3 for row in doc["segments"]))

    def test_eval_interrupt_exact_resume_matches_uninterrupted_parameters(self):
        with tempfile.TemporaryDirectory(prefix="fh20r1-full-") as full, tempfile.TemporaryDirectory(prefix="fh20r1-resume-") as resumed:
            a, b = SyntheticRun(Path(full)), SyntheticRun(Path(resumed))
            def noisy_eval(*args, **kwargs):
                torch.rand(23); np.random.rand(4); random.random()
                return fake_result()
            a.run(noisy_eval)
            def broken(*args, **kwargs):
                torch.rand(31)
                raise RuntimeError("simulated evaluation interruption")
            with self.assertRaisesRegex(RuntimeError, "evaluation interruption"):
                b.run(broken)
            saved_ids = {row["id"] for row in b.state()["timing_segments"]}
            self.assertEqual(b.run(noisy_eval, resume=True), 0)
            sa, sb = a.state(), b.state()
            self.assertEqual(sa["update"], sb["update"])
            for key, value in sa["model_state"].items():
                torch.testing.assert_close(value, sb["model_state"][key], rtol=0, atol=0)
            self.assertEqual(sa["scheduler"], sb["scheduler"])
            self.assertTrue(saved_ids.issubset({row["id"] for row in sb["timing_segments"]}))

    def test_work_journal_disallows_overlapping_segments(self):
        journal = T.WorkJournal(); journal.start("train", 0)
        with self.assertRaises(ValueError): journal.start("eval", 1)
        row = journal.finish(1)
        self.assertEqual(row["update_to"], 1)
        self.assertGreaterEqual(journal.totals()["train"], 0.)
        with self.assertRaises(ValueError): T.WorkJournal([row, row])

    def test_actual_committed_trainer_timing_ingests_once_into_ledger(self):
        from fh20r1 import ledger
        with tempfile.TemporaryDirectory(prefix="fh20r1-ledger-training-") as name:
            run = SyntheticRun(Path(name))
            budget_path = ledger.camp(run.root, "s1") / "campaign_budget.json"
            ledger.write(budget_path, dict(campaign_id=ledger.CAMPAIGN_ID, actual_start_authorized=True,
                                          started_at_utc=ledger.iso(), minimum_effective_hours=20))
            self.assertEqual(run.run(lambda *a, **kw: fake_result()), 0)
            doc = json.loads((run.wd / "meta/timing_committed.json").read_text())
            self.assertEqual(ledger.ingest_timing(run.root, "s1", run.case.run_id), len(doc["segments"]))
            self.assertEqual(ledger.ingest_timing(run.root, "s1", run.case.run_id), 0)
            result = ledger.report(run.root, "s1")
            self.assertGreater(result["effective_seconds"], 0.)
            self.assertAlmostEqual(result["effective_seconds"], ledger.union_seconds(
                (row["start_utc"], row["end_utc"]) for row in doc["segments"]), places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
