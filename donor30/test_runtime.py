"""CPU-only controller tests: no data access, GPU, Google or long training."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import plan
from . import runtime as R


def point(server="s1", *, phase="donor", qualified=True):
    threshold = plan.LANES[server]["hqnr_min"]
    return {"HQNR": threshold + .001 if qualified else threshold - .001, "ERGAS": 2.2, "SCC": .98,
        "run_id": "fake_" + phase, "server": server, "dataset": plan.LANES[server]["dataset"],
        "phase": phase, "completed_update": 8000, "checkpoint_path": "/tmp/fake_donor30.pt",
        "state_hash": "a" * 64, "file_sha256": "b" * 64,
        "checkpoint_verified": True, "metrics_complete": True, "metrics_finite": True,
        "same_state_verified": True}


def selected(server="s1", *, qualified=True):
    result = plan.select_donor([point(server, qualified=qualified)], plan.LANES[server]["hqnr_min"])
    result.update(calibration_path="/tmp/fake_calibration.json", calibration_sha256="c" * 64)
    return result


def eval_fixture(step=1):
    identity = {"run_id": "test", "completed_update": step, "model_state_sha256": "a" * 64}
    return {"state_before": identity,
        "rr": {"metrics": {"ERGAS": 2.2, "SCC": .98}, "per_scene": [
            {"scene_id": f"rr:{i}", "raw_prediction_sha256": "a" * 64} for i in range(20)]},
        "fr": {"metrics": {"HQNR": .96}, "per_scene": [
            {"scene_id": f"fr:{i}", "raw_prediction_sha256": "b" * 64} for i in range(20)]}}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(R, "code_pin", return_value={"test": "a" * 64}))
        self.stack.enter_context(patch.object(R, "evaluator_identity", return_value={"test": "b" * 64}))
        self.stack.enter_context(patch.object(R, "emit_status"))
        self.stack.enter_context(patch.object(R, "emit_evaluation"))
        self.c = self.campaign()

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def campaign(self, server="s1"):
        c = R.Campaign(plan.default_config(server), output=self.temp.name, device="cpu")
        c.state["timing"] = {"donor_update_seconds": .001, "stage2_update_seconds": .001,
            "evaluation_seconds": .001, "calibration_seconds": .001, "checkpoint_seconds_estimate": .001}
        return c

    def tiny_run(self, *, fail_trace=False, fail_evaluate=False):
        cfg = plan.resolve_donor_case(self.c.config)
        cfg["planned_horizon"], cfg["evaluation_steps"] = 3, [1, 2, 3]
        self.stack.enter_context(patch.object(R.plan, "validate_case"))
        built = []
        def builder(config, sources):
            result = SimpleNamespace(updates=0, poisoned=False, config=deepcopy(config), stage=object())
            built.append(result)
            return result
        build = self.stack.enter_context(patch.object(R, "build_trainer", side_effect=builder))
        self.stack.enter_context(patch.object(R, "RRStream", return_value=SimpleNamespace()))
        self.stack.enter_context(patch.object(self.c, "cpu_trainer_to_gpu", side_effect=lambda x: x))
        self.stack.enter_context(patch.object(self.c, "release"))
        self.stack.enter_context(patch.object(self.c, "verify_code"))
        def trace(trainer, stream, directory):
            if fail_trace and trainer.updates == 1:
                raise RuntimeError("injected training failure")
            trainer.updates += 1
            return .001
        self.stack.enter_context(patch.object(self.c, "trace", side_effect=trace))
        def evaluate(trainer):
            if fail_evaluate and trainer.updates == 2:
                raise RuntimeError("injected evaluation failure")
            return eval_fixture(trainer.updates)
        self.stack.enter_context(patch.object(self.c, "evaluate", side_effect=evaluate))
        save = self.stack.enter_context(patch("donor30.artifacts.save_candidate", return_value=None))
        resume = self.stack.enter_context(patch("donor30.artifacts.save_resume", return_value={"resume_only": True}))
        return cfg, build, save, resume, built

    def test_official_reverification_requires_all_scenes_and_identity(self):
        first = eval_fixture()
        R.same_predictions(first, deepcopy(first))
        for mutate in (
                lambda x: x["rr"]["per_scene"].pop(),
                lambda x: x["fr"]["per_scene"][0].update(raw_prediction_sha256="c" * 64),
                lambda x: x["state_before"].update(completed_update=2)):
            changed = deepcopy(first)
            mutate(changed)
            with self.assertRaises(ValueError):
                R.same_predictions(first, changed)

    def test_queue_two_donors_then_shared_donor_stage2_both_lanes(self):
        for server in ("s1", "s3"):
            with self.subTest(server=server):
                c = self.campaign(server)
                cases = []
                def run_case(cfg, **kwargs):
                    cases.append(deepcopy(cfg))
                    c.state["runs"].append({"run_id": cfg["run_id"], "attempt": 1,
                        "phase": cfg["phase"], "status": "COMPLETED", "step": 50000})
                    record = point(server, phase=cfg["phase"])
                    record["run_id"] = cfg["run_id"]
                    c.state["observations"].append(record)
                donor = selected(server)
                with patch.object(c, "admission"), patch.object(c, "preflight"), \
                        patch.object(c, "run_case", side_effect=run_case), \
                        patch.object(c, "prepare_selected_donor", return_value=donor) as prepare:
                    result = c.run(catalog_path="unused")
                self.assertEqual([x["phase"] for x in cases[:2]], ["donor", "donor"])
                self.assertEqual([x["replica"] for x in cases[:2]], ["R1", "R2"])
                self.assertEqual(len(cases), 8)
                self.assertEqual([x["stage2_mode"] for x in cases[2:5]], c.config["stage2"]["core_order"])
                self.assertEqual([x["stage2_mode"] for x in cases[5:7]], c.config["stage2"]["second_seed_order"])
                self.assertEqual(cases[-1]["stage2_mode"], "uniform")
                self.assertEqual({x["donor_state_hash"] for x in cases[2:]}, {donor["state_hash"]})
                self.assertTrue(all(x["donor_checkpoint_sha256"] == donor["file_sha256"] for x in cases[2:]))
                prepare.assert_called_once()
                self.assertEqual(result["status"], "FINISHED_QUEUE")

    def test_fallback_flows_to_every_stage2_case_unqualified(self):
        seen = []
        donor = selected(qualified=False)
        def run_case(cfg, **kwargs):
            seen.append(deepcopy(cfg))
            self.c.state["runs"].append({"run_id": cfg["run_id"], "attempt": 1,
                "phase": cfg["phase"], "status": "COMPLETED", "step": 50000})
        with patch.object(self.c, "admission"), patch.object(self.c, "preflight"), \
                patch.object(self.c, "run_case", side_effect=run_case), \
                patch.object(self.c, "prepare_selected_donor", return_value=donor):
            self.c.run(catalog_path="unused")
        self.assertEqual(len(seen), 8)
        self.assertTrue(all(x["donor_selection_status"] == plan.EXPLORATORY_DONOR for x in seen[2:]))
        self.assertTrue(all(x["donor_hqnr_qualified"] is False for x in seen[2:]))

    def test_deadline_starts_once_and_downtime_counts(self):
        self.c.state["started_unix"] = 1000.
        with patch.object(R.time, "time", return_value=4600.), patch.object(R.time, "monotonic", return_value=10.):
            self.c.persist()
            restarted = self.campaign()
            self.assertEqual(restarted.state["started_unix"], 1000.)
            self.assertEqual(restarted.elapsed(), 3600.)
            self.assertEqual(restarted.deadline, 109000.)
        with patch.object(R.time, "time", return_value=100.), patch.object(R.time, "monotonic", return_value=20.):
            self.assertEqual(restarted.elapsed(), 3610.)
            restarted.persist()
            second_restart = self.campaign()
            self.assertGreaterEqual(second_restart.elapsed(), 3610.)

    def test_incomplete_second_donor_cannot_silently_start_stage2(self):
        cases = []
        def run_case(cfg, **kwargs):
            cases.append(cfg)
            self.c.state["runs"].append({"run_id": cfg["run_id"], "attempt": 1,
                "phase": cfg["phase"], "status": "COMPLETED" if cfg["replica"] == "R1" else "FAILED"})
        with patch.object(self.c, "admission"), patch.object(self.c, "preflight"), \
                patch.object(self.c, "run_case", side_effect=run_case), \
                patch.object(self.c, "prepare_selected_donor") as prepare:
            with self.assertRaisesRegex(RuntimeError, "Both independent donor runs"):
                self.c.run(catalog_path="unused")
        self.assertEqual(len(cases), 2)
        self.assertEqual(self.c.state["status"], "FAILED")
        prepare.assert_not_called()

    def test_required_failure_skips_optional_and_is_not_clean_finish(self):
        modes = []
        def run_case(cfg, **kwargs):
            modes.append(cfg.get("stage2_mode", "donor"))
            failed = cfg.get("stage2_mode") == "e"
            self.c.state["runs"].append({"run_id": cfg["run_id"], "attempt": 1,
                "phase": cfg["phase"], "status": "FAILED" if failed else "COMPLETED"})
        with patch.object(self.c, "admission"), patch.object(self.c, "preflight"), \
                patch.object(self.c, "run_case", side_effect=run_case), \
                patch.object(self.c, "prepare_selected_donor", return_value=selected()):
            self.c.run(catalog_path="unused")
        self.assertNotIn("uniform", modes)
        self.assertEqual(len(modes), 7)
        self.assertEqual(self.c.state["status"], "FINISHED_WITH_FAILURES_OR_BUDGET_SKIPS")

    def test_process_exit_old_attempt_is_marked_in_its_own_file(self):
        cfg = plan.resolve_donor_case(self.c.config)
        old = self.c.root / "runs" / cfg["run_id"] / "attempt_01"
        old.mkdir(parents=True)
        R.atomic_json(old / "config.json", cfg)
        status = {"run_id": cfg["run_id"], "attempt": 1, "phase": "donor",
                  "directory": str(old), "status": "RUNNING", "step": 1000}
        R.atomic_json(old / "run_state.json", status)
        self.c.state["runs"].append(status)
        new = self.c._attempt(cfg)
        self.assertEqual(new["attempt"], 2)
        self.assertEqual(json.loads((old / "run_state.json").read_text())["status"], "INTERRUPTED_PROCESS_EXIT")

    def test_check_time_reserves_closeout_and_signal_stops(self):
        with patch.object(self.c, "elapsed", return_value=104400.):
            with self.assertRaises(R.BudgetExpired):
                self.c.check_time()
            self.c.check_time(closeout=False)
        self.c.stop_requested = True
        with self.assertRaisesRegex(R.BudgetExpired, "STOP_REQUESTED"):
            self.c.check_time(closeout=False)

    def test_run_case_evaluates_then_saves_and_retains_attempt_files(self):
        cfg, builder, save, resume, built = self.tiny_run()
        self.c.run_case(cfg)
        self.assertEqual(self.c.state["runs"][0]["status"], "COMPLETED")
        self.assertEqual(save.call_count, 3)
        self.assertEqual(resume.call_count, 1)
        directory = Path(self.c.state["runs"][0]["directory"])
        self.assertEqual(len(list(directory.glob("evaluation_*.json"))), 3)
        self.assertEqual(json.loads((directory / "run_state.json").read_text())["step"], 3)
        self.c.run_case(cfg)
        self.assertEqual(builder.call_count, 1, "Completed run must not restart or overwrite")

    def test_failure_visible_and_next_attempt_is_fresh(self):
        cfg, builder, save, resume, built = self.tiny_run(fail_evaluate=True)
        self.c.run_case(cfg)
        original = Path(self.c.state["runs"][0]["directory"])
        first_eval = (original / "evaluation_0000001.json").read_bytes()
        self.assertEqual(self.c.state["runs"][0]["status"], "FAILED")
        self.assertIn("injected evaluation failure", self.c.state["runs"][0]["error"])
        self.assertEqual(resume.call_count, 1)
        self.c.run_case(cfg)
        self.assertEqual([r["attempt"] for r in self.c.state["runs"]], [1, 2])
        self.assertEqual(builder.call_count, 2)
        self.assertEqual(built[1].config["attempt"], 2)
        self.assertEqual((original / "evaluation_0000001.json").read_bytes(), first_eval)
        self.assertTrue((original.parent / "attempt_02" / "evaluation_0000001.json").is_file())

    def test_budget_stop_writes_resume_not_completed(self):
        cfg, builder, save, resume, built = self.tiny_run()
        with patch.object(self.c, "check_time", side_effect=[None, R.BudgetExpired("deadline")]):
            self.c.run_case(cfg)
        self.assertEqual(self.c.state["runs"][0]["status"], "STOPPED_BUDGET")
        self.assertEqual(self.c.state["runs"][0]["step"], 1)
        resume.assert_called_once()
        self.assertEqual(save.call_count, 1)

    def test_admission_skip_never_builds_training_model(self):
        cfg, builder, save, resume, built = self.tiny_run()
        with patch.object(self.c, "admitted", return_value=False):
            self.c.run_case(cfg)
        self.assertEqual(self.c.state["runs"][0]["status"], "SKIPPED_BUDGET")
        builder.assert_not_called()
        save.assert_not_called()
        resume.assert_not_called()

    def test_poisoned_optimizer_is_never_saved_as_resume(self):
        cfg, builder, save, resume, built = self.tiny_run()
        def poison(trainer, stream, directory):
            trainer.poisoned = True
            raise RuntimeError("incomplete optimizer update")
        with patch.object(self.c, "trace", side_effect=poison):
            self.c.run_case(cfg)
        self.assertEqual(self.c.state["runs"][0]["status"], "FAILED")
        resume.assert_not_called()

    def test_resume_save_failure_is_visible(self):
        cfg, builder, save, resume, built = self.tiny_run()
        resume.side_effect = OSError("disk full")
        self.c.run_case(cfg)
        self.assertIn("disk full", self.c.state["runs"][0]["resume_save_error"])

    def test_missing_verified_donor_does_not_build_stage2(self):
        with self.assertRaisesRegex(RuntimeError, "No verified saved local donor"):
            self.c.prepare_selected_donor()


if __name__ == "__main__":
    unittest.main()
