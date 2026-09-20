"""Controller integration tests with isolated files and mocked external effects.

All roots are temporary; subprocesses, training, GPU inventory, preflight,
reference access, recovery registration and wall-clock reads are mocked.
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from qg40 import controller, deployment
from qg40.common import atomic_json, camp, read_json
from qg40.plan import CASES, SERVERS, active_cases, blocks_for, build_config, case_for, sensor_spec, teacher_for
from qg40.policy import CampaignWindow, admission


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qg40-controller-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.window = CampaignWindow("2026-09-20T00:00:00Z")
        self.now = self.window.t0_utc
        clock = patch.object(controller, "utcnow", side_effect=lambda: self.now.isoformat())
        clock.start()
        self.addCleanup(clock.stop)
        test = self
        class TestClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return test.now if tz is None else test.now.astimezone(tz)
        datetime_clock = patch.object(controller.dt, "datetime", TestClock)
        datetime_clock.start()
        self.addCleanup(datetime_clock.stop)
        for name in ("run", "Popen", "check_output"):
            guard = patch(f"subprocess.{name}", side_effect=AssertionError("External process forbidden in CPU test"))
            guard.start()
            self.addCleanup(guard.stop)
        self.legacy = SimpleNamespace(HOLD="FH20R1_FUTURE_ADMISSIONS_HOLD_v1",
                                      inventory=Mock(return_value=[]), gpu_processes=Mock(return_value=[]))
        injected = patch.dict(sys.modules, {"tools.fh20r1_runner": self.legacy})
        injected.start()
        self.addCleanup(injected.stop)

    def window_file(self):
        return controller.write_window(self.root, self.window.t0_utc)

    def sensor_file(self, server="s1"):
        sensor = case_for(controller.cases_for(server)[0].run_id).sensor
        bindings = {name: dict(path=str(self.root / f"{sensor}_{name}.h5"),
                               sha256="1" * 64, source_identity="cpu-test-source")
                    for name in ("train", "val", "rr", "fr")}
        spec = sensor_spec(sensor).bind(band_order=("B", "G", "R", "NIR"), splits=bindings,
                                        source_provenance={"units": "DN"}, verify_files=False)
        path = self.root / "sensor.json"
        atomic_json(path, spec.to_dict())
        return path

    def test_common_window_is_explicit_immutable_and_not_reset_by_late_server(self):
        with self.assertRaises(ValueError):
            controller.shared_window(self.root)
        receipt = self.window_file()
        self.assertFalse(receipt["training_started"])
        self.now += timedelta(hours=12)
        self.assertEqual(controller.shared_window(self.root), self.window)
        self.assertEqual(controller.status(self.root, "s5")["remaining_hours"], 28.)
        controller.write_window(self.root, self.window.t0_utc)
        with self.assertRaises(ValueError):
            controller.write_window(self.root, self.now)
        self.assertEqual(controller.shared_window(self.root), self.window)

    def test_registry_generation_and_dry_run_do_not_activate(self):
        generated = controller.build_artifacts(self.root)
        self.assertEqual(generated["configs"], 89)
        self.assertFalse(generated["activated"])
        self.assertEqual(len(list((self.root / "config/qg40").glob("*.yaml"))), 89)
        self.assertEqual(controller.build_artifacts(self.root), generated)
        total_enabled = 0
        for server in SERVERS:
            result = controller.start(self.root, server, dry_run=True)
            self.assertEqual(result["definitions"], 89)
            self.assertFalse(result["registered"])
            self.assertFalse(result["training_started"])
            self.assertEqual(result["window"]["status"], "UNBOUND")
            self.assertLessEqual(len(result["next_two_run_ids"]), 2)
            for run in result["next_two_run_ids"]:
                self.assertEqual(case_for(run).server_id, server)
            enabled = [run for block in result["queue"] for run in block["run_ids"]]
            total_enabled += len(enabled)
            self.assertTrue(all(case_for(run).profile == "BASE" for run in enabled))
        self.assertEqual(total_enabled, 59)  # 31 primary + 28 finite reserve, no alternatives.
        self.assertFalse((self.root / "work_dir").exists())

    def test_packaged_clock_is_common_and_conflicting_runtime_cannot_replace_it(self):
        atomic_json(self.root / 'qg40/campaign_window.json', self.window.to_dict())
        self.assertEqual(controller.shared_window(self.root), self.window)
        self.now += timedelta(hours=5)
        self.assertEqual(controller.status(self.root, 's5')['remaining_hours'], 35.)
        with self.assertRaisesRegex(ValueError, 'clock reset'):
            controller.write_window(self.root, self.now)
        self.assertFalse((self.root / 'work_dir/_qg40/campaign_window.json').exists())
        atomic_json(self.root / 'work_dir/_qg40/campaign_window.json', CampaignWindow(self.now).to_dict())
        with self.assertRaisesRegex(ValueError, 'clock reset'):
            controller.shared_window(self.root)

    def test_missing_local_binding_uses_verified_catalog_without_new_user_step(self):
        self.window_file()
        spec = controller.SensorSpec.from_dict(read_json(self.sensor_file()))
        with patch('qg40.bootstrap.default_sensor_spec', return_value=spec) as bind, \
                patch.object(controller, 'source_identity', return_value={'files': {}}), \
                patch.object(controller, 'run', return_value=0), \
                patch('qg40.deployment.install_recovery', return_value={'status': 'MOCK_ONLY'}):
            self.assertEqual(controller.start(self.root, 's1', foreground=True), 0)
        bind.assert_called_once_with(self.root, 'QB')
        self.assertEqual(read_json(camp(self.root, 's1') / 'sensor_spec.json'), spec.to_dict())

    def test_selected_axis_is_paired_and_membership_frozen_once_admitted(self):
        folder = camp(self.root, "s2")
        atomic_json(folder / "branch_receipt.json", {"profile": "B20"})
        queue = controller.effective_queue(self.root, "s2")
        pair = next(item for item in queue if item["block_id"] == "s2_SCREEN_82006")
        self.assertEqual([case_for(run).profile for run in pair["run_ids"]], ["BASE", "B20"])
        state = controller._state(self.root, "s2")
        state["blocks"][pair["block_id"]] = {"status": "ADMITTED", "run_ids": pair["run_ids"]}
        state["runs"][pair["run_ids"][0]] = {"status": "OFFICIAL_EVAL_COMPLETE"}
        queue = controller.effective_queue(self.root, "s2", state)
        self.assertEqual(queue[0]["pending_run_ids"], pair["run_ids"][1:])
        self.assertEqual(queue[0]["run_ids"], pair["run_ids"])
        atomic_json(folder / "branch_receipt.json", {"profile": "E10"})
        with self.assertRaises(ValueError):
            controller.effective_queue(self.root, "s2", state)

    def test_foreign_hold_is_preserved_and_no_registration_or_launch(self):
        self.window_file()
        path = self.root / "work_dir/_eval_phase/hold.json"
        original = {"protocol_id": "UNRELATED_OWNER", "reason": "user maintenance"}
        atomic_json(path, original)
        spec = self.sensor_file()
        with patch.object(controller, "source_identity", return_value={"files": {}}), \
                patch.object(controller, "run") as run:
            with self.assertRaisesRegex(ValueError, "Foreign hold"):
                controller.start(self.root, "s1", spec_path=spec, foreground=True)
            run.assert_not_called()
        self.assertEqual(read_json(path), original)
        self.assertFalse((camp(self.root, "s1") / "registration.json").exists())

    def test_nonempty_hold_without_protocol_is_not_treated_as_unowned(self):
        self.window_file()
        path = self.root / "work_dir/_eval_phase/hold.json"
        original = {"owner": "operator", "reason": "maintenance"}
        atomic_json(path, original)
        with patch.object(controller, "source_identity", return_value={"files": {}}), \
                patch.object(controller, "run") as run, \
                patch("qg40.deployment.install_recovery", return_value={"status": "MOCK_ONLY"}) as recovery:
            with self.assertRaisesRegex(ValueError, "[Ff]oreign|[Uu]nknown|owner"):
                controller.start(self.root, "s1", spec_path=self.sensor_file(), foreground=True)
            run.assert_not_called()
            recovery.assert_not_called()
        self.assertEqual(read_json(path), original)

    def test_known_wv3_hold_recorded_and_existing_files_preserved(self):
        self.window_file()
        path = self.root / "work_dir/_eval_phase/hold.json"
        original = {"protocol_id": self.legacy.HOLD, "old_ledger": "unchanged"}
        atomic_json(path, original)
        old_ledger = self.root / "work_dir/_fh20r1/s1/status.json"
        atomic_json(old_ledger, {"run": "active_old_run", "status": "RUNNING"})
        prior = old_ledger.read_bytes()
        with patch.object(controller, "source_identity", return_value={"files": {}}), \
                patch.object(controller, "run", return_value=0) as run, \
                patch("qg40.deployment.install_recovery", return_value={"status": "MOCK_ONLY"}):
            controller.start(self.root, "s1", spec_path=self.sensor_file(), foreground=True)
            run.assert_called_once_with(self.root, "s1")
        self.assertEqual(old_ledger.read_bytes(), prior)
        registration = read_json(camp(self.root, "s1") / "registration.json")
        self.assertEqual(registration["original_wv3_hold"], original)
        self.assertEqual(read_json(path)["protocol_id"], controller.HOLD)

    def test_busy_gpu_is_waited_on_without_kill_until_shared_deadline(self):
        state = controller._state(self.root, "s1")
        self.legacy.inventory.return_value = [{"pid": 123, "run_id": "old_run"}]
        self.legacy.gpu_processes.return_value = [{"pid": 123}]
        def expire(_seconds):
            self.now = self.window.deadline_utc
        with patch.object(controller.time, "sleep", side_effect=expire) as sleep:
            self.assertFalse(controller._wait_local_idle(self.root, "s1", state, self.window))
            sleep.assert_called_once_with(15)
        self.assertEqual(state["status"], "WAIT_LOCAL_RESOURCE")
        self.assertEqual(state["active_processes"][0]["pid"], 123)

    def test_cutoff_admits_no_training_and_closes_finite_queue(self):
        self.window_file()
        self.now = self.window.admission_cutoff_utc
        preflight = Mock()
        atomic_json(camp(self.root, "s1") / "preflight.json", {"complete": True})
        with patch.dict(sys.modules, {"qg40.preflight": SimpleNamespace(execute=preflight)}), \
                patch.object(controller, "verify_registration"), \
                patch.object(controller, "reconcile"), patch.object(controller, "retry_uploads"), \
                patch.object(controller, "_wait_local_idle", return_value=True), \
                patch.object(controller, "_branch_decisions"), \
                patch.object(controller, "ensure_reference", return_value=True), \
                patch.object(controller, "_command", return_value=0), \
                patch.object(controller, "run_case") as train:
            self.assertEqual(controller.run(self.root, "s1"), 0)
            train.assert_not_called()
        state = controller._state(self.root, "s1")
        self.assertEqual(state["status"], "FINITE_QUEUE_FINISHED")
        self.assertTrue(state["blocks"])
        self.assertTrue(all(block["status"] == "NOT_ADMITTED_CUTOFF" for block in state["blocks"].values()))
        closeout = read_json(camp(self.root, "s1") / "closeout.json")
        self.assertEqual(len(closeout["runs"]), len(controller.cases_for("s1")))
        self.assertEqual(closeout["runs"][teacher_for("s1").run_id]["status"], "NOT_ADMITTED_CUTOFF")

    def test_deadline_closeout_preserves_incomplete_and_condition_status(self):
        self.window_file()
        self.now = self.window.deadline_utc
        state = controller._state(self.root, "s1")
        case = teacher_for("s1")
        state["runs"][case.run_id] = {"status": "INCOMPLETE_AT_DEADLINE", "actual_updates": 42000}
        controller._write_state(self.root, "s1", state)
        with patch.object(controller, "verify_registration"), \
                patch.object(controller, "reconcile"), patch.object(controller, "retry_uploads"), \
                patch.object(controller, "_wait_local_idle", return_value=False), \
                patch.object(controller, "run_case") as train:
            self.assertEqual(controller.run(self.root, "s1"), 0)
            train.assert_not_called()
        report = read_json(camp(self.root, "s1") / "closeout.json")
        self.assertEqual(report["runs"][case.run_id]["actual_updates"], 42000)
        self.assertEqual(report["runs"][case.run_id]["status"], "INCOMPLETE_AT_DEADLINE")
        c3 = next(case for case in CASES if case.server_id == "s1" and case.profile == "C3")
        self.assertEqual(report["runs"][c3.run_id]["status"], "NOT_ADMITTED_CONDITION")

    def test_interrupted_run_requires_exact_fullstate(self):
        case = active_cases("s2")[0]
        state = controller._state(self.root, "s2")
        state["runs"][case.run_id] = {"training_started": True, "status": "PAUSED"}
        with patch.object(controller, "_resolve_config", return_value=self.root / "config.yaml"), \
                patch.object(controller, "_command") as command:
            with self.assertRaisesRegex(ValueError, "fresh restart prohibited"):
                controller.run_case(self.root, "s2", case, state, self.window)
            command.assert_not_called()

    def test_integrity_error_is_not_automatically_retried_as_training(self):
        case = active_cases("s2")[0]
        state = controller._state(self.root, "s2")
        with patch.object(controller, "_resolve_config", return_value=self.root / "config.yaml"), \
                patch.object(controller, "_command", return_value=2):
            self.assertFalse(controller.run_case(self.root, "s2", case, state, self.window))
        self.assertEqual(state["runs"][case.run_id]["status"], "BLOCKED_INTEGRITY")
        with patch.object(controller, "verify_registration"), patch("qg40.deployment.spawn") as spawn:
            result = controller.ensure(self.root, "s2")
            spawn.assert_not_called()
            self.assertEqual(result["status"], "BLOCKED_INTEGRITY")

    def test_signal_pause_uses_exact_resume_and_remains_recoverable(self):
        case = active_cases("s2")[0]
        state = controller._state(self.root, "s2")
        state["runs"][case.run_id] = {"training_started": True, "status": "PAUSED"}
        checkpoint = self.root / "work_dir" / case.run_id / "last/training_state.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        with patch.object(controller, "_resolve_config", return_value=self.root / "config.yaml"), \
                patch.object(controller, "_command", return_value=75) as command:
            self.assertFalse(controller.run_case(self.root, "s2", case, state, self.window))
            self.assertIn("--resume", command.call_args.args[1])
        self.assertEqual(state["runs"][case.run_id]["status"], "PAUSED")
        with patch.object(controller, "verify_registration"), \
                patch("qg40.deployment.spawn", return_value={"status": "SUBMITTED"}) as spawn:
            self.assertEqual(controller.ensure(self.root, "s2")["status"], "SUBMITTED")
            spawn.assert_called_once()

    def test_second_fresh_member_of_admitted_pair_stops_at_36h(self):
        self.window_file()
        self.now = self.window.admission_cutoff_utc
        block = next(block for block in blocks_for("s1") if block.block_id == "s1_BASE_01")
        state = controller._state(self.root, "s1")
        state["blocks"][block.block_id] = {"status": "ADMITTED", "run_ids": list(block.run_ids)}
        state["runs"][block.run_ids[0]] = {"status": "OFFICIAL_EVAL_COMPLETE"}
        controller._write_state(self.root, "s1", state)
        atomic_json(camp(self.root, "s1") / "preflight.json", {"complete": True})
        with patch.object(controller, "verify_registration"), \
                patch.object(controller, "reconcile"), patch.object(controller, "retry_uploads"), \
                patch.object(controller, "_wait_local_idle", return_value=True), \
                patch.object(controller, "_branch_decisions"), \
                patch.object(controller, "ensure_reference", return_value=True), \
                patch.object(controller, "_command", return_value=0), \
                patch.object(controller, "run_case") as train:
            self.assertEqual(controller.run(self.root, "s1"), 0)
            train.assert_not_called()
        saved = controller._state(self.root, "s1")
        self.assertEqual(saved["runs"][block.run_ids[1]]["status"], "NOT_ADMITTED_CUTOFF")

    def test_closed_campaign_retries_upload_without_training_or_clock_reset(self):
        self.window_file()
        self.now = self.window.deadline_utc + timedelta(hours=1)
        case = active_cases("s2")[0]
        state = controller._state(self.root, "s2")
        state["status"] = "WINDOW_CLOSED"
        state["runs"][case.run_id] = {"status": "OFFICIAL_EVAL_COMPLETE", "upload_pending": True}
        controller._write_state(self.root, "s2", state)
        observed = []
        def upload_only(_root, args, _log):
            observed.append(args)
            self.assertEqual(args, ["upload", "--run", case.run_id])
            official = self.root / "work_dir" / case.run_id / "official"
            atomic_json(official / "postrun_status.json", {"official_complete": True, "sheet_uploaded": True})
            atomic_json(official / "upload_receipt.json", {"readback_verified": True})
            return 0
        with patch.object(controller, "verify_registration"), \
                patch.object(controller, "_command", side_effect=upload_only), \
                patch.object(controller, "run_case") as train, \
                patch("qg40.deployment.spawn") as spawn:
            result = controller.ensure(self.root, "s2")
            self.assertEqual(result["status"], "CLOSED")
            self.assertEqual(result["pending_uploads"], [])
            train.assert_not_called()
            spawn.assert_not_called()
        self.assertEqual(len(observed), 1)
        self.assertEqual(controller._state(self.root, "s2")["status"], "WINDOW_CLOSED")
        self.assertEqual(controller.shared_window(self.root), self.window)

    def test_train_authorization_requires_registered_admitted_current_active_case(self):
        self.window_file()
        case = teacher_for("s1")
        cfg = build_config(case)
        cfg["qg40"]["dataset_manifest"] = str(camp(self.root, "s1") / "dataset_manifest.json")
        path = self.root / "work_dir" / case.run_id / "meta/config.resolved.yaml"
        atomic_json(path, cfg)  # JSON is valid YAML; no training or data is loaded.
        state = controller._state(self.root, "s1")
        state["status"] = "RUNNING"
        state["runs"][case.run_id] = {"status": "RUNNING", "training_started": True}
        state["blocks"][case.block_id] = dict(status="ADMITTED", run_ids=[case.run_id],
            reservation=admission(self.window, self.now, [case], p0_ready=True, evaluation_debt_hours=0.))
        controller._write_state(self.root, "s1", state)
        release = {"files": {}, "content_sha256": "cpu-mock-release"}
        preflight = camp(self.root, "s1") / "preflight.json"
        atomic_json(preflight, {"complete": True, "source_identity": release})
        with patch.object(controller, "verify_registration") as registration, \
                patch.dict(sys.modules, {"qg40.training": SimpleNamespace(validate_config=lambda cfg: case_for(Path(cfg["work_dir"]).name))}), \
                patch.object(controller, "source_identity", return_value=release):
            self.assertEqual(controller.authorize_train(self.root, "s1", path), case)
            registration.assert_called_with(self.root, "s1")
            with self.assertRaisesRegex(ValueError, "another server"):
                controller.authorize_train(self.root, "s2", path)
            state["blocks"][case.block_id]["status"] = "DEFINED_NOT_DEPLOYED"
            controller._write_state(self.root, "s1", state)
            with self.assertRaisesRegex(ValueError, "atomic block admission"):
                controller.authorize_train(self.root, "s1", path)
            state["blocks"][case.block_id]["status"] = "ADMITTED"
            state["runs"][case.run_id]["status"] = "PAUSED"
            controller._write_state(self.root, "s1", state)
            with self.assertRaisesRegex(ValueError, "active training case"):
                controller.authorize_train(self.root, "s1", path)
            state["runs"][case.run_id]["status"] = "RUNNING"
            controller._write_state(self.root, "s1", state)
            atomic_json(preflight, {"complete": True, "source_identity": {"files": {"changed": "sha"}}})
            with self.assertRaisesRegex(ValueError, "local preflight"):
                controller.authorize_train(self.root, "s1", path)

    def test_unselected_alternative_cannot_bypass_branch_with_admitted_state(self):
        case = next(case for case in CASES if case.server_id == "s2" and case.profile == "B20")
        cfg = build_config(case)
        cfg["qg40"]["dataset_manifest"] = str(camp(self.root, "s2") / "dataset_manifest.json")
        path = self.root / "work_dir" / case.run_id / "meta/config.resolved.yaml"
        atomic_json(path, cfg)
        block = next(block for block in blocks_for("s2", screen="B20") if block.block_id == case.block_id)
        state = controller._state(self.root, "s2")
        state["status"] = "RUNNING"
        state["runs"][case.run_id] = {"status": "RUNNING"}
        state["blocks"][case.block_id] = dict(status="ADMITTED", run_ids=list(block.run_ids),
                                              reservation={"allowed": True})
        controller._write_state(self.root, "s2", state)
        with patch.object(controller, "verify_registration"), \
                patch.dict(sys.modules, {"qg40.training": SimpleNamespace(validate_config=lambda cfg: case_for(Path(cfg["work_dir"]).name))}):
            with self.assertRaisesRegex(ValueError, "atomic block admission"):
                controller.authorize_train(self.root, "s2", path)

    def test_cli_train_calls_authorization_before_loading_training(self):
        from tools import qg40_runner
        train = Mock()
        with patch.object(qg40_runner, "ROOT", self.root), \
                patch.object(sys, "argv", ["qg40_runner.py", "train", "--server", "s2", "--config", "unadmitted.yaml"]), \
                patch.dict(sys.modules, {"tools.fh12_runner": SimpleNamespace(detect_server=lambda root, server: server),
                                        "qg40.training": SimpleNamespace(train_run=train)}), \
                patch.object(controller, "shared_window", return_value=self.window), \
                patch.object(controller, "authorize_train", side_effect=ValueError("unadmitted case")) as authorize:
            with self.assertRaisesRegex(ValueError, "unadmitted case"):
                qg40_runner.main()
            authorize.assert_called_once_with(self.root, "s2", "unadmitted.yaml")
            train.assert_not_called()

    def test_upload_failure_preserves_complete_run_and_does_not_retrain(self):
        case = active_cases("s2")[0]
        state = controller._state(self.root, "s2")
        atomic_json(self.root / "work_dir" / case.run_id / "meta/training_status.json",
                    {"training_complete": True, "actual_updates": 50000})
        atomic_json(self.root / "work_dir" / case.run_id / "official/postrun_status.json",
                    {"official_complete": True})
        commands = []
        def execute(_root, args, _log):
            commands.append(args[0])
            return 1 if args[0] == "upload" else 0
        with patch.object(controller, "_resolve_config", return_value=self.root / "config.yaml"), \
                patch.object(controller, "_command", side_effect=execute):
            self.assertTrue(controller.run_case(self.root, "s2", case, state, self.window))
        self.assertEqual(commands, ["postrun", "upload"])
        self.assertEqual(state["runs"][case.run_id]["status"], "OFFICIAL_EVAL_COMPLETE")
        self.assertTrue(state["runs"][case.run_id]["upload_pending"])

    def test_status_reports_existing_training_truthfully(self):
        case = teacher_for("s1")
        state = controller._state(self.root, "s1")
        state["status"] = "RUNNING"
        state["runs"][case.run_id] = {"status": "RUNNING", "training_started": True}
        controller._write_state(self.root, "s1", state)
        self.assertTrue(controller.status(self.root, "s1")["training_started"])

    def test_completed_run_observations_allow_real_next_block_admission(self):
        case = active_cases('s2')[0]
        state = controller._state(self.root, 's2')
        wd = self.root / 'work_dir' / case.run_id
        atomic_json(wd / 'meta/training_status.json', dict(training_complete=True,
            actual_updates=50000, training_seconds=3600., evaluation_seconds=1800.,
            peak_memory_bytes=1000, peak_memory_scope='training+scheduled_eval+diagnostics'))
        atomic_json(wd / 'official/postrun_status.json', dict(official_complete=True))
        with patch.object(controller, '_resolve_config', return_value=wd / 'config.yaml'), \
                patch.object(controller, '_command', return_value=0):
            self.assertTrue(controller.run_case(self.root, 's2', case, state, self.window))
        restored = controller._state(self.root, 's2')
        self.assertIn('profile', restored['observations'][0])
        next_case = next(c for c in active_cases('s2') if c.queue_rank == 50)
        decision = admission(self.window, self.now, [next_case], p0_ready=True,
                             evaluation_debt_hours=0., observations=restored['observations'])
        self.assertTrue(decision['allowed'])
        self.assertGreater(decision['block_hours'], 1.875 - 1e-6)

    def test_resource_hold_records_only_local_block_without_training(self):
        case = teacher_for("s1")
        state = controller._state(self.root, "s1")
        decision = dict(allowed=False, status="WAIT_LOCAL_RESOURCE", reasons=["insufficient disk"])
        fake = SimpleNamespace(assess_block=Mock(return_value=decision))
        with patch.dict(sys.modules, {"qg40.resources": fake}), patch.object(controller, "run_case") as train:
            self.assertFalse(controller._resources_available(self.root, "s1", [case], state, case.block_id))
            self.assertEqual(state["status"], "WAIT_LOCAL_RESOURCE")
            self.assertEqual(state["blocks"][case.block_id]["resource_wait"], decision)
            self.assertFalse(state["runs"])
            train.assert_not_called()
            fake.assess_block.return_value = dict(allowed=True, status="AVAILABLE")
            self.assertTrue(controller._resources_available(self.root, "s1", [case], state, case.block_id))
            self.assertNotIn("resource_wait", state["blocks"][case.block_id])
        self.assertTrue(controller.effective_queue(self.root, "s1", state))


if __name__ == "__main__":
    unittest.main()
