"""Mocked boundary safety checks; no GPU, signals, network, or real work_dir writes."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from panda_sensor_focus import boundary as B


def configs():
    common = {"replica": "R1", "planned_horizon": 30, "dataset": "GF2"}
    return [dict(common, run_id="donor", case="S1_D", phase="donor"),
            dict(common, run_id="student", case="S_D", phase="stage2", donor_version="S1_D"),
            dict(common, run_id="extra", case="S_EXTRA", phase="stage2", donor_version="S1_D"),
            dict(common, run_id="other", case="S_OTHER", phase="stage2", donor_version="S1_OTHER")]


class BoundarySelection(unittest.TestCase):
    def test_started_donor_keeps_only_its_first_registered_student(self):
        state = {"runs": [{"run_id": "donor", "status": "STOPPED"}]}
        self.assertEqual([c["run_id"] for c in B.select_configs(
            configs(), state, ["donor", "student"])], ["donor", "student"])
        for ids in (["donor"], ["donor", "extra"], ["donor", "student", "extra"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                B.select_configs(configs(), state, ids)

    def test_started_student_never_admits_additional_pending_experiments(self):
        state = {"runs": [{"run_id": "student", "status": "STOPPED"}]}
        self.assertEqual(len(B.select_configs(configs(), state, ["student"])), 1)
        with self.assertRaises(ValueError):
            B.select_configs(configs(), state, ["student", "extra"])
        with self.assertRaisesRegex(ValueError, "already-started"):
            B.select_configs(configs(), {"runs": []}, ["student"])

    def test_live_duplicate_unknown_and_unselected_stopped_rows_are_rejected(self):
        for rows in (
            [{"run_id": "student", "status": "RUNNING"}],
            [{"run_id": "student", "status": "STOPPED"}] * 2,
            [{"run_id": "student", "status": "COMPLETED"},
             {"run_id": "other", "status": "STOPPED"}],
            [{"run_id": "student", "status": "STOPPED"},
             {"run_id": "unknown", "status": "COMPLETED"}],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                B.select_configs(configs(), {"runs": rows}, ["student"])

    def test_completed_donor_and_stopped_child_support_idempotent_recovery(self):
        rows = [{"run_id": "donor", "status": "COMPLETED"},
                {"run_id": "student", "status": "STOPPED"}]
        self.assertEqual(len(B.select_configs(configs(), {"runs": rows}, ["donor", "student"])), 2)

    def test_known_support_rejects_unclassified_errors(self):
        self.assertTrue(B.known_support(B.STAGE1_SUPPORT))
        self.assertTrue(B.known_support(B.STAGE2_SUPPORT + json.dumps(
            {"action": "STOP_BEFORE_UPDATE_NO_CLAMP_OR_ADAPTIVE_ROI"})))
        for message in ("CUDA OOM", "SUPPORT_VIOLATION: unknown", B.STAGE2_SUPPORT + "{}"):
            self.assertFalse(B.known_support(message))
        self.assertEqual(B.row_support_message({"error": repr(ValueError(B.STAGE1_SUPPORT))}), B.STAGE1_SUPPORT)
        self.assertIsNone(B.row_support_message({"error": "RuntimeError('OOM')"}))


class BoundaryAuthorization(unittest.TestCase):
    def test_authorization_binds_server_exact_ids_sources_and_owned_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "work_dir/panda_sensor_focus_launchers/s3"
            directory.mkdir(parents=True)
            takeover = {"server": "s3", "controller_pin": {"file.py": "known"}}
            plan = {"server": "s3", "entries": [{"package": "panda_sensor_refine",
                                                  "run_ids": ["donor", "student"]}]}
            (directory / "takeover.json").write_text(json.dumps(takeover))
            (directory / "boundary_plan.json").write_text(json.dumps(plan))
            fake_queue = SimpleNamespace(controller_pin=lambda: {"file.py": "known"})
            with patch.object(B, "ROOT", root), patch.dict("sys.modules", {"panda_sensor_focus.queue": fake_queue}):
                receipt = directory / "boundary_refine.json"
                proof = B.authorization("s3", "panda_sensor_refine", ["donor", "student"], receipt)
                self.assertIn("controller_pin_sha256", proof)
                with self.assertRaisesRegex(ValueError, "uniquely authorized"):
                    B.authorization("s3", "panda_sensor_refine", ["student"], receipt)
                with self.assertRaisesRegex(ValueError, "inside its launcher"):
                    B.authorization("s3", "panda_sensor_refine", ["donor", "student"], root / "escape.json")
                fake_queue.controller_pin = lambda: {"file.py": "changed"}
                with self.assertRaisesRegex(ValueError, "controller source"):
                    B.authorization("s3", "panda_sensor_refine", ["donor", "student"], receipt)
                fake_queue.controller_pin = lambda: {"file.py": "known"}
                plan["entries"] *= 2
                (directory / "boundary_plan.json").write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, "uniquely authorized"):
                    B.authorization("s3", "panda_sensor_refine", ["donor", "student"], receipt)


class BoundaryRuntime(unittest.TestCase):
    def make_campaign(self, root, rows, *, failure=None, mutate_admission=False):
        camp_root = root / "original"
        camp_root.mkdir()
        (camp_root / "admission.json").write_text('{"immutable":true}')
        state_path = camp_root / "campaign_state.json"
        state_path.write_text(json.dumps({"runs": rows, "donors": {}, "groups": []}))
        calls = []
        class BudgetExpired(RuntimeError):
            pass
        class Campaign:
            def __init__(self, server):
                self.server, self.root, self.state_path = server, camp_root, state_path
                self.state = B.read(state_path)
                self.code, self.stop_requested = {"original.py": "unchanged"}, False
            def check(self, **kwargs):
                if self.stop_requested:
                    raise BudgetExpired("STOP_REQUESTED")
            def verify_code(self):
                calls.append("verify_code")
            def persist(self):
                state_path.write_text(json.dumps(self.state))
            def admission(self, catalog):
                calls.append("admission")
                if mutate_admission:
                    (camp_root / "admission.json").write_text('{"immutable":false}')
            def register_donor(self, case):
                calls.append("register:" + case)
            def admit_group(self, group):
                calls.append("admit:" + group["runs"][0][0])
            def validate_preflight(self, cfg):
                calls.append("preflight:" + cfg["run_id"])
            def run_case(self, cfg):
                calls.append("run:" + cfg["run_id"])
                row = next((r for r in self.state["runs"] if r["run_id"] == cfg["run_id"]), None)
                if row is None:
                    row = {"run_id": cfg["run_id"]}
                    self.state["runs"].append(row)
                if failure is not None:
                    row.update(status="FAILED", step=5, error=repr(failure), poisoned=False)
                    raise failure
                row.update(status="COMPLETED", step=cfg["planned_horizon"])
        return SimpleNamespace(Campaign=Campaign, BudgetExpired=BudgetExpired), calls, state_path

    def invoke(self, root, runtime, ids):
        (root / "gspread").mkdir()
        (root / "gspread/server.txt").write_text("s3")
        frozen = {"controller_pin_sha256": "same", "boundary_plan_sha256": "plan", "takeover_sha256": "takeover"}
        with patch.object(B, "ROOT", root), \
                patch.object(B, "authorization", return_value=frozen), \
                patch.object(B, "registered_configs", return_value=configs()), \
                patch.object(B.importlib, "import_module", return_value=runtime), \
                patch.object(B, "authenticate_stop", return_value={"checkpoint_sha256": "safe"}), \
                patch.object(B, "final_evidence", side_effect=lambda path, cfg: {"run_id": cfg["run_id"]}), \
                patch.object(B.early_stop, "finish_stopped", return_value=False), \
                patch.object(B.early_stop, "run_case", side_effect=lambda camp, cfg, exc: camp.run_case(cfg)), \
                patch.object(B.signal, "signal", return_value="old-handler"):
            return B.run_boundary("s3", "panda_s1loss9", ids, root / "receipt.json")

    def test_original_runtime_finishes_only_donor_and_child_and_pauses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime, calls, path = self.make_campaign(root, [{"run_id": "donor", "status": "STOPPED", "step": 5}])
            result = self.invoke(root, runtime, ["donor", "student"])
            self.assertEqual([x for x in calls if x.startswith("run:")], ["run:donor", "run:student"])
            self.assertIn("register:S1_D", calls)
            self.assertEqual(result["status"], B.PAUSED)
            self.assertFalse(result["all_original_runs_completed"])
            self.assertEqual(B.read(path)["status"], B.PAUSED)
            self.assertEqual({v["status"] for v in result["outcomes"].values()}, {"COMPLETED"})

    def test_known_donor_failure_holds_child_without_starting_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime, calls, path = self.make_campaign(root, [{"run_id": "donor", "status": "STOPPED", "step": 5}],
                                                       failure=ValueError(B.STAGE1_SUPPORT))
            result = self.invoke(root, runtime, ["donor", "student"])
            self.assertEqual(result["outcomes"]["donor"]["status"], "FAILED_SUPPORT")
            self.assertEqual(result["outcomes"]["student"]["status"], "HELD_FAILED_DONOR")
            self.assertNotIn("run:student", calls)
            self.assertEqual(len(B.read(path)["runs"]), 1)

    def test_unknown_failure_is_fatal_and_never_has_completion_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime, calls, path = self.make_campaign(root, [{"run_id": "student", "status": "STOPPED", "step": 5}],
                                                       failure=RuntimeError("CUDA OOM"))
            with self.assertRaisesRegex(RuntimeError, "CUDA OOM"):
                self.invoke(root, runtime, ["student"])
            self.assertEqual(B.read(path)["status"], "FAILED")
            self.assertFalse((root / "receipt.json").exists())

    def test_changed_original_admission_blocks_before_any_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime, calls, path = self.make_campaign(root, [{"run_id": "student", "status": "STOPPED", "step": 5}],
                                                       mutate_admission=True)
            with self.assertRaisesRegex(ValueError, "admission bytes changed"):
                self.invoke(root, runtime, ["student"])
            self.assertFalse(any(call.startswith("run:") for call in calls))
            self.assertFalse((root / "receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
