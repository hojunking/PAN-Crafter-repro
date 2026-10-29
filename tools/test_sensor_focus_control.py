"""Offline safety checks for the focused sensor queue, 2026-10-29.

These tests do not signal processes, touch training artifacts, or call Sheets.
The historical digests are evidence that the successor leaves the original
training and deferred WV3 registries intact.
"""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


LEGACY_SOURCE_PINS = {
    "panda_sensor_refine": "05d05db03c17aa428a61c27d4b7c8e17af3b6b46b980ae4f036b80aa51d5eca1",
    "panda_s1loss9": "25eef0736dfa8c680864f9bd2f9a9f8b8bcd5486005a19851ba4595e42bf036d",
    "panda_tune30": "f879f4218fc2b7c96116c02f11b646afc655017e85dbb200c02e482b9cad3aab",
    "panda_hard36": "2e2e2ad19416d6d1532df6f50aea393f6be686c3f6810e9af34cc9f6a752a1b5",
}
WV3_REGISTRIES = {
    ("panda_tune30", "s1"): "ba8326c309d8f0b3c085e80e94c52f5e889d963ebb2fdbb02eb37ef05e46bcd4",
    ("panda_tune30", "s3"): "8dd49d7a6271cc7c66d60a2cdb417621f71577b748768da08437eed160bee1c8",
    ("panda_hard36", "s1"): "51b4c03f88c3076a7b5aa5ad354e001a170ff5d02c8e82dc2348cee59c741648",
    ("panda_hard36", "s3"): "3bb5580c44a3f99a71e48a2803960abbd5abef05e9c2e353a7ffd6b01e8b8304",
}


class HistoricalRegistrySafety(unittest.TestCase):
    def test_original_training_source_pins_remain_unchanged(self):
        for package, expected in LEGACY_SOURCE_PINS.items():
            with self.subTest(package=package):
                runtime = importlib.import_module(package + ".runtime")
                self.assertEqual(digest(runtime.code_pin()), expected)

    def test_deferred_wv3_cases_and_settings_remain_unchanged(self):
        for (package, server), expected in WV3_REGISTRIES.items():
            with self.subTest(package=package, server=server):
                plan = importlib.import_module(package + ".plan")
                self.assertEqual(digest(plan.default_config(server)), expected)


class FocusControlSafety(unittest.TestCase):
    def test_only_old_queue_watchers_in_this_checkout_and_server_match(self):
        from panda_sensor_focus import queue as Q
        for module in ("panda_sensor_refine.queue", "panda_s1loss9.queue", "panda_sensor5.queue"):
            argv = ["python", "-m", module, "--server", "s1", "--watch"]
            self.assertTrue(Q.watcher_matches(argv, Q.ROOT, "s1"))
            self.assertFalse(Q.watcher_matches(argv, Q.ROOT, "s3"))
            self.assertFalse(Q.watcher_matches(argv, "/tmp/another-checkout", "s1"))
        for module in ("panda_sensor_refine", "panda_sensor_refine.best_publication",
                       "panda_sensor_focus.queue", "panda_sensor_focus.best_publication"):
            self.assertFalse(Q.watcher_matches(
                ["python", "-m", module, "--server", "s1"], Q.ROOT, "s1"))

    def test_container_inventory_includes_retained_loss9_adapter(self):
        from panda_sensor_focus import queue as Q
        def container(module, server="s3", root=None):
            return {"Config": {"Cmd": ["python", "-m", module, "--server", server]},
                    "Mounts": [{"Type": "bind", "Destination": "/workspace",
                                "Source": str(root or Q.ROOT)}]}
        for module in ("panda_sensor_refine", "panda_sensor_refine.orchestration",
                       "panda_s1loss9", "panda_sensor5"):
            with self.subTest(module=module):
                self.assertTrue(Q.managed_container(container(module), "s3"))
                self.assertFalse(Q.managed_container(container(module, server="s1"), "s3"))
                self.assertFalse(Q.managed_container(container(module, root="/tmp/other"), "s3"))
        self.assertFalse(Q.managed_container(container("panda_sensor_refine.best_publication"), "s3"))

    def test_natural_tail_only_preserves_s1_final_pair(self):
        from panda_sensor_focus import queue as Q
        cfgs = [dict(run_id="old", case="QB_C40"),
                dict(run_id="donor", case="S1_A100_E030_R3"),
                dict(run_id="student", case="QB_A100_E030_R3")]
        containers = [{"Config": {"Cmd": ["python", "-m", "panda_sensor_refine", "--server", "s1"]}}]
        snapshot = {"panda_sensor_refine": {"runs": [dict(run_id="old", status="COMPLETED"),
                                                       dict(run_id="donor", status="RUNNING")]}}
        with patch.object(Q, "registered", return_value=cfgs):
            self.assertTrue(Q.natural_refine_tail("s1", snapshot, containers))
            self.assertFalse(Q.natural_refine_tail("s3", snapshot, containers))
            self.assertFalse(Q.natural_refine_tail("s1", snapshot, []))
            snapshot["panda_sensor_refine"]["runs"] = [dict(run_id="old", status="RUNNING")]
            self.assertFalse(Q.natural_refine_tail("s1", snapshot, containers))

    def test_boundary_selection_retains_only_started_run_and_first_child(self):
        from panda_sensor_focus import orchestration as O
        cfgs = [dict(run_id="d", case="D", phase="donor"),
                dict(run_id="s", case="S", phase="stage2", donor_version="D"),
                dict(run_id="extra", case="EXTRA", phase="stage2", donor_version="D")]
        with patch.object(O, "registered", return_value=cfgs):
            state = {"runs": [dict(run_id="d", status="STOPPED")]}
            self.assertEqual([c["run_id"] for c in O.boundary_configs("panda_sensor5", "s3", state)], ["d", "s"])
            state = {"runs": [dict(run_id="d", status="COMPLETED"), dict(run_id="s", status="RUNNING")]}
            self.assertEqual([c["run_id"] for c in O.boundary_configs("panda_sensor5", "s3", state)], ["s"])
            state["runs"].append(dict(run_id="extra", status="STOPPED"))
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                O.boundary_configs("panda_sensor5", "s3", state)

    def test_trainer_identity_change_prevents_any_signal(self):
        from panda_sensor_focus import queue as Q
        saved = dict(name="trainer", id="original", image="image", argv=["python", "train"])
        current = dict(Id="replacement", Image="image", Config={"Cmd": saved["argv"]},
                       State={"Running": True})
        with patch.object(Q.launch, "inspect_optional", return_value=current), \
                patch.object(Q.subprocess, "run") as signal:
            with self.assertRaisesRegex(ValueError, "identity changed"):
                Q.advance("s3", Path("/tmp"), {"stage": "STOP_OLD"}, {"containers": [saved]})
            signal.assert_not_called()

    def test_running_natural_tail_never_stops_or_launches_new_job(self):
        from panda_sensor_focus import queue as Q
        saved = dict(name="trainer", id="original", image="image", argv=["python", "train"])
        current = dict(Id="original", Image="image", Config={"Cmd": saved["argv"]}, State={"Running": True})
        chain = {"stage": "WAIT_RETAINED_REFINE"}
        with patch.object(Q.launch, "inspect_optional", return_value=current), \
                patch.object(Q, "safe_boundary_plan") as boundary, \
                patch.object(Q.launch, "launch_job") as launch, patch.object(Q.subprocess, "run") as signal:
            Q.advance("s1", Path("/tmp"), chain, {"containers": [saved]})
            self.assertEqual(chain["stage"], "WAIT_RETAINED_REFINE")
            boundary.assert_not_called()
            launch.assert_not_called()
            signal.assert_not_called()

    def test_focus_admission_waits_for_all_boundaries_and_valid_predecessor(self):
        from panda_sensor_focus import queue as Q
        entry = {"package": "panda_sensor_refine", "run_ids": ["d", "s"]}
        chain = {"stage": "BOUNDARIES", "boundaries": [entry], "boundary_index": 0}
        with patch.object(Q.launch, "launch_job") as launch, \
                patch.object(Q.launch, "wait_job", return_value=False), \
                patch.object(Q, "write_predecessor") as predecessor:
            Q.advance("s3", Path("/tmp"), chain, {})
            self.assertEqual(launch.call_args.args[1], "boundary_00")
            self.assertEqual(chain["stage"], "BOUNDARIES")
            predecessor.assert_not_called()
        chain["boundary_index"] = 1
        with patch.object(Q, "write_predecessor", side_effect=ValueError("unfinished old run")), \
                patch.object(Q.launch, "launch_job") as launch:
            with self.assertRaisesRegex(ValueError, "unfinished"):
                Q.advance("s3", Path("/tmp"), chain, {})
            launch.assert_not_called()
            self.assertEqual(chain["stage"], "BOUNDARIES")
        chain["stage"] = "START_FOCUS"
        with patch.object(Q, "predecessor_receipt", side_effect=ValueError("state changed")), \
                patch.object(Q.launch, "launch_job") as launch:
            with self.assertRaisesRegex(ValueError, "state changed"):
                Q.advance("s3", Path("/tmp"), chain, {})
            launch.assert_not_called()

    def test_existing_launch_receipt_does_not_spawn_duplicate_job(self):
        from panda_sensor_focus import launch as L
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            package, args = L.arguments("s1", "focus")
            (directory / "focus.json").write_text(json.dumps({"args": args, "package": package}))
            with patch.object(L, "launcher_root", return_value=directory), \
                    patch.object(L, "command") as command, patch.object(L.subprocess, "run") as run:
                self.assertEqual(L.launch_job("s1", "focus")["args"], args)
                command.assert_not_called()
                run.assert_not_called()

    def test_retirement_cannot_hide_any_incomplete_started_run(self):
        from panda_sensor_focus import orchestration as O
        cfg = dict(run_id="started", case="S", phase="stage2", replica="R1", planned_horizon=30)
        with patch.object(O, "registered", return_value=[cfg]):
            for status in ("RUNNING", "STOPPED", "UNKNOWN"):
                with self.subTest(status=status), self.assertRaisesRegex(ValueError, "incomplete"):
                    O.retirement_accounting("/tmp", "panda_sensor_refine", "s3",
                                            {"runs": [dict(run_id="started", status=status)]})

    def test_boundary_completion_requires_all_authorization_hashes_and_outcomes(self):
        from panda_sensor_focus import queue as Q
        entry = {"package": "panda_sensor_refine", "run_ids": ["d", "s"]}
        proof = {"schema": "SENSOR_FOCUS_BOUNDARY_COMPLETION_V1", "server": "s3",
                 **entry, "status": "PAUSED_FOR_SENSOR_FOCUS",
                 "outcomes": {"d": {"status": "COMPLETED"}, "s": {"status": "COMPLETED"}},
                 "controller_pin_sha256": digest({"source": "fixed"}),
                 "takeover_sha256": "takeover.json", "boundary_plan_sha256": "boundary_plan.json",
                 "original_admission_sha256": "admission.json", "original_training_registry_unchanged": True}
        with patch.object(Q, "controller_pin", return_value={"source": "fixed"}), \
                patch.object(Q, "campaign_root", return_value=Path("/tmp/old")), \
                patch.object(Q, "sha", side_effect=lambda path: Path(path).name):
            Q.validate_boundary_completion("s3", Path("/tmp/launcher"), entry, proof)
            for field in ("schema", "server", "package", "status", "controller_pin_sha256",
                          "takeover_sha256", "boundary_plan_sha256", "original_admission_sha256"):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity/evidence"):
                    Q.validate_boundary_completion("s3", Path("/tmp/launcher"), entry,
                                                   dict(proof, **{field: "tampered"}))
            for outcomes in ({"d": {"status": "COMPLETED"}},
                             {"d": {"status": "COMPLETED"}, "s": {"status": "RUNNING"}}):
                with self.subTest(outcomes=outcomes), self.assertRaises(ValueError):
                    Q.validate_boundary_completion("s3", Path("/tmp/launcher"), entry,
                                                   dict(proof, outcomes=outcomes))

    def test_retirement_preserves_explicit_boundary_hold_and_preflight_failure(self):
        from panda_sensor_focus import orchestration as O
        cfgs = [dict(run_id=key, case=key, phase="stage2", replica="R1", planned_horizon=30)
                for key in ("held", "preflight", "legacy_preflight", "excluded")]
        state = {"runs": [], "sensor_focus_boundary": {"outcomes": {
            "held": {"status": "HELD_FAILED_DONOR"},
            "preflight": {"status": "FAILED_PREFLIGHT_SUPPORT"}}},
            "sensor_refine_control": {"preflight_failures": {"legacy_preflight": {"proof": "saved"}}}}
        with patch.object(O, "registered", return_value=cfgs):
            result = O.retirement_accounting("/tmp", "panda_sensor_refine", "s3", state)
            self.assertEqual({row["run_id"]: row["status"] for row in result},
                             {"held": "PRESERVED_HELD", "preflight": "PRESERVED_PREFLIGHT_FAILURE",
                              "legacy_preflight": "PRESERVED_PREFLIGHT_FAILURE", "excluded": "EXCLUDED_UNSTARTED"})

    def test_terminal_focus_registry_rejects_duplicate_run_records(self):
        from panda_sensor_focus import queue as Q, plan
        cfg = dict(run_id="same", planned_horizon=30)
        state = {"status": "FINISHED_REGISTERED_QUEUE",
                 "runs": [dict(run_id="same", status="COMPLETED", step=30)] * 2}
        with patch.object(Q, "read", return_value=state), \
                patch.object(plan, "groups", return_value=[{"runs": [("case", "R1")]}]), \
                patch.object(plan, "resolve_case", return_value=cfg):
            with self.assertRaisesRegex(ValueError, "exact terminal"):
                Q.focus_complete("s1")


if __name__ == "__main__":
    unittest.main()
