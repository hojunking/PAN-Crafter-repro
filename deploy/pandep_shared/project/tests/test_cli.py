from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pan_shared import cli
from pan_shared.common import CAMPAIGN, atomic_json, read_json
from pan_shared.safety import SafetyStop, control, exclusive_lock


class CliTests(unittest.TestCase):
    def args(self, command, work, *extra):
        values = [command, "--work-root", str(work)]
        if command in ("start", "resume", "run"):
            values += ["--server", "s2", "--until-operator-stop"]
        if command == "start":
            values += ["--legacy-root", "/fixture/legacy", "--data-root", "/fixture/data",
                       "--server-identity", "/fixture/server", "--credentials", "/fixture/credential"]
        return cli.parser().parse_args(values + list(extra))

    def manifest(self, microbatch=48):
        return dict(campaign_id=CAMPAIGN, server="s2", hostname="fixture-s2", microbatch=microbatch,
                    data_root="/fixture/data", server_identity_file="/fixture/server",
                    legacy_roots=["/fixture/legacy"], data_catalog=str(cli.PROJECT / "vendor_reference/model_source/ablr2/sensor_sources.json"))

    def test_parser_requires_explicit_indefinite_and_s2(self):
        with redirect_stderr(io.StringIO()):
            for values in (["run", "--server", "s2"], ["run", "--server", "s1", "--until-operator-stop"],
                           ["verify"], ["stop"], ["stop", "--after-current-update", "--after-current-block"]):
                with self.assertRaises(SystemExit):
                    cli.parser().parse_args(values)

    def test_parser_resume_does_not_invent_microbatch_change(self):
        args = self.args("resume", Path("/fixture/work"))
        self.assertIsNone(args.microbatch)

    def test_cold_start_order_preflight_gates_register_then_run(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory); calls = []
            def prepared(*args, **kwargs):
                calls.append("prepare")
                atomic_json(work / "campaign_manifest.json", self.manifest())
                return self.manifest()
            with mock.patch.object(cli, "isolated_paths"), mock.patch.object(cli, "prepare", side_effect=prepared), \
                 mock.patch.object(cli, "wait_for_resource", side_effect=lambda *a: calls.append("wait")), \
                 mock.patch("pan_shared.verify.verify", side_effect=lambda *a, **k: calls.append("verify")), \
                 mock.patch.object(cli, "register", side_effect=lambda *a: calls.append("register")), \
                 mock.patch.object(cli, "run_campaign", side_effect=lambda *a: calls.append("run") or {"status": "TEST_ONLY"}):
                result = cli.execute(self.args("start", work))
            self.assertEqual(calls, ["prepare", "wait", "verify", "register", "run"])
            self.assertEqual(result["status"], "TEST_ONLY")

    def test_failed_gate_cannot_register_or_run(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with mock.patch.object(cli, "isolated_paths"), mock.patch.object(cli, "prepare", return_value=self.manifest()), \
                 mock.patch.object(cli, "wait_for_resource"), \
                 mock.patch("pan_shared.verify.verify", side_effect=SafetyStop("BLOCKED_GATES: fixture")), \
                 mock.patch.object(cli, "register") as register, mock.patch.object(cli, "run_campaign") as run:
                with self.assertRaises(SafetyStop):
                    cli.execute(self.args("start", work))
            register.assert_not_called(); run.assert_not_called()

    def test_register_checks_unregistered_identity_and_gates_first(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with mock.patch.object(cli, "isolated_paths"), \
                 mock.patch.object(cli, "check_identity", return_value=self.manifest()) as identity, \
                 mock.patch("pan_shared.verify.check_gates") as gates, mock.patch.object(cli, "register") as register:
                cli.execute(self.args("register", work))
            identity.assert_called_once_with(work, require_registered=False)
            gates.assert_called_once_with(work, require_registered=False)
            self.assertEqual(len(register.call_args.args[1]["runs"]), 18)

    def test_incomplete_campaign_resume_finishes_gates_and_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory); atomic_json(work / "campaign_manifest.json", self.manifest())
            control(work, "PAUSE")
            calls = []
            with mock.patch.object(cli, "isolated_paths"), \
                 mock.patch.object(cli, "check_identity", return_value=self.manifest()) as identity, \
                 mock.patch.object(cli, "wait_for_resource", side_effect=lambda *a: calls.append("wait")), \
                 mock.patch("pan_shared.verify.verify", side_effect=lambda *a, **k: calls.append("verify")), \
                 mock.patch.object(cli, "register", side_effect=lambda *a: calls.append("register")), \
                 mock.patch.object(cli, "run_campaign", side_effect=lambda *a: calls.append("run") or {"status": "TEST_ONLY"}):
                cli.execute(self.args("resume", work, "--credentials", "/fixture/credential"))
            self.assertEqual(identity.call_args.kwargs, {"require_registered": False})
            self.assertEqual(calls, ["wait", "verify", "register", "run"])
            self.assertEqual(control(work)["action"], "RUN")

    def test_resume_without_microbatch_preserves_registered_24(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory); atomic_json(work / "cases.json", {"fixture": True})
            atomic_json(work / "campaign_manifest.json", self.manifest(24))
            control(work, "PAUSE")
            with mock.patch.object(cli, "isolated_paths"), mock.patch.object(cli, "check_identity", return_value=self.manifest(24)), \
                 mock.patch("pan_shared.verify.check_gates"), \
                 mock.patch("pan_shared.preflight.reconfigure_preformal_microbatch") as reconfigure, \
                 mock.patch.object(cli, "run_campaign", return_value={"status": "TEST_ONLY"}):
                cli.execute(self.args("resume", work))
            reconfigure.assert_not_called()

    def test_registered_resume_checks_gates_without_rerunning_q00(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory); atomic_json(work / "cases.json", {"fixture": True})
            atomic_json(work / "campaign_manifest.json", self.manifest())
            atomic_json(work / "controller_state.json", {"campaign_id": CAMPAIGN, "status": "PAUSED_SAFE", "progress": {"run": 137}})
            control(work, "PAUSE")
            with mock.patch.object(cli, "isolated_paths"), mock.patch.object(cli, "check_identity", return_value=self.manifest()), \
                 mock.patch("pan_shared.verify.check_gates") as gate, mock.patch("pan_shared.verify.verify") as verify, \
                 mock.patch.object(cli, "register") as register, mock.patch.object(cli, "run_campaign", return_value={}):
                cli.execute(self.args("resume", work, "--microbatch", "48"))
            gate.assert_called_once_with(work)
            verify.assert_not_called(); register.assert_not_called()
            state = read_json(work / "controller_state.json")
            self.assertEqual(state["status"], "READY")
            self.assertEqual(state["progress"], {"run": 137})

    def test_resume_changed_server_binding_cannot_authorize(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory); control(work, "STOP")
            atomic_json(work / "campaign_manifest.json", self.manifest())
            with mock.patch.object(cli, "isolated_paths"), mock.patch.object(cli, "check_identity", return_value=self.manifest()), \
                 mock.patch.object(cli, "authorize_resume") as authorize:
                with self.assertRaisesRegex(SafetyStop, "server mapping"):
                    cli.execute(self.args("resume", work, "--server-identity", "/different/server", "--microbatch", "48"))
            authorize.assert_not_called()
            self.assertEqual(control(work)["action"], "STOP")

    def test_start_cannot_override_persistent_operator_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory); control(work, "STOP")
            with mock.patch.object(cli, "isolated_paths"), mock.patch.object(cli, "prepare") as prepare:
                with self.assertRaisesRegex(SafetyStop, "Explicit resume"):
                    cli.execute(self.args("start", work))
            prepare.assert_not_called()

    def test_pause_and_both_stop_modes_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory); atomic_json(work / "campaign_manifest.json", self.manifest())
            with mock.patch.object(cli, "isolated_paths"):
                for command, extra, expected in (("pause", [], "PAUSE"),
                        ("stop", ["--after-current-update"], "STOP"),
                        ("stop", ["--after-current-block"], "STOP_AFTER_BLOCK")):
                    result = cli.execute(self.args(command, work, *extra))
                    self.assertEqual(result["action"], expected)

    def test_authorize_resume_refuses_live_controller_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory); control(work, "PAUSE")
            with exclusive_lock(work / "controller.lock"):
                with self.assertRaisesRegex(SafetyStop, "WRITER_CONFLICT"):
                    cli.authorize_resume(work)
            self.assertEqual(control(work)["action"], "PAUSE")

    def test_run_cannot_instantiate_runtime_before_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("pan_shared.verify.check_gates", side_effect=SafetyStop("BLOCKED_GATES")), \
                 mock.patch("pan_shared.runtime.CampaignRuntime") as runtime:
                with self.assertRaises(SafetyStop):
                    cli.run_campaign(self.args("run", Path(directory)))
            runtime.assert_not_called()

    def test_duplicate_startup_lock_blocks_before_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with exclusive_lock(work / "startup.lock"), mock.patch.object(cli, "isolated_paths"), \
                 mock.patch.object(cli, "execute") as execute, redirect_stdout(io.StringIO()):
                result = cli.main(["run", "--work-root", str(work), "--server", "s2", "--until-operator-stop"])
            self.assertEqual(result, 2)
            execute.assert_not_called()

    def test_wait_resource_checks_stop_without_starting_any_model(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory); control(work, "STOP")
            with mock.patch.object(cli, "inventory") as inventory:
                with self.assertRaisesRegex(SafetyStop, "STOPPED_BY_OPERATOR"):
                    cli.wait_for_resource(self.args("run", work), self.manifest())
            inventory.assert_not_called()

    def test_stop_during_cold_preflight_prevents_q00_and_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            def prepared(*args, **kwargs):
                self.assertTrue((work / "startup_manifest.json").exists())
                self.assertFalse((work / "campaign_manifest.json").exists())
                cli.execute(self.args("stop", work, "--after-current-update"))
                return self.manifest()
            with mock.patch.object(cli, "isolated_paths"), mock.patch.object(cli, "prepare", side_effect=prepared), \
                 mock.patch("pan_shared.verify.verify") as verify, mock.patch.object(cli, "register") as register, \
                 mock.patch.object(cli, "run_campaign") as run:
                with self.assertRaisesRegex(SafetyStop, "STOPPED_BY_OPERATOR"):
                    cli.execute(self.args("start", work))
            verify.assert_not_called(); register.assert_not_called(); run.assert_not_called()
            self.assertEqual(control(work)["action"], "STOP")

    def test_resume_bootstrap_after_interrupted_first_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory); boot = self.manifest(24)
            atomic_json(work / "startup_manifest.json", boot); control(work, "STOP")
            def prepared(*args, **kwargs):
                self.assertEqual(kwargs["microbatch"], 24)
                atomic_json(work / "campaign_manifest.json", boot)
                return boot
            with mock.patch.object(cli, "isolated_paths"), mock.patch.object(cli, "prepare", side_effect=prepared) as prepare, \
                 mock.patch.object(cli, "wait_for_resource"), mock.patch("pan_shared.verify.verify") as verify, \
                 mock.patch.object(cli, "register") as register, mock.patch.object(cli, "run_campaign", return_value={"status": "TEST_ONLY"}):
                result = cli.execute(self.args("resume", work, "--credentials", "/fixture/credential"))
            prepare.assert_called_once(); verify.assert_called_once(); register.assert_called_once()
            self.assertEqual(result["status"], "TEST_ONLY")
            self.assertEqual(register.call_args.args[1]["runs"][0]["microbatch"], 24)

    def test_explicit_preformal_microbatch_change_uses_registered_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory); atomic_json(work / "campaign_manifest.json", self.manifest())
            control(work, "PAUSE")
            with mock.patch.object(cli, "isolated_paths"), mock.patch.object(cli, "check_identity", return_value=self.manifest()), \
                 mock.patch("pan_shared.preflight.reconfigure_preformal_microbatch", return_value=self.manifest(12)) as reconfigure, \
                 mock.patch.object(cli, "wait_for_resource"), mock.patch("pan_shared.verify.verify"), \
                 mock.patch.object(cli, "register") as register, mock.patch.object(cli, "run_campaign", return_value={}):
                cli.execute(self.args("resume", work, "--credentials", "/fixture/credential", "--microbatch", "12"))
            reconfigure.assert_called_once_with(work, 12)
            self.assertEqual(register.call_args.args[1]["runs"][0]["microbatch"], 12)


if __name__ == "__main__":
    unittest.main()
