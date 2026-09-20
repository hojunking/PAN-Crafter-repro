"""Portable overlay and recovery tests; all writes are temporary or mocked."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qg40 import deployment
from qg40.common import atomic_json
from qg40.plan import CASES, SOURCE_SHAS
from qg40.policy import CampaignWindow


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qg40-deployment-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("run", "Popen", "check_output"):
            guard = patch(f"subprocess.{name}", side_effect=AssertionError("Real subprocess forbidden in deployment test"))
            guard.start()
            self.addCleanup(guard.stop)

    def fixture(self, name, contents):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        return path

    def source_fixture(self):
        included = {
            "qg40/__init__.py": "# isolated package\n",
            "qg40/controller.py": "# isolated new controller\n",
            "qg40/plan.py": "# isolated plan\n",
            "qg40/test_deployment.py": "# isolated CPU verification\n",
            "qg40/README.md": "Only explicit QG40 entrypoints.\n",
            "tools/qg40_runner.py": "# isolated explicit entrypoint\n",
            "tools/qg40_start.sh": "#!/bin/sh\n# isolated launcher\n",
            "config/qg40/QG40_Registry.json": '{"campaign": "isolated fixture"}\n',
        }
        included.update({f"config/qg40/{case.run_id}.yaml": "trainer: qg40\n" for case in CASES})
        included.update({path: "source fixture: " + path + "\n" for path in SOURCE_SHAS})
        for name, contents in included.items():
            self.fixture(name, contents)
        excluded = {
            "fh12/training.py": "active-fh12-source\n",
            "fh20r1/training.py": "active-fh20-source\n",
            "tools/fh20r1_runner.py": "active-fh20-runner\n",
            "tools/watchdog.sh": "active-watchdog\n",
            "model/pancrafter.py": "active-model-source\n",
            "pa/aligner.py": "active-aligner-source\n",
            "data/train.h5": "private-data\n",
            "gspread/credentials.json": '{"private_key": "SECRET_NEVER_BUNDLE"}\n',
            "config/qg40/QB_sensor.json": '{"bound_local_path": "/private/sensor"}\n',
            "config/qg40/GF2_sensor.json": '{"bound_local_path": "/private/sensor"}\n',
            "config/qg40/credentials.json": '{"token": "SECRET_NEVER_BUNDLE"}\n',
            ".git/HEAD": "ref: refs/heads/active-wv3\n",
            "work_dir/live_run/last/model.safetensors": "active-weights\n",
            "work_dir/_fh20r1/s1/status.json": '{"status": "RUNNING"}\n',
            "qg40/__pycache__/plan.pyc": "compiled-cache\n",
        }
        for name, contents in excluded.items():
            self.fixture(name, contents)
        return included, excluded

    def unpack(self, result):
        archive = Path(result["path"])
        with tarfile.open(archive, "r:gz") as stream:
            members = stream.getmembers()
            for member in members:
                self.assertTrue(member.isfile())
                self.assertFalse(member.name.startswith("/"))
                self.assertNotIn("..", Path(member.name).parts)
                self.assertEqual(member.mode, 0o644)
            contents = {member.name: stream.extractfile(member).read() for member in members}
        manifest = json.loads(contents.pop("qg40/package_manifest.json"))
        self.assertEqual(result["sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
        self.assertEqual(result["file_count"], len(contents) + 1)
        self.assertEqual(manifest["schema"], "QG40_SOURCE_BUNDLE_v1")
        self.assertTrue(manifest["keep_existing_git_head"])
        self.assertEqual(manifest["files"], {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()})
        return contents

    def test_overlay_allowlist_and_every_checksum_without_activation(self):
        included, excluded = self.source_fixture()
        with patch.object(deployment, "spawn") as spawn, patch.object(deployment, "install_recovery") as cron:
            result = deployment.bundle(self.root)
            spawn.assert_not_called()
            cron.assert_not_called()
        self.assertFalse(result["activation_performed"])
        contents = self.unpack(result)
        self.assertEqual(set(contents), set(included))
        self.assertTrue(all(contents[name] == text.encode() for name, text in included.items()))
        self.assertTrue(set(contents).isdisjoint(excluded))
        self.assertNotIn(b"SECRET_NEVER_BUNDLE", b"".join(contents.values()))
        for name, original in excluded.items():
            self.assertEqual((self.root / name).read_text(), original)
        self.assertFalse((self.root / "work_dir/_qg40/campaign_window.json").exists())

    def test_optional_common_window_is_validated_and_copied_exactly(self):
        included, _ = self.source_fixture()
        path = self.root / "work_dir/_qg40/campaign_window.json"
        window = CampaignWindow("2026-09-20T00:00:00Z").to_dict()
        atomic_json(path, window)
        original = path.read_bytes()
        contents = self.unpack(deployment.bundle(self.root))
        self.assertEqual(contents["qg40/campaign_window.json"], original)
        self.assertEqual(set(contents), set(included) | {"qg40/campaign_window.json"})
        self.assertEqual(path.read_bytes(), original)
        atomic_json(path, dict(window, deadline_utc="2026-09-23T00:00:00+00:00"))
        with self.assertRaises(ValueError):
            deployment.bundle(self.root)

    def test_bundle_refuses_source_symlink_to_credentials(self):
        self.source_fixture()
        leaked = self.root / "qg40/credentials.py"
        leaked.symlink_to(self.root / "gspread/credentials.json")
        with self.assertRaisesRegex(ValueError, "[Ss]ymlink|[Oo]utside|[Uu]nsafe"):
            deployment.bundle(self.root)

    def test_committed_clock_and_sensor_catalog_survive_new_clone_bundle(self):
        included, _ = self.source_fixture()
        window = CampaignWindow('2026-09-20T00:00:00Z').to_dict()
        atomic_json(self.root / 'qg40/campaign_window.json', window)
        self.fixture('qg40/sensor_sources.json', '{"schema":"test-fixture"}\n')
        self.fixture('qg40/SENSOR_SOURCE_EVIDENCE.md', '# test evidence\n')
        contents = self.unpack(deployment.bundle(self.root))
        self.assertEqual(json.loads(contents['qg40/campaign_window.json']), window)
        self.assertIn('qg40/sensor_sources.json', contents)
        self.assertIn('qg40/SENSOR_SOURCE_EVIDENCE.md', contents)
        self.assertFalse((self.root / 'work_dir/_qg40/campaign_window.json').exists())
        atomic_json(self.root / 'work_dir/_qg40/campaign_window.json',
                    CampaignWindow('2026-09-20T01:00:00Z').to_dict())
        with self.assertRaisesRegex(ValueError, 'clock reset'):
            deployment.bundle(self.root)

    def test_bundle_does_not_include_unregistered_qg_named_config(self):
        self.source_fixture()
        self.fixture("config/qg40/QG_PRIVATE.yaml", "token: SECRET_NEVER_BUNDLE\n")
        contents = self.unpack(deployment.bundle(self.root))
        self.assertNotIn("config/qg40/QG_PRIVATE.yaml", contents)

    def test_bundle_refuses_symlinked_archive_destination_without_overwrite(self):
        self.source_fixture()
        credential = self.root / "gspread/credentials.json"
        original = credential.read_bytes()
        destination = self.root / "work_dir/_qg40/deployment/qg40-overlay.tar.gz"
        destination.parent.mkdir(parents=True)
        destination.symlink_to(credential)
        with self.assertRaisesRegex(ValueError, "[Ss]ymlink|[Oo]utside|[Uu]nsafe"):
            deployment.bundle(self.root)
        self.assertEqual(credential.read_bytes(), original)

    def test_recovery_appends_preserving_existing_cron_then_is_idempotent(self):
        original = "MAILTO=operator@example.invalid\n# WV3 jobs must remain\n*/2 * * * * /srv/wv3/watchdog --keep-running\n"
        current = [original]
        writes = []
        def write(args, **kwargs):
            self.assertEqual(args, ["crontab", "-"])
            self.assertTrue(kwargs["check"])
            current[0] = kwargs["input"]
            writes.append(kwargs["input"])
            return SimpleNamespace(returncode=0)
        with patch.object(deployment, "_read_crontab", side_effect=lambda: (True, current[0])), \
                patch.object(deployment.subprocess, "run", side_effect=write):
            result = deployment.install_recovery(self.root, "s2")
            self.assertEqual(result, {"status": "INSTALLED", "readback_verified": True})
            self.assertTrue(current[0].startswith(original))
            self.assertEqual(len(current[0].splitlines()), len(original.splitlines()) + 2)
            self.assertIn("*/5 * * * * ", current[0])
            self.assertIn("@reboot sleep 120 && ", current[0])
            self.assertEqual(current[0].count("ensure --server s2"), 2)
            self.assertNotIn("kill", current[0])
            self.assertEqual(deployment.install_recovery(self.root, "s2")["status"], "ALREADY_INSTALLED")
            self.assertEqual(len(writes), 1)

    def test_foreign_changes_and_conflicting_own_cron_are_preserved(self):
        old = "*/2 * * * * /srv/wv3/watchdog\n"
        changed = old + "0 4 * * * /srv/operator/backup\n"
        with patch.object(deployment, "_read_crontab", side_effect=[(True, old), (True, changed)]), \
                patch.object(deployment.subprocess, "run") as write:
            with self.assertRaisesRegex(ValueError, "Crontab changed"):
                deployment.install_recovery(self.root, "s1")
            write.assert_not_called()
        tag = f"# PANDA-QG40-s1-{hashlib.sha256(str(self.root.resolve()).encode()).hexdigest()[:12]}"
        conflicting = old + "* * * * * /custom/qg40 " + tag + "\n"
        with patch.object(deployment, "_read_crontab", return_value=(True, conflicting)), \
                patch.object(deployment.subprocess, "run") as write:
            with self.assertRaisesRegex(ValueError, "entry differs"):
                deployment.install_recovery(self.root, "s1")
            write.assert_not_called()

    def test_crontab_permission_error_is_not_assumed_empty(self):
        failure = SimpleNamespace(returncode=1, stdout="", stderr="permission denied")
        with patch.object(deployment.subprocess, "run", return_value=failure) as command:
            with self.assertRaisesRegex(RuntimeError, "Cannot read existing crontab"):
                deployment._read_crontab()
            self.assertEqual(command.call_args.args[0], ["crontab", "-l"])
        empty = SimpleNamespace(returncode=1, stdout="", stderr="no crontab for example")
        with patch.object(deployment.subprocess, "run", return_value=empty):
            self.assertEqual(deployment._read_crontab(), (False, ""))

    def test_cron_readback_mismatch_is_not_reported_success(self):
        original = "# original jobs\n"
        with patch.object(deployment, "_read_crontab", side_effect=[(True, original), (True, original), (True, "different\n")]), \
                patch.object(deployment.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
            with self.assertRaisesRegex(ValueError, "readback mismatch"):
                deployment.install_recovery(self.root, "s1")

    def test_spawn_uses_argument_vector_detached_process_and_only_qg40_log(self):
        with patch.object(deployment.subprocess, "Popen", return_value=SimpleNamespace(pid=12345)) as launch:
            result = deployment.spawn(self.root, "s5")
        args, kwargs = launch.call_args
        self.assertEqual(args[0][1:], [str(self.root / "tools/qg40_runner.py"), "run", "--server", "s5"])
        self.assertNotIn("shell", kwargs)
        self.assertEqual(kwargs["cwd"], self.root)
        self.assertTrue(kwargs["start_new_session"])
        self.assertTrue(kwargs["close_fds"])
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(result["status"], "SUBMITTED")
        self.assertEqual(result["pid"], 12345)
        self.assertTrue((self.root / "work_dir/_qg40/s5/runner.log").exists())
        self.assertFalse((self.root / "work_dir/_fh20r1").exists())


if __name__ == "__main__":
    unittest.main()
