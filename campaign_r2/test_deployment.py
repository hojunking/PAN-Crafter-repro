"""Temporary-directory deployment tests; no live holds, cron, queues or network."""
import os
from pathlib import Path
import stat
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from campaign_r2 import deployment as D


WATCHDOG = '''#!/usr/bin/env bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "${1:-}" = "--install" ]; then
  echo INSTALL_ONLY; exit 0
fi
LOG="$REPO/work_dir/cases_chain.log"
# User-local deployment edits must survive byte-for-byte.
CUSTOM_KEEP="preserved"
flock -n "$REPO/work_dir/fixture.lock" true || exit 0
echo LEGACY_QUEUE
'''


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="r2-deployment-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "tools").mkdir()
        (self.root / "work_dir/_fh20r1").mkdir(parents=True)
        self.watchdog = self.root / "tools/_watchdog.sh"
        self.watchdog.write_text(WATCHDOG)
        self.watchdog.chmod(0o751)
        self.pointer = self.root / "work_dir/_fh20r1/r2_local_server.txt"

    def run_watchdog(self, *args):
        return subprocess.run(
            ["bash", str(self.watchdog), *args], cwd=self.root,
            env=dict(os.environ, PYTHON=sys.executable), capture_output=True,
            text=True, timeout=5,
        )

    def runner(self, returncode=0):
        (self.root / "tools/r2_runner.py").write_text(
            'import sys\nprint("R2_ROUTE", *sys.argv[1:])\n'
            f'sys.exit({returncode})\n'
        )

    def test_install_preserves_original_bytes_mode_and_dirty_lines(self):
        before = self.watchdog.read_bytes()
        result = D.install_watchdog(self.root)
        after = self.watchdog.read_bytes()
        self.assertTrue(result["changed"])
        self.assertNotEqual(result["before_sha256"], result["after_sha256"])
        self.assertEqual(after.replace(D.ROUTE.encode(), b""), before)
        self.assertEqual(stat.S_IMODE(self.watchdog.stat().st_mode), 0o751)
        self.assertLess(after.index(b"INSTALL_ONLY"), after.index(D.BEGIN.encode()))
        self.assertLess(after.index(D.END.encode()), after.index(D.ANCHOR.encode()))
        self.assertEqual(list(self.watchdog.parent.glob("._watchdog.r2.*")), [])

    def test_repeated_install_is_noop(self):
        first = D.install_watchdog(self.root)
        before = self.watchdog.stat().st_mtime_ns
        second = D.install_watchdog(self.root)
        self.assertFalse(second["changed"])
        self.assertEqual(second["before_sha256"], first["after_sha256"])
        self.assertEqual(second["after_sha256"], first["after_sha256"])
        self.assertEqual(self.watchdog.stat().st_mtime_ns, before)

    def test_shell_syntax_no_registration_and_install_remain_legacy(self):
        D.install_watchdog(self.root)
        result = subprocess.run(["bash", "-n", str(self.watchdog)], capture_output=True)
        self.assertEqual(result.returncode, 0)
        result = self.run_watchdog()
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "LEGACY_QUEUE")
        self.pointer.write_text("broken")
        result = self.run_watchdog("--install")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "INSTALL_ONLY")

    def test_valid_pointer_executes_ensure_without_legacy_fallback(self):
        D.install_watchdog(self.root)
        self.runner()
        for server in sorted(D.SERVERS):
            with self.subTest(server=server):
                self.pointer.write_text(server + "\n")
                result = self.run_watchdog()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "R2_ROUTE ensure --server " + server)

    def test_empty_invalid_and_multiline_pointer_fail_closed(self):
        D.install_watchdog(self.root)
        self.runner()
        for value in ("", "\n", "s6\n", "s1\ns2\n", " s1\n", "../s1\n"):
            with self.subTest(value=value):
                self.pointer.write_text(value)
                result = self.run_watchdog()
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("LEGACY_QUEUE", result.stdout)
                self.assertNotIn("R2_ROUTE", result.stdout)

    def test_dangling_pointer_and_directory_pointer_fail_closed(self):
        D.install_watchdog(self.root)
        self.runner()
        self.pointer.symlink_to(self.root / "missing")
        result = self.run_watchdog()
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("LEGACY_QUEUE", result.stdout)
        self.pointer.unlink()
        self.pointer.mkdir()
        result = self.run_watchdog()
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("LEGACY_QUEUE", result.stdout)

    def test_missing_runner_and_failed_ensure_do_not_fall_back(self):
        D.install_watchdog(self.root)
        self.pointer.write_text("s3\n")
        result = self.run_watchdog()
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("LEGACY_QUEUE", result.stdout)
        self.runner(returncode=7)
        result = self.run_watchdog()
        self.assertEqual(result.returncode, 7)
        self.assertNotIn("LEGACY_QUEUE", result.stdout)

    def test_partial_modified_or_duplicate_hooks_are_preserved(self):
        for value in (WATCHDOG + D.BEGIN, WATCHDOG + D.END,
                      WATCHDOG + D.ROUTE.replace("ensure --server", "run --server"),
                      WATCHDOG + D.ROUTE + D.ROUTE):
            with self.subTest(value=value[-80:]):
                self.watchdog.write_text(value)
                with self.assertRaisesRegex(ValueError, "partial or modified"):
                    D.install_watchdog(self.root)
                self.assertEqual(self.watchdog.read_text(), value)

    def test_missing_ambiguous_or_unrecognized_anchor_is_preserved(self):
        for value in (WATCHDOG.replace(D.ANCHOR, "LOG=another"),
                      WATCHDOG + D.ANCHOR + "\n",
                      WATCHDOG.replace('"--install"', '"--other"')):
            with self.subTest(value=value):
                self.watchdog.write_text(value)
                with self.assertRaises(ValueError):
                    D.install_watchdog(self.root)
                self.assertEqual(self.watchdog.read_text(), value)

    def test_watchdog_symlink_is_not_overwritten(self):
        target = self.root / "original_watchdog.sh"
        self.watchdog.rename(target)
        self.watchdog.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "nonsymlink"):
            D.install_watchdog(self.root)
        self.assertTrue(self.watchdog.is_symlink())
        self.assertEqual(target.read_text(), WATCHDOG)

    def test_misplaced_complete_route_is_not_accepted(self):
        self.watchdog.write_text(WATCHDOG + D.ROUTE)
        before = self.watchdog.read_bytes()
        with self.assertRaisesRegex(ValueError, "partial or modified"):
            D.install_watchdog(self.root)
        self.assertEqual(self.watchdog.read_bytes(), before)

    def test_concurrent_watchdog_change_is_not_overwritten(self):
        original_fsync = D.os.fsync
        edited = WATCHDOG + "# Concurrent local edit\n"
        changed = False

        def concurrent_edit(fd):
            nonlocal changed
            original_fsync(fd)
            if not changed:
                self.watchdog.write_text(edited)
                changed = True

        with patch.object(D.os, "fsync", side_effect=concurrent_edit):
            with self.assertRaisesRegex(ValueError, "changed during installation"):
                D.install_watchdog(self.root)
        self.assertEqual(self.watchdog.read_text(), edited)
        self.assertEqual(list(self.watchdog.parent.glob("._watchdog.r2.*")), [])

    def test_start_wrapper_delegates_from_another_directory(self):
        self.runner()
        wrapper_source = Path(__file__).resolve().parents[1] / "tools/r2_start.sh"
        wrapper = self.root / "tools/r2_start.sh"
        wrapper.write_bytes(wrapper_source.read_bytes())
        result = subprocess.run(
            ["bash", str(wrapper), "--server", "s4", "--dry-run"],
            cwd=self.root.parent, env=dict(os.environ, PYTHON=sys.executable),
            capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "R2_ROUTE start --server s4 --dry-run")
        self.assertFalse(self.pointer.exists())
        self.assertEqual(self.watchdog.read_text(), WATCHDOG)

    def test_spawn_uses_background_controller_only(self):
        self.runner()
        with patch.object(D.subprocess, "Popen", return_value=types.SimpleNamespace(pid=123)) as popen:
            result = D.spawn(self.root, "s2")
        self.assertEqual(result["status"], "R2_RUNNER_SUBMITTED")
        self.assertEqual(result["pid"], 123)
        self.assertEqual(popen.call_args.args[0],
                         [sys.executable, str(self.root / "tools/r2_runner.py"), "run", "--server", "s2"])
        options = popen.call_args.kwargs
        self.assertTrue(options["start_new_session"])
        self.assertTrue(options["close_fds"])
        self.assertEqual(options["stdin"], subprocess.DEVNULL)
        self.assertEqual(options["stderr"], subprocess.STDOUT)
        self.assertEqual(options["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertTrue(Path(result["log_path"]).is_file())
        self.assertFalse(self.pointer.exists())
        self.assertEqual(self.watchdog.read_text(), WATCHDOG)

    def test_spawn_rejects_invalid_server_or_missing_script_before_launch(self):
        with patch.object(D.subprocess, "Popen") as popen:
            for server in ("s0", "s6", "../s1", ""):
                with self.assertRaises(ValueError):
                    D.spawn(self.root, server)
            with self.assertRaisesRegex(ValueError, "missing"):
                D.spawn(self.root, "s1")
            popen.assert_not_called()
        self.assertFalse((self.root / "work_dir/_fh20r1/s1").exists())


class CronTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="r2-cron-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repository with spaces"
        (self.root / "tools").mkdir(parents=True)
        self.watchdog = self.root / "tools/_watchdog.sh"
        self.watchdog.write_text(WATCHDOG)
        self.watchdog.chmod(0o751)
        command = shlex.quote(str(self.watchdog))
        self.periodic = f"*/15 * * * * {command} # PANCRAFTER-WATCHDOG\n"
        self.reboot = f"@reboot sleep 120 && {command} # PANCRAFTER-WATCHDOG\n"
        self.table = None
        self.commands = []
        self.writes = []

    def result(self, code=0, stdout="", stderr=""):
        return types.SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)

    def fake_cron(self, args, **kwargs):
        self.commands.append(args)
        self.assertEqual(kwargs["env"]["LC_ALL"], "C")
        self.assertEqual(kwargs["timeout"], 15)
        if args == ["crontab", "-l"]:
            if self.table is None:
                return self.result(1, stderr="no crontab for fixture\n")
            return self.result(stdout=self.table)
        self.assertEqual(args, ["crontab", "-"])
        self.table = kwargs["input"]
        self.writes.append(self.table)
        return self.result()

    def test_absent_crontab_installs_two_quoted_jobs_with_readback(self):
        with patch.object(D.subprocess, "run", side_effect=self.fake_cron):
            result = D.install_watchdog_cron(self.root)
        self.assertEqual(self.table, self.periodic + self.reboot)
        self.assertEqual(result["status"], "INSTALLED")
        self.assertTrue(result["created_new_crontab"])
        self.assertTrue(result["readback_verified"])
        self.assertTrue(result["recover_on_reboot"])
        self.assertEqual(len(self.writes), 1)
        self.assertEqual(len(self.commands), 4)

    def test_unrelated_jobs_comments_other_repositories_and_nonowned_recipes_survive(self):
        escaped = shlex.quote(str(self.watchdog))
        untouched = (
            "# Preserve human comments and blank lines.\n\nMAILTO=fixture@example.invalid\n"
            "13 2 * * * /usr/bin/true # a human job\n"
            "*/15 * * * * /other/repository/tools/_watchdog.sh # PANCRAFTER-WATCHDOG\n"
            "@reboot sleep 120 && /other/repository/tools/_watchdog.sh # PANCRAFTER-WATCHDOG\n"
            f"0 * * * * {escaped} # PANCRAFTER-WATCHDOG\n"
            f"*/15 * * * * echo {escaped} # PANCRAFTER-WATCHDOG\n"
            f"# */15 * * * * {escaped} # PANCRAFTER-WATCHDOG\n"
        )
        self.table = untouched + self.periodic + self.periodic + f"@reboot {escaped} # PANCRAFTER-WATCHDOG\n"
        with patch.object(D.subprocess, "run", side_effect=self.fake_cron):
            result = D.install_watchdog_cron(self.root)
            again = D.install_watchdog_cron(self.root)
        self.assertEqual(self.table, untouched + self.periodic + self.reboot)
        self.assertTrue(result["unrelated_entries_preserved"])
        self.assertFalse(result["created_new_crontab"])
        self.assertEqual(again["status"], "ALREADY_INSTALLED")
        self.assertFalse(again["changed"])
        self.assertEqual(len(self.writes), 1)

    def test_already_valid_jobs_are_not_moved_or_rewritten(self):
        self.table = "# first\n" + self.reboot + "17 * * * * true\n" + self.periodic + "# last\n"
        original = self.table
        with patch.object(D.subprocess, "run", side_effect=self.fake_cron):
            result = D.install_watchdog_cron(self.root)
        self.assertEqual(self.table, original)
        self.assertFalse(result["changed"])
        self.assertEqual(self.writes, [])
        self.assertEqual(self.commands, [["crontab", "-l"]])

    def test_read_errors_never_become_an_empty_crontab(self):
        cases = ((1, "Permission denied"), (1, ""), (2, "no crontab for fixture"),
                 (1, "no crontab for fixture\npermission denied"))
        for code, error in cases:
            with self.subTest(code=code, error=error), \
                    patch.object(D.subprocess, "run", return_value=self.result(code, stderr=error)) as run:
                with self.assertRaisesRegex(RuntimeError, "Cannot read existing crontab"):
                    D.install_watchdog_cron(self.root)
                run.assert_called_once()
                self.assertEqual(run.call_args.args[0], ["crontab", "-l"])

    def test_no_crontab_diagnostic_does_not_hide_stdout_or_tool_failure(self):
        with patch.object(D.subprocess, "run", return_value=self.result(
                1, stdout="unexpected existing data", stderr="no crontab for fixture")) as run:
            with self.assertRaisesRegex(RuntimeError, "Cannot read existing crontab"):
                D.install_watchdog_cron(self.root)
            run.assert_called_once()
        with patch.object(D.subprocess, "run", side_effect=FileNotFoundError("crontab unavailable")) as run:
            with self.assertRaises(FileNotFoundError):
                D.install_watchdog_cron(self.root)
            run.assert_called_once()

    def test_concurrent_edit_between_reads_is_preserved_without_a_write(self):
        original = "11 * * * * true\n"
        concurrent = original + "12 * * * * true\n"
        results = [self.result(stdout=original), self.result(stdout=concurrent)]
        with patch.object(D.subprocess, "run", side_effect=results) as run:
            with self.assertRaisesRegex(RuntimeError, "changed during R2 registration"):
                D.install_watchdog_cron(self.root)
        self.assertEqual([call.args[0] for call in run.call_args_list],
                         [["crontab", "-l"], ["crontab", "-l"]])

    def test_failed_write_is_reported_without_retry_or_readback_claim(self):
        results = [self.result(stdout=""), self.result(stdout=""),
                   self.result(1, stderr="cron installation denied")]
        with patch.object(D.subprocess, "run", side_effect=results) as run:
            with self.assertRaisesRegex(RuntimeError, "cron registration failed"):
                D.install_watchdog_cron(self.root)
        self.assertEqual(run.call_count, 3)
        self.assertEqual(run.call_args.args[0], ["crontab", "-"])

    def test_readback_mismatch_is_reported_without_rollback(self):
        results = [self.result(stdout=""), self.result(stdout=""), self.result(),
                   self.result(stdout="# a subsequent human edit\n")]
        with patch.object(D.subprocess, "run", side_effect=results) as run:
            with self.assertRaisesRegex(RuntimeError, "readback differs"):
                D.install_watchdog_cron(self.root)
        self.assertEqual(run.call_count, 4)
        self.assertEqual(sum(call.args[0] == ["crontab", "-"] for call in run.call_args_list), 1)

    def test_missing_or_unsafe_watchdog_path_never_accesses_cron(self):
        self.watchdog.unlink()
        with patch.object(D.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "watchdog is required"):
                D.install_watchdog_cron(self.root)
            run.assert_not_called()
        unsafe = self.root / "percent%repository"
        (unsafe / "tools").mkdir(parents=True)
        (unsafe / "tools/_watchdog.sh").write_text(WATCHDOG)
        with patch.object(D.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "cron metacharacters"):
                D.install_watchdog_cron(unsafe)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
