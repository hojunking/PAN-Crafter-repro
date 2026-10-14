"""Shell launch checks using fake executables; no real GPU or Docker calls."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"; self.bin.mkdir()
        self.capture = self.root / "captured-args.txt"
        self.env = {**os.environ, "PATH": str(self.bin) + ":" + os.environ["PATH"],
                    "DIRECT_TEST_CAPTURE": str(self.capture), "DIRECT_TEST_GPU": ""}
        self.executable("nvidia-smi", '#!/bin/bash\nprintf "%s" "$DIRECT_TEST_GPU"\n')
        self.executable("fake-python", '#!/bin/bash\nprintf "%s\\n" "$@" > "$DIRECT_TEST_CAPTURE"\n')
        self.env["PAN_PYTHON"] = str(self.bin / "fake-python")
        self.executable("docker", '''#!/bin/bash
if [[ "$1 $2" == "image inspect" ]]; then exit 0; fi
if [[ "$1 $2" == "container inspect" ]]; then exit 1; fi
if [[ "$1" == run ]]; then printf '%s\\n' "$@" > "$DIRECT_TEST_CAPTURE"; exit 0; fi
exit 99
''')
        mount = self.root / "data.h5"; mount.touch()
        self.env["DIRECT_TEST_MOUNT"] = str(mount)
        self.executable("fake-host-python", '#!/bin/bash\nprintf "%s\\n" "$DIRECT_TEST_MOUNT"\n')
        self.env["PAN_HOST_PYTHON"] = str(self.bin / "fake-host-python")
        dlpan = self.root / "dlpan"
        wald = dlpan / "01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py"
        wald.parent.mkdir(parents=True); wald.touch()
        self.env["PANCRAFTER_DLPAN"] = str(dlpan)

    def executable(self, name, source):
        path = self.bin / name; path.write_text(source); path.chmod(0o700)

    def run_script(self, name, *args):
        return subprocess.run(["bash", str(ROOT / "scripts" / name), *args], env=self.env,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_shell_syntax(self):
        for name in ("run_directshift48.sh", "run_directshift48_docker.sh"):
            subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], check=True)

    def test_unsupported_lane_stops_before_any_launch(self):
        for name in ("run_directshift48.sh", "run_directshift48_docker.sh"):
            self.assertEqual(self.run_script(name, "s2").returncode, 2)
        self.assertFalse(self.capture.exists())

    def test_busy_gpu_refuses_host_and_docker(self):
        self.env["DIRECT_TEST_GPU"] = "1234, active-training, 4000 MiB\n"
        for name in ("run_directshift48.sh", "run_directshift48_docker.sh"):
            result = self.run_script(name, "s1")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("active", result.stderr.lower())
        self.assertFalse(self.capture.exists())

    def test_gpu_query_failure_is_not_treated_as_idle(self):
        self.executable("nvidia-smi", "#!/bin/bash\nexit 2\n")
        self.assertNotEqual(self.run_script("run_directshift48.sh", "s1").returncode, 0)
        self.assertFalse(self.capture.exists())

    def test_host_training_does_not_start_publisher(self):
        result = self.run_script("run_directshift48.sh", "s3")
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.capture.read_text().splitlines()
        self.assertEqual(argv, ["-m", "directshift48", "run", "--server", "s3"])
        self.assertNotIn("publish", argv)

    def test_cpu_diagnostic_run_ignores_unrelated_gpu_workload(self):
        self.env["DIRECT_TEST_GPU"] = "1234, active-training, 4000 MiB\n"
        result = self.run_script("run_directshift48.sh", "s1", "--device", "cpu")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_docker_evaluator_and_data_mounts_readonly(self):
        result = self.run_script("run_directshift48_docker.sh", "s3")
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.capture.read_text().splitlines()
        self.assertIn("PANCRAFTER_DLPAN=/opt/pancrafter-eval/DLPan-Toolbox", argv)
        self.assertIn("type=bind,src=" + self.env["PANCRAFTER_DLPAN"] +
                      ",dst=/opt/pancrafter-eval/DLPan-Toolbox,readonly", argv)
        self.assertIn("type=bind,src=" + self.env["DIRECT_TEST_MOUNT"] + ",dst=" +
                      self.env["DIRECT_TEST_MOUNT"] + ",readonly", argv)
        self.assertEqual(argv[-3:], ["bash", "scripts/run_directshift48.sh", "s3"])
        self.assertNotIn("publish", argv)

    def test_missing_dlpan_fails_before_container_run(self):
        self.env["PANCRAFTER_DLPAN"] = str(self.root / "missing")
        result = self.run_script("run_directshift48_docker.sh", "s1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DLPan", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_host_absolute_output_remapped_to_persistent_container_path(self):
        result = self.run_script("run_directshift48_docker.sh", "s1", "--output",
                                 str(ROOT / "work_dir/direct-test"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.capture.read_text().splitlines()[-2:], ["--output", "/workspace/work_dir/direct-test"])

    def test_unmounted_outputs_or_catalogs_are_rejected(self):
        for args in (("--output", "/tmp/ephemeral"), ("--sources", "custom.json"),
                     ("--output=/tmp/ephemeral",)):
            result = self.run_script("run_directshift48_docker.sh", "s1", *args)
            self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.capture.exists())

    def test_cpu_docker_does_not_request_gpu(self):
        result = self.run_script("run_directshift48_docker.sh", "s1", "--device=cpu")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--gpus", self.capture.read_text().splitlines())


if __name__ == "__main__":
    unittest.main()
