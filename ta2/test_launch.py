import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ta2.launch import (LABEL, assert_gpu_free, build_command, build_mounts, resolve_work,
                        select_image, source_snapshot, validate_gpu, validate_server)
from ta2.common import source_manifest


class LauncherTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)/"repo"
        (root/"ablr2").mkdir(parents=True)
        (root/"ta2").mkdir()
        (root/"ta2/start.sh").write_text("#!/bin/bash\n")
        (root/"ta2/controller.py").write_text("# synthetic fixture\n")
        (root/"research_log/PAN_Aligner_TeacherOnly_v2").mkdir(parents=True)
        (root/"research_log/PAN_Aligner_TeacherOnly_v2/experiment_spec.json").write_text("{}")
        (root/"gspread").mkdir()
        (root/"gspread/account.json").write_text("NOT_A_REAL_CREDENTIAL")
        (root/"gspread/scripts.py").write_text("# uploader helpers are not snapshot imports\n")
        native = Path(folder)/"native"
        native.mkdir()
        (root/"data").mkdir()
        splits = {}
        for split in ("train", "val", "rr", "fr"):
            path = native/(split+".h5"); path.write_bytes(b"fixture")
            (root/"data"/path.name).symlink_to(path)
            splits[split] = dict(path="data/"+path.name)
        catalog = {"sensors": {s: dict(splits=splits) for s in ("WV3", "QB", "GF2")}}
        (root/"ablr2/sensor_sources.json").write_text(json.dumps(catalog))
        dlpan = Path(folder)/"DLPan"; dlpan.mkdir()
        return root, native, dlpan

    def test_protected_servers_and_single_gpu(self):
        for server in ("s2", "s4", "all"):
            with self.assertRaises(ValueError):
                validate_server(server)
        for gpu in ("all", "0,1", "0; rm", ""):
            with self.assertRaises(ValueError):
                validate_gpu(gpu)
        self.assertEqual(validate_gpu("GPU-aaa-bbb"), "GPU-aaa-bbb")

    def test_dataset_symlink_mounts_and_only_owned_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root, native, dlpan = self.fixture(folder)
            work, container_work = resolve_work(root, "s3")
            mounts = build_mounts(root, work, "s3", dlpan)
            self.assertEqual([(a, b) for a, b, mode in mounts if mode == "rw"], [(str(work), str(container_work))])
            self.assertIn((str(native/"train.h5"), str(native/"train.h5"), "ro"), mounts)
            self.assertIn((str(root), "/workspace", "ro"), mounts)
            with self.assertRaises(ValueError):
                resolve_work(root, "s1", root/"work_dir")
            with self.assertRaises(ValueError):
                resolve_work(root, "s1", Path(folder)/"other")

    def test_content_pinned_command_campaign_root_and_no_takeover(self):
        command = build_command(server="s5", command="run", image_id="sha256:"+"a"*64,
                    name="ta2-s5-test", mounts=[("/safe", "/workspace", "ro"), ("/owned", "/workspace/work_dir/ta2/s5", "rw")],
                    container_work="/workspace/work_dir/ta2/s5", dlpan="/DLPan", gpu="GPU-aaa",
                    source_root="/ta2-source", micro_batch=12, publish=True, gpu_uuid="GPU-aaa")
        self.assertIn("--detach", command)
        self.assertNotIn("--privileged", command)
        self.assertNotIn("--pid=host", command)
        self.assertEqual(command[command.index("--work-root")+1], "/workspace/work_dir/ta2")
        self.assertEqual(command[command.index("--workdir")+1], "/ta2-source")
        self.assertIn("--publish", command)
        self.assertIn("device=GPU-aaa", command)
        self.assertIn("TA2_GPU_UUID=GPU-aaa", command)
        self.assertIn("TA2_GOOGLE_QUOTA_DIR=/workspace/work_dir/ta2/s5/google_quota", command)
        self.assertEqual(command[command.index("--restart")+1], "no")
        with self.assertRaises(ValueError):
            build_command(server="s1", command="run", image_id="mutable:latest", name="x", mounts=[],
                          container_work="/workspace/work_dir/ta2/s1", dlpan="/DLPan")

    def test_non_gpu_status(self):
        command = build_command(server="s1", command="status", image_id="sha256:"+"a"*64,
                    name="x", mounts=[], container_work="/workspace/work_dir/ta2/s1", dlpan="/DLPan")
        self.assertNotIn("--gpus", command)
        self.assertNotIn("--detach", command)
        self.assertEqual(command[command.index("--device")+1], "cpu")

    def test_busy_gpu_fail_without_any_stop(self):
        calls = []
        def fake(argv, **kwargs):
            calls.append(argv)
            return "GPU-aaa" if "--query-gpu=uuid" in argv else "GPU-aaa, 123, ongoing_training"
        with patch("ta2.launch.output", side_effect=fake):
            with self.assertRaisesRegex(ValueError, "GPU_BUSY"):
                assert_gpu_free("0")
        self.assertTrue(all(c[0] == "nvidia-smi" for c in calls))

    def test_explicit_image_probe_uses_content_id_and_never_pull(self):
        calls = []
        def fake(argv, **kwargs):
            calls.append(argv)
            return "sha256:"+"f"*64 if argv[1:3] == ["image", "inspect"] else '{"status":"PASS","gpu":"RTX 5090"}'
        with patch("ta2.launch.output", side_effect=fake):
            identity, tag, proof = select_image("server-local:cu128", gpu="0", probe=True)
        self.assertEqual(identity, "sha256:"+"f"*64)
        self.assertEqual(tag, "server-local:cu128")
        self.assertEqual(proof["status"], "PASS")
        self.assertFalse(any("pull" in c for c in calls))
        self.assertIn(identity, calls[-1])

    def test_frozen_source_ignores_future_repo_edits_and_excludes_secret(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _, _ = self.fixture(folder)
            lane, _ = resolve_work(root, "s1")
            predicted, manifest = source_snapshot(root, lane, write=False)
            self.assertFalse(predicted.exists())
            frozen, actual = source_snapshot(root, lane)
            self.assertEqual(manifest, actual)
            self.assertEqual((frozen/"ta2/controller.py").read_text(), "# synthetic fixture\n")
            self.assertNotIn("gspread/account.json", actual["files"])
            self.assertNotIn("gspread/scripts.py", actual["files"])
            self.assertEqual((frozen/"data").readlink(), Path("/workspace/data"))
            self.assertEqual(source_snapshot(root, lane)[0], frozen)
            (root/"ta2/controller.py").write_text("# changed code\n")
            newer, _ = source_snapshot(root, lane)
            self.assertNotEqual(newer, frozen)
            self.assertEqual((frozen/"ta2/controller.py").read_text(), "# synthetic fixture\n")

    def test_detail_registry_is_frozen_and_changes_snapshot_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _, _ = self.fixture(folder)
            lane, _ = resolve_work(root, "s1")
            registry = root/"ta2/detail_books.json"
            registry.write_text(json.dumps({"entries": {"TA2_s1_B01_P01": {"spreadsheet_id": "fake-a"}}}))
            for relative in ("fh20r1/training.py", "utils.py", "tools/eval_dlpan.py", "feeders/feeder.py"):
                path = root/relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# synthetic source identity fixture\n")
            runtime_manifest = source_manifest(root)
            self.assertIn("ta2/detail_books.json", runtime_manifest)
            frozen, manifest = source_snapshot(root, lane)
            self.assertIn("ta2/detail_books.json", manifest["files"])
            self.assertEqual((frozen/"ta2/detail_books.json").read_text(), registry.read_text())
            registry.write_text(json.dumps({"entries": {"TA2_s1_B01_P01": {"spreadsheet_id": "fake-b"}}}))
            self.assertNotEqual(runtime_manifest["ta2/detail_books.json"], source_manifest(root)["ta2/detail_books.json"])
            changed, changed_manifest = source_snapshot(root, lane)
            self.assertNotEqual(frozen, changed)
            self.assertNotEqual(manifest["files"]["ta2/detail_books.json"], changed_manifest["files"]["ta2/detail_books.json"])
            self.assertEqual(json.loads((frozen/"ta2/detail_books.json").read_text())["entries"]["TA2_s1_B01_P01"]["spreadsheet_id"], "fake-a")


if __name__ == "__main__":
    unittest.main()
