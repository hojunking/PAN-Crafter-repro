"""Controller contracts without GPU work, donor downloads, or sheet writes."""
import copy
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import uuid

from panpv.common import atomic_json, file_sha, load_json
from panpv.campaign import Campaign, REQUIRED_SOURCE_FILES, stage_plan, validate_registry


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.source = Path(__file__).resolve().parents[1]
        self.output = self.source / "work_dir" / ("panpv_unit_" + uuid.uuid4().hex)
        self.patcher = patch("panpv.campaign.REQUIRED_SOURCE_FILES", ("panpv/campaign.py",))
        self.patcher.start()
        self.registry = self.fixture()

    def tearDown(self):
        self.patcher.stop()
        # Only this test's explicit UUID directory; never a broad work_dir.
        if self.output.exists():
            import shutil
            shutil.rmtree(self.output)
        self.temporary.cleanup()

    def bound(self, name, value):
        path = self.base / name
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            atomic_json(path, value)
        return dict(path=str(path), sha256=file_sha(path))

    def fixture(self):
        scenes = []
        for index in range(5):
            record = self.bound(f"scene{index}.h5", b"not opened by registry-only tests")
            scenes.append(dict(scene_id=f"scene{index}", file=f"scene{index}.h5",
                               sha256=record["sha256"], patch_index_sha256="a" * 64))
        donor = self.bound("donor.safetensors", b"not deserialized by controller tests")
        provenance = self.bound("donor.json", dict(source="test-only"))
        donor.update(provenance_path=provenance["path"], provenance_sha256=provenance["sha256"])
        protocol = self.bound("protocol.json", dict(scenes={s["scene_id"]: {} for s in scenes}))
        preflight = self.bound("preflight.json", dict(status="PASS", donor_sha256=donor["sha256"],
                                                     micro_batch=12, ft0_zs_parity=True,
                                                     microbatch_verified=True))
        return dict(schema="PANPV_CAMPAIGN_v1", campaign_id="TEST_ONLY", execution_server="s1",
                    target_sensor="WV2", seed=261002, device="cuda", micro_batch=12,
                    source_root=str(self.source), source_sha256={"panpv/campaign.py": file_sha(self.source / "panpv/campaign.py")},
                    input_root=str(self.base), work_root=str(self.output), scenes=scenes, donor=donor,
                    protocol=protocol, preflight=preflight)

    def campaign(self):
        path = self.base / "registry.json"
        atomic_json(path, self.registry)
        return Campaign(path)

    def test_queue_is_ten_baselines_then_interleaved_2k_then_two_selectors(self):
        queue = stage_plan([f"scene{i}" for i in range(5)])
        self.assertEqual(len(queue), 130)
        self.assertEqual([s["method"] for s in queue[:4]], ["BROVEY8", "ZS_WV3", "BROVEY8", "ZS_WV3"])
        self.assertTrue(all(s["kind"] == "baseline" for s in queue[:10]))
        self.assertTrue(all(s["target_step"] == 2000 for s in queue[10:20]))
        self.assertEqual([s["method"] for s in queue[10:12]], ["SC_SCENE", "FT_WV3_SCENE"])
        self.assertEqual(queue[109]["target_step"], 20000)
        self.assertEqual([s["selector"] for s in queue[110:112]], ["EXACT20K", "BEST_RRVAL_TO_20K"])

    def test_registry_good_and_does_not_create_outputs(self):
        validate_registry(self.registry)
        self.assertFalse(self.output.exists())

    def test_missing_donor_or_changed_bytes_is_rejected(self):
        Path(self.registry["donor"]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            validate_registry(self.registry)

    def test_false_preflight_cannot_become_admission(self):
        path = Path(self.registry["preflight"]["path"])
        payload = load_json(path)
        payload["ft0_zs_parity"] = False
        atomic_json(path, payload)
        self.registry["preflight"]["sha256"] = file_sha(path)
        with self.assertRaisesRegex(ValueError, "admission"):
            validate_registry(self.registry)

    def test_invalid_output_and_scene_paths_rejected(self):
        for changed in (dict(work_root=str(self.source / "work_dir")), dict(work_root=str(self.base))):
            value = dict(self.registry, **changed)
            with self.assertRaises(ValueError):
                validate_registry(value)
        self.registry["scenes"][0]["file"] = "../outside.h5"
        with self.assertRaisesRegex(ValueError, "leaves"):
            validate_registry(self.registry)

    def test_protocol_must_include_all_scenes(self):
        self.registry["protocol"] = self.bound("broken_protocol.json", {"scenes": {}})
        with self.assertRaisesRegex(ValueError, "all five"):
            validate_registry(self.registry)

    def test_freeze_prevents_changed_identity(self):
        campaign = self.campaign()
        self.output.mkdir(parents=True)
        atomic_json(self.output / "campaign_manifest.json", {"registry_sha256": "wrong"})
        with patch("torch.cuda.is_available", return_value=True):
            with self.assertRaisesRegex(ValueError, "identity"):
                campaign.run()

    def test_success_receipt_and_resume_do_not_repeat_stage(self):
        campaign = self.campaign()
        campaign.queue = campaign.queue[:1]
        calls = []
        def stage(value):
            calls.append(value["id"])
            atomic_json(self.output / "evaluations" / value["id"] / "report.json", {"status": "COMPLETE"})
            return {"result": {"status": "COMPLETE"}}
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.empty_cache"), \
             patch.object(campaign, "_stage", side_effect=stage), patch.object(campaign, "_publish") as publisher:
            self.assertEqual(campaign.run(), 0)
            self.assertEqual(campaign.run(), 0)
        # After the stage, at first completion, and at completed-only resume.
        self.assertEqual(publisher.call_count, 3)
        self.assertEqual(len(calls), 1)
        self.assertEqual(load_json(self.output / "campaign_status.json")["state"], "LOCAL_COMPUTE_COMPLETE")
        self.assertEqual(load_json(self.output / "campaign_status.json")["sheet_status"], "CHECK_REPORTING_OUTBOX")
        self.assertEqual(len(list((self.output / "outbox").glob("*.json"))), 1)

    def test_completed_artifact_corruption_stops_resume(self):
        campaign = self.campaign()
        self.output.mkdir(parents=True)
        stage = campaign.queue[0]
        artifact = self.output / "evaluations" / stage["id"] / "report.json"
        atomic_json(artifact, {"status": "COMPLETE"})
        atomic_json(campaign._receipt(stage), dict(state="COMPLETE", registry_sha256=campaign.registry_sha,
                    stage=stage, artifacts=[dict(path=str(artifact.relative_to(self.output)), sha256=file_sha(artifact))]))
        atomic_json(artifact, {"status": "changed"})
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            campaign._completed(stage)

    def test_signal_does_not_mark_incomplete_stage_complete(self):
        campaign = self.campaign()
        campaign.queue = campaign.queue[:1]
        def pause(_):
            campaign.stop_requested = True
            campaign.stop_signal = 15
            return {"status": "PAUSED_SIGNAL"}
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.empty_cache"), \
             patch.object(campaign, "_stage", side_effect=pause):
            self.assertEqual(campaign.run(), 130)
        self.assertFalse(campaign._receipt(campaign.queue[0]).exists())
        self.assertEqual(load_json(self.output / "campaign_status.json")["state"], "PAUSED_SIGNAL")

    def test_disk_guard_never_deletes_artifacts(self):
        campaign = self.campaign()
        self.output.mkdir(parents=True)
        sentinel = self.output / "keep.txt"
        sentinel.write_text("preserve")
        with patch("panpv.campaign.shutil.disk_usage") as usage:
            usage.return_value.free = 10
            with self.assertRaisesRegex(OSError, "SAFE_STOP_DISK"):
                campaign._space()
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_export_allowlist_excludes_credentials_and_weights(self):
        campaign = self.campaign()
        stage = next(s for s in campaign.queue if s["kind"] == "train")
        run = campaign._run_root(stage["scene_id"], stage["method"])
        run.mkdir(parents=True)
        (run / "train_steps.jsonl").write_text('{"step":1}\n')
        (run / "credentials.json").write_text('{"secret":"excluded"}')
        (run / "model.safetensors").write_bytes(b"excluded")
        receipt = campaign._export(stage)
        with tarfile.open(receipt["path"], "r:gz") as archive:
            names = archive.getnames()
        self.assertEqual(len(names), 1)
        self.assertTrue(names[0].endswith("train_steps.jsonl"))

    def test_real_engine_adapter_resumes_fullstate_and_preserves_metadata(self):
        import torch
        from types import SimpleNamespace
        from panpv.test_training import ToyData, ToyModel
        from panpv.training import TrainingEngine
        campaign = self.campaign()
        self.output.mkdir(parents=True)
        scene = SimpleNamespace(scene_id="scene0", sha256=self.registry["scenes"][0]["sha256"])
        def make_model(_):
            model = ToyModel()
            model.architecture_manifest = lambda: {"model": "toy_test_only"}
            return model
        def make_engine(*args, **kwargs):
            kwargs.update(device="cpu", max_updates=4, block_updates=2,
                          validation_interval=2, probe_interval=2, diagnostic_steps=(0, 2, 4))
            return TrainingEngine(*args, **kwargs)
        def evaluate(*args, **kwargs):
            return {"status": "COMPLETE", "rr": {"metrics": {"ergas": 1., "scc": .8}}, "HQNR": None}
        def dataset(*args):
            data = ToyData(3)
            data.manifest = {"identity_sha256": "test-only"}
            return data
        torch.set_num_threads(1)
        with patch.object(campaign, "_scene", return_value=scene), \
             patch.object(campaign, "_model", side_effect=make_model), \
             patch.object(campaign, "_evaluate", side_effect=evaluate), \
             patch.object(campaign, "_dataset", side_effect=dataset), \
             patch("panpv.training.TrainingEngine", side_effect=make_engine):
            first = campaign._train(dict(scene_id="scene0", method="SC_SCENE", target_step=2))
            second = campaign._train(dict(scene_id="scene0", method="SC_SCENE", target_step=4))
        self.assertEqual(first["completed_step"], 2)
        self.assertEqual(second["completed_step"], 4)
        root = campaign._run_root("scene0", "SC_SCENE")
        self.assertEqual(load_json(root / "run_manifest.json")["target_sensor"], "WV2")
        self.assertTrue(load_json(root / "config.resolved.json")["diagnostics_configured"])
        self.assertTrue((root / "resume_events.jsonl").exists())

    def test_campaign_reuses_same_scene_split_cache_without_second_h5_read(self):
        from .test_v3_data import write_scene
        from .v3_data import SceneV3
        campaign = self.campaign()
        self.output.mkdir(parents=True)
        path = write_scene(self.base / "real_synthetic.h5")
        with SceneV3(path, expected_sha256=file_sha(path)) as scene:
            first = campaign._dataset(scene, "train")
            with patch.object(scene, "read_region", side_effect=AssertionError("Repeated H5 read")):
                second = campaign._dataset(scene, "train")
                second[0]
            self.assertIs(first, second)
        self.assertEqual(len((self.output / "cache_events.jsonl").read_text().splitlines()), 1)

    def test_recreated_process_cache_preserves_frozen_content_identity(self):
        from .test_v3_data import write_scene
        from .v3_data import SceneV3
        campaign = self.campaign()
        self.output.mkdir(parents=True)
        path = write_scene(self.base / "real_synthetic.h5")
        with SceneV3(path, expected_sha256=file_sha(path)) as scene:
            first = campaign._dataset(scene, "train")
            restarted = self.campaign()
            second = restarted._dataset(scene, "train")
            self.assertEqual(first.manifest["identity_sha256"], second.manifest["identity_sha256"])
            self.assertIsNot(first, second)
        self.assertEqual(len((self.output / "cache_events.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
