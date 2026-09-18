#!/usr/bin/env python
"""CPU-only tests of launch config preparation; all writes use temp dirs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from kdv import mix20h_plan as plan
from tools import mix20h_launch_config as launch


class LaunchConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mix20-launch-config-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cases = plan.cases_for("s5")
        self.build = patch.object(launch.gen, "build_config", side_effect=self.config)
        self.render = patch.object(launch.gen, "render_config", side_effect=self.rendered)
        self.build.start()
        self.render.start()
        self.addCleanup(self.build.stop)
        self.addCleanup(self.render.stop)

    @staticmethod
    def config(run_id, root):
        root = Path(root)
        result = dict(work_dir=str(root / "work_dir" / run_id), seed=plan.case_for(run_id).seed,
                      immutable_recipe=dict(beta=plan.case_for(run_id).beta, iterations=50000))
        for field in launch.DATA_FIELDS:
            result[field] = dict(dataroot=str(root / "data/WV3" / (field + ".h5")), untouched=True)
        return result

    @classmethod
    def rendered(cls, run_id, root):
        return "# Generated config\n" + yaml.safe_dump(cls.config(run_id, root), sort_keys=False)

    def put_configs(self, old_root=None):
        for case in self.cases:
            path = self.root / "config" / (case.run_id + ".yaml")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.rendered(case.run_id, old_root or self.root))

    def first(self):
        return self.root / "config" / (self.cases[0].run_id + ".yaml")

    def test_01_missing_configs_materialized_without_runtime_activation(self):
        summary = launch.ensure_configs(self.root, "s5")
        self.assertEqual(len(summary["generated"]), 6)
        self.assertEqual(summary["queue"]["action"], "generated")
        self.assertFalse((self.root / plan.PLAN_MANIFEST).exists())
        self.assertEqual(yaml.safe_load(self.first().read_text()), self.config(self.cases[0].run_id, self.root))

    def test_02_dryrun_has_no_files_or_directories(self):
        summary = launch.ensure_configs(self.root, "s5", dry_run=True)
        self.assertEqual(len(summary["generated"]), 6)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_03_root_rebase_has_exact_original_byte_backups(self):
        self.put_configs(Path("/different/checkout/PAN-Crafter"))
        before = self.first().read_bytes()
        preview = launch.ensure_configs(self.root, "s5", dry_run=True)
        self.assertEqual(len(preview["rebased"]), 6)
        self.assertEqual(self.first().read_bytes(), before)
        self.assertFalse((self.root / plan.STATE_DIR).exists())
        summary = launch.ensure_configs(self.root, "s5")
        self.assertEqual(len(summary["rebased"]), 6)
        self.assertEqual((self.root / summary["backups"][0]).read_bytes(), before)
        self.assertEqual(yaml.safe_load(self.first().read_bytes()), self.config(self.cases[0].run_id, self.root))

    def test_04_changed_recipe_fails_before_any_config_or_queue_write(self):
        self.put_configs(Path("/old/checkout"))
        before = self.first().read_bytes()
        last = self.root / "config" / (self.cases[-1].run_id + ".yaml")
        changed = yaml.safe_load(last.read_text())
        changed["immutable_recipe"]["iterations"] = 100
        last.write_text(yaml.safe_dump(changed))
        with self.assertRaisesRegex(ValueError, "beyond checkout paths"):
            launch.ensure_configs(self.root, "s5")
        self.assertEqual(self.first().read_bytes(), before)
        self.assertFalse((self.root / plan.STATE_DIR / "config_before").exists())
        self.assertFalse((self.root / "config/queues").exists())

    def test_05_inconsistent_data_root_is_not_silently_repaired(self):
        self.put_configs(Path("/old/checkout"))
        changed = yaml.safe_load(self.first().read_text())
        changed[launch.DATA_FIELDS[0]]["dataroot"] = "/unrelated/custom-train.h5"
        self.first().write_text(yaml.safe_dump(changed))
        before = self.first().read_bytes()
        with self.assertRaisesRegex(ValueError, "beyond checkout paths"):
            launch.ensure_configs(self.root, "s5")
        self.assertEqual(self.first().read_bytes(), before)

    def test_06_runtime_manifest_freezes_rebase(self):
        self.put_configs(Path("/old/checkout"))
        manifest = self.root / plan.PLAN_MANIFEST
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps(dict(campaign_id=plan.CAMPAIGN_ID)))
        with self.assertRaisesRegex(ValueError, "frozen"):
            launch.ensure_configs(self.root, "s5")

    def test_07_any_existing_local_campaign_run_freezes_rebase(self):
        self.put_configs(Path("/old/checkout"))
        (self.root / "work_dir" / plan.CASES[0].run_id).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            launch.ensure_configs(self.root, "s5")

    def test_08_canonical_config_bytes_remain_unchanged_even_when_frozen(self):
        self.put_configs()
        before = b"# Retain user comments and byte identity\n" + self.first().read_bytes()
        self.first().write_bytes(before)
        (self.root / "work_dir" / self.cases[0].run_id).mkdir(parents=True)
        summary = launch.ensure_configs(self.root, "s5")
        self.assertEqual(len(summary["unchanged"]), 6)
        self.assertEqual(summary["rebased"], [])
        self.assertEqual(self.first().read_bytes(), before)

    def test_09_missing_config_cannot_be_reconstructed_after_activation(self):
        (self.root / "work_dir" / self.cases[0].run_id).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "frozen; missing"):
            launch.ensure_configs(self.root, "s5")
        self.assertFalse((self.root / "config").exists())

    def test_10_bad_queue_fails_before_config_relocation(self):
        self.put_configs(Path("/old/checkout"))
        before = self.first().read_bytes()
        queue = self.root / "config/queues/qrc24_mix20h_s5.txt"
        queue.parent.mkdir(parents=True)
        queue.write_text("legacy_run\n")
        with self.assertRaisesRegex(ValueError, "static queue differs"):
            launch.ensure_configs(self.root, "s5")
        self.assertEqual(self.first().read_bytes(), before)
        self.assertEqual(queue.read_text(), "legacy_run\n")

    def test_11_idempotent_and_no_backup_for_canonical_configs(self):
        launch.ensure_configs(self.root, "s5")
        summary = launch.ensure_configs(self.root, "s5")
        self.assertEqual(len(summary["unchanged"]), 6)
        self.assertEqual(summary["generated"], [])
        self.assertEqual(summary["rebased"], [])
        self.assertEqual(summary["backups"], [])
        self.assertEqual(summary["queue"]["action"], "unchanged")

    def test_12_invalid_server_does_not_create_lock(self):
        with self.assertRaises(ValueError):
            launch.ensure_configs(self.root, "s99")
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
