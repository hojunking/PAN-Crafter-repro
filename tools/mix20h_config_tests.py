#!/usr/bin/env python
"""CPU-only MIX20H registry/config regression tests; no training or queue activation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kdv import mix20h_plan as plan
from kdv.registry import resolve
from tools import gen_mix20h_configs as gen
from tools import gen_pakd50_configs as legacy


class Mix20ConfigTests(unittest.TestCase):
    def test_01_exact_case_registry_matches_plan(self):
        import re
        text = (ROOT / plan.SOURCE_PLAN).read_text()
        ids = re.findall(r"`(PAKD50_QRC24_S[1-5]_(?:G23|B20A03)_W104_D121_WV3_T0_S52\d{3}_FRESH50_v4)`", text)
        self.assertEqual(ids, [c.run_id for c in plan.CASES])
        self.assertEqual(len(set(ids)), 46)
        self.assertEqual([len(plan.cases_for(s)) for s in plan.SERVERS], [8, 8, 12, 12, 6])
        self.assertEqual(sum(c.tier == "base" for c in plan.CASES), 36)
        self.assertEqual(len(plan.GRID_STEPS), 50)
        self.assertEqual(plan.GRID_STEPS[-2:], (49490, 50000))
        for case in plan.CASES:
            pair = plan.pair_for(case.run_id)
            self.assertEqual(len(pair), 2)
            self.assertEqual({p.profile for p in pair}, {"G23", "B20A03"})
            self.assertEqual(pair[1].order, pair[0].order + 1)
            self.assertEqual(case.pairmate_run_id, next(p.run_id for p in pair if p != case))

    def test_02_lock_independent_all_configs_and_queue_unchanged(self):
        from kdv.mix20h_runtime import validate_definition
        old_priority = copy.deepcopy(legacy.PRIORITY_BY_SERVER)
        with patch.object(legacy, "qrc24_recipe_lock", side_effect=AssertionError("MIX20H read historical lock")), \
             patch.object(legacy, "qrc24_control_run", side_effect=AssertionError("virtual control resolver")), \
             patch.object(legacy, "qrc24_seed_profile", side_effect=AssertionError("local profile resolver")):
            for case in plan.CASES:
                cfg = gen.build_config(case.run_id)
                self.assertEqual(validate_definition(cfg)["run_id"], case.run_id)
                k = cfg["kdv"]
                spec = resolve(k)
                self.assertEqual(spec["protocol"], "I-NATIVE-TRANSFER")
                self.assertTrue(spec["aligner_trainable"])
                self.assertEqual(spec["offset_weight_effective"], 0)
                self.assertEqual(k["rec"]["tau"], plan.TAU_R)
                self.assertEqual(k["qrecon"]["q_ref"], plan.Q_REF)
                self.assertEqual(k["stat"]["outer_weight"], .002)
                self.assertEqual(k["rec"]["kd_weight"], case.beta)
                self.assertEqual(k["teacher"]["expected_sha256"], plan.TEACHER_SHA256)
                self.assertEqual(k["donor"]["expected_sha256"], plan.TEACHER_SHA256)
                self.assertNotIn("budget", k)
                self.assertEqual(k["mix20h"]["reservation_hours"], plan.reservation_hours(case.server_id))
                self.assertEqual(k["mix20h"]["expected_runtime"]["mixed_precision"], "no")
                self.assertEqual(len(k["mix20h"]["expected_dataset_hashes"]), 4)
        self.assertEqual(legacy.PRIORITY_BY_SERVER, old_priority)

    def test_03_pair_training_diff_is_only_beta(self):
        def without_identity(cfg):
            cfg = copy.deepcopy(cfg)
            cfg.pop("work_dir")
            k = cfg["kdv"]
            for key in ("case_id", "mix20h", "control_runs"):
                k.pop(key)
            k["rec"].pop("kd_weight")
            for key in ("profile", "canonical", "beta"):
                k["qrc24"].pop(key)
            return cfg
        for case in plan.CASES:
            if case.pair_position == 1:
                self.assertEqual(without_identity(gen.build_config(case.run_id)),
                                 without_identity(gen.build_config(case.pairmate_run_id)))

    def test_04_old_v1_v2_v3_generation_unchanged(self):
        # Execute the actual pre-edit source, rather than duplicating expected
        # dictionaries which could accidentally bless old-campaign mutations.
        source = subprocess.check_output(["git", "show", "HEAD:tools/gen_pakd50_configs.py"], cwd=ROOT, text=True)
        old = types.ModuleType("_mix20_test_pre_edit_generator")
        old.__file__ = str(ROOT / "tools/gen_pakd50_configs.py")
        exec(compile(source, old.__file__, "exec"), old.__dict__)
        args = dict(cal={"tau_R": plan.TAU_R}, arch="W104_D121", branch="QRECON24")
        template = (ROOT / "config/PO10_N1_REC_W112_D123_WV3_S2025_R200_FRSTAT.yaml").read_text()
        for server, profile, seed, version in (("s1", "G23", 1234, "v1"),
                                                ("s1", "B20A03", 1234, "v2"),
                                                ("s1", "B20A03", 1234, "v3"),
                                                ("s3", "G23", 41003, "v1")):
            case = f"QRC24_{server.upper()}_{profile}"
            now = legacy.kdv_block(case, seed, server, version=version, **args)
            before = old.kdv_block(case, seed, server, version=version, **args)
            self.assertEqual(now, before)
            tag = legacy.run_name(case, seed, version, arch="W104_D121")
            self.assertEqual(legacy.render(tag, case, seed, server, now, 50000, 5, template,
                                           arch="W104_D121", branch="QRECON24"),
                             old.render(tag, case, seed, server, before, 50000, 5, template,
                                        arch="W104_D121", branch="QRECON24"))

    def test_05_generation_is_idempotent_and_never_overwrites_collision(self):
        with tempfile.TemporaryDirectory(prefix="mix20-config-test-") as tmp:
            out = Path(tmp)
            made = gen.generate("s5", out)
            self.assertEqual(len(made), 6)
            self.assertEqual(gen.generate("s5", out), made)
            target = out / (made[0] + ".yaml")
            target.write_text("user modified config\n")
            with self.assertRaisesRegex(ValueError, "collision"):
                gen.generate("s5", out)
            self.assertEqual(target.read_text(), "user modified config\n")
            self.assertFalse((out / plan.PLAN_MANIFEST).exists())

    def test_06_identity_and_wrong_route_rejected(self):
        first = plan.CASES[0]
        cfg = gen.build_config(first.run_id)
        for key, wrong in (("profile", "B20A03"), ("seed", 52006), ("server_id", "s2"),
                           ("version", "v1"), ("selector", "HQNR959_SCC_v1"), ("eval_mode", "NOA")):
            bad = copy.deepcopy(cfg)
            bad["kdv"]["mix20h"][key] = wrong
            with self.assertRaises(ValueError):
                gen.validate_config(bad)
        with self.assertRaises(ValueError):
            plan.case_for(first.run_id.replace("G23", "G22"))
        with self.assertRaisesRegex(ValueError, "--mix20"):
            legacy.kdv_block("QRC24_S1_G23", 52001, "s1", version="v4", arch="W104_D121")

    def test_07_legacy_hold_only_after_explicit_activation(self):
        with tempfile.TemporaryDirectory(prefix="mix20-hold-test-") as tmp, patch.object(legacy, "ROOT", tmp):
            self.assertEqual(legacy.eval_hold(), {})
            manifest = Path(tmp) / plan.PLAN_MANIFEST
            manifest.parent.mkdir(parents=True)
            for status in ("READY", "BUDGET_CLOSED"):
                manifest.write_text(json.dumps({"campaign_id": plan.CAMPAIGN_ID, "status": status}))
                self.assertEqual(legacy.eval_hold()["phase"], "SUPERSEDED_BY_MIX20H")
            manifest.write_text("broken json")
            self.assertEqual(legacy.eval_hold()["phase"], "MIX20H_MANIFEST_INVALID")

    def test_08_reference_runtime_and_cue_are_actual_bytes(self):
        import hashlib
        ref, _ = gen.reference_recipe()
        source = ROOT / ref["source_training_file"]
        if source.exists():
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), ref["source_training_sha256"])
            self.assertEqual(json.loads(source.read_text())["training"], ref["training"])
        cue = ROOT / "assets/qedge9/cue_T0_AXIS16_v1.json"
        self.assertEqual(hashlib.sha256(cue.read_bytes()).hexdigest(), ref["cue_manifest_sha256"])
        self.assertEqual(json.loads(cue.read_text())["npz_sha256"], ref["cue_sha256"])
        self.assertEqual(hashlib.sha256(cue.with_suffix('.npz').read_bytes()).hexdigest(), ref["cue_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
