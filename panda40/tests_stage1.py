"""Synthetic CPU/H5 restart, pairing and frozen-donor contract checks.

Metric fixtures below are intentionally synthetic and never published as
experimental observations. These tests do not contact a sheet or start a run.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from donor24 import plan as old_plan
from donor30.artifacts import _same, capture_rng
from donor30.evaluation import model_state_hash
from panda_method.calibration import state_dict_sha256
from panda_stage2_condq.donor import load_donor
from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
from s2dshift48.donor import load_donor as load_s2d_donor
from ta2.evaluation import file_sha

from .stage1 import (Stage1Stream, advance_stage1, build_stage1,
                    export_s2d_donor, load_stage1_checkpoint, save_stage1_checkpoint)


PINS = {"code_pin": {"synthetic_test": "a" * 64},
        "evaluator_pin": {"synthetic_test_not_official": "b" * 64}}


def config(kind="direct"):
    cfg = old_plan.resolve_case(old_plan.default_config("s1"), "A1")
    cfg.update(aligner_kind=kind, mask_policy="MASK2", run_id="SYNTHETIC_NOT_A_RUN_" + kind)
    cfg["baseline_resolved"].update(effective_batch_RR=1, microbatch_RR=1, accumulation_RR=1)
    return cfg


def evaluation(trainer):
    """Unit fixture only; no test result is official or selectable externally."""
    cfg = trainer.config
    identity = {key: cfg[key] for key in ("run_id", "attempt", "seed", "dataset")}
    identity.update(completed_update=trainer.updates, model_state_sha256=model_state_hash(trainer.stage))
    result = {key: deepcopy(identity) for key in ("state_before", "rr_state", "fr_state", "state_after")}
    result["qualification"] = {k: True for k in ("same_state_verified", "metrics_complete", "metrics_finite")}
    for split in ("rr", "fr"):
        result[split] = {"metrics": {"HQNR": .95, "SCC": .98, "ERGAS": 2.1},
                         "per_scene": [{"scene_id": f"SYNTHETIC_{split}_{i}",
                                        "raw_prediction_sha256": f"{i:064x}"} for i in range(20)]}
    return result


class Stage1CampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registered = synthetic_sources(self.root, count=3)

    def make(self, kind="direct"):
        cfg = config(kind)
        return (build_stage1(cfg, self.registered.sources),
                Stage1Stream(self.registered.sources, cfg, device="cpu"), cfg)

    def save(self, trainer, stream, cfg, name="state.pt", attached=False):
        return save_stage1_checkpoint(trainer, stream, self.root / name, cfg,
            evaluation(trainer) if attached else None, **PINS)

    def test_fresh_matched_teacher_data_augmentation_epsilon_rng(self):
        rng = capture_rng()
        left, ls, lc = self.make("direct")
        right, rs, rc = self.make("s2d")
        self.assertTrue(_same(rng, capture_rng()))
        self.assertEqual(state_dict_sha256(left.stage.teacher), state_dict_sha256(right.stage.teacher))
        self.assertNotEqual(state_dict_sha256(left.stage.aligner), state_dict_sha256(right.stage.aligner))
        self.assertTrue(_same(ls.state_dict(), rs.state_dict()))
        lb, rb = ls.next_update()[0], rs.next_update()[0]
        self.assertEqual(lb.lineage, rb.lineage)
        self.assertTrue(torch.equal(lb.pan, rb.pan))
        lo, ro = left.stage(lb), right.stage(rb)
        for field in ("epsilon_rr", "loss", "rec", "abs_rr", "eps_rr"):
            self.assertTrue(torch.equal(lo[field], ro[field]), field)
        self.assertEqual(lc["stream_seeds"], rc["stream_seeds"])

    def test_both_architectures_exact_next_update_after_complete_resume(self):
        for kind in ("direct", "s2d"):
            with self.subTest(kind=kind):
                trainer, stream, cfg = self.make(kind)
                advance_stage1(trainer, stream)
                before_rng = capture_rng()
                receipt = self.save(trainer, stream, cfg, name=kind + ".pt")
                self.assertTrue(_same(before_rng, capture_rng()))
                actual_next = advance_stage1(trainer, stream)
                expected_state = trainer.state_dict()
                expected_stream = stream.state_dict()
                restored, sampler, payload = load_stage1_checkpoint(
                    receipt["checkpoint_path"], receipt["file_sha256"], cfg,
                    self.registered.sources, **PINS)
                replay = advance_stage1(restored, sampler)
                self.assertTrue(_same(actual_next, replay))
                self.assertTrue(_same(expected_state, restored.state_dict()))
                self.assertTrue(_same(expected_stream, sampler.state_dict()))
                self.assertTrue(payload["campaign_resume_complete"])
                self.assertTrue(payload["resume_only"])

    def test_immutable_direct_candidate_is_same_pair_stage2_import(self):
        trainer, stream, cfg = self.make()
        advance_stage1(trainer, stream)
        receipt = self.save(trainer, stream, cfg, attached=True)
        self.assertTrue(receipt["direct_stage2_import_compatible"])
        self.assertFalse(receipt["resume_only"])
        loaded = load_donor(receipt["checkpoint_path"], receipt["file_sha256"], self.registered)
        self.assertEqual(state_dict_sha256(loaded.aligner), state_dict_sha256(trainer.stage.aligner))
        self.assertEqual(state_dict_sha256(loaded.teacher), state_dict_sha256(trainer.stage.teacher))
        loaded.assert_frozen()
        with self.assertRaises(FileExistsError):
            self.save(trainer, stream, cfg, attached=True)

    def test_s2d_explicit_paired_export_and_old_donor_cross_load_rejected(self):
        trainer, stream, cfg = self.make("s2d")
        advance_stage1(trainer, stream)
        receipt = self.save(trainer, stream, cfg, attached=True)
        self.assertFalse(receipt["direct_stage2_import_compatible"])
        with self.assertRaises(ValueError):
            load_donor(receipt["checkpoint_path"], receipt["file_sha256"], self.registered)
        exported = export_s2d_donor(self.root / "paired.pt", trainer, self.registered, run_id=cfg["run_id"])
        loaded = load_s2d_donor(exported["path"], exported["sha256"], self.registered)
        self.assertEqual(state_dict_sha256(loaded.aligner), state_dict_sha256(trainer.stage.aligner))
        self.assertEqual(state_dict_sha256(loaded.teacher), state_dict_sha256(trainer.stage.teacher))
        self.assertFalse(exported["qualified_donor"])
        self.assertFalse(exported["campaign_resume_complete"])

    def test_sampler_corruption_fails_before_mutation(self):
        _, stream, _ = self.make()
        state = stream.state_dict()
        mutations = [lambda x: x.update(cursor=3), lambda x: x.update(epoch=-1),
                     lambda x: x.update(order=[0, 1, 1]), lambda x: x.update(count=4),
                     lambda x: x.update(augmentation_rng={"bad": "rng"})]
        for mutate in mutations:
            corrupted = deepcopy(state)
            mutate(corrupted)
            with self.assertRaises(ValueError):
                stream.load_state_dict(corrupted)
            self.assertTrue(_same(state, stream.state_dict()))

    def test_preoptimizer_failure_restores_sampler_and_lr(self):
        trainer, stream, cfg = self.make()
        stream_before = stream.state_dict()
        lrs = [group["lr"] for group in trainer.optimizer.param_groups]
        with patch.object(trainer, "step_accumulated", side_effect=ValueError("before optimizer")):
            with self.assertRaises(ValueError):
                advance_stage1(trainer, stream)
        self.assertTrue(_same(stream_before, stream.state_dict()))
        self.assertEqual(lrs, [group["lr"] for group in trainer.optimizer.param_groups])
        self.assertEqual(trainer.updates, 0)
        self.assertFalse(trainer.poisoned)

    def test_checkpoint_pin_config_scheduler_optimizer_and_pair_corruption_rejected(self):
        trainer, stream, cfg = self.make()
        advance_stage1(trainer, stream)
        receipt = self.save(trainer, stream, cfg)
        path, sha = receipt["checkpoint_path"], receipt["file_sha256"]
        with self.assertRaisesRegex(ValueError, "SHA"):
            load_stage1_checkpoint(path, "0" * 64, cfg, self.registered.sources, **PINS)
        wrong_pins = deepcopy(PINS)
        wrong_pins["code_pin"]["synthetic_test"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "contract"):
            load_stage1_checkpoint(path, sha, cfg, self.registered.sources, **wrong_pins)
        wrong_cfg = deepcopy(cfg)
        wrong_cfg["seed"] += 1
        with self.assertRaisesRegex(ValueError, "contract"):
            load_stage1_checkpoint(path, sha, wrong_cfg, self.registered.sources, **PINS)
        original = torch.load(path, map_location="cpu", weights_only=False)
        mutations = [
            lambda p: p["scheduler"].update(completed_update=2),
            lambda p: p["trainer"]["optimizer"]["param_groups"][0].update(lr=1.),
            lambda p: p["trainer"]["optimizer"]["state"][0]["step"].fill_(7),
            lambda p: p["trainer"]["stage"]["aligner.fc2.bias"].add_(1.),
            lambda p: p["stream_protocol"].update(source_train_manifest_sha256="0" * 64),
            lambda p: p["stream"].update(cursor=2),
        ]
        for i, mutate in enumerate(mutations):
            payload = deepcopy(original)
            mutate(payload)
            altered = self.root / f"corrupt_{i}.pt"
            torch.save(payload, altered)
            with self.assertRaises(ValueError):
                load_stage1_checkpoint(altered, file_sha(altered), cfg, self.registered.sources, **PINS)

    def test_sampler_optimizer_boundary_must_agree_before_checkpoint(self):
        trainer, stream, cfg = self.make()
        advance_stage1(trainer, stream)
        stream.next_update()
        with self.assertRaisesRegex(ValueError, "one boundary"):
            self.save(trainer, stream, cfg)
        self.assertFalse((self.root / "state.pt").exists())

    def test_candidate_rejects_stale_incomplete_mixed_state_metrics(self):
        trainer, stream, cfg = self.make()
        advance_stage1(trainer, stream)
        mutations = [lambda p: p["qualification"].update(metrics_complete=False),
                     lambda p: p["fr_state"].update(completed_update=2),
                     lambda p: p["fr"]["metrics"].update(HQNR=float("nan")),
                     lambda p: p["rr"]["per_scene"].pop()]
        for i, mutate in enumerate(mutations):
            value = evaluation(trainer)
            mutate(value)
            path = self.root / f"rejected_{i}.pt"
            with self.assertRaises(ValueError):
                save_stage1_checkpoint(trainer, stream, path, cfg, value, **PINS)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
