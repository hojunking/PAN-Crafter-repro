"""Regression guards for exact resume, attached evaluation, and FP32 policy.

All tensors/checkpoints here are synthetic CPU fixtures in temporary folders;
no live model, GPU job, official evaluation, or cloud result is touched.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from panda_method.calibration import canonical_sha256, state_dict_sha256
from panda_stage2_condq.checkpoint import save_checkpoint, load_checkpoint
from panda_stage2_condq.evaluation import model_identity, validate_result
from panda_stage2_condq.spec import METHOD_VERSION
from panda_stage2_condq.training import SELECTION_RULE, configure_precision
from panda_stage2_condq.tests.test_training_evaluation import stage_and_trainer, rr_batch, EvalSources
from ta2.evaluation import file_sha


def evaluation_fixture(trainer):
    """Complete but explicitly synthetic record; it cannot select a real best."""
    identity = model_identity(trainer, EvalSources(), {})
    record = {"schema": "PANDA_S2_CONDQ_OFFICIAL_EVAL_V1", "method_version": METHOD_VERSION,
        "sensor": "WV3", "completed_update": trainer.updates,
        "eval_protocol": {}, "selection_rule": SELECTION_RULE,
        "checkpoint_selection": trainer.selection_policy.manifest(),
        "pipeline": deepcopy(trainer.stage.pipeline_manifest()),
        "metric_function_injection": True, "same_state_verified": True,
        "metrics_complete": True, "test_aware_selection": True}
    for key in ("state_before", "rr_state", "fr_state", "state_after"):
        record[key] = deepcopy(identity)
    values = {"rr": {"ERGAS": 2.1, "SCC": .98, "SAM": 2., "PSNR": 39.,
                      "SSIM": .97, "Q8": .95, "RMSE": 11., "CC": .99},
              "fr": {"HQNR": .958, "D_s": .03, "D_lambda": .01, "JQM": .86}}
    for split, metrics in values.items():
        record[split] = {"metrics": deepcopy(metrics), "per_scene": [
            {"scene_id": f"WV3:{split}:{i}", "metrics": deepcopy(metrics)} for i in range(20)]}
    validate_result(record)
    return record


def change_identities(record, field, value):
    for key in ("state_before", "rr_state", "fr_state", "state_after"):
        record[key][field] = deepcopy(value)


class CheckpointGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_same_weights_wrong_loss_contract_or_protocol_cannot_attach(self):
        trainer = stage_and_trainer()
        valid = evaluation_fixture(trainer)
        wrong_pipeline = deepcopy(valid)
        wrong_pipeline["pipeline"]["alpha"] = 99.
        wrong_contract = deepcopy(valid)
        change_identities(wrong_contract, "training_contract_sha256", "f"*64)
        wrong_protocol = deepcopy(valid)
        wrong_protocol["eval_protocol"] = {"different_evaluator": True}
        change_identities(wrong_protocol, "eval_protocol_sha256", canonical_sha256(wrong_protocol["eval_protocol"]))
        wrong_step = deepcopy(valid)
        wrong_step["completed_update"] += 1
        change_identities(wrong_step, "completed_update", wrong_step["completed_update"])
        with tempfile.TemporaryDirectory() as temp:
            for i, malformed in enumerate((wrong_pipeline, wrong_contract, wrong_protocol, wrong_step)):
                with self.subTest(case=i):
                    path = Path(temp) / f"invalid_{i}.pt"
                    with self.assertRaises(ValueError):
                        save_checkpoint(path, trainer, evaluation=malformed)
                    self.assertFalse(path.exists())
            good = Path(temp) / "valid.pt"
            written = save_checkpoint(good, trainer, evaluation=valid)
            self.assertEqual(written["sha256"], file_sha(good))

    def test_eval_schema_top_level_step_and_pin_digest_must_match(self):
        valid = evaluation_fixture(stage_and_trainer())
        mutations = []
        for key, value in (("schema", "LEGACY_MASK2"), ("sensor", "BAD_SENSOR"),
                           ("completed_update", 7), ("selection_rule", "ERGAS_FIRST"),
                           ("eval_protocol", {"changed": True})):
            bad = deepcopy(valid); bad[key] = value; mutations.append(bad)
        for i, bad in enumerate(mutations):
            with self.subTest(case=i), self.assertRaises(ValueError):
                validate_result(bad)

    def test_checkpoint_load_rejects_wrong_attached_eval_before_restoring_model(self):
        source = stage_and_trainer()
        source.step(rr_batch())
        evaluation = evaluation_fixture(source)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "original.pt"
            save_checkpoint(path, source, evaluation=evaluation)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            change_identities(payload["evaluation"], "training_contract_sha256", "f"*64)
            malformed = Path(temp) / "different_eval_contract.pt"
            torch.save(payload, malformed)
            target = stage_and_trainer()
            before = state_dict_sha256(target.stage.student)
            with self.assertRaises(ValueError):
                load_checkpoint(malformed, file_sha(malformed), target)
            self.assertEqual(target.updates, 0)
            self.assertEqual(before, state_dict_sha256(target.stage.student))
            self.assertFalse(target.poisoned)

    def test_valid_attached_eval_round_trip_for_nonzero_update(self):
        source = stage_and_trainer()
        source.step(rr_batch())
        expected = evaluation_fixture(source)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "valid_step1.pt"
            written = save_checkpoint(path, source, evaluation=expected)
            target = stage_and_trainer()
            loaded = load_checkpoint(path, written["sha256"], target)
            self.assertEqual(target.updates, 1)
            self.assertEqual(loaded["evaluation"], expected)
            self.assertEqual(state_dict_sha256(target.stage.student), state_dict_sha256(source.stage.student))

    def test_precision_drift_is_rejected_before_optimizer_update(self):
        for namespace, field in ((torch.backends.cuda.matmul, "allow_tf32"),
                                 (torch.backends.cudnn, "allow_tf32"),
                                 (torch.backends.cudnn, "benchmark")):
            with self.subTest(field=field, namespace=str(namespace)):
                trainer = stage_and_trainer()
                before = state_dict_sha256(trainer.stage.student)
                try:
                    setattr(namespace, field, True)
                    with patch.object(trainer.optimizer, "step", wraps=trainer.optimizer.step) as step:
                        with self.assertRaisesRegex(ValueError, "Precision policy"):
                            trainer.step(rr_batch())
                        step.assert_not_called()
                    self.assertEqual(trainer.updates, 0)
                    self.assertEqual(before, state_dict_sha256(trainer.stage.student))
                finally:
                    configure_precision()

    def test_nonfinite_moments_after_optimizer_step_poison_the_trainer(self):
        trainer = stage_and_trainer()
        original_step = trainer.optimizer.step
        def corrupt_second_moment(*args, **kwargs):
            result = original_step(*args, **kwargs)
            first = next(iter(trainer.optimizer.state.values()))
            first["exp_avg_sq"].fill_(float("inf"))
            return result
        with patch.object(trainer.optimizer, "step", side_effect=corrupt_second_moment):
            with self.assertRaises((ValueError, FloatingPointError)):
                trainer.step(rr_batch())
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in trainer.stage.student.parameters()))
        self.assertTrue(trainer.poisoned)
        self.assertEqual(trainer.updates, 0)
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            trainer.state_dict()


if __name__ == "__main__":
    unittest.main()
