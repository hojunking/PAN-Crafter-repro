"""Real models, synthetic CPU scenes/metrics; never official quality evidence."""
from copy import deepcopy
from dataclasses import asdict, replace
import tempfile
import unittest

import numpy as np
import torch

from panda_stage2_condq.tests.test_training_evaluation import EvalSources, metrics
from panda_stage2_condq.tests.test_method import batch
from panda_stage2_condq.training import TrainingSpec
from panda_mature30 import evaluation as E, plan, reporting as R, publication as P
from panda_mature30.engine import build_fresh_stage2
from panda_mature30.stage1 import build_stage1
from .test_schedule_core import make_fixture
from .test_reporting import persist


def student_fixture(policy="DOWN"):
    cfg = plan.resolve_case("s1", policy)
    unused, donor, bank = make_fixture(policy, shift=(2., -1.))
    del unused
    cfg["donor_binding"] = deepcopy(donor.binding)
    cfg["training_spec"] = asdict(TrainingSpec(total_updates=50000, warmup_updates=0,
        microbatch_size=1, effective_batch_size=1, full_audit_interval=1))
    trainer = build_fresh_stage2(cfg, donor, bank, evaluator_pin={"test":"explicit"})
    trainer.step_accumulated([batch()])
    return trainer


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def evaluate(self, trainer):
        sources, functions = EvalSources(), metrics()
        seen = {"rr":0, "fr":0}
        for split in ("rr","fr"):
            original = functions[split]
            def inspect(raw, arrays, split=split, original=original):
                index = seen[split]
                expected = sources.raw(split,[index])
                self.assertEqual(set(arrays),set(expected))
                for key in arrays:
                    np.testing.assert_array_equal(arrays[key],expected[key][0])
                if split=="fr":
                    self.assertNotIn("gt",arrays)
                seen[split] += 1
                return original(raw,arrays)
            functions[split]=inspect
        before=E.model_state_hash(trainer.stage)
        record=E.official_evaluate(trainer,sources,metric_functions=functions,
                                  evaluator_revision="TEST")
        self.assertEqual(seen,{"rr":20,"fr":20})
        self.assertEqual(E.model_state_hash(trainer.stage),before)
        self.assertEqual(record["state_before"],record["state_after"])
        self.assertFalse(record["full_method_eligible"])
        self.assertTrue(record["qualification"]["same_state_verified"])
        self.assertTrue(record["qualification"]["metrics_complete"])
        self.assertFalse(record["reference_warp"])
        self.assertFalse(record["inverse_output_warp"])
        self.assertEqual(record["output_coordinates"],"MS")
        for key in ("RMSE","CC"): self.assertIn(key,record["rr"]["metrics"])
        self.assertIn("JQM",record["fr"]["metrics"])
        return record

    def test_native_stage1_uses_full_u_and_original_refs(self):
        cfg=plan.resolve_case("s3","G8_MATURE_U23")
        trainer=build_stage1(cfg,EvalSources())
        trainer.updates=1  # Evaluation-only synthetic fixture, not training evidence.
        record=self.evaluate(trainer)
        for split in ("rr","fr"):
            for row in record[split]["per_scene"]:
                self.assertEqual(row["correction_dy_dx"],row["pan_applied_dy_dx"])
                self.assertEqual(row["ms_applied_dy_dx"],[0.,0.])

    def test_stage2_scale_cap_and_step_bound_deployment(self):
        for policy in ("FULL","DOWN","CAP085"):
            with self.subTest(policy=policy):
                trainer=student_fixture(policy)
                record=self.evaluate(trainer)
                self.assertEqual(record["mature30_deployment"],trainer.stage.deployment_manifest())
                P._verify_corrections(record,trainer.config)
                for split in ("rr","fr"):
                    for row in record[split]["per_scene"]:
                        self.assertEqual(row["ms_applied_dy_dx"],[0.,0.])
                        self.assertEqual(row["output_applied_dy_dx"],[0.,0.])
                        if policy=="CAP085":
                            self.assertAlmostEqual(np.linalg.norm(row["pan_applied_dy_dx"]),.85,places=6)
                # Even real forward outputs with injected metrics must never publish.
                with tempfile.TemporaryDirectory() as root:
                    persist(root,trainer.config,step=trainer.updates,ev=record)
                    view=P.project_campaign(root,"s1")
                    self.assertTrue(view["conflicts"])
                    self.assertFalse(view["rows"])

    def test_wrong_config_policy_and_fr_gt_fail_closed(self):
        trainer=student_fixture()
        trainer.config["policy_id"]="HALF"
        with self.assertRaises(ValueError):
            E.configured_pipeline(trainer,EvalSources())
        trainer=student_fixture()
        class BadSources(EvalSources):
            def batch(self,split,rows,**kwargs):
                value=super().batch(split,rows,**kwargs)
                return replace(value,gt=value.native_lms) if split=="fr" else value
        with self.assertRaisesRegex(ValueError,"FR inference"):
            E.official_evaluate(trainer,BadSources(),metric_functions=metrics(),
                                evaluator_revision="TEST")

    def test_phase_counter_tamper_during_evaluation_poisoned(self):
        trainer=student_fixture("DOWN")
        with self.assertRaisesRegex(RuntimeError,"changed during paused"):
            with E.paused_model(trainer):
                trainer.stage.set_completed_update(10001)
        self.assertTrue(trainer.poisoned)


if __name__=="__main__":
    unittest.main()
