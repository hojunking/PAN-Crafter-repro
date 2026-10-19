"""CPU full40: effective operators, same state, original refs and no FR GT."""
from copy import deepcopy
from dataclasses import asdict, replace
import unittest

import numpy as np
import torch

from directshift48.method import ScharrAbsolute
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.donor import LoadedDonor
from panda_stage2_condq.losses import reference_maps
from panda_stage2_condq.tests.test_method import fixture as prior_fixture, batch
from panda_stage2_condq.tests.test_training_evaluation import EvalSources, metrics
from panda_stage2_condq.training import TrainingSpec
from panda_warp30 import evaluation as E, plan
from panda_warp30.calibration import bind_donor
from panda_warp30.contract import CAMPAIGN_ID
from panda_warp30.engine import build_fresh_stage2
from panda_warp30.method import frozen_predictions
from panda_warp30.spec import WarpSpec
from panda_warp30.stage1 import build_stage1


def student_fixture(case='U23_MID_RETURN'):
    cfg = plan.resolve_case('s3', case)
    spec = WarpSpec(**cfg['stage2_spec'])
    original = prior_fixture()
    binding = deepcopy(original.donor_binding)
    binding['warp30_teacher'] = {'upsampler': spec.upsampler, 'campaign_id': CAMPAIGN_ID}
    donor = bind_donor(LoadedDonor(original.aligner, original.teacher, binding, {}), spec)
    rr = batch()
    out = frozen_predictions(donor.aligner, donor.teacher, rr, spec=spec)
    e, g, q = reference_maps(ScharrAbsolute('WV3'), out['teacher_y'], out['gt_loss'], out['pan_loss'])
    builder = CalibrationBuilder(['WV3:train:0'], donor.binding, k=5,
        memory_budget_bytes=4*1024**2, include_condq=True, mask_policy=plan.LOSS_MASK_POLICY)
    builder.add(['WV3:train:0'], e, g, q, sensor='WV3', loss_mask=out['loss_mask'])
    cfg.update(donor_binding=donor.binding, planned_horizon=3, evaluation_steps=[1,2,3])
    cfg['training_spec'] = asdict(TrainingSpec(total_updates=3, warmup_updates=0,
        microbatch_size=1, effective_batch_size=1, full_audit_interval=1))
    return build_fresh_stage2(cfg, donor, builder.finalize(), evaluator_pin={'test': 'explicit'})


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def evaluate_and_check(self, trainer):
        sources, funcs = EvalSources(), metrics()
        seen = {'rr': 0, 'fr': 0}
        for split in ('rr', 'fr'):
            original = funcs[split]
            def inspect(raw, arrays, split=split, original=original):
                idx = seen[split]
                expected = sources.raw(split, [idx])
                self.assertEqual(set(arrays), set(expected))
                for key in arrays:
                    np.testing.assert_array_equal(arrays[key], expected[key][0])
                if split == 'fr':
                    self.assertNotIn('gt', arrays)
                    self.assertIsNone(sources.batch(split, [idx]).gt)
                out = trainer.stage.native_prediction(sources.batch(split, [idx]))
                prediction = ((out['y'][0] + 1) * (sources.peak/2)).numpy()
                np.testing.assert_array_equal(raw, prediction)
                seen[split] += 1
                return original(raw, arrays)
            funcs[split] = inspect
        before = E.model_state_hash(trainer.stage)
        result = E.official_evaluate(trainer, sources, metric_functions=funcs, evaluator_revision='TEST')
        self.assertEqual(seen, {'rr': 20, 'fr': 20})
        self.assertEqual(E.model_state_hash(trainer.stage), before)
        self.assertTrue(result['qualification']['same_state_verified'])
        self.assertTrue(result['qualification']['metrics_complete'])
        self.assertFalse(result['full_method_eligible'])
        self.assertFalse(result['reference_warp'])
        self.assertEqual(result['state_before'], result['state_after'])
        self.assertIn('RMSE', result['rr']['metrics'])
        self.assertIn('CC', result['rr']['metrics'])
        self.assertIn('JQM', result['fr']['metrics'])
        return result

    def test_both_fresh_stage1_upsamplers_full40_immutable(self):
        for case in ('S1_U23', 'S1_BIC'):
            with self.subTest(case=case):
                trainer = build_stage1(plan.resolve_case('s3', case), EvalSources())
                trainer.updates = 1  # Evaluation-only fixture; never claimed official evidence.
                result = self.evaluate_and_check(trainer)
                self.assertEqual(result['upsampler'], trainer.stage.upsampler)
                self.assertEqual(result['output_coordinates'], 'MS')
                self.assertFalse(result['inverse_output_warp'])
                row = result['fr']['per_scene'][0]
                self.assertEqual(row['pan_applied_dy_dx'], row['correction_dy_dx'])
                self.assertEqual(row['ms_applied_dy_dx'], [0., 0.])

    def test_mid_return_stage2_full40_and_actual_effective_shifts(self):
        trainer = student_fixture()
        trainer.step_accumulated([batch()])
        result = self.evaluate_and_check(trainer)
        self.assertTrue(result['inverse_output_warp'])
        self.assertEqual(result['output_coordinates'], 'MS')
        row = result['fr']['per_scene'][0]
        u = np.array(row['correction_dy_dx'])
        np.testing.assert_allclose(row['pan_applied_dy_dx'], u*.5)
        np.testing.assert_allclose(row['ms_applied_dy_dx'], u*-.5)
        np.testing.assert_allclose(row['output_applied_dy_dx'], u*.5)

    def test_all_frames_bind_real_pipeline_and_reject_wrong_strength(self):
        for case in ('U23_PAN','U23_MID','U23_MID_RETURN','U23_MS_RETURN','U23_PAN_S050'):
            with self.subTest(case=case):
                trainer = student_fixture(case)
                pipeline = E.configured_pipeline(trainer, EvalSources())
                self.assertEqual(pipeline['warp30_geometry'], trainer.config['geometry'])
                trainer.config['stage2_spec']['shift_scale'] = .25
                with self.assertRaises(ValueError):
                    E.configured_pipeline(trainer, EvalSources())

    def test_metric_injection_mutation_and_fr_gt_fail_closed(self):
        trainer = student_fixture()
        with self.assertRaisesRegex(ValueError, 'synthetic TEST'):
            E.official_evaluate(trainer, EvalSources(), metric_functions=metrics())
        class BadSources(EvalSources):
            def batch(self, split, rows, **kwargs):
                value = super().batch(split, rows, **kwargs)
                return replace(value, gt=value.native_lms) if split == 'fr' else value
        with self.assertRaisesRegex(ValueError, 'FR inference must not receive GT'):
            E.official_evaluate(trainer, BadSources(), metric_functions=metrics(), evaluator_revision='TEST')
        with self.assertRaisesRegex(RuntimeError, 'changed during paused'):
            with E.paused_model(trainer):
                trainer.stage.student.backbone.output[-1].bias.add_(1.)
        self.assertTrue(trainer.poisoned)

    def test_incomplete_or_mixed_state_never_qualifies(self):
        from .test_reporting import evaluation
        cfg = plan.resolve_case('s3', 'U23_PAN')
        ev = evaluation(cfg)
        for kind in ('partial','duplicate','mixed'):
            bad = deepcopy(ev)
            if kind == 'partial': bad['fr']['per_scene'].pop()
            if kind == 'duplicate': bad['rr']['per_scene'][-1] = deepcopy(bad['rr']['per_scene'][0])
            if kind == 'mixed': bad['fr_state']['model_state_sha256'] = 'f'*64
            gate = E.qualify_evaluation(bad, server='s3',
                expected_rr_ids=[f'WV3:rr:{i}' for i in range(20)],
                expected_fr_ids=[f'WV3:fr:{i}' for i in range(20)])
            self.assertFalse(gate['joint_pass'])
            self.assertFalse(gate['metrics_complete'] and gate['same_state_verified'])


if __name__ == '__main__':
    unittest.main()
