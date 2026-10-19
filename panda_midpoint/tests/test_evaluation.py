"""Full40 same-state evaluator keeps all official original references intact."""
from copy import deepcopy
import unittest

import numpy as np
import torch

from panda_stage2_condq.tests.test_training_evaluation import EvalSources, metrics
from panda_stage2_condq.tests.test_method import batch
from panda_midpoint.tests.test_engine import fixture
from panda_midpoint import evaluation as E


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_midpoint_output_original_unshifted_rr_fr_and_frozen_state(self):
        trainer = fixture()
        trainer.step_accumulated([batch()])
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
        self.assertFalse(result['inverse_output_warp'])
        self.assertEqual(result['output_coordinates'], 'MIDPOINT')
        self.assertEqual(result['state_before'], result['state_after'])
        row = result['rr']['per_scene'][0]
        np.testing.assert_allclose(row['pan_applied_dy_dx'], np.array(row['correction_dy_dx']) * .5)
        np.testing.assert_allclose(row['ms_applied_dy_dx'], np.array(row['correction_dy_dx']) * -.5)

    def test_wrong_geometry_engine_and_original_stage_not_admitted(self):
        trainer = fixture()
        trainer.config['engine_kind'] = 'stage2_condq'
        with self.assertRaises(ValueError):
            E.configured_pipeline(trainer, EvalSources())
        from panda_stage2_condq.tests.test_training_evaluation import stage_and_trainer
        original = stage_and_trainer()
        original.config = deepcopy(fixture().config)
        with self.assertRaisesRegex(ValueError, 'Midpoint Stage 2'):
            E.configured_pipeline(original, EvalSources())

    def test_evaluation_mutation_poisoned_and_injected_metrics_cannot_publish(self):
        trainer = fixture()
        with self.assertRaisesRegex(ValueError, 'synthetic TEST'):
            E.official_evaluate(trainer, EvalSources(), metric_functions=metrics())
        with self.assertRaisesRegex(RuntimeError, 'changed during paused'):
            with E.paused_model(trainer):
                trainer.stage.student.backbone.output[-1].bias.add_(1.)
        self.assertTrue(trainer.poisoned)


if __name__ == '__main__':
    unittest.main()
