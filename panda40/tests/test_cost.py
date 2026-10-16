from copy import deepcopy
import tempfile
import unittest

import torch
from panda_stage2_condq.tests.test_training_evaluation import EvalSources
from panda40.tests.test_reporting import config
from panda40 import cost
from s2dshift48.aligner import S2DGlobalPANAligner


class CostTests(unittest.TestCase):
    def test_signature_separates_actual_graph_and_server_not_seed_or_loss(self):
        cfg = config(); reference = cost.signature(cfg)
        same = deepcopy(cfg); same['seed'] += 1; same['lambda_abs'] = .25
        self.assertEqual(reference, cost.signature(same))
        for key, value in (('aligner_kind', 's2d'), ('server', 's3')):
            other = deepcopy(cfg); other[key] = value
            self.assertNotEqual(reference, cost.signature(other))
        self.assertNotEqual(reference, cost.signature(config('s3', 'stage2')))

    def test_fresh_graph_profiles_actual_s2d_and_excludes_kd_teacher(self):
        for phase in ('donor', 'stage2'):
            for kind in ('direct', 's2d'):
                cfg = config('s1' if phase == 'donor' else 's3', phase)
                cfg['aligner_kind'] = kind
                graph = cost.build_deployment(cfg, EvalSources())
                self.assertEqual(isinstance(graph.aligner, S2DGlobalPANAligner), kind == 's2d')
                # The lightweight profiler wrapper names its sole deployable
                # reconstructor 'teacher'; Stage2 labels relabel it Student.
                self.assertFalse(hasattr(graph, 'student'))
                self.assertEqual(graph.teacher.spec.width, 112)
                self.assertEqual(sum(p.numel() for p in graph.parameters()),
                                 sum(p.numel() for p in graph.aligner.parameters()) +
                                 sum(p.numel() for p in graph.teacher.parameters()))

    def test_cpu_cannot_claim_production_cost(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'GPU'):
                cost.ensure_cost(config(), EvalSources(), retained_trainer=None,
                                 device='cpu', deadline_unix=float('inf'), cost_root=root)
            self.assertIsNone(cost.find_profile(config(), root))


if __name__ == '__main__':
    unittest.main()
