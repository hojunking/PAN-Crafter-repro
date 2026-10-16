from copy import deepcopy
import tempfile
import unittest

from directshift48.method import DirectPANAligner
from panda_stage2_condq.tests.test_training_evaluation import EvalSources
from pandas4.tests.test_reporting import config
from pandas4 import cost


class CostTests(unittest.TestCase):
    def test_signature_separates_phase_not_seed_or_objective(self):
        cfg = config(); reference = cost.signature(cfg)
        same = deepcopy(cfg); same['seed'] += 1; same['lambda_abs'] = .25
        self.assertEqual(reference, cost.signature(same))
        self.assertNotEqual(reference, cost.signature(config(phase='stage2')))
        for key, value in (('aligner_kind', 's2d'), ('server', 's1')):
            other = deepcopy(cfg); other[key] = value
            with self.assertRaises(ValueError):
                cost.signature(other)

    def test_fresh_graph_is_direct_and_excludes_training_teacher(self):
        for phase in ('donor', 'stage2'):
            cfg = config(phase=phase)
            graph = cost.build_deployment(cfg, EvalSources())
            self.assertIs(type(graph.aligner), DirectPANAligner)
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
