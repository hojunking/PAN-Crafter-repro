from copy import deepcopy
import tempfile
import unittest
import torch

from directshift48.method import DirectPANAligner
from panda_stage2_condq.tests.test_training_evaluation import EvalSources
from panda30r.tests.test_reporting import config
from panda30r import cost


class CostTests(unittest.TestCase):
    def test_signature_separates_phase_not_seed_or_objective(self):
        cfg = config(); reference = cost.signature(cfg)
        same = deepcopy(cfg); same['seed'] += 1; same['lambda_abs'] = .25
        self.assertEqual(reference, cost.signature(same))
        self.assertNotEqual(reference, cost.signature(config(phase='stage2')))
        for key, value in (('aligner_kind', 's2d'), ('server', 's2')):
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

    def test_native_profile_graph_matches_scheduled_inference_not_training_teacher(self):
        from panda30r.tests.test_engine import fixture
        from panda_stage2_condq.tests.test_training_evaluation import rr_batch
        for kind in ('mature_teacher','tail_fc2_eps','tail_fc2_abs','tail_gt'):
            with self.subTest(kind=kind):
                trainer=fixture(kind)
                cfg=config(phase='donor' if kind=='mature_teacher' else 'stage2')
                cfg['engine_kind']=kind
                graph=cost.build_deployment(cfg,EvalSources())
                graph.aligner.load_state_dict(trainer.stage.aligner.state_dict(),strict=True)
                graph.teacher.load_state_dict(trainer.stage.reconstruction.state_dict(),strict=True)
                batch=rr_batch()
                expected=trainer.stage.native_prediction(batch)
                observed=graph.native_prediction(batch)
                torch.testing.assert_close(observed['y'],expected['y'],rtol=0,atol=0)
                torch.testing.assert_close(observed['u'],expected['u'],rtol=0,atol=0)

    def test_actual_registry_zero_loss_coefficients_do_not_break_profile_constructor(self):
        from panda30r import plan
        cases=[('s1','FRESH_A1',None),('s1','MATURE',7000),('s1','E_Q',None),
               ('s1','M_G',None),('s3','T40_G',None),('s3','T40_EPS',None),
               ('s3','T40_ABS',None),('s4','M_FLIP',None)]
        for server,case,start in cases:
            with self.subTest(server=server,case=case):
                cfg=plan.resolve_case(server,case,source_update=start)
                graph=cost.build_deployment(cfg,EvalSources())
                self.assertEqual(graph.teacher.spec.width,112)
                self.assertIs(type(graph.aligner),DirectPANAligner)
