"""Static campaign tests: no model, GPU, assets, sheets or production writes."""
from copy import deepcopy
import unittest

from maina_hqnr import plan


class PlanTests(unittest.TestCase):
    def test_seed_blocks_and_anchor(self):
        self.assertEqual([plan.seed_for('s4',i) for i in range(4)], [73101,2100000,2100002,2100004])
        self.assertEqual([plan.seed_for('s5',i) for i in range(4)], [2100001,2100003,2100005,2100007])
        seeds={plan.seed_for(s,c) for s in ('s4','s5') for c in range(1000)}
        self.assertEqual(len(seeds),2000)
        self.assertTrue(plan.make_case('s4',0,'BASE')['anchor'])
        self.assertFalse(plan.make_case('s4',1,'BASE')['anchor'])
        self.assertFalse(plan.make_case('s5',0,'BASE')['anchor'])

    def test_order_rotation_and_reverse(self):
        self.assertEqual(plan.order_for('s4',0),['BASE','AL05','AL15','BE005','BE020','ED0006','ED006'])
        self.assertEqual(plan.order_for('s5',0),['BASE','ED006','ED0006','AL15','AL05','BE020','BE005'])
        self.assertEqual(plan.order_for('s4',1),['BASE','BE020','BE005','ED006','ED0006','AL15','AL05'])
        for server in ('s4','s5'):
            for cycle in range(12):
                self.assertEqual(sorted(plan.order_for(server,cycle)),sorted(plan.CASES))

    def test_case_roundtrip_and_single_factor(self):
        all_cases=[c for s in ('s4','s5') for cycle in range(3) for c in plan.cycle_cases(s,cycle)]
        self.assertEqual(len(all_cases),42)
        self.assertEqual(len({c['run_id'] for c in all_cases}),42)
        for c in all_cases:
            self.assertEqual(plan.case_for(c['run_id']),c)
            self.assertEqual(plan.validate_case(c),c)
            base=plan.make_case(c['server'],c['cycle'],'BASE')
            self.assertEqual(c['local_baseline_run_id'],base['run_id'])
            differences=sum(c[k]!=base[k] for k in ('alpha','beta','lambda_E'))
            self.assertEqual(differences,int(c['code']!='BASE'))
            self.assertEqual(c['teacher_alias'],'F1')
            self.assertEqual(c['teacher_seed'],71001)

    def test_protected_servers_and_seed_exhaustion(self):
        for server in ('s1','s2','s3','S4',None):
            with self.assertRaises(ValueError): plan.seed_for(server,0)
        for cycle in (-1,True,1.2):
            with self.assertRaises(ValueError): plan.seed_for('s4',cycle)
        with self.assertRaises(OverflowError): plan.seed_for('s4',2**32)
        with self.assertRaises(ValueError): plan.make_case('s4',0,'QREF2')

    def test_cursor_and_no_preview_limit(self):
        self.assertEqual(plan.next_cursor('s4',2,6),{'cycle':3,'position':0})
        self.assertEqual(plan.next_cursor('s5',40,3),{'cycle':40,'position':4})
        self.assertEqual(plan.make_case('s4',99,'BASE')['cycle'],99)

    def test_fifty_fixed_candidates_hqnr_only(self):
        self.assertEqual(len(plan.GRID_STEPS),50)
        self.assertEqual(len(set(plan.GRID_STEPS)),50)
        self.assertEqual(plan.GRID_STEPS[-2:],(49490,50000))
        self.assertFalse(plan.SELECTION['ergas_used_for_selection'])
        self.assertTrue(plan.SELECTION['test_aware'])
        self.assertEqual(plan.SELECTION['primary_selection'],'HQNR_MAX50')

    def test_original_main_builder_numeric_defaults(self):
        from fh20r1.plan import build_config,case_for
        old=build_config(case_for(plan.ORIGINAL_RUN))
        for server in ('s4','s5'):
            for c in plan.cycle_cases(server,0):
                cfg=plan.build_config(c)
                self.assertEqual(plan.validate_config(cfg),c)
                for key in ('batch_size','num_iter','num_warmup','learning_rate','num_worker',
                    'mixed_precision','weight_decay','eps','betas','model_args'):
                    self.assertEqual(cfg[key],old[key])
                f=cfg['fh20r1']
                self.assertEqual(f['input_layout'],'PLH')
                self.assertEqual(f['teacher_alias'],'F1')
                self.assertEqual(f['teacher_input_layout'],'P0')
                self.assertEqual(f['teacher_run_id'],plan.TEACHER_RUN)
                self.assertEqual(f['aligner_lr'],3e-6)
                self.assertEqual(f['offset_weight'],0.)
                self.assertNotIn('hqnr_threshold',f)
                self.assertNotIn('ergas_goal',f)
                self.assertNotIn('budget_path',f)
                for key in ('alpha','beta','lambda_E'): self.assertEqual(f[key],c[key])

    def test_config_mutation_fails_closed(self):
        cfg=plan.build_config(plan.make_case('s5',0,'BASE'))
        cfg['fh20r1']['reference_alias']='F5'
        with self.assertRaises(ValueError): plan.validate_config(cfg)
        cfg=plan.build_config(plan.make_case('s5',0,'BASE'))
        with self.assertRaises(ValueError): plan.validate_config(cfg,require_bound=True)
        case=deepcopy(plan.make_case('s4',0,'BASE'));case['alpha']=.7
        with self.assertRaises(ValueError): plan.validate_case(case)

    def test_actual_original_source_hashes(self):
        self.assertEqual(plan.verify_sources()['git_release'],plan.ORIGINAL_RELEASE)


if __name__=='__main__': unittest.main()
