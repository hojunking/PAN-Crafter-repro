"""Preregistered scientific comparisons, not production result assertions."""
from copy import deepcopy
import csv
import io
import unittest

from panda30r import plan
from panda_stage2_condq.stream import StreamSpec
from panda_stage2_condq.training import TrainingSpec


class PlanTests(unittest.TestCase):
    def test_servers_are_wv3_with_same_architecture_and_paired_seeds(self):
        names=set()
        for server in plan.SERVERS:
            cfg=plan.default_config(server)
            plan.validate_config(cfg)
            self.assertEqual(cfg['budget']['total_seconds'],30*3600)
            self.assertEqual(cfg['baseline_resolved']['dataset'],'WV3')
            for group in plan.groups(server):
                self.assertEqual([r for _,r in group['runs']],['R1','R2'])
                for case,replica in group['runs']:
                    c=plan.resolve_case(server,case,replica)
                    self.assertEqual(c['dataset'],'WV3')
                    self.assertEqual(c['loss_mask_policy'],plan.LOSS_MASK_POLICY)
                    self.assertEqual(c['student_architecture'],{'bands':8,'width':112,'depth':[1,2,3]})
                    self.assertNotIn(c['run_id'],names); names.add(c['run_id'])
                    self.assertEqual(c['start_update']+c['planned_horizon'],50000)
                    self.assertEqual(c['evaluation_steps'][-1],c['planned_horizon'])
                    TrainingSpec(**c['training_spec'])
                    if c['engine_kind']=='stage2_condq':
                        StreamSpec(1,2,**c['augmentation'])

    def test_single_factor_s4_controls(self):
        base=plan.resolve_case('s4','M_Q')
        beta=plan.resolve_case('s4','M_B2')
        lr=plan.resolve_case('s4','M_LR5')
        flip=plan.resolve_case('s4','M_FLIP')
        expected=deepcopy(base['stage2_spec']); expected['beta']=.2
        self.assertEqual(beta['stage2_spec'],expected)
        self.assertEqual(lr['stage2_spec'],base['stage2_spec'])
        self.assertEqual(lr['training_spec']['lr'],5e-5)
        self.assertEqual(flip['stage2_spec'],base['stage2_spec'])
        self.assertEqual(flip['training_spec'],base['training_spec'])
        self.assertEqual(flip['augmentation'],{'hflip_probability':.5,'vflip_probability':.5,'rot90':True})
        self.assertEqual(flip['stream_seeds'],base['stream_seeds'])

    def test_donor_causal_comparison_changes_no_student_setting(self):
        cases=[plan.resolve_case('s1',name) for name in ('E_Q','L_Q','M_Q')]
        for key in ('stream_seeds','stage2_spec','training_spec','augmentation','loss_mask_policy'):
            self.assertEqual(cases[0][key],cases[1][key]); self.assertEqual(cases[0][key],cases[2][key])
        self.assertEqual([c['donor_version'] for c in cases],['E','L','M'])

    def test_tails_share_parent_no_teacher_and_exact_local_lifetime(self):
        for start in (20,40):
            cases=[plan.resolve_case('s3',f'T{start}_{name}') for name in ('G','EPS','ABS')]
            for c in cases:
                self.assertEqual(c['parent_case'],'M_Q')
                self.assertEqual(c['start_update'],start*1000)
                self.assertEqual(c['planned_horizon'],50000-start*1000)
                self.assertEqual(c['training_spec']['lr'],1e-5)
                self.assertNotIn('stage2_spec',c)
            self.assertEqual(cases[1]['lambda_abs'],0.)
            self.assertEqual(cases[2]['lambda_abs'],.2)
            self.assertEqual(cases[1]['adapt_updates'],5000)
        parent=plan.resolve_case('s3','M_Q')
        self.assertTrue({20000,40000,50000}<=set(parent['retention_steps']))

    def test_maturation_preserves_global_horizon_and_requires_early_source(self):
        c=plan.resolve_case('s1','MATURE',source_update=7000)
        self.assertEqual(c['planned_horizon'],43000)
        self.assertEqual(c['training_spec']['total_updates'],50000)
        self.assertEqual(c['training_spec']['warmup_updates'],100)
        self.assertEqual((c['lr_A'],c['lambda_abs'],c['lambda_epsilon']),(0.,0.,0.))
        for value in (None,0,50000,50001):
            with self.assertRaises(ValueError): plan.resolve_case('s1','MATURE',source_update=value)

    def test_catalog_is_full_server_queue_not_prep_or_pruned_seed(self):
        for server in plan.SERVERS:
            rows=list(csv.DictReader(io.StringIO(plan.catalog_csv(server))))
            self.assertEqual(len(rows),2*len(plan.groups(server)))
            self.assertEqual([int(r['order']) for r in rows],list(range(1,len(rows)+1)))

    def test_optional_early_tail_study_is_one_six_run_budget_block(self):
        block=[g for g in plan.groups('s3') if g['budget_block']=='T20_TRIO']
        self.assertEqual([g['name'] for g in block],['T20_G','T20_EPS','T20_ABS'])
        self.assertEqual(sum(len(g['runs']) for g in block),6)
        self.assertTrue(all(g['optional'] for g in block))

    def test_native_declared_and_executed_settings_cannot_drift(self):
        from panda30r.engine import _validate_config
        for name in ('T40_G','T40_EPS','T40_ABS'):
            cfg=plan.resolve_case('s3',name)
            _validate_config(cfg)
            cfg['lambda_epsilon']=1.
            with self.assertRaisesRegex(ValueError,'Declared native setting'):
                _validate_config(cfg)


if __name__=='__main__': unittest.main()
