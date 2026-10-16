import unittest
from . import plan
from .runtime import improvement


class PlanTests(unittest.TestCase):
    def test_budget_and_pairs(self):
        for server in plan.SERVERS:
            cfg=plan.default_config(server)
            plan.validate_config(cfg)
            self.assertEqual(cfg['budget']['total_seconds'],144000)
            for group in plan.groups(server):
                for c,r in group['runs']:
                    case=plan.resolve_case(server,c,r)
                    self.assertEqual(case['planned_horizon'],50000)
                    self.assertIn(50000,case['evaluation_steps'])
            self.assertEqual(len(set(plan.SEEDS.values())),2)

    def test_s1_geometry_pairs(self):
        for label in ('A1','A3','A5'):
            for replica in plan.SEEDS:
                d=plan.resolve_case('s1','D_'+label,replica)
                s=plan.resolve_case('s1','S_'+label,replica)
                for key in ('stream_seeds','lambda_abs','lambda_epsilon','lr_A','baseline_resolved'):
                    self.assertEqual(d[key],s[key])

    def test_s3_incremental_and_full(self):
        cases={c:plan.resolve_case('s3',c) for c in plan.STAGE2}
        self.assertEqual(cases['G']['stage2_spec']['variant'],'gt_only')
        q=dict(cases['Q']['stage2_spec']); f=dict(cases['Q_FULL']['stage2_spec'])
        self.assertNotEqual(q.pop('mask_policy'),f.pop('mask_policy'))
        self.assertEqual(q,f)
        self.assertEqual(cases['E']['stage2_spec']['beta'],0)
        self.assertEqual(cases['ES']['stage2_spec']['gamma'],0)
        self.assertTrue(all(c['student_architecture']['width']==112 for c in cases.values()))

    def test_gate_then_ergas_never_scc(self):
        a={'HQNR':.960,'ERGAS':2.3,'SCC':1.,'completed_update':1000}
        b={'HQNR':.9591,'ERGAS':2.1,'SCC':0.,'completed_update':2000}
        c={'HQNR':.9589,'ERGAS':1.,'SCC':1.,'completed_update':3000}
        self.assertEqual(plan.choose([a,b,c]),b)

    def test_backup_and_primary_separate(self):
        old={'HQNR':.960,'ERGAS':2.3,'completed_update':1000}
        ev={'fr':{'metrics':{'HQNR':.9591}},'rr':{'metrics':{'ERGAS':2.1,'SCC':.8}},
            'state_before':{'completed_update':2000}}
        row,reasons=improvement(ev,{'best_selected':old,'best_hqnr':old})
        self.assertEqual(reasons,['HQNR_GATE_ERGAS_SELECTOR'])


if __name__=='__main__':
    unittest.main()
