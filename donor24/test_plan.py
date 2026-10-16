from copy import deepcopy
import json
from pathlib import Path
import unittest
from . import plan


class PlanTests(unittest.TestCase):
    def test_registered_both_wv3_and_versioned_files(self):
        for s in plan.LANES:
            c=plan.default_config(s)
            self.assertEqual(c,json.loads((plan.ROOT/'config/donor24'/f'{s}.json').read_text()))
            self.assertEqual(c['dataset'],'WV3')
            self.assertEqual(c['baseline_resolved']['teacher_architecture']['bands'],8)
            self.assertEqual(len(c['search_order']),6)
        self.assertEqual((plan.ROOT/'config/donor24/cases.csv').read_text(),plan.catalog_csv().replace('\r\n','\n'))

    def test_common_anchors_paired_every_rng_and_unique_names(self):
        ids=[]
        for s in plan.LANES:
            for case in plan.QUEUES[s]:
                for replica in plan.SEEDS:
                    cfg=plan.resolve_case(plan.default_config(s),case,replica)
                    plan.validate_case(cfg);ids.append(cfg['run_id'])
                    for word in ('eps','lrA','R3','M2'):
                        self.assertIn(word,cfg['run_display']);self.assertIn(word,cfg['run_id'])
        self.assertEqual(len(ids),len(set(ids)))
        for case in ('A0','A1'):
            a=plan.resolve_case(plan.default_config('s1'),case)
            b=plan.resolve_case(plan.default_config('s3'),case)
            for key in ('stream_seeds','lambda_abs','lambda_epsilon','lr_A','alignment_protocol_digest'):
                self.assertEqual(a[key],b[key])

    def test_no_schedule_shortening_no_stage2_and_dense_evaluation(self):
        c=plan.resolve_case(plan.default_config(), 'A0')
        self.assertEqual(c['planned_horizon'],50000)
        self.assertFalse(c['stage2'])
        self.assertEqual(len(plan.evaluation_steps()),35)
        self.assertTrue(set(range(4000,20001,1000)) <= set(plan.evaluation_steps()))
        for key,value in [('dataset','GF2'),('lambda_epsilon',0),('planned_horizon',20000),('lr_A',1e-3),('phase','stage2')]:
            x=deepcopy(c);x[key]=value
            with self.assertRaises(ValueError):plan.validate_case(x)

    def test_one_confirmation_reserved_not_24h_plus_confirm(self):
        estimate=9200
        self.assertTrue(plan.admission_decision(0,7*estimate,budget=plan.BUDGET)['admitted'])
        self.assertFalse(plan.admission_decision(70000,2*estimate,budget=plan.BUDGET)['admitted'])

    def test_gate_and_failed_trajectories(self):
        def r(case,h,e):
            return dict(run_id=case,attempt=1,replica='R1',case=case,HQNR=h,ERGAS=e,SCC=.98,
                completed_update=1000,checkpoint_path='/tmp/test.pt',state_hash='a'*64,
                checkpoint_verified=True,metrics_complete=True,metrics_finite=True,same_state_verified=True)
        obs=[r('A0',.958,2.),r('A1',.960,2.2),r('A2',.959,2.1),r('A3',.99,1.)]
        runs=[dict(run_id=r['run_id'],attempt=1,status='COMPLETED' if r['run_id']!='A3' else 'FAILED') for r in obs]
        self.assertEqual(plan.choose_completed(obs,runs)['run_id'],'A2')
        self.assertEqual(plan.choose_completed(obs[:1],runs)['donor_selection_status'],'BEST_HQNR_EXPLORATORY')
        self.assertIsNone(plan.choose_completed(obs,[]))


if __name__=='__main__':unittest.main()
