from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from panda_band30.tests.test_reporting import evaluation as original_evaluation
from panda_mature30.tests.test_reporting import reused_receipt
from panda_qb_s2 import best_publication as B, plan, reporting as R
from panda_qb_s2.spec import geometry_manifest

def persist(root, case=None, *, server='s2', step=1000, h=None, e=None):
    sensor = R.DATASETS[server]
    case = case or 'QB6_S2_A200_E030_D101'
    hmin, emax = R.thresholds(server)
    h, e = hmin if h is None else h, emax if e is None else e
    cfg = plan.resolve_case(server, case, 'R1')
    ev = original_evaluation(cfg, step, h, e)
    key = cfg['policy_id'].lower()
    ev['mature30_deployment'] = {'schema':'PANDA_NEXT30_DEPLOYMENT_V1',
        'policy_id':cfg['policy_id'], 'completed_update':step, 'geometry_key':key,
        'geometry':geometry_manifest(key), 'student_sampling':deepcopy(cfg['student_sampling']),
        'student_input':deepcopy(cfg['student_input']), 'student_warp':deepcopy(cfg['student_warp'])}
    for split in ('rr', 'fr'):
        for scene in ev[split]['per_scene']:
            scene['correction_dy_dx'] = [.3,-.4]
            scene['pan_applied_dy_dx'] = [.3,-.4]
    directory = Path(root)/'runs'/cfg['run_id']/'attempt_01'
    directory.mkdir(parents=True, exist_ok=True)
    R.atomic_json(directory/'config.json', cfg)
    R.atomic_json(directory/f'evaluation_{step:07d}.json', ev)
    return R.emit_evaluation(root, cfg, ev, timestamp='2026-10-26T00:00:00+00:00')


class PublicationTests(unittest.TestCase):
    def test_all_six_student_rows_thresholds_and_deduplication(self):
        with tempfile.TemporaryDirectory() as root:
            cases=[case for case,_ in plan.G.RUNS['s2'] if case not in plan.DONORS]
            for case in cases:
                persist(root,case,step=1000,h=.9315,e=3.5)
                persist(root,case,step=2000,h=.932,e=3.49)
                persist(root,case,step=3000,h=.9314999,e=3.4)
            view=B.project(root,'s2')
            self.assertFalse(view['conflicts'],view['conflicts'])
            self.assertEqual(len(view['rows']),12)
            for row in view['rows']:
                B.verify_owned(row,view)
                self.assertEqual(row[R.INDEX['Server']],'s2')
                self.assertEqual(row[R.INDEX['Selected_step']],2000 if row[R.INDEX['Selector']]==B.SELECTOR else 1000)
                self.assertIn('S1seed=',row[R.INDEX['Approach']])
                self.assertIn('Sseed=261018101',row[R.INDEX['Approach']])

    def test_fallback_retained_until_hqnr_qualified_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root,step=1000,h=.92,e=3.4)
            persist(root,step=2000,h=.93,e=3.6)
            first=B.project(root,'s2')
            self.assertFalse(first['conflicts'],first['conflicts'])
            self.assertEqual(first['selected'][0]['step'],2000)
            self.assertEqual(first['rows'][0][R.INDEX['Selector']],B.FALLBACK_SELECTOR)
            persist(root,step=3000,h=.9315,e=3.7)
            after=B.project(root,'s2')
            self.assertEqual(after['selected'][0]['step'],3000)
            self.assertEqual(after['rows'][0][R.INDEX['Result_ID']],first['rows'][0][R.INDEX['Result_ID']])
    def test_fresh_donor_selected_rre_receipt_is_counted_as_new_training(self):
        for server in R.DATASETS:
            cfg=plan.resolve_case(server,plan.DONORS[0])
            with tempfile.TemporaryDirectory() as root:
                receipt,ev=reused_receipt(root,server=server,source_config=cfg)
                original=deepcopy((receipt,ev))
                R.emit_evaluation(root,cfg,ev,selector_receipt=receipt)
                view=B.project(root,server)
                self.assertFalse(view['conflicts'],view['conflicts'])
                self.assertEqual((receipt,ev),original)
                row=view['rows'][0]
                self.assertEqual(row[R.INDEX['Selector']],'BEST_VAL_RRE')
                self.assertEqual(row[R.INDEX['Updates']],50000)
                note=json.loads(row[R.INDEX['Original description / notes']])
                self.assertEqual(note['publication']['locally_completed_updates'],50000)
                self.assertFalse(note['imported_source_not_new_training'])

    def test_two_donor_pairs_publish_distinct_rows_and_cannot_swap_receipts(self):
        for server in R.DATASETS:
            with tempfile.TemporaryDirectory() as root:
                first=plan.resolve_case(server,plan.DONORS[0])
                second=plan.resolve_case(server,plan.DONORS[1])
                receipt1,ev1=reused_receipt(root,server=server,source_config=first)
                receipt2,ev2=reused_receipt(root,server=server,source_config=second)
                R.emit_evaluation(root,first,ev1,selector_receipt=receipt1)
                R.emit_evaluation(root,second,ev2,selector_receipt=receipt2)
                view=B.project(root,server)
                self.assertFalse(view['conflicts'],view['conflicts'])
                self.assertEqual(len(view['rows']),2)
                self.assertEqual(len({r[R.INDEX['Result_ID']] for r in view['rows']}),2)
                with self.assertRaisesRegex(ValueError,'another loss pair'):
                    R.emit_evaluation(root,first,ev2,selector_receipt=receipt2)
