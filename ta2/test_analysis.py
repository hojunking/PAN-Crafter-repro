import copy
import tempfile
import unittest
from pathlib import Path

from ta2.analysis import collect_runs,refresh_analysis
from ta2.common import atomic_json,load_json
from ta2.evaluation import seal
from ta2.plan import LANES
from ta2.reporting import Outbox


class FakePlan:
    def __init__(self,cases=('TA2-B00','TA2-M05','TA2-M07')):
        self.rows=[dict(case_id=c,server='s1',dataset='WV3',replica=i,seed=s,
                       run_id=f'run_{c}_{i}',updates=50000,control_id='' if c=='TA2-B00' else 'TA2-B00')
                   for c in cases for i,s in enumerate(LANES['s1'][3],1)]
    def queue(self,server):return copy.deepcopy(self.rows)


def completed(lane,row,value=.95):
    run=lane/'runs'/row['run_id'];identity={'source':'source'}
    atomic_json(run/'identity.json',identity)
    atomic_json(run/'initialization.json',dict(U_sha256='u'+str(row['seed']),
        A_sha256=None if row['case_id']=='TA2-B00' else 'a'+str(row['seed']),native_stream_sha256='stream'+str(row['seed'])))
    atomic_json(run/'status.json',dict(status='COMPLETE'))
    atomic_json(run/'selection/selection.json',dict(status='HQNR_SELECTION_COMPLETE',candidates_complete=50,candidates_expected=50,
        test_aware=True,completed_step=1010,checkpoint_sha256='selected',exact_final_sha256='exact'))
    for selector,step,cp in [('HQNR_MAX50',1010,'selected'),('EXACT_FINAL',50000,'exact')]:
        report=seal(dict(record_type='NATIVE',complete=True,rr_fr_same_checkpoint=True,gt_in_inference=False,context=identity,
            completed_step=step,checkpoint_sha256=cp,
            rr=dict(n_scenes=20,ergas=2.,scc=.9,per_scene=[dict(scene_index=i,ergas=2.,scc=.9) for i in range(20)]),
            fr=dict(n_scenes=20,hqnr=value,masking=False,support='full512',reference='original_PAN_and_native_LMS',
                    per_scene=[dict(scene_index=i,hqnr=value) for i in range(20)])))
        atomic_json(run/'exports'/selector/'report.json',report)
        probe=seal(dict(status='COMPLETE',checkpoint_sha256=cp,response_summary=[],geometry_summary=[],correction_summary={},cost=[]))
        atomic_json(run/'diagnostics'/selector/'diagnostics'/f'step_{step:06d}'/'report.json',probe)


class AnalysisTests(unittest.TestCase):
    def test_empty_lane_no_fabricated_records(self):
        with tempfile.TemporaryDirectory() as temp:
            lane=Path(temp);box=Outbox(lane/'outbox')
            result=refresh_analysis(lane,FakePlan(),'s1',box)
            self.assertEqual(result['record_count'],0)
            self.assertFalse(result['network_writes'])

    def test_three_seeds_and_no_aligner_baseline_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            lane=Path(temp);plan=FakePlan();box=Outbox(lane/'outbox')
            for row in plan.rows:
                completed(lane,row,.95+(row['replica']-1)*.001+(0 if row['case_id']=='TA2-B00' else .001))
            result=refresh_analysis(lane,plan,'s1',box)
            self.assertGreater(result['record_count'],0)
            artifacts=[load_json(lane/p) for p in result['artifact_files'] if 'seed_TA2-B00_EXACT_FINAL' in p]
            h=next(a for a in artifacts[0]['aggregates'] if a['metric']=='hqnr')
            self.assertEqual(h['n_success'],3);self.assertEqual(h['status'],'COMPLETE_3_SEEDS')
            pairs=[load_json(lane/p) for p in result['artifact_files'] if 'pair_TA2-M07_TA2-B00_EXACT_FINAL' in p]
            h=next(a for a in pairs[0]['aggregates'] if a['metric']=='hqnr')
            self.assertAlmostEqual(h['mean'],.001);self.assertEqual(h['n_pairs'],3)
            before=sorted((lane/'outbox').rglob('*.json'))
            repeat=refresh_analysis(lane,plan,'s1',box)
            self.assertEqual(before,sorted((lane/'outbox').rglob('*.json')))
            self.assertEqual(result['artifact_files'],repeat['artifact_files'])

    def test_failed_seed_is_not_third_success(self):
        with tempfile.TemporaryDirectory() as temp:
            lane=Path(temp);plan=FakePlan(('TA2-B00',));box=Outbox(lane/'outbox')
            for row in plan.rows[:2]:completed(lane,row)
            failed=lane/'runs'/plan.rows[-1]['run_id']
            atomic_json(failed/'identity.json',dict(source='source'))
            atomic_json(failed/'status.json',dict(status='NUMERICAL_FAILURE',error='diverged'))
            result=refresh_analysis(lane,plan,'s1',box)
            artifacts=[load_json(lane/p) for p in result['artifact_files'] if 'seed_TA2-B00_EXACT_FINAL' in p]
            h=next(a for a in artifacts[0]['aggregates'] if a['metric']=='hqnr')
            self.assertEqual(h['n_success'],2);self.assertEqual(h['n_failed'],1)
            self.assertEqual(h['status'],'INCOMPLETE_3_SEEDS')

    def test_reject_masked_wrong_selector_and_partial_diagnostic(self):
        for mutation in ('mask','selector','partial'):
            with tempfile.TemporaryDirectory() as temp:
                lane=Path(temp);plan=FakePlan(('TA2-B00',));row=plan.rows[0];completed(lane,row)
                if mutation=='partial':
                    path=lane/'runs'/row['run_id']/'diagnostics/HQNR_MAX50/diagnostics/step_001010/report.json'
                    report=load_json(path);report.pop('payload_sha256');report['status']='PARTIAL'
                else:
                    path=lane/'runs'/row['run_id']/'exports/HQNR_MAX50/report.json'
                    report=load_json(path);report.pop('payload_sha256')
                    if mutation=='mask':report['fr']['masking']=True
                    else:report['checkpoint_sha256']='wrong'
                atomic_json(path,seal(report))
                with self.assertRaises(ValueError):collect_runs(lane,plan.rows)


if __name__=='__main__':unittest.main()
