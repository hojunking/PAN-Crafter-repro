import csv
import io
from pathlib import Path
import tempfile
import unittest
from panda_wv3_ablation import plan,runtime


class QueueTests(unittest.TestCase):
    def test_round_major_all_cases_and_server_isolation(self):
        for server,number in [('s1',9),('s3',5)]:
            q=plan.queue(server)
            self.assertEqual(len(q),number*3)
            for i,repeat in enumerate((1,2,3)):
                section=q[i*number:(i+1)*number]
                self.assertEqual({c['repeat'] for c in section},{repeat})
                self.assertEqual([c['case_id'] for c in section],[c['case_id'] for c in q[:number]])
            for c in q:
                plan.validate(c)
                self.assertEqual(c['server'],server)
                if server=='s3':self.assertNotIn('donor_checkpoint',c)
            self.assertEqual(plan.catalog(server),(plan.CONFIG/f'cases_{server}.csv').read_text())

    def test_exact_anchor_seeds_and_schedule(self):
        for stage in ('stage1','stage2'):
            ref=plan.REFERENCES[stage]
            self.assertEqual(plan.seeds(ref['seed']),ref['stream_seeds'])
        for c in plan.queue('s1'):
            self.assertEqual((c['scale'],c['beta'],c['horizon'],c['primary_step']),(1,0,30000,29000))
            self.assertIn(c['gamma'],(0,.25))
        with self.assertRaises(ValueError):plan.resolve('s3','S2_B0_CORR_GT',1)

    def test_resume_keeps_position_and_failure_does_not_skip_independent_cases(self):
        st=runtime.initial_state('s3');first=runtime.next_job(st)
        first.update(status='PAUSED',step=1400)
        self.assertIs(runtime.next_job(st),first)
        first['status']='ERROR'
        self.assertIs(runtime.next_job(st),first) # repairable errors must not skip this case
        first['status']='FAILED'
        self.assertEqual(runtime.next_job(st)['case'],'S1_A1_SCHARR')
        st['runs'][1].update(status='COMPLETED',step=50000)
        self.assertEqual(runtime.next_job(st)['case'],'S1_A2_REC')
        for r in st['runs']:r.update(status='COMPLETED',step=50000)
        self.assertIsNone(runtime.next_job(st))

    def test_no_phantom_completed_run_or_amended_order(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'campaign_state.json';st=runtime.initial_state('s1')
            st['runs'][0].update(status='COMPLETED',step=1000);p.write_text(json.dumps(st))
            with self.assertRaisesRegex(ValueError,'Premature'):runtime.read_state(tmp,'s1')
            st['runs'][0]['status']='PAUSED';st['runs'].reverse();p.write_text(json.dumps(st))
            with self.assertRaisesRegex(ValueError,'order'):runtime.read_state(tmp,'s1')

    def test_crash_trace_suffix_is_archived_before_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'train.jsonl'
            p.write_bytes(b'{"step":1}\n{"step":2}\n{"step":3')
            runtime.rewind_trace(root,2)
            self.assertEqual(p.read_bytes(),b'{"step":1}\n{"step":2}\n')
            self.assertEqual(len(list(root.glob('train_before_resume_*.jsonl'))),1)
            with self.assertRaisesRegex(ValueError,'incomplete'):runtime.rewind_trace(root,3)
