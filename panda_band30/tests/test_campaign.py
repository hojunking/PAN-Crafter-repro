import csv
import io
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_band30 import plan, runtime
from panda_band30.diagnostics import paired_summary
from panda_band30.contract import file_identity


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.now=1000.
        for target in ('code_pin','evaluator_identity'):
            p=patch.object(runtime,target,return_value={}); p.start(); self.addCleanup(p.stop)

    def campaign(self):
        return runtime.Campaign('s1',output=self.tmp.name,device='cpu',clock=lambda:self.now)

    def test_plan_exact_interleaved_14_rows_and_8_priority(self):
        cfg=plan.default_config('s1'); plan.validate_config(cfg)
        rows=list(csv.DictReader(io.StringIO(plan.catalog_csv())))
        original=plan.registered_rows()
        self.assertEqual([(r['case'],r['replica']) for r in rows],
                         [(r['case_id'],r['replica']) for r in original])
        self.assertEqual(len(rows),14)
        self.assertEqual(sum(r['optional']=='False' for r in rows),8)
        self.assertEqual([r['case'] for r in rows if r['phase']=='donor'],list(plan.DONORS))
        self.assertEqual(len(cfg['groups'][-1]['runs']),3)
        for server in ('s2','s3','s4','s5'):
            with self.assertRaises(ValueError):plan.default_config(server)
        changed=deepcopy(cfg);changed['budget']['total_seconds']+=1
        with self.assertRaises(ValueError):plan.validate_config(changed)

    def test_cases_share_declared_arch_seed_loss_horizon(self):
        donors=[plan.resolve_case('s1',c) for c in plan.DONORS]
        self.assertTrue(all(c['stream_seeds']==donors[0]['stream_seeds'] for c in donors))
        for row in plan.registered_rows():
            cfg=plan.resolve_case('s1',row['case_id'],row['replica'])
            self.assertEqual(cfg['planned_horizon'],50000)
            self.assertEqual(len(cfg['evaluation_steps']),35)
            self.assertEqual(cfg['evaluation_steps'][-1],50000)
            self.assertEqual(cfg['student_architecture'],{'bands':8,'width':112,'depth':[1,2,3]})
            self.assertEqual(cfg['pan_channels']+8,row['unet_input_channels'])
            self.assertEqual(cfg['stage2_spec']['beta'],.05)
            self.assertNotEqual(cfg['stream_seeds']['rr_band_augmentation'],cfg['stream_seeds']['rr_epsilon'])
        with self.assertRaises(ValueError):plan.resolve_case('s1','G8_MS_RETURN')

    def test_hqnr_first_no_threshold_override(self):
        a={'HQNR':.96,'ERGAS':2.0,'completed_update':1000}
        b={'HQNR':.961,'ERGAS':2.3,'completed_update':2000}
        self.assertGreater(plan.rank(b),plan.rank(a))
        b['HQNR']=a['HQNR'];self.assertLess(plan.rank(b),plan.rank(a))

    def test_durable_deadline_no_backward_clock_extension(self):
        c=self.campaign();c.persist();deadline=c.deadline_unix
        self.now+=20*3600;restored=self.campaign();restored.persist()
        self.now-=3600;restored=self.campaign()
        self.assertGreaterEqual(restored.elapsed_seconds(),20*3600)
        self.assertEqual(restored.deadline_unix,deadline)
        self.now=1000+29.5*3600
        with self.assertRaises(runtime.BudgetExpired):restored.check()
        restored.closeout=True;restored.check()
        self.now=deadline
        with self.assertRaises(runtime.BudgetExpired):restored.check()

    def test_whole_three_repeat_admission(self):
        c=self.campaign();group=plan.groups()[-1];self.now+=27*3600
        for case,replica in group['runs']:
            cfg=plan.resolve_case('s1',case,replica)
            c.state['timing'][c.timing_key(cfg)]={'updates_observed':100,'eval_seconds':10}
        with patch.object(c,'estimate_seconds',return_value=3600):
            self.assertFalse(c.admit_group(group))
            self.assertEqual(c.state['groups'][-1]['required_seconds'],10800)
            case,replica=group['runs'][0]
            c.state['runs'].append({'run_id':plan.resolve_case('s1',case,replica)['run_id'],'status':'COMPLETED'})
            self.assertTrue(c.admit_group(group))

    def test_interleaved_queue_registers_completed_donor_before_dependents(self):
        c=self.campaign();calls=[]
        def run(cfg):
            calls.append(cfg['case'])
            c.state['runs'].append({'run_id':cfg['run_id'],'status':'COMPLETED','residual_registration':{'status':'COMPLETE'}})
        with patch.object(c,'admission'),patch.object(c,'run_case',side_effect=run), \
             patch.object(c,'register_donor',side_effect=lambda case:calls.append('REGISTER:'+case)), \
             patch.object(c,'estimate_seconds',return_value=1e9):
            state=c.run()
        self.assertEqual(calls[:3],['G8_S1_U23','REGISTER:G8_S1_U23','G8_PAN_U23'])
        self.assertLess(calls.index('REGISTER:B8A_S1_U23'),calls.index('B8A_PAN_U23'))
        self.assertEqual(state['status'],'FINISHED_CORE_OPTIONALS_BUDGET_SKIPPED')

    def test_started_optional_group_is_recognized_for_resume(self):
        c=self.campaign();group=plan.groups()[-1]
        self.assertFalse(c.group_started(group))
        case,replica=group['runs'][0]
        c.state['runs'].append({'run_id':plan.resolve_case('s1',case,replica)['run_id'],
                               'status':'STOPPED','step':20000})
        self.assertTrue(c.group_started(group))

    def test_diagnostic_ignores_invalid_zero_fallback(self):
        good={'valid':True,'shift':[3.,4.],'fallback_reason':None}
        bad={'valid':False,'shift':[0.,0.],'fallback_reason':'LOW_TEXTURE'}
        rows=[{'band':0,'before':good,'after':good},
              {'band':0,'before':good,'after':bad}]
        summary=paired_summary(rows)['bands']['0']
        self.assertEqual(summary['paired_valid_count'],1)
        self.assertEqual(summary['after']['norm_median'],5.)
        self.assertEqual(summary['failures']['after'],{'LOW_TEXTURE':1})

    def test_file_identity_excludes_access_time_but_detects_actual_writes(self):
        from types import SimpleNamespace
        first=SimpleNamespace(st_dev=1,st_ino=2,st_size=3,st_mtime_ns=4,st_ctime_ns=5,st_atime_ns=6)
        second=deepcopy(first);second.st_atime_ns=9
        fake=SimpleNamespace(stat=lambda:first)
        before=file_identity(fake)
        fake.stat=lambda:second
        self.assertEqual(file_identity(fake),before)
        second.st_mtime_ns+=1
        self.assertNotEqual(file_identity(fake),before)


if __name__=='__main__':unittest.main()
