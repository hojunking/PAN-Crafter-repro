import copy
import datetime as dt
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
from gfp40 import controller as C
from gfp40.common import atomic_json,read_json,camp,run_dir,object_sha
from gfp40.plan import *
from gfp40.policy import CampaignWindow
from gfp40.upload import apply_upsert,flatten_summary

class Worksheet:
    title='GF2-P40-s3';id=123;col_count=1;row_count=1
    def __init__(self):self.rows=[];self.formats=[]
    def row_values(self,n,**kwargs):return self.rows[n-1].copy() if n<=len(self.rows) else []
    def add_cols(self,n):self.col_count+=n
    def add_rows(self,n):self.row_count+=n
    def get_all_values(self,**kwargs):return copy.deepcopy(self.rows)
    def update(self,range_name,values,**kwargs):
        letters,num=re.fullmatch(r'([A-Z]+)(\d+)',range_name).groups();col=0;row=int(num)
        for c in letters:col=col*26+ord(c)-64
        while len(self.rows)<row:self.rows.append([])
        while len(self.rows[row-1])<col-1+len(values[0]):self.rows[row-1].append('')
        self.rows[row-1][col-1:col-1+len(values[0])]=values[0]
    def batch_format(self,formats):self.formats.extend(formats)

class OperationsTests(unittest.TestCase):
    def test_build83_no_clock_no_training(self):
        with tempfile.TemporaryDirectory() as d,patch.object(C,'verify_sources'):
            result=C.build(Path(d))
            self.assertEqual(result['cases'],83);self.assertFalse(result['clock_created'])
            self.assertFalse((Path(d)/'work_dir').exists())
            for p in (Path(d)/'config/gfp40').glob('*.yaml'):validate_config(read_json(p))
    def test_clock_immutable_no_per_server_reset_or_future_start(self):
        with tempfile.TemporaryDirectory() as d:
            now=dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=1)
            C.write_window(d,now,'s3')
            with self.assertRaises(ValueError):C.write_window(d,now+dt.timedelta(minutes=1),'s3')
            with patch.object(C,'register_runtime') as register:
                with self.assertRaises(ValueError):C.start(d,'s3')
                register.assert_not_called()
    def test_no_remote_gate_after_local_parent_completion(self):
        b=block_for('S3_CONFIRM_1');state=dict(runs={})
        self.assertEqual(C.ready_case(b,state).case_id,'A11')
        state['runs'][case_for('A11').run_id]=dict(terminal=True,training_complete=True)
        self.assertIsNone(C.ready_case(b,state))
        state['runs'][case_for('A11').run_id]['complete']=True
        self.assertEqual(C.ready_case(b,state).case_id,'A12')
        self.assertEqual(C.ready_case(block_for('S3_CONFIRM_2'),state).case_id,'A14')
    def test_independent_local_clocks_auto_start_and_resume_without_reset(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.assertFalse(C.status(root,'s3')['window'])
            with patch.object(C,'utcnow',return_value='2026-09-22T00:00:00Z'):
                a=C.local_window(root,'s3',initialize=True)
            with patch.object(C,'utcnow',return_value='2026-09-23T00:00:00Z'):
                b=C.local_window(root,'s5',initialize=True)
                self.assertEqual(C.local_window(root,'s3',initialize=True),a)
            self.assertNotEqual(a.t0_utc,b.t0_utc)
            self.assertEqual((b.deadline_utc-a.deadline_utc).total_seconds(),86400)
            self.assertFalse((root/'work_dir/_gfp40/s4').exists())
            with self.assertRaises(ValueError):C.write_window(root,b.t0_utc,'s3')
    def test_local_clock_rejects_other_server_and_obsolete_shared_state(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            atomic_json(C.window_path(root,'s3'),CampaignWindow('2026-09-22T00:00:00Z','s4').to_dict())
            with self.assertRaises(ValueError):C.local_window(root,'s3')
            atomic_json(root/'work_dir/_gfp40/campaign_window.json',dict(obsolete=True))
            with self.assertRaises(ValueError):C.local_window(root,'s5',initialize=True)
    def test_start_needs_only_server_no_common_t0_or_selection_transport(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(C,'register_runtime') as register,patch.object(C,'run',return_value=0) as run:
                self.assertEqual(C.start(root,'s5',foreground=True,upload=False),dict(exit_code=0))
            register.assert_called_once_with(root,'s5')
            run.assert_called_once_with(root,'s5',None,False)
            self.assertEqual(C.local_window(root,'s5').server,'s5')
            self.assertFalse(C.window_path(root,'s3').exists())
            self.assertFalse(C.window_path(root,'s4').exists())
            self.assertFalse(hasattr(C,'tick_lock'))
            self.assertFalse(hasattr(C,'import_lock'))
    def test_cache_cost_cannot_be_spent_away_by_slow_preflight(self):
        with tempfile.TemporaryDirectory() as d:
            state=dict(blocks={},setup_hours=100,family_hours=100)
            self.assertGreater(C.pending_cache_hours(d,'s4',state),0.)
            self.assertGreater(C.pending_cache_hours(d,'s5',state),2.)
            self.assertAlmostEqual(C.pending_cache_hours(d,'s3',state),3*.75+.2)
    def test_native_confirmation_reserves_fixed_g025_disk(self):
        with tempfile.TemporaryDirectory() as d:
            required,n=C.block_cache_requirement(d,'s3',block_for('S3_CONFIRM_1'),dict(runs={}))
            self.assertTrue(required);self.assertEqual(n,3)
    def test_failed_replay_is_retried_and_reported_not_completed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);state=dict(runs={case_for(c).run_id:dict(complete=True) for c in ('A02','A03')})
            w=CampaignWindow(dt.datetime.now(dt.timezone.utc))
            with patch.object(C,'_command',side_effect=[(74,.1),(0,.1)]),patch.object(C,'_save'):
                self.assertTrue(C.run_replay(root,'s3',state,w));self.assertFalse(state['replay_complete'])
                self.assertEqual(state['replay_status'],'DIAGNOSTICS_PENDING')
                self.assertTrue(C.run_replay(root,'s3',state,w));self.assertTrue(state['replay_complete'])
            self.assertEqual(state['replay_attempts'],2)
    def test_completed_training_crash_goes_only_to_postrun(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);case=case_for('A01');window=CampaignWindow(dt.datetime.now(dt.timezone.utc))
            state=dict(runs={},observations=[]);calls=[]
            def read(path,default=None):
                if Path(path).name=='training_status.json':return dict(training_complete=True,actual_updates=case.updates)
                if Path(path).name=='summary.json':return dict(complete=True)
                return {} if default is None else default
            def command(root,s,args,log,deadline):calls.append(args[0]);return 0,.01
            with patch.object(C,'resolve_config',return_value=root/'config.yaml'),patch.object(C,'read',side_effect=read),\
                patch.object(C,'_command',side_effect=command),patch.object(C,'_save'):
                self.assertTrue(C._run_case(root,'s3',state,case,window,False))
            self.assertEqual(calls,['postrun'])
    def test_safe_pause_and_divergence_not_silent_terminal_skip(self):
        for reason,code in (('PAUSED_SIGNAL',75),('ABORTED_VAL_DIVERGENCE',76)):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);case=case_for('A01');window=CampaignWindow(dt.datetime.now(dt.timezone.utc));state=dict(runs={},observations=[])
                def read(path,default=None):
                    if Path(path).name=='training_status.json':return dict(training_complete=False,actual_updates=123,status=reason)
                    return {} if default is None else default
                with patch.object(C,'resolve_config',return_value=root/'cfg'),patch.object(C,'read',side_effect=read),\
                    patch.object(C,'_command',return_value=(code,.01)),patch.object(C,'_save'),patch.object(C,'append_event'),\
                    patch('gfp40.training.recover_exact_endpoint',return_value=False):
                    self.assertEqual(C._run_case(root,'s3',state,case,window,False),'PAUSED_SAFE')
                self.assertFalse(state['runs'][case.run_id]['terminal'])
    def test_new_sheet_precision_readback_and_old_tab_protection(self):
        ws=Worksheet();value=dict(campaign_id=CAMPAIGN_ID,run_id='x',server='s3',readback_status='READBACK_PENDING',
            **{'EXACT_FINAL hqnr':.9512345678912,'EXACT_FINAL rmse':7.123456789})
        self.assertTrue(apply_upsert(ws,value)['readback_verified']);self.assertEqual(ws.rows[1][4],value['EXACT_FINAL hqnr'])
        apply_upsert(ws,value);self.assertEqual(len(ws.rows),2)
        self.assertEqual(ws.formats[0]['format']['numberFormat']['pattern'],'0.0000')
        for title in ('GF2-B20-s3','GF2-s3(5090)'):
            ws.title=title
            with self.assertRaises(ValueError):apply_upsert(ws,value)
    def test_flatten_rmse_cc_jqm_selectors_samepoint(self):
        with tempfile.TemporaryDirectory() as d:
            r=dict(update=60000,val_ergas=.54,rr=dict(ergas=.55,rmse=4.123456789,cc=.98),fr=dict(hqnr=.95,d_s=.03,d_lambda=.02,jqm=.88,jqm_variant='SRF-substitute'))
            summary=dict(profile='CTRL',local_updates=60000,lifetime_updates=160000,
                selections={k:r for k in ('EXACT_FINAL','RR_VAL_SELECTED','RAW_AUX')})
            values=flatten_summary(summary,case_for('A07'),d)
            self.assertEqual(values['EXACT_FINAL rmse'],4.123456789);self.assertEqual(values['EXACT_FINAL cc'],.98)
            self.assertEqual(values['EXACT_FINAL jqm'],.88);self.assertEqual(values['lifetime_student_updates'],160000)
    def test_three_lanes_complete_finite_mock_dag_without_training(self):
        for s in SERVERS:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);folder=camp(root,s);C.write_window(root,dt.datetime.now(dt.timezone.utc),s)
                atomic_json(folder/'parent_bindings.json',{p:dict(status='BOUND') for p in PARENTS})
                for f in FAMILIES:atomic_json(folder/'families'/f'{f}.json',dict(manifest='fixture'))
                actual=[]
                def fake_case(root,server,state,case,window,upload):
                    actual.append(case.case_id);state['runs'][case.run_id]=dict(terminal=True,complete=True,training_complete=True)
                    return True
                def replay(root,server,state,window):state['replay_complete']=True;state['replay_attempted']=True;return True
                with patch.object(C,'verify_registration'),\
                    patch.object(C,'_command',return_value=(0,.01)),patch.object(C,'_run_case',side_effect=fake_case),\
                    patch.object(C,'evaluation_debt',return_value=0.),patch.object(C,'run_replay',side_effect=replay),\
                    patch.object(C,'pending_cache_hours',return_value=2.),\
                    patch.object(C,'closeout',return_value=0),\
                    patch('gfp40.resources.idle_evidence',return_value={'idle':True}),\
                    patch('gfp40.resources.assess_block',return_value={'allowed':True}):
                    self.assertEqual(C.run(root,s,upload=False),0)
                primary=[c.case_id for c in cases_for(s) if c.priority=='PRIMARY']
                self.assertTrue(set(primary)<=set(actual))
                self.assertEqual(len(actual),len(set(actual)))
                self.assertFalse(any((camp(root,other)/'status.json').exists() for other in SERVERS if other!=s))
                self.assertFalse((folder/'selection_lock.json').exists())
                if s=='s5':self.assertFalse({'C31','C32','C33'} & set(actual))

if __name__=='__main__':unittest.main()
