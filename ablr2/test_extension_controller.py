"""Offline extension admission regressions; no workers, GPUs, or remote APIs."""
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ablr2 import controller as ctl
from ablr2.common import RuntimePaused, camp, read_json
from ablr2.plan import case_for, cases_for, full_wave, recheck_cases


class ExtensionControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.patches=ExitStack();self.addCleanup(self.patches.close)
        for name in ('verify_sources','verify_extension_sources'):
            self.patches.enter_context(patch.object(ctl,name,return_value={}))
        ctl.build(self.root,'s1')
        self.boot=cases_for('s1')

    def invoke(self,worker):
        def anchor(case,root):
            original=case_for(case.run_id.replace('_C17_','_C03_'),root)
            self.assertTrue(ctl._state(root,'s1')['runs'][original.run_id]['complete'])
            return original
        def preflight(root,server,args,log,deadline):
            self.assertEqual(args,['preflight'])
            return 0,0.
        with ExitStack() as stack:
            stack.enter_context(patch.object(ctl,'verify_registration',return_value={}))
            stack.enter_context(patch.object(ctl,'_lease',return_value=dict(sequence=1,
                expires_utc=None,service_mode='UNTIL_OPERATOR_STOP')))
            stack.enter_context(patch.object(ctl,'_command',side_effect=preflight))
            stack.enter_context(patch.object(ctl,'_run_case',side_effect=worker))
            stack.enter_context(patch('ablr2.extension.prepare_anchor',side_effect=anchor))
            stack.enter_context(patch('ablr2.resources.idle_evidence',return_value={'idle':True}))
            stack.enter_context(patch('ablr2.resources.assess_case',return_value={'allowed':True}))
            stack.enter_context(patch('ablr2.resources.assess_block',return_value={'allowed':True}))
            stack.enter_context(patch('ablr2.analysis.refresh_extended_reports',return_value={}))
            stack.enter_context(patch('ablr2.reporting.rebuild',return_value={}))
            return ctl.run(self.root,'s1',upload=False)

    def test_a17_recheck_runs_scheduled_c03_before_c17_without_repair(self):
        state=ctl._state(self.root,'s1')
        state['runs']={c.run_id:dict(complete=True) for c in self.boot}
        state['stages'][0]['complete']=True
        jobs=recheck_cases('s1','A17','R00','r000','RECHECK_A17',
            ctl._teacher_panels(self.boot),1234,[])
        self.assertEqual([c.case_id for c in jobs[:3]],['C17','C03','C07'])
        ctl._register_stage(self.root,'s1',state,jobs,'RECHECK_A17','RECHECK5',relation_id='A17')
        actual=[]
        def worker(root,server,state,case,upload):
            actual.append(case);state['runs'][case.run_id]=dict(complete=True)
            if case.case_id=='C17':ctl.control(root,server,'STOP_AFTER_RUN')
        self.assertEqual(self.invoke(worker),75)
        self.assertEqual([c.case_id for c in actual],['C03','C17'])
        self.assertEqual({c.sweep for c in actual},{'P01'})
        self.assertTrue(all(c.phase=='RECHECK5' for c in actual))
        registry=read_json(camp(self.root,'s1')/'cases.json')['cases']
        self.assertEqual(len(registry),115)
        self.assertFalse(any(c['phase']=='PAIR_REPAIR' for c in registry.values()))
        self.assertEqual(ctl._state(self.root,'s1')['reason'],'STOP_AFTER_RUN')

    def test_legacy_debt_limit_two_survives_restart_before_next_core_job(self):
        state=ctl._state(self.root,'s1');stage=state['stages'][0]
        legacy=[c for c in self.boot if c.case_id!='C17']
        stage['run_ids']=[c.run_id for c in legacy]
        # A historical95 task list is retained; C17 jobs exist only in its overlay.
        ctl.atomic_json(camp(self.root,'s1')/'stages/BOOT5.json',dict(stage_id='BOOT5',
            kind='BOOT5',cases=[asdict(c) for c in legacy]))
        core=next(c for c in legacy if c.sweep=='P04' and c.case_id=='C04')
        state['runs']={c.run_id:dict(complete=True) for c in legacy if c!=core}
        ctl._save(self.root,'s1',state)
        original=(camp(self.root,'s1')/'stages/BOOT5.json').read_bytes()
        completed=[];attempted=[]
        def crash_before_core(root,server,state,case,upload):
            attempted.append(case)
            if case.case_id!='C17':raise RuntimePaused('SIMULATED_RESTART_BEFORE_CORE')
            completed.append(case);state['runs'][case.run_id]=dict(complete=True)
        self.assertEqual(self.invoke(crash_before_core),75)
        self.assertEqual([(c.case_id,c.sweep) for c in completed],[('C17','P01'),('C17','P02')])
        self.assertEqual(attempted[-1],core)
        self.assertEqual(ctl._state(self.root,'s1')['extension_debt_streak'],2)
        resumed=[]
        def finish_core(root,server,state,case,upload):
            resumed.append(case);state['runs'][case.run_id]=dict(complete=True)
            ctl.control(root,server,'STOP_AFTER_RUN')
        self.assertEqual(self.invoke(finish_core),75)
        self.assertEqual(resumed,[core])
        final=ctl._state(self.root,'s1')
        self.assertEqual(final['extension_debt_streak'],0)
        self.assertEqual(len(final['stages'][0]['run_ids']),95)
        self.assertEqual((camp(self.root,'s1')/'stages/BOOT5.json').read_bytes(),original)
        pending=next(c for c in self.boot if c.case_id=='C17' and c.sweep=='P03')
        self.assertFalse(final['runs'].get(pending.run_id,{}).get('complete',False))

    def test_stop_after_sweep_defers_ready_debt_from_different_historical_sweep(self):
        state=ctl._state(self.root,'s1')
        old_debt=next(c for c in self.boot if c.case_id=='C17' and c.sweep=='P01')
        state['runs']={c.run_id:dict(complete=True) for c in self.boot if c!=old_debt}
        state['stages'][0]['complete']=True
        jobs=full_wave('s1','REFRESH5','R00','r000','NEXT_DEV',1234,[])
        ctl._register_stage(self.root,'s1',state,jobs,'NEXT_DEV','REFRESH5',recipe_id='R00')
        core=next(c for c in jobs if c.sweep=='P02' and c.case_id=='C04')
        for c in jobs:
            if (c.sweep in ('P01','P02') and c!=core) or c.case_id=='C17':
                state['runs'][c.run_id]=dict(complete=True)
        state['last_completed_run']=next(c.run_id for c in jobs if c.sweep=='P02' and c.case_id=='C03')
        ctl._save(self.root,'s1',state);ctl.control(self.root,'s1','STOP_AFTER_SWEEP')
        actual=[]
        def worker(root,server,state,case,upload):
            actual.append(case);state['runs'][case.run_id]=dict(complete=True)
        self.assertEqual(self.invoke(worker),75)
        self.assertEqual(actual,[core])
        final=ctl._state(self.root,'s1')
        self.assertFalse(final['runs'].get(old_debt.run_id,{}).get('complete',False))
        self.assertEqual(final['reason'],'STOP_AFTER_SWEEP')
        self.assertFalse(next(s for s in final['stages'] if s['stage_id']=='NEXT_DEV')['complete'])


if __name__=='__main__':unittest.main()
