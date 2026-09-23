"""CPU-only durable lane orchestration checks; no subprocesses/GPU/Sheets."""
from dataclasses import asdict
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ablr2 import controller as ctl
from ablr2.common import camp, read_json, run_dir, RuntimePaused
from ablr2.plan import CASES, GRAPH, LEGACY_BOOT_CASES, cases_for, case_for, full_wave, object_sha, CAMPAIGN_ID


def thresholds(sensor='WV3'):
    return dict(schema='ABLR2_THRESHOLDS_v1', sensor=sensor, epsilon_H=1e-5,
                epsilon_E=1e-4, delta_H=.001, delta_E=.003)


def relation_reports(reversal=None):
    reports = {r: dict(classification='NEAR_ZERO', median_delta_H=0., median_rE=0., flags=[])
               for r in GRAPH}
    if reversal:
        reports[reversal].update(classification='REVERSAL', median_delta_H=-.002, median_rE=.01)
    return reports


def observation(case, *, h=.95, e=3., ev=3.):
    metrics = dict(HQNR=h, ERGAS=e, E_val=ev, D_lambda=.02)
    return dict(run_id=case.run_id, case_id=case.case_id, VAL=metrics, EXACT50K=dict(metrics),
        source_identity={'content_sha256':'a'*64},data_sha256='b'*64,
        init_U_sha256='c'*64,sampler_sha256='d'*64,rng_roles={'seed':case.seed})


def mock_wave_report(root, stage, cases, reversal=None):
    report=dict(report_path=str(camp(root,'s1')/(stage+'_report.json')),
        thresholds=thresholds(),relations=relation_reports(reversal),
        panelrows=[observation(c) for c in cases if c.role=='S'])
    Path(report['report_path']).write_text(json.dumps(report))
    return report


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_patch = patch.object(ctl, 'verify_sources', return_value={})
        self.source_patch.start(); self.addCleanup(self.source_patch.stop)
        self.extension_source_patch=patch.object(ctl,'verify_extension_sources',return_value={})
        self.extension_source_patch.start();self.addCleanup(self.extension_source_patch.stop)
        self.no_command = patch.object(ctl, '_command', side_effect=AssertionError('No real worker in orchestration tests'))
        self.command = self.no_command.start(); self.addCleanup(self.no_command.stop)
        self.cumulative_patch = patch('ablr2.analysis.cumulative_balanced_reports', return_value={'mock': True})
        self.cumulative_patch.start(); self.addCleanup(self.cumulative_patch.stop)
        self.block_resources_patch=patch('ablr2.resources.assess_block',return_value={'allowed':True})
        self.block_resources_patch.start();self.addCleanup(self.block_resources_patch.stop)

    def build(self, server='s1'):
        ctl.build(self.root, server)
        return ctl._state(self.root, server)

    def test_build_only_registers_exact_hundred_without_lease_or_launch(self):
        report = ctl.build(self.root, 's1')
        self.assertFalse(report['activated'])
        self.assertFalse(report['lease_created'])
        folder = camp(self.root, 's1')
        self.assertFalse((folder/'lease.json').exists())
        self.assertFalse((folder/'dataset_manifest.json').exists())
        self.assertFalse((folder/'registration.json').exists())
        state = ctl._state(self.root, 's1')
        self.assertEqual(state['active_stage'], 'BOOT5')
        self.assertEqual(len(state['stages'][0]['run_ids']), 100)
        self.assertEqual(len(read_json(folder/'cases.json')['cases']), 100)
        self.assertEqual(len(read_json(folder/'seed_ledger.json')), 10)
        self.command.assert_not_called()

    def test_build_is_idempotent_and_preserves_live_state(self):
        state = self.build()
        state.update(status='PAUSED_SAFE', cycle=7)
        state['observations'].append(dict(hours=12.5,completed=True))
        ctl._save(self.root,'s1',state)
        original = (camp(self.root,'s1')/'state.json').read_bytes()
        ledger = (camp(self.root,'s1')/'task_ledger.jsonl').read_bytes()
        ctl.build(self.root,'s1')
        self.assertEqual((camp(self.root,'s1')/'state.json').read_bytes(),original)
        self.assertEqual((camp(self.root,'s1')/'task_ledger.jsonl').read_bytes(),ledger)

    def test_build_preserves_actual_old95_stage_and_adds_only_five_c17_debts(self):
        folder=camp(self.root,'s1');old=tuple(c for c in LEGACY_BOOT_CASES if c.server_id=='s1')
        stage=dict(stage_id='BOOT5',kind='BOOT5',run_ids=[c.run_id for c in old],complete=False,recipe_id='R00',recipe_revision='r000')
        original_state=dict(campaign_id=CAMPAIGN_ID,server='s1',sensor='WV3',status='PAUSED_SAFE',
            incumbent='R00',recipe_revision='r000',cycle=7,active_stage='BOOT5',stages=[stage],
            runs={old[0].run_id:{'complete':True,'train_hours':2.4}},observations=[{'hours':12.5}],
            rechecked={},tried_recipes=[],verification_recipes=[],low_information_cycles=1)
        ctl.atomic_json(folder/'state.json',original_state)
        ctl.atomic_json(folder/'cases.json',dict(cases={c.run_id:asdict(c) for c in old}))
        ctl.atomic_json(folder/'stages/BOOT5.json',dict(stage_id='BOOT5',kind='BOOT5',cases=[asdict(c) for c in old]))
        seeds=[]
        for case in old:
            key=f'BOOT5|s1|{case.sweep}|{case.role}'
            if not any(r['key']==key for r in seeds):seeds.append(dict(key=key,seed=case.seed,phase='BOOT5',sensor='WV3',selection='AUTHOR_SUPPLIED_CSV'))
        ctl.atomic_json(folder/'seed_ledger.json',seeds);ctl._sync_seed_log(folder,seeds)
        ctl.atomic_json(folder/'thresholds_v1.json',thresholds())
        preserved={name:(folder/name).read_bytes() for name in ('state.json','stages/BOOT5.json','seed_ledger.json','seed_ledger.jsonl','thresholds_v1.json')}
        ctl.build(self.root,'s1')
        for name,before in preserved.items():self.assertEqual((folder/name).read_bytes(),before,name)
        registry=read_json(folder/'cases.json')['cases']
        self.assertEqual(len(registry),100)
        for case in old:self.assertEqual(registry[case.run_id],asdict(case))
        debts=read_json(folder/'extension/debts.json')['debts']
        self.assertEqual(len(debts),5);self.assertTrue(all('_C17_' in run for run in debts))
        self.assertEqual(len(ctl._state(self.root,'s1')['stages'][0]['run_ids']),95)
        before=(folder/'extension/task_ledger.jsonl').read_bytes()
        ctl.build(self.root,'s1')
        self.assertEqual((folder/'extension/task_ledger.jsonl').read_bytes(),before)
        self.command.assert_not_called()

    def test_build_crash_before_state_publication_recovers_same_seed_panel(self):
        with patch.object(ctl,'_sync_seed_log',side_effect=OSError('simulated storage interruption')):
            with self.assertRaises(OSError): ctl.build(self.root,'s1')
        folder=camp(self.root,'s1')
        seed_bytes=(folder/'seed_ledger.json').read_bytes()
        self.assertFalse((folder/'state.json').exists())
        ctl.build(self.root,'s1')
        self.assertEqual((folder/'seed_ledger.json').read_bytes(),seed_bytes)
        self.assertEqual(len(read_json(folder/'cases.json')['cases']),100)
        self.assertEqual(len((folder/'seed_ledger.jsonl').read_text().splitlines()),10)

    def test_partial_seed_jsonl_is_repaired_without_reallocation(self):
        state=self.build()
        folder=camp(self.root,'s1')
        path=folder/'seed_ledger.jsonl'
        original=path.read_text().splitlines()
        path.write_text('\n'.join(original[:5])+'\n')
        seeds=(folder/'seed_ledger.json').read_bytes()
        ctl.build(self.root,'s1')
        repaired=[json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(repaired),10)
        self.assertEqual({r['key'] for r in repaired},{json.loads(line)['key'] for line in original})
        self.assertEqual((folder/'seed_ledger.json').read_bytes(),seeds)
        self.assertEqual(ctl._state(self.root,'s1')['active_stage'],'BOOT5')

    def test_duplicate_boot_seed_allocation_is_rejected(self):
        self.build()
        path=camp(self.root,'s1')/'seed_ledger.json'
        seeds=read_json(path)
        seeds.append(dict(seeds[0]))
        path.write_text(json.dumps(seeds))
        with self.assertRaises(ValueError): ctl.build(self.root,'s1')

    def test_seed_jsonl_conflict_is_not_silently_repaired(self):
        self.build()
        path=camp(self.root,'s1')/'seed_ledger.jsonl'
        records=[json.loads(line) for line in path.read_text().splitlines()]
        records[0]['seed']+=1
        path.write_text('\n'.join(json.dumps(row) for row in records)+'\n')
        corrupted=path.read_bytes()
        with self.assertRaises(ValueError): ctl.build(self.root,'s1')
        self.assertEqual(path.read_bytes(),corrupted)

    def test_lane_build_does_not_touch_other_lanes(self):
        self.build('s1')
        self.assertFalse(camp(self.root,'s2').exists())
        self.assertFalse(camp(self.root,'s3').exists())
        self.build('s3')
        self.assertEqual(ctl._state(self.root,'s3')['sensor'],'GF2')
        for server in ('s4','s5'):
            with self.assertRaises(ValueError): ctl.build(self.root,server)

    def test_lease_requires_operator_and_renewal_preserves_origin_and_compute(self):
        state = self.build()
        state['observations'].append(dict(hours=4.2));ctl._save(self.root,'s1',state)
        with self.assertRaises(RuntimePaused): ctl._lease(self.root,'s1')
        first = ctl.grant_lease(self.root,'s1',72,now='2030-01-01T00:00:00+00:00')
        second = ctl.grant_lease(self.root,'s1',72,now='2030-01-02T00:00:00+00:00')
        self.assertEqual(second['first_granted_at_utc'],first['first_granted_at_utc'])
        self.assertEqual(second['sequence'],2)
        self.assertEqual(second['expires_utc'],'2030-01-05T00:00:00+00:00')
        self.assertFalse(second['automatic_renewal'])
        self.assertEqual(ctl._state(self.root,'s1')['observations'],[dict(hours=4.2)])
        self.assertEqual(len(list((camp(self.root,'s1')/'leases').glob('*.json'))),2)
        with self.assertRaises(ValueError): ctl.grant_lease(self.root,'s1',1,now='2030-01-02T00:00:00+00:00')
        for bad in (0,-1,73,float('nan')):
            with self.assertRaises(ValueError): ctl.grant_lease(self.root,'s1',bad)

    def test_expired_lease_does_not_auto_renew(self):
        self.build()
        ctl.grant_lease(self.root,'s1',1,now='2000-01-01T00:00:00+00:00')
        lease = (camp(self.root,'s1')/'lease.json').read_bytes()
        with self.assertRaises(RuntimePaused): ctl._lease(self.root,'s1')
        self.assertEqual((camp(self.root,'s1')/'lease.json').read_bytes(),lease)

    def test_control_is_lane_local_and_append_only(self):
        self.build()
        ctl.control(self.root,'s1','STOP_NOW_SAFE')
        ctl.control(self.root,'s1','CONTINUE')
        rows = [json.loads(line) for line in (camp(self.root,'s1')/'control_ledger.jsonl').read_text().splitlines()]
        self.assertEqual([r['command'] for r in rows],['STOP_NOW_SAFE','CONTINUE'])
        with self.assertRaises(ValueError): ctl.control(self.root,'s1','KILL_OTHER_GPU')
        self.assertFalse(camp(self.root,'s2').exists())

    def test_whole_stage_registration_is_idempotent_and_recoverable(self):
        state = self.build()
        cases = full_wave('s1','REFRESH5','R00','r000','W001',1234,[])
        ctl._register_stage(self.root,'s1',state,cases,'W001','REFRESH5',recipe_id='R00')
        events = (camp(self.root,'s1')/'task_ledger.jsonl').read_bytes()
        ctl._register_stage(self.root,'s1',state,cases,'W001','REFRESH5',recipe_id='R00')
        self.assertEqual(len(state['stages']),2)
        self.assertEqual(len(read_json(camp(self.root,'s1')/'cases.json')['cases']),200)
        self.assertEqual((camp(self.root,'s1')/'task_ledger.jsonl').read_bytes(),events)
        self.assertEqual(case_for(cases[-1].run_id,self.root),cases[-1])

    def test_stage_registration_rejects_cross_lane_and_duplicate_jobs(self):
        state = self.build()
        with self.assertRaises(ValueError):
            ctl._register_stage(self.root,'s1',state,cases_for('s2'),'WRONG','BOOT5')
        with self.assertRaises(ValueError):
            ctl._register_stage(self.root,'s1',state,(cases_for('s1')[0],)*2,'DUP','FIT_ROUND')
        with self.assertRaises(ValueError):
            ctl._register_stage(self.root,'s1',state,cases_for('s1'),'../escape','BOOT5')

    def test_boot_recheck_fit_refresh_pure_stage_cycle(self):
        state = self.build()
        boot = state['stages'][0]
        report = mock_wave_report(self.root,'BOOT5',cases_for('s1'),'L04')
        with patch('ablr2.analysis.analyze_wave',return_value=report):
            ctl._analyze_and_advance(self.root,'s1',state,boot)
        self.assertEqual(state['cycle'],1)
        stage = state['stages'][-1]
        self.assertEqual(stage['kind'],'RECHECK5')
        self.assertEqual(stage['relation_id'],'L04')
        self.assertEqual(len(stage['run_ids']),15)
        self.assertEqual(stage['teacher_pool'],'BOOT5')
        def negative(case, root=None):
            return observation(case,h=.948 if case.case_id=='C04' else .95,
                e=3.03 if case.case_id=='C04' else 3.)
        with patch('ablr2.analysis.case_report',side_effect=negative):
            ctl._analyze_and_advance(self.root,'s1',state,stage)
        self.assertEqual(state['rechecked']['r000'],['L04'])
        fit = state['stages'][-1]
        self.assertEqual(fit['kind'],'FIT_ROUND')
        self.assertEqual(fit['candidates'],['R01','R02'])
        self.assertEqual(len(fit['run_ids']),18)
        def screened(case,root=None):
            # Absolute FULL validation quality selects R01, irrespective of gap.
            return observation(case,h=.951 if case.recipe_id=='R01' else .95,
                               ev=2.97 if case.recipe_id=='R01' else 3.)
        with patch('ablr2.analysis.case_report',side_effect=screened):
            ctl._analyze_and_advance(self.root,'s1',state,fit)
        self.assertEqual(state['incumbent'],'R01')
        self.assertEqual(state['recipe_revision'],'r001')
        self.assertEqual(state['tried_recipes'],['R01','R02'])
        refresh = state['stages'][-1]
        self.assertEqual(refresh['kind'],'REFRESH5')
        self.assertEqual(len(refresh['run_ids']),100)
        self.assertEqual({case_for(r,self.root).recipe_id for r in refresh['run_ids']},{'R01'})
        self.command.assert_not_called()

    def test_exhausted_bank_continues_balanced_refresh_not_new_coefficients(self):
        state = self.build()
        state['cycle']=6;state['tried_recipes']=['R01','R02','R03','R04','R05']
        ctl._schedule_fit(self.root,'s1',state,'D15')
        self.assertEqual(state['alert'],'PLATEAU_UNRESOLVED')
        self.assertEqual(state['stages'][-1]['kind'],'REFRESH5')
        self.assertEqual(len(state['stages'][-1]['run_ids']),100)
        self.assertEqual(state['incumbent'],'R00')

    def test_same_recipe_relation_cannot_be_rechecked_again(self):
        state = self.build()
        state['rechecked']['r000']=list(GRAPH)
        report = mock_wave_report(self.root,'BOOT5',cases_for('s1'),'L04')
        with patch('ablr2.analysis.analyze_wave',return_value=report):
            ctl._analyze_and_advance(self.root,'s1',state,state['stages'][0])
        self.assertEqual(state['stages'][-1]['kind'],'FIT_ROUND')

    def test_completed_stage_crash_before_pointer_commit_reuses_same_seeds_and_jobs(self):
        state=self.build()
        state['stages'][0]['complete']=True
        ctl._save(self.root,'s1',state)
        report=mock_wave_report(self.root,'BOOT5',cases_for('s1'),'L04')
        real_save=ctl._save
        def crash_new_pointer(root,server,value):
            if value['active_stage'].startswith('RECHECK'):
                raise OSError('simulated process loss before state pointer commit')
            return real_save(root,server,value)
        with patch('ablr2.analysis.analyze_wave',return_value=report), \
             patch.object(ctl,'_save',side_effect=crash_new_pointer):
            with self.assertRaises(OSError):
                ctl._analyze_and_advance(self.root,'s1',state,state['stages'][0])
        folder=camp(self.root,'s1')
        seeds=read_json(folder/'seed_ledger.json')
        self.assertEqual(len(seeds),15)
        restarted=ctl._state(self.root,'s1')
        self.assertEqual(restarted['active_stage'],'BOOT5')
        self.assertEqual(restarted['cycle'],0)
        with patch('ablr2.analysis.analyze_wave',return_value=report):
            ctl._analyze_and_advance(self.root,'s1',restarted,restarted['stages'][0])
        self.assertEqual(restarted['active_stage'],'RECHECK0001_L04')
        self.assertEqual(restarted['cycle'],1)
        self.assertEqual(len(restarted['stages']),2)
        self.assertEqual(read_json(folder/'seed_ledger.json'),seeds)
        self.assertEqual(len(read_json(folder/'cases.json')['cases']),115)
        self.assertEqual(len({r['key'] for r in seeds}),len(seeds))

    def test_technical_train_retry_must_resume_existing_fullstate(self):
        state = self.build();case = cases_for('s1')[0]
        state.update(status='TRAINING',active_run=case.run_id)
        folder = run_dir(case.run_id,self.root)
        calls=[]
        def command(root,server,args,log,deadline):
            calls.append(list(args))
            if len(calls)==1:
                (folder/'last').mkdir(parents=True)
                (folder/'last/training_state.pt').write_bytes(b'test fullstate marker')
                return 74,.25
            return 0,.5
        with patch.object(ctl,'_lease',return_value={'expires_utc':'2030-01-01T00:00:00+00:00'}), \
             patch.object(ctl,'_command',side_effect=command):
            ctl._record_action(self.root,'s1',state,case,'train',['train','--config','registered.yaml'])
        self.assertEqual(len(calls),2)
        self.assertIn('--resume',calls[1])
        self.assertEqual(state['runs'][case.run_id]['train_technical_retries'],1)
        self.assertEqual(state['runs'][case.run_id]['train_hours'],.75)

    def test_technical_retries_are_finite_and_low_scores_are_not_retries(self):
        state = self.build();case = cases_for('s1')[0]
        with patch.object(ctl,'_lease',return_value={'expires_utc':'2030-01-01T00:00:00+00:00'}), \
             patch.object(ctl,'_command',return_value=(74,.1)) as command:
            with self.assertRaises(RuntimePaused):
                ctl._record_action(self.root,'s1',state,case,'postrun',['postrun','--run',case.run_id])
            self.assertEqual(command.call_count,3)
        state['runs'].clear()
        with patch.object(ctl,'_lease',return_value={'expires_utc':'2030-01-01T00:00:00+00:00'}), \
             patch.object(ctl,'_command',return_value=(0,.1)) as command:
            ctl._record_action(self.root,'s1',state,case,'train',['train','--config','registered.yaml'])
            self.assertEqual(command.call_count,1)
            self.assertNotIn('train_technical_retries',state['runs'][case.run_id])

    def test_parse_error_two_is_not_successful_training(self):
        state = self.build();case = cases_for('s1')[0]
        with patch.object(ctl,'_lease',return_value={'expires_utc':'2030-01-01T00:00:00+00:00'}), \
             patch.object(ctl,'_command',return_value=(2,.01)):
            with self.assertRaises(ValueError):
                ctl._record_action(self.root,'s1',state,case,'train',['train','--config','registered.yaml'])

    def test_verification_cannot_replace_already_registered_same_recipe(self):
        state = self.build();state['verification_recipes']=['r000'];ctl._save(self.root,'s1',state)
        with self.assertRaises(ValueError): ctl.request_verify(self.root,'s1')

    def test_loop_stop_after_sweep_finishes_all_registered_components_only(self):
        self.build()
        called=[]
        lease=dict(sequence=1,expires_utc='2030-01-01T00:00:00+00:00')
        def fake_run(root,server,state,case,upload):
            called.append(case)
            state['runs'][case.run_id]=dict(complete=True,status='COMPLETE')
            if len(called)==1:
                # Mid-sweep operator request preserves C00-C17, including the overlay.
                ctl.control(root,server,'STOP_AFTER_SWEEP')
        def ready(root,server,state):
            result=[]
            for c17 in (c for c in cases_for(server) if c.case_id=='C17'):
                anchor=case_for(c17.run_id.replace('_C17_','_C03_'),root)
                if state['runs'].get(anchor.run_id,{}).get('complete') and not state['runs'].get(c17.run_id,{}).get('complete'):
                    result.append((c17,anchor))
            return result
        with patch.object(ctl,'verify_registration',return_value={}), \
             patch.object(ctl,'_lease',return_value=lease), \
             patch.object(ctl,'_command',return_value=(0,0.)), \
             patch.object(ctl,'_run_case',side_effect=fake_run), \
             patch('ablr2.extension.ready_debts',side_effect=ready), \
             patch('ablr2.extension.prepare_anchor',side_effect=lambda c,root:case_for(c.run_id.replace('_C17_','_C03_'),root)), \
             patch('ablr2.resources.idle_evidence',return_value={'idle':True}), \
             patch('ablr2.resources.assess_case',return_value={'allowed':True}):
            code=ctl.run(self.root,'s1',upload=False)
        self.assertEqual(code,75)
        self.assertEqual(len(called),20)
        self.assertEqual({case.sweep for case in called},{'P01'})
        self.assertEqual({case.case_id for case in called},set(('TPLUS','TZERO'))|{f'C{k:02}' for k in range(18)})
        state=ctl._state(self.root,'s1')
        self.assertEqual(state['status'],'PAUSED_SAFE')
        self.assertFalse(state['stages'][0]['complete'])

    def test_loop_lease_admission_does_not_drop_or_replace_registered_case(self):
        state=self.build()
        expected=list(state['stages'][0]['run_ids'])
        lease=dict(sequence=1,expires_utc=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=1)).isoformat())
        with patch.object(ctl,'verify_registration',return_value={}), \
             patch.object(ctl,'_lease',return_value=lease), \
             patch.object(ctl,'_command',return_value=(0,0.)), \
             patch.object(ctl,'_run_case') as worker, \
             patch('ablr2.resources.idle_evidence',return_value={'idle':True}):
            code=ctl.run(self.root,'s1',upload=False)
        self.assertEqual(code,75)
        worker.assert_not_called()
        state=ctl._state(self.root,'s1')
        self.assertEqual(state['stages'][0]['run_ids'],expected)
        self.assertIn('LEASE_ADMISSION_RESERVE',state['reason'])

    def test_until_stop_authorization_has_no_time_reservation_but_obeys_operator_stop(self):
        self.build('s3');called=[]
        authorization=dict(sequence=1,expires_utc=None,service_mode='UNTIL_OPERATOR_STOP')
        def work(root,server,state,case,upload):
            called.append(case);state['runs'][case.run_id]=dict(complete=True)
            ctl.control(root,server,'STOP_AFTER_RUN')
        with patch.object(ctl,'verify_registration',return_value={}), \
             patch.object(ctl,'_lease',return_value=authorization), \
             patch.object(ctl,'_command',return_value=(0,0.)), \
             patch.object(ctl,'_run_case',side_effect=work), \
             patch('ablr2.resources.idle_evidence',return_value={'idle':True}), \
             patch('ablr2.resources.assess_case',return_value={'allowed':True}):
            self.assertEqual(ctl.run(self.root,'s3',upload=False),75)
        self.assertEqual(len(called),1);self.assertEqual(called[0].sensor,'GF2')
        state=ctl._state(self.root,'s3')
        self.assertEqual(state['reason'],'STOP_AFTER_RUN')
        self.assertFalse(camp(self.root,'s1').exists());self.assertFalse(camp(self.root,'s2').exists())

    def test_verify_returns_to_saved_dev_stage_without_replacing_incumbent_panel(self):
        state=self.build()
        state.update(cycle=1,latest_full_stage='BOOT5',latest_relations=relation_reports('D11'))
        ctl._seed_wave(self.root,'s1',state,'REFRESH5','NEXT_DEV')
        state['resume_stage_after_verify']='NEXT_DEV'
        ctl._seed_wave(self.root,'s1',state,'VERIFY5','VERIFY_r000')
        verify=state['stages'][-1]
        report=dict(report_path=str(camp(self.root,'s1')/'verify_report.json'),
            thresholds=thresholds(),relations=relation_reports('L04'))
        with patch('ablr2.analysis.analyze_wave',return_value=report):
            ctl._analyze_and_advance(self.root,'s1',state,verify)
        self.assertEqual(state['active_stage'],'NEXT_DEV')
        self.assertEqual(state['cycle'],1)
        self.assertEqual(state['latest_full_stage'],'BOOT5')
        self.assertEqual(state['latest_relations']['D11']['classification'],'REVERSAL')

    def _pending_older_recipe_verify(self):
        state=self.build()
        state['stages'][0]['complete']=True
        state.update(cycle=1,latest_full_stage='BOOT5',latest_relations=relation_reports('D11'))
        ctl.atomic_json(camp(self.root,'s1')/'thresholds_v1.json',thresholds())
        ctl._save(self.root,'s1',state)
        identity={'content_sha256':'a'*64}
        with patch.object(ctl,'source_identity',return_value=identity):
            ctl.request_verify(self.root,'s1')
        # A completed fitting round promoted R01 after the operator locked R00.
        state.update(incumbent='R01',recipe_revision='r001')
        ctl._seed_wave(self.root,'s1',state,'REFRESH5','NEXT_DEV')
        return state,identity

    def test_pending_verify_preempts_untouched_next_stage_once_only(self):
        state,identity=self._pending_older_recipe_verify()
        folder=camp(self.root,'s1')
        with patch.object(ctl,'source_identity',return_value=identity):
            self.assertTrue(ctl._activate_pending_verify(self.root,'s1',state))
            seeds=read_json(folder/'seed_ledger.json')
            self.assertFalse(ctl._activate_pending_verify(self.root,'s1',state))
        self.assertEqual(state['active_stage'],'VERIFY_r000')
        self.assertEqual(state['resume_stage_after_verify'],'NEXT_DEV')
        self.assertEqual(state['incumbent'],'R01')
        self.assertEqual(state['recipe_revision'],'r001')
        self.assertEqual(state['verification_recipes'],['r000'])
        verify=state['stages'][-1]
        self.assertEqual(len(verify['run_ids']),100)
        self.assertEqual({case_for(run,self.root).recipe_id for run in verify['run_ids']},{'R00'})
        self.assertEqual(read_json(folder/'seed_ledger.json'),seeds)
        self.assertEqual(len(state['stages']),3)

    def test_pending_verify_defers_partially_started_registered_block(self):
        state,identity=self._pending_older_recipe_verify()
        next_stage=state['stages'][-1]
        state['runs'][next_stage['run_ids'][0]]=dict(train_attempts=1)
        seeds=read_json(camp(self.root,'s1')/'seed_ledger.json')
        with patch.object(ctl,'source_identity',return_value=identity):
            self.assertFalse(ctl._activate_pending_verify(self.root,'s1',state))
            state['runs'][next_stage['run_ids'][0]]=dict(complete=True)
            self.assertFalse(ctl._activate_pending_verify(self.root,'s1',state))
        self.assertEqual(state['active_stage'],'NEXT_DEV')
        self.assertEqual(state['verification_recipes'],[])
        self.assertEqual(read_json(camp(self.root,'s1')/'seed_ledger.json'),seeds)

    def test_verify_pointer_crash_restores_newer_dev_recipe_after_locked_old_recipe(self):
        state,identity=self._pending_older_recipe_verify()
        real_save=ctl._save
        def crash_after_verify_pointer(root,server,value):
            if value['active_stage']=='VERIFY_r000' and value['incumbent']=='R01':
                raise OSError('simulated loss after VERIFY pointer publication')
            return real_save(root,server,value)
        with patch.object(ctl,'source_identity',return_value=identity), \
             patch.object(ctl,'_save',side_effect=crash_after_verify_pointer):
            with self.assertRaises(OSError):
                ctl._activate_pending_verify(self.root,'s1',state)
        restarted=ctl._state(self.root,'s1')
        self.assertEqual(restarted['active_stage'],'VERIFY_r000')
        self.assertEqual(restarted['incumbent'],'R00')
        self.assertEqual(restarted['verify_resume_recipe'],dict(recipe_id='R01',recipe_revision='r001'))
        with patch.object(ctl,'source_identity',return_value=identity):
            self.assertFalse(ctl._activate_pending_verify(self.root,'s1',restarted))
        verify=next(s for s in restarted['stages'] if s['stage_id']=='VERIFY_r000')
        report=dict(report_path=str(camp(self.root,'s1')/'verify_crash_report.json'),
            thresholds=thresholds(),relations=relation_reports('L04'))
        with patch('ablr2.analysis.analyze_wave',return_value=report):
            ctl._analyze_and_advance(self.root,'s1',restarted,verify)
        final=ctl._state(self.root,'s1')
        self.assertEqual(final['active_stage'],'NEXT_DEV')
        self.assertEqual((final['incumbent'],final['recipe_revision']),('R01','r001'))
        self.assertEqual(final['verification_recipes'],['r000'])
        self.assertNotIn('verify_resume_recipe',final)
        self.assertEqual(final['latest_full_stage'],'BOOT5')


if __name__=='__main__': unittest.main()
