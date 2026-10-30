"""Synthetic CPU correctness checks, never reported as experiment results."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from panda_qb_refit import registry as G


class CampaignTests(unittest.TestCase):
    def subprocess(self, code):
        completed = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_finite_registry_single_axes_and_committed_manifest(self):
        self.assertEqual(len(G.RUNS), 10)
        self.assertEqual(sum(G.cases()[c]['total_updates'] for c, _ in G.RUNS), 290000)
        from panda_sensor_focus.contract import _fresh
        control = _fresh('QB')
        for case, settings in G.cases().items():
            changed = {k for k in control if control[k] != settings[k]}
            self.assertEqual(changed, set(G.CHANGES[case]['axes']))
            self.assertEqual((settings['scale'], settings['epsilon_radius_rr']), (1., 3.))
            self.assertEqual(settings['input_variant'], 'CORRECTED_DIFF')
            self.assertEqual((settings['lambda_abs'], settings['lambda_epsilon']), (2., .03))
            self.assertEqual(settings['inherited_updates'], 0)
        manifest_path = Path(__file__).resolve().parents[1] / 'experiment_manifest.json'
        self.assertEqual(json.loads(manifest_path.read_text()), json.loads(json.dumps(G.manifest())))
        self.assertIsNone(G.manifest()['time_cap_seconds'])

    def test_dedicated_process_configs_pins_and_exact_matched_controls(self):
        self.subprocess('''
from panda_qb_refit.adapter import install
p=install()
assert install() is p
from panda_qb_refit import registry as G
from panda_sensor_focus import reporting as R
from panda_sensor_focus.runtime import code_pin
from donor30.plan import canonical_sha256
import importlib
assert 'panda_qb_refit/registry.py' in code_pin()
assert 'panda_qb_refit/experiment_manifest.json' in code_pin()
for c,r in G.RUNS:
    cfg=p.resolve_case('s1',c,r);R._config(cfg)
    assert cfg['config_digest']==canonical_sha256(p.default_config('s1'))
    assert cfg['joint_target']==G.TARGET and cfg['stage2_spec']['beta']==0
    assert cfg['initialization']=='FRESH' and not cfg['stage1_retrained']
    assert cfg['fixed_donor_source']['selected_step']==44000
    assert cfg['fixed_donor_source']['checkpoint_sha256']=='7fa4491f0c0e7b8c5d559263c50fff18f84abd7a820852a0d24666928191ed34'
    assert cfg['evaluation_steps'][-1]==cfg['planned_horizon']
    b=G.CONTROLS[r];control=importlib.import_module(b['package']+'.plan').resolve_case('s1',b['case'],r)
    assert control['campaign_id']==b['campaign_id'] and cfg['stream_seeds']==control['stream_seeds']
    assert not cfg['comparison_control']['student_seed_differs_from_control']
    if c=='QB2_C30AG000': assert cfg['stage2_spec']['variant']=='panda_e'
try: p.resolve_case('s3','GF2_C60','R2')
except ValueError: pass
else: raise AssertionError('s3 must be rejected')
from pathlib import Path
assert Path('panda_qb_refit/cases_s1.csv').read_text()==p.catalog_csv('s1')
''')
        from panda_sensor_focus.contract import CAMPAIGN_ID, REGISTERED_RUNS
        self.assertEqual(CAMPAIGN_ID, G.OLD_CAMPAIGN)
        self.assertEqual(REGISTERED_RUNS['s1'], G.OLD_RUNS)

    def test_refuse_shared_focus_runtime_process(self):
        self.subprocess('''
import panda_sensor_focus.plan
from panda_qb_refit.adapter import install
try: install()
except RuntimeError: pass
else: raise AssertionError('shared process must be rejected')
''')

    def test_train_exact_resume_ag000_mask_and_existing_sheet_policy(self):
        self.subprocess('''
from panda_qb_refit.adapter import install
p=install()
import tempfile,torch
from copy import deepcopy
from pathlib import Path
from panda_sensor_focus.tests.test_core import sources,donor,bank,PINS
from panda_sensor_focus.calibration import bind_donor
from panda_sensor_focus.spec import MatureSpec
from panda_sensor_focus.engine import build_fresh_stage2,advance,save_checkpoint,load_checkpoint
from panda_stage2_condq.stream import TrainStream,StreamSpec
from panda_sensor_focus.cost.profiling import registered_profiles
from panda_sensor_focus.cost import signature
from donor30.artifacts import _same
torch.set_num_threads(1)
profiles=registered_profiles('s1')
assert {c['student_architecture']['width'] for c,_ in profiles}=={112}
for c,s in profiles: signature(c,s)
for case in p.CASES:
    with tempfile.TemporaryDirectory() as temp:
        reg=sources(temp,'QB',count=1);cfg=p.resolve_case('s1',case,'R1')
        cfg['training_spec'].update(effective_batch_size=1,microbatch_size=1)
        d=donor('s1',reg);spec=MatureSpec(**cfg['stage2_spec']);d=bind_donor(d,spec)
        stats=bank(d,spec,reg);cfg['donor_binding']=deepcopy(d.binding)
        t=build_fresh_stage2(cfg,d,stats,evaluator_pin=PINS['evaluator_pin'])
        ss=StreamSpec(cfg['stream_seeds']['rr_data'],cfg['stream_seeds']['rr_augmentation'],**cfg['augmentation'])
        stream=TrainStream(reg,ss,microbatch_size=1,effective_batch_size=1)
        loss=advance(t,stream)
        assert loss['loss']==loss['hard_weighted'] and loss['soft_weighted']==0
        if case=='QB2_C30AG000':
            assert loss['w_h_mean']==1 and loss['w_h_max']==1
            assert loss['loss']==loss['gt_raw']
            assert t.stage.spec.mask_policy==cfg['stage2_spec']['mask_policy']
        assert t.stage.student.backbone.input.in_channels==6 and t.stage.student.spec.width==112
        assert t.stage.spec.policy_id=='S100'
        path=Path(temp)/'resume.pt';saved=save_checkpoint(path,t,stream,**PINS)
        expected=advance(t,stream);state=deepcopy(t.state_dict())
        fresh=build_fresh_stage2(cfg,d,stats,evaluator_pin=PINS['evaluator_pin'])
        restream=TrainStream(reg,ss,microbatch_size=1,effective_batch_size=1)
        fresh,restream=load_checkpoint(path,saved['file_sha256'],fresh,restream,**PINS)
        assert _same(advance(fresh,restream),expected) and _same(fresh.state_dict(),state)
from panda_sensor_focus.best_publication import project,SELECTOR,FALLBACK_SELECTOR
with tempfile.TemporaryDirectory() as temp:
    value=project(temp,'s1')
    assert not value['conflicts'] and not value['rows']
    assert value['publication_thresholds']['hqnr_min']==.9315 and value['publication_thresholds']['ergas_max']==3.5
    assert value['publication_thresholds']['publish_all_joint_passes']
    assert SELECTOR=='HQNR_QUALIFIED_MIN_ERGAS' and FALLBACK_SELECTOR=='BEST_HQNR_FALLBACK'
''')

    def test_admission_waits_existing_wv3(self):
        from panda_qb_refit.prerequisites import evidence
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'work_dir/panda_sensor_focus_launchers/s1/status.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'server': 's1', 'stage': 'RESUME_HARD36', 'status': 'RESUME_HARD36'}))
            with self.assertRaisesRegex(ValueError, 'WAIT:'):
                evidence(root)

    def test_authenticated_early_stop_is_terminal_but_not_completed(self):
        from panda_sensor_focus.tests.test_early_stop import GateTests
        from panda_sensor_focus import early_stop as E
        from panda_qb_refit.prerequisites import registry_completion
        helper = GateTests()
        helper.setUp()
        try:
            root = Path(helper.temp.name)
            folder = root / 'campaign'
            helper.directory = folder / 'runs/test/attempt_01'
            (helper.directory / 'models').mkdir(parents=True)
            helper.cfg.update(campaign_id=G.CAMPAIGN_ID, server='s1', case='QB2_C30LR080', replica='R1')
            helper.write_window([.89, .87, .88])
            camp, row = helper.camp(step=10000)
            row.update(case=helper.cfg['case'], replica='R1')
            E.atomic_json(helper.directory / 'run_state.json', row)
            E.atomic_json(helper.directory / 'config.json', helper.cfg)
            with patch.object(E, 'authenticate_stop', side_effect=helper.safe), patch.object(E, 'emit_status'):
                self.assertTrue(E.finish_stopped(camp, helper.cfg))
            E.atomic_json(folder / 'campaign_state.json', {'status': 'FINISHED_WITH_EXPLICIT_EARLY_STOPS', 'runs': [row]})
            result = registry_completion(root, folder, G.CAMPAIGN_ID,
                                         [('QB2_C30LR080', 'R1')], {'QB2_C30LR080': 30000})
            self.assertEqual((result['completed_count'], result['early_stopped_count']), (0, 1))
            self.assertTrue(result['all_registered_accounted'])
            self.assertFalse(result['full_queue_completed_without_early_stops'])
            self.assertFalse(result['early_stopped_runs'][0]['full_horizon_completed'])
            (helper.directory / 'early_stop.json').write_text('{}')
            with self.assertRaises(ValueError):
                registry_completion(root, folder, G.CAMPAIGN_ID,
                                     [('QB2_C30LR080', 'R1')], {'QB2_C30LR080': 30000})
        finally:
            helper.doCleanups()

    def test_waiter_rejects_registry_or_source_changes(self):
        from panda_qb_refit.enqueue import verify_pin
        original = {'files': {'source.py': 'a'}, 'registry_digest': 'registered'}
        with patch('panda_qb_refit.enqueue.source_pin', return_value=deepcopy(original)):
            verify_pin(original)
        for changed in ({'files': {'source.py': 'b'}, 'registry_digest': 'registered'},
                        {'files': original['files'], 'registry_digest': 'different'}):
            with patch('panda_qb_refit.enqueue.source_pin', return_value=changed):
                with self.assertRaisesRegex(ValueError, 'versioned boundary'):
                    verify_pin(original)

    def test_owned_container_requires_nonce_and_exact_environment(self):
        from panda_qb_refit.enqueue import authenticate_container, TRAIN_ARGS
        identity = {'image': 'sha256:fixed', 'user': '1000', 'working_dir': '/workspace',
                    'env': ['ENV=1'], 'mounts': [{'Source': '/repo', 'Destination': '/workspace', 'RW': False}]}
        intent = {'identity': identity, 'token': 'nonce'}
        current = {'Image': identity['image'], 'Id': 'owned', 'Config': {
            'Cmd': TRAIN_ARGS, 'User': identity['user'], 'WorkingDir': identity['working_dir'],
            'Env': identity['env'], 'Labels': {'panda.campaign': G.CAMPAIGN_ID, 'panda.launch-token': 'nonce'}},
            'Mounts': identity['mounts'], 'State': {'Running': True}}
        authenticate_container(current, intent)
        for changed in ('token', 'image', 'env', 'mount'):
            other = deepcopy(current)
            if changed == 'token': other['Config']['Labels']['panda.launch-token'] = 'foreign'
            if changed == 'image': other['Image'] = 'sha256:other'
            if changed == 'env': other['Config']['Env'] = ['ENV=2']
            if changed == 'mount': other['Mounts'][0]['RW'] = True
            with self.assertRaises(ValueError): authenticate_container(other, intent)
        current['State']['Running'] = False
        with self.assertRaisesRegex(ValueError, 'exited'):
            authenticate_container(current, intent)
        authenticate_container(current, intent, {'container_id': 'owned'})
        with self.assertRaises(ValueError):
            authenticate_container(current, intent, {'container_id': 'replacement'})

    def test_docker_writes_only_new_output_and_pins_launch_token(self):
        from panda_qb_refit.enqueue import command
        source = {'Config': {'User': '1000:1000', 'WorkingDir': '/workspace', 'Env': ['SAFE=1']},
                  'Image': 'sha256:fixed', 'Mounts': [
                      {'Type': 'bind', 'Source': '/repo', 'Destination': '/workspace', 'RW': False},
                      {'Type': 'bind', 'Source': '/old', 'Destination': '/workspace/work_dir/old', 'RW': True}]}
        with tempfile.TemporaryDirectory() as temp, patch('panda_qb_refit.enqueue.ROOT', Path(temp)), \
                patch('panda_sensor_refine.launch.source_container', return_value=('original', source)):
            cmd, identity = command('test', token='nonce')
        mounts = [cmd[i + 1] for i, v in enumerate(cmd) if v == '--mount']
        self.assertEqual(len(mounts), 2)
        self.assertTrue(mounts[0].endswith(',readonly'))
        self.assertIn('/workspace/work_dir/panda_qb_refit', mounts[1])
        self.assertNotIn('/workspace/work_dir/old', ' '.join(cmd))
        self.assertIn('panda.launch-token=nonce', cmd)
        self.assertEqual(identity['source_container'], 'original')


if __name__ == '__main__':
    unittest.main()
