"""CPU and isolated-process checks; not scientific performance measurements."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from panda_qb_deepen import registry as G


class CampaignTests(unittest.TestCase):
    def child(self, code):
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_registry_and_exact_source_provenance(self):
        self.assertEqual(len(G.RUNS), 8)
        self.assertEqual(len(G.manifest()['ordered_runs']), 9)
        self.assertEqual(sum(G.cases()[c]['total_updates'] for c, _ in G.RUNS), 240000)
        baseline = G.cases()['QB3_CONTROL']
        for case, settings in G.cases().items():
            self.assertEqual({k for k in baseline if baseline[k] != settings[k]}, set(G.CHANGES[case]['axes']))
        manifest = Path(__file__).resolve().parents[1] / 'experiment_manifest.json'
        self.assertEqual(json.loads(manifest.read_text()), json.loads(json.dumps(G.manifest())))
        self.child('''
from copy import deepcopy
from pathlib import Path
from panda_qb_deepen.adapter import install
from panda_qb_deepen import registry as G
p=install()
from panda_s1loss9 import reporting as R
from panda_s1loss9 import publication
assert publication.LANES['s3']==('QB','QB-ablations',261001011,.9315,3.5)
from panda_s1loss9.runtime import code_pin
from donor30.plan import canonical_sha256
from panda_sensor_refine import contract as legacy
assert legacy.LANES['s3']['dataset']=='GF2'
assert p.LANES['s3']['dataset']=='QB' and p.LANES['s3']['peak']==2047
stage1=p.resolve_case('s3','S1_A200_E030')
assert stage1['seed']==261016301 and stage1['planned_horizon']==50000
assert len(stage1['evaluation_steps'])==50 and stage1['stage1_retrained']
from donor30.plan import stream_seeds
assert all(stage1['stream_seeds'][key]==value for key,value in stream_seeds(261016301).items())
assert stage1['lambda_abs']==2 and stage1['lambda_epsilon']==.03
assert p.default_config('s3')['stage1_runs']==1 and not p.default_config('s3')['allow_cross_server_donor']
assert p.default_config('s3')['before_campaign']=='NONE_INDEPENDENT_LOCAL_STAGE1'
assert 'queue_change_policy' not in p.default_config('s3')
from panda_sensor_focus.early_stop import eligible
assert not eligible(stage1)
assert R.destination('s3')['sheet_title']=='QB-ablations'
assert Path('panda_qb_deepen/cases_s3.csv').read_text()==p.catalog_csv('s3')
for name,rep in G.RUNS:
 c=p.resolve_case('s3',name,rep);R._config(c)
 assert c['server']=='s3' and c['donor_origin_server']=='s3' and c['dataset']=='QB'
 assert c['config_digest']==canonical_sha256(p.default_config('s3'))
 assert c['training_spec']['weight_decay']==G.cases()[name]['weight_decay']
 assert c['training_spec']['effective_batch_size']==48
 assert c['stage2_spec']['beta']==0 and c['shift_scale']==1 and c['inherited_updates']==0
 assert c['comparison_control']['server']=='s3' and c['stage1_retrained']
 assert 'AdamW wd=' in R.approach_description(c)
 bad=deepcopy(c);bad['training_spec']['weight_decay']=.123
 try:R._config(bad)
 except ValueError:pass
 else:raise AssertionError('unregistered WD must fail')
assert 'panda_qb_deepen/runner.py' in code_pin()
assert 'panda_qb_deepen/experiment_manifest.json' in code_pin()
assert not any(name.startswith('panda_gf2_closeout/') for name in code_pin())
from panda_s1loss9.orchestration import predecessor_receipt
assert predecessor_receipt(Path('/does-not-exist'), 's3')['external_dependencies']==[]
from panda_s1loss9.cost.profiling import registered_profiles
assert {cfg['phase'] for cfg,step in registered_profiles('s3')}=={'donor','stage2'}
try:p.resolve_case('s1','QB3_CONTROL','R1')
except ValueError:pass
else:raise AssertionError('consumer s1 must fail')
from panda_s1loss9.best_publication import project
import tempfile
with tempfile.TemporaryDirectory() as d:
 v=project(d,'s3');assert not v['conflicts']
 assert v['publication_thresholds']['hqnr_min']==.9315
 assert v['publication_thresholds']['ergas_max']==3.5
''')

    def test_fresh_stage1_updates_both_models_and_exact_resume(self):
        self.child('''
from panda_qb_deepen.adapter import install
p=install()
import torch,tempfile
from pathlib import Path
from types import SimpleNamespace
from panda_s1loss9.tests.test_loss_contract import sources,PINS
from panda_s1loss9.stage1 import build_stage1,Stage1Stream,advance_stage1,save_stage1,load_stage1
from panda_method.calibration import state_dict_sha256
from donor30.artifacts import _same
from copy import deepcopy
torch.set_num_threads(1)
with tempfile.TemporaryDirectory() as temp:
 reg=sources(temp,'QB',count=1);cfg=p.resolve_case('s3','S1_A200_E030')
 cfg['baseline_resolved'].update(effective_batch_RR=1,microbatch_RR=1,accumulation_RR=1)
 trainer=build_stage1(cfg,reg.sources);stream=Stage1Stream(reg.sources,cfg,device='cpu')
 before={k:state_dict_sha256(getattr(trainer.stage,k)) for k in ('aligner','teacher')}
 advance_stage1(trainer,stream)
 assert all(state_dict_sha256(getattr(trainer.stage,k))!=before[k] for k in before)
 path=Path(temp)/'resume.pt';saved=save_stage1(path,trainer,stream,**PINS)
 first_state=deepcopy(trainer.state_dict());first_stream=deepcopy(stream.state_dict())
 expected=advance_stage1(trainer,stream);expected_state=deepcopy(trainer.state_dict())
 loaded,restream,_=load_stage1(path,saved['file_sha256'],cfg,reg.sources,**PINS)
 assert _same(loaded.state_dict(),first_state) and _same(restream.state_dict(),first_stream)
 actual=advance_stage1(loaded,restream)
 assert _same(actual,expected) and _same(loaded.state_dict(),expected_state)
''')

    def test_selection_requires_complete50_and_uses_validation_order(self):
        self.child('''
from panda_qb_deepen.adapter import install
p=install()
import json,tempfile
from pathlib import Path
from panda_s1loss9.donor import complete_history
from panda_s1loss9.selection import rank
from ta2.evaluation import file_sha
cfg=p.resolve_case('s3','S1_A200_E030')
with tempfile.TemporaryDirectory() as temp:
 folder=Path(temp);models=folder/'models';models.mkdir()
 (folder/'config.json').write_text(json.dumps(cfg))
 state=dict(status='COMPLETED',step=50000,run_id=cfg['run_id'],phase='donor')
 (folder/'run_state.json').write_text(json.dumps(state))
 paths=[];rows=[]
 for step in cfg['evaluation_steps']:
  path=models/f'step_{step:07d}.pt';path.write_bytes(f'synthetic-{step}'.encode());paths.append(path)
  rows.append(dict(completed_update=step,file_sha256=file_sha(path),run_id=cfg['run_id'],
    checkpoint_verified=True,metrics_complete=True,metrics_finite=True,same_state_verified=True))
 (models/'selection.json').write_text(json.dumps({'saved':rows}))
 assert complete_history(paths,server='s3')['candidate_steps']==list(range(1000,50001,1000))
 try:complete_history(paths[:-1],server='s3')
 except ValueError:pass
 else:raise AssertionError('incomplete50K candidates must fail')
 state['step']=49000;(folder/'run_state.json').write_text(json.dumps(state))
 try:complete_history(paths,server='s3')
 except ValueError:pass
 else:raise AssertionError('49K donor must fail')
def row(r50,r90,er,step):return dict(validation=dict(RRE50=r50,RRE90=r90,Val_ERGAS=er),completed_update=step)
assert min([row(.2,.4,3,44000),row(.1,.3,4,37000)],key=rank)['completed_update']==37000
assert min([row(.1,.3,4,44000),row(.1,.3,3,22000)],key=rank)['completed_update']==22000
''')

    def test_actual_adamw_update_and_exact_resume_all_four_settings(self):
        self.child('''
from panda_qb_deepen.adapter import install
p=install()
import tempfile,torch
from copy import deepcopy
from pathlib import Path
from panda_s1loss9.tests.test_loss_contract import sources,paired_donor,exact_one_image_bank,PINS
from panda_s1loss9.calibration import bind_donor
from panda_s1loss9.spec import MatureSpec
from panda_s1loss9.engine import build_fresh_stage2,advance,save_checkpoint,load_checkpoint
from panda_s1loss9.evaluation import configured_pipeline
from panda_stage2_condq.stream import TrainStream,StreamSpec
from donor30.artifacts import _same
torch.set_num_threads(1)
for case in p.CASES:
 with tempfile.TemporaryDirectory() as temp:
  reg=sources(temp,'QB',count=1);cfg=p.resolve_case('s3',case,'R1')
  cfg['training_spec'].update(effective_batch_size=1,microbatch_size=1)
  d=paired_donor(p.resolve_case('s3','S1_A200_E030'),reg);spec=MatureSpec(**cfg['stage2_spec']);d=bind_donor(d,spec)
  stats=exact_one_image_bank(d,spec,reg);cfg['donor_binding']=deepcopy(d.binding)
  t=build_fresh_stage2(cfg,d,stats,evaluator_pin=PINS['evaluator_pin'])
  configured_pipeline(t,reg.sources)
  assert all(g['weight_decay']==p.CASES[case]['weight_decay'] for g in t.optimizer.param_groups)
  ss=StreamSpec(cfg['stream_seeds']['rr_data'],cfg['stream_seeds']['rr_augmentation'],**cfg['augmentation'])
  stream=TrainStream(reg,ss,microbatch_size=1,effective_batch_size=1)
  result=advance(t,stream)
  assert result['loss']==result['hard_weighted'] and result['soft_weighted']==0
  path=Path(temp)/'resume.pt';saved=save_checkpoint(path,t,stream,**PINS)
  expected=advance(t,stream);state=deepcopy(t.state_dict())
  fresh=build_fresh_stage2(cfg,d,stats,evaluator_pin=PINS['evaluator_pin'])
  restream=TrainStream(reg,ss,microbatch_size=1,effective_batch_size=1)
  fresh,restream=load_checkpoint(path,saved['file_sha256'],fresh,restream,**PINS)
  assert _same(advance(fresh,restream),expected) and _same(fresh.state_dict(),state)
''')

    def test_launch_without_gf2_or_other_campaign_records_and_ignore_old_controller_pin(self):
        from panda_qb_deepen import enqueue as E
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); control = root / G.OUTPUT / 'launcher_s3'
            control.mkdir(parents=True)
            (root / 'gspread').mkdir(); (root / 'gspread/server.txt').write_text('s3')
            stale_pin = control / 'controller_pin.json'
            stale_pin.write_text('{"old_source": "not-current"}')
            output = root / G.OUTPUT / G.CAMPAIGN_ID / 's3/QB'
            publication = output / 'sheet_publish_panda_s1loss9/terminal_upload.json'
            publication.parent.mkdir(parents=True)
            publication.write_text(json.dumps({'readback_verified': True,
                                               'campaign_status': 'FINISHED_REGISTERED_QUEUE'}))
            identity = {'source_container': 'local-image', 'image': 'image', 'user': '1:1',
                        'working_dir': '/workspace', 'env': [], 'mounts': []}
            old_intent = {'campaign_id': G.CAMPAIGN_ID, 'registry_digest': 'old-unlaunched-plan',
                          'identity': identity, 'token': 'old'}
            (control / 'launch_intent.json').write_text(json.dumps(old_intent))
            running = {'Id': 'test-owned-container', 'State': {'Running': True}}
            exited = {'Id': 'test-owned-container', 'State':
                      {'Running': False, 'Status': 'exited', 'ExitCode': 0, 'OOMKilled': False}}
            completed = {'campaign_status': 'FINISHED_REGISTERED_QUEUE',
                         'completed_count': 9, 'early_stopped_count': 0}
            with patch.object(E, 'ROOT', root), patch.object(E, 'CONTROL', control), \
                 patch.object(sys, 'argv', ['enqueue']), \
                 patch.object(E, 'command', return_value=(['docker', 'test-only'], identity)), \
                 patch('panda_sensor_refine.launch.inspect_optional', side_effect=[None, running, exited]), \
                 patch.object(E, 'authenticate_container', side_effect=lambda value, *_: value), \
                 patch.object(E, 'ensure_publisher', return_value=123), \
                 patch.object(E, 'terminal', return_value=completed), \
                 patch('directshift48.__main__.require_idle_gpu') as idle, \
                 patch.object(E.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run, \
                 patch('panda_gf2_closeout.verify_retirement', side_effect=AssertionError('No GF2 dependency')):
                self.assertEqual(E.main(), 0)
            self.assertEqual(idle.call_count, 2)
            self.assertEqual([call.args[0][-1] for call in run.call_args_list], ['check', 'cost', 'run'])
            self.assertFalse((control / 'predecessor.json').exists())
            self.assertEqual(stale_pin.read_text(), '{"old_source": "not-current"}')
            history = list(control.glob('previous_launch_intent_*.json'))
            self.assertEqual(len(history), 1)
            self.assertEqual(json.loads(history[0].read_text()), old_intent)
            self.assertNotEqual(json.loads((control / 'launch_intent.json').read_text())['registry_digest'],
                                old_intent['registry_digest'])

    def test_wait_is_only_for_local_gpu_occupancy(self):
        from panda_qb_deepen import enqueue as E
        busy = RuntimeError('Active GPU compute workload; test')
        with patch('directshift48.__main__.require_idle_gpu', side_effect=[busy, None]) as idle, \
             patch.object(E.time, 'sleep') as sleep:
            status = []
            E.wait_for_gpu(True, lambda stage, **kw: status.append(stage))
        self.assertEqual(idle.call_count, 2)
        self.assertEqual(status, ['WAIT_LOCAL_GPU'])
        sleep.assert_called_once_with(30)
        for wait in (False, True):
            with patch('directshift48.__main__.require_idle_gpu',
                       side_effect=RuntimeError('Cannot verify idle GPU')) as idle, \
                 patch.object(E.time, 'sleep') as sleep:
                with self.assertRaisesRegex(RuntimeError, 'Cannot verify'):
                    E.wait_for_gpu(wait, lambda *a, **kw: None)
                sleep.assert_not_called()
        with patch('directshift48.__main__.require_idle_gpu', side_effect=busy), \
             patch.object(E.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'Active GPU'):
                E.wait_for_gpu(False, lambda *a, **kw: None)
            sleep.assert_not_called()

    def test_launcher_adds_actual_qb_symlink_targets(self):
        from panda_qb_deepen import enqueue as E
        source = {'Image': 'image', 'Config': {'User':'1:1','WorkingDir':'/workspace','Env':[]},
            'Mounts':[{'Type':'bind','Source':str(E.ROOT),'Destination':'/workspace','RW':False}]}
        with patch('panda_sensor_refine.launch.source_container', return_value=('old-gf2',source)), \
             patch('edgeres48.docker_mounts.targets', return_value=[Path('/data/qb_train.h5')]):
            command, ident = E.command()
        self.assertIn('type=bind,src=/data/qb_train.h5,dst=/data/qb_train.h5,readonly',command)
        self.assertIn({'Source':'/data/qb_train.h5','Destination':'/data/qb_train.h5','RW':False},ident['mounts'])


if __name__ == '__main__': unittest.main()
