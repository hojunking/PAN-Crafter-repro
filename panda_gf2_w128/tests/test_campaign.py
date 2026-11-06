import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from panda_gf2_w128 import registry as G
from panda_gf2_w128.publication import three_metric_pass, annotate


class CampaignTests(unittest.TestCase):
    def process(self, code):
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_six_recipes_two_rounds_and_target_boundaries(self):
        self.assertEqual(len(G.cases()), 6)
        self.assertEqual(len(G.RUNS), 12)
        self.assertEqual([r for _, r in G.RUNS], ['R1'] * 6 + ['R2'] * 6)
        self.assertEqual({(p['lr_S'], p['alpha'], p['gamma']) for p in G.cases().values()},
                         {(lr, a, g) for lr in (1.5e-4, 2e-4, 2.5e-4) for a, g in ((.25, .5), (.125, .25))})
        for p in G.cases().values():
            self.assertEqual((p['student_width'], p['total_updates'], p['scale']), (128, 60000, 1.))
        self.assertTrue(three_metric_pass(.9645, .522, 45.08))
        self.assertFalse(three_metric_pass(.9644999, .521, 45.1))
        self.assertFalse(three_metric_pass(.97, .5220001, 45.1))
        self.assertFalse(three_metric_pass(.97, .52, 45.07999))

    def test_all_configs_and_no_shared_process_mutation(self):
        self.process('''
from panda_gf2_w128.adapter import install
p=install()
assert install() is p
from panda_gf2_w128 import registry as G
from panda_sensor_focus import reporting as R
from donor30.plan import canonical_sha256
for c,r in G.RUNS:
 cfg=p.resolve_case('s3',c,r); R._config(cfg)
 assert cfg['config_digest']==canonical_sha256(p.default_config('s3'))
 assert cfg['joint_target']==G.TARGET and cfg['psnr_min']==45.08
 assert cfg['student_input']['channels']==6 and cfg['student_architecture']['bands']==4
 assert cfg['stage2_spec']['beta']==0 and cfg['shift_scale']==1
 assert cfg['stage1_retrained'] is False and cfg['initialization']=='FRESH'
 assert cfg['fixed_donor_source']['selected_step']==42000
 assert cfg['planned_horizon']==60000 and len(cfg['evaluation_steps'])==60
 assert cfg['comparison_control']['seed']==p.SEEDS[r]
 assert cfg['run_id'].startswith('PSFC_GF2_S3_G')
try: p.resolve_case('s1','G1_LR200')
except ValueError: pass
else: raise AssertionError('Wrong server accepted')
''')
        from panda_sensor_focus.contract import CAMPAIGN_ID
        self.assertEqual(CAMPAIGN_ID, 'PANDA_S13_SENSOR_FOCUS_FIXEDDONOR_20261029_v1')
        self.process('''
import panda_sensor_focus.plan
from panda_gf2_w128.adapter import install
try: install()
except RuntimeError: pass
else: raise AssertionError('Shared training process accepted')
''')

    def test_six_scientific_graphs_frozen_donor_and_exact_resume(self):
        self.process('''
from panda_gf2_w128.adapter import install
p=install()
import tempfile,torch
from copy import deepcopy
from pathlib import Path
from panda_sensor_focus.tests.test_core import sources,donor,bank,PINS
from panda_sensor_focus.calibration import bind_donor
from panda_sensor_focus.spec import MatureSpec
from panda_sensor_focus.engine import build_fresh_stage2,advance,save_checkpoint,load_checkpoint
from panda_stage2_condq.stream import TrainStream,StreamSpec
from panda_method.calibration import state_dict_sha256
from panda_sensor_focus.cost.profiling import registered_profiles
from donor30.artifacts import _same
torch.set_num_threads(1)
profiles=registered_profiles('s3')
assert {c['student_architecture']['width'] for c,_ in profiles}=={128}
for case in p.CASES:
 with tempfile.TemporaryDirectory() as temp:
  reg=sources(temp,'GF2',count=1);cfg=p.resolve_case('s3',case,'R1')
  cfg['training_spec'].update(effective_batch_size=1,microbatch_size=1)
  d=donor('s3',reg);spec=MatureSpec(**cfg['stage2_spec']);d=bind_donor(d,spec)
  stats=bank(d,spec,reg);cfg['donor_binding']=deepcopy(d.binding)
  t=build_fresh_stage2(cfg,d,stats,evaluator_pin=PINS['evaluator_pin'])
  before=[state_dict_sha256(t.stage.aligner),state_dict_sha256(t.stage.teacher)]
  assert t.stage.student.backbone.input.in_channels==6
  assert t.stage.student.spec.width==128 and t.stage.teacher.spec.width==112
  assert sum(x.numel() for x in t.stage.student.parameters())==cfg['student_sampling']['student_parameters']
  ss=StreamSpec(cfg['stream_seeds']['rr_data'],cfg['stream_seeds']['rr_augmentation'],**cfg['augmentation'])
  stream=TrainStream(reg,ss,microbatch_size=1,effective_batch_size=1)
  result=advance(t,stream)
  assert result['loss']==result['hard_weighted'] and result['soft_weighted']==0
  assert before==[state_dict_sha256(t.stage.aligner),state_dict_sha256(t.stage.teacher)]
  path=Path(temp)/'resume.pt';saved=save_checkpoint(path,t,stream,**PINS)
  expected=advance(t,stream);state=deepcopy(t.state_dict())
  fresh=build_fresh_stage2(cfg,d,stats,evaluator_pin=PINS['evaluator_pin'])
  restream=TrainStream(reg,ss,microbatch_size=1,effective_batch_size=1)
  fresh,restream=load_checkpoint(path,saved['file_sha256'],fresh,restream,**PINS)
  assert _same(advance(fresh,restream),expected) and _same(fresh.state_dict(),state)
from panda_gf2_w128.publication import install as pub
pub();pub()
from panda_sensor_focus.best_publication import project
with tempfile.TemporaryDirectory() as temp:
 value=project(temp,'s3')
 assert not value['conflicts'] and not value['rows']
 assert value['additional_goal']['PSNR']==45.08
''')

    def test_only_new_output_mount_writable(self):
        from panda_gf2_w128 import enqueue
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'work_dir/panda_sensor5').mkdir(parents=True)
            with patch.object(enqueue, 'ROOT', root), \
                 patch.object(enqueue, 'inspect', return_value={'Id': 'sha256:test', 'Config': {'Env': ['PATH=/bin']}}), \
                 patch('edgeres48.docker_mounts.targets', return_value=[]):
                cmd, identity = enqueue.command('test', gpu=False)
            writable = [m['Destination'] for m in identity['mounts'] if m['RW']]
            self.assertEqual(writable, ['/workspace/' + G.OUTPUT])
            self.assertNotIn('--gpus', cmd)

    def test_publication_keeps_historical_selector_and_notes_goal(self):
        from panda_sensor_focus import reporting as R
        row = [''] * 64
        for k,v in {'HQNR':.97, 'ERGAS':.52, 'PSNR':45.07999,
                    'Selection@Step':'대표 · 조건 충족', 'Result_ID':'stable',
                    'Selector':'HQNR_QUALIFIED_MIN_ERGAS'}.items():
            row[R.INDEX[R.METRICS.get(k,k)]] = v
        row[R.INDEX['Original description / notes']] = json.dumps({'publication':{'publication_joint_pass':True}})
        value=annotate({'rows':[row],'notes':['{}']},R)
        note=json.loads(value['rows'][0][R.INDEX['Original description / notes']])
        self.assertFalse(note['publication']['three_metric_target_pass'])
        self.assertEqual(value['rows'][0][R.INDEX['Result_ID']],'stable')
        self.assertEqual(value['rows'][0][R.INDEX['Selector']],'HQNR_QUALIFIED_MIN_ERGAS')


if __name__ == '__main__': unittest.main()
