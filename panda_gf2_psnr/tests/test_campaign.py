import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from panda_gf2_psnr import registry as G
from panda_gf2_psnr.publication import three_metric_pass, annotate


class CampaignTests(unittest.TestCase):
    def test_exact_registry_and_single_knob_comparisons(self):
        self.assertEqual(len(G.RUNS), 8)
        self.assertEqual(sum(G.cases()[c]['total_updates'] for c,r in G.RUNS), 460000)
        for p in G.cases().values():
            self.assertEqual((p['scale'], p['gamma'], p['epsilon_radius_rr']), (1., .5, 3.))
            self.assertEqual(p['input_variant'], 'CORRECTED_DIFF')
            self.assertEqual(p['inherited_updates'], 0)
            self.assertEqual(p['lambda_abs'], 2.)
            self.assertEqual(p['lambda_epsilon'], .003)

    def test_process_isolation_valid_configs_and_new_pin(self):
        code = '''
import json
from panda_gf2_psnr.adapter import install
p=install()
assert install() is p
from panda_gf2_psnr import registry as G
from panda_sensor_focus import reporting as R
from panda_sensor_focus.runtime import code_pin
from donor30.plan import canonical_sha256
from panda_gf2_psnr.enqueue import source_pin
assert 'panda_gf2_psnr/registry.py' in code_pin()
assert source_pin()==code_pin()
for c,r in G.RUNS:
    cfg=p.resolve_case('s3',c,r)
    R._config(cfg)
    assert cfg['config_digest']==canonical_sha256(p.default_config('s3'))
    assert cfg['psnr_min']==45.08 and cfg['stage2_spec']['beta']==0
    assert cfg['joint_target']==G.TARGET
    assert cfg['initialization']=='FRESH'
    assert cfg['stage2_spec']['total_updates']==cfg['planned_horizon']
    assert cfg['fixed_donor_source']['selected_step']==42000
    assert cfg['evaluation_steps'][-1]==cfg['planned_horizon']
    assert cfg['stage1_retrained'] is False
    assert cfg['comparison_control']['same_exact_donor_checkpoint']
    assert cfg['comparison_control']['campaign_id']==G.OLD_CAMPAIGN
    assert cfg['comparison_control']['case']=='GF2_C60LR150'
    assert cfg['comparison_control']['replica']==r
    assert cfg['comparison_control']['seed']==p.SEEDS[r]
    assert cfg['comparison_control']['student_seed_differs_from_control'] is False
    assert cfg['fixed_donor_source']==p.SOURCE_DONORS['s3']
try: p.resolve_case('s1','QB_W128','R1')
except ValueError: pass
else: raise AssertionError('s1 must be rejected')
print('OK')
'''
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
        from panda_sensor_focus.contract import CAMPAIGN_ID, REGISTERED_RUNS
        self.assertEqual(CAMPAIGN_ID, G.OLD_CAMPAIGN)
        self.assertEqual(REGISTERED_RUNS['s3'], G.OLD_RUNS)

    def test_refuse_shared_training_process(self):
        code = '''
import panda_sensor_focus.plan
from panda_gf2_psnr.adapter import install
try: install()
except RuntimeError: pass
else: raise AssertionError('shared process must be rejected')
'''
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)

    def test_new_configs_train_and_resume_existing_method_on_cpu(self):
        code = '''
from panda_gf2_psnr.adapter import install
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
from panda_gf2_psnr.publication import install as pub
from donor30.artifacts import _same
torch.set_num_threads(1)
profiles=registered_profiles('s3')
assert {c['student_architecture']['width'] for c,_ in profiles}=={112,128}
for c,s in profiles: signature(c,s)
for case in p.CASES:
    with tempfile.TemporaryDirectory() as temp:
        reg=sources(temp,'GF2',count=1)
        cfg=p.resolve_case('s3',case,'R1')
        cfg['training_spec'].update(effective_batch_size=1,microbatch_size=1)
        d=donor('s3',reg);spec=MatureSpec(**cfg['stage2_spec']);d=bind_donor(d,spec)
        stats=bank(d,spec,reg);cfg['donor_binding']=deepcopy(d.binding)
        t=build_fresh_stage2(cfg,d,stats,evaluator_pin=PINS['evaluator_pin'])
        ss=StreamSpec(cfg['stream_seeds']['rr_data'],cfg['stream_seeds']['rr_augmentation'],**cfg['augmentation'])
        stream=TrainStream(reg,ss,microbatch_size=1,effective_batch_size=1)
        loss=advance(t,stream)
        assert loss['loss']==loss['hard_weighted'] and loss['soft_weighted']==0
        assert t.stage.student.backbone.input.in_channels==6
        assert t.stage.student.spec.width==cfg['student_architecture']['width']
        path=Path(temp)/'resume.pt';saved=save_checkpoint(path,t,stream,**PINS)
        expected=advance(t,stream);state=deepcopy(t.state_dict())
        fresh=build_fresh_stage2(cfg,d,stats,evaluator_pin=PINS['evaluator_pin'])
        restream=TrainStream(reg,ss,microbatch_size=1,effective_batch_size=1)
        fresh,restream=load_checkpoint(path,saved['file_sha256'],fresh,restream,**PINS)
        assert _same(advance(fresh,restream),expected) and _same(fresh.state_dict(),state)
pub()
from panda_sensor_focus.best_publication import project
with tempfile.TemporaryDirectory() as temp:
    value=project(temp,'s3')
    assert not value['conflicts'] and not value['rows']
    assert value['additional_goal']['PSNR']==45.08
'''
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)

    def test_targets_use_full_precision_inclusive_and_all_three(self):
        self.assertTrue(three_metric_pass(.9645, .522, 45.08))
        self.assertFalse(three_metric_pass(.9645, .522, 45.0799999))
        self.assertFalse(three_metric_pass(.9644999, .521, 45.2))
        self.assertFalse(three_metric_pass(.97, .5220001, 45.2))
        for p in (float('nan'), float('inf'), '45.08'):
            with self.assertRaises(ValueError): three_metric_pass(.97,.52,p)

    def test_annotation_keeps_historical_selection_and_row_identity(self):
        from panda_sensor_focus import reporting as R
        row = [''] * 64
        for key,value in {'HQNR':.97,'ERGAS':.52,'PSNR':45.035,'Selection@Step':'대표 · ERGAS 최소 · 조건 충족 · 47,000',
                          'Result_ID':'unchanged','Selector':'HQNR_QUALIFIED_MIN_ERGAS'}.items():
            row[R.INDEX[R.METRICS.get(key,key)]]=value
        row[R.INDEX['Original description / notes']]=json.dumps({'publication':{'publication_joint_pass':True}})
        value={'rows':[row], 'notes':[json.dumps({'publication':{}})], 'conflicts':[]}
        annotated=annotate(value,R)
        pub=json.loads(row[R.INDEX['Original description / notes']])['publication']
        self.assertTrue(pub['publication_joint_pass'])
        self.assertFalse(pub['three_metric_target_pass'])
        self.assertIn('3목표 미달',row[R.INDEX['Selection@Step']])
        self.assertEqual(row[R.INDEX['Result_ID']], 'unchanged')
        self.assertEqual(row[R.INDEX['Selector']], 'HQNR_QUALIFIED_MIN_ERGAS')
        self.assertEqual(len(annotated['rows']),1)

    def test_admission_waits_existing_wv3(self):
        from panda_gf2_psnr.prerequisites import evidence
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            path=root/'work_dir/panda_sensor_focus_launchers/s3/status.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'server':'s3','stage':'RESUME_HARD36','status':'RESUME_HARD36'}))
            with self.assertRaisesRegex(ValueError,'WAIT:'): evidence(root)

    def test_source_change_while_waiting_refuses_launch(self):
        from panda_gf2_psnr import enqueue
        original = {'panda_gf2_psnr/registry.py': 'original'}
        changed = {'panda_gf2_psnr/registry.py': 'changed'}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'gspread').mkdir()
            (root / 'gspread/server.txt').write_text('s3')
            control = root / 'launcher'
            with patch.object(enqueue, 'ROOT', root), patch.object(enqueue, 'CONTROL', control), \
                 patch.object(enqueue, 'source_pin', side_effect=[original, original, changed]), \
                 patch.object(enqueue, 'evidence', side_effect=ValueError('WAIT: original queue running')) as evidence, \
                 patch.object(enqueue.time, 'sleep') as sleep, \
                 patch.object(enqueue.subprocess, 'run') as launch, \
                 patch.object(sys, 'argv', ['enqueue', '--wait']):
                with self.assertRaisesRegex(RuntimeError, 'source changed'):
                    enqueue.main()
            evidence.assert_called_once_with(root)
            sleep.assert_called_once_with(30)
            launch.assert_not_called()
            self.assertEqual(json.loads((control / 'status.json').read_text())['stage'], 'BLOCKED_SOURCE_CHANGED')

    def test_existing_admission_refuses_changed_source(self):
        from panda_gf2_psnr import enqueue
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'gspread').mkdir()
            (root / 'gspread/server.txt').write_text('s3')
            admission = root / G.OUTPUT / G.CAMPAIGN_ID / 's3/GF2/admission.json'
            admission.parent.mkdir(parents=True)
            admission.write_text(json.dumps({'code': {'source.py': 'old'}}))
            with patch.object(enqueue, 'ROOT', root), patch.object(enqueue, 'CONTROL', root / 'launcher'), \
                 patch.object(enqueue, 'source_pin', return_value={'source.py': 'new'}), \
                 patch.object(enqueue, 'evidence') as evidence, \
                 patch.object(enqueue.subprocess, 'run') as launch, \
                 patch.object(sys, 'argv', ['enqueue', '--wait']):
                with self.assertRaisesRegex(ValueError, 'admission has different source'):
                    enqueue.main()
            evidence.assert_not_called()
            launch.assert_not_called()

    def test_only_new_output_is_writable_docker_mount(self):
        from panda_gf2_psnr.enqueue import command
        source={'Config':{'User':'1000:1000','WorkingDir':'/workspace','Env':['SAFE=1']},
                'Image':'sha256:fixed','Mounts':[
                    {'Type':'bind','Source':'/repo','Destination':'/workspace','RW':False},
                    {'Type':'bind','Source':'/old','Destination':'/workspace/work_dir/old','RW':True}]}
        with tempfile.TemporaryDirectory() as temp, patch('panda_gf2_psnr.enqueue.ROOT',Path(temp)), \
             patch('panda_sensor_refine.launch.source_container',return_value=('original',source)):
            cmd,_,_=command('test')
        mounts=[cmd[i+1] for i,v in enumerate(cmd) if v=='--mount']
        self.assertEqual(len(mounts),2)
        self.assertTrue(mounts[0].endswith(',readonly'))
        self.assertIn('/workspace/work_dir/panda_gf2_psnr',mounts[1])
        self.assertNotIn('/workspace/work_dir/old',' '.join(cmd))


if __name__ == '__main__':
    unittest.main()
