"""Offline campaign/reporting/launcher contracts; no real GPU, Sheets or SSH."""
from copy import deepcopy
from contextlib import redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from panda_midpoint import plan, reporting as R, publication as P
from panda_midpoint.__main__ import main
from panda30r.tests.test_reporting import evaluation

ROOT = Path(__file__).resolve().parents[2]


def config():
    return plan.resolve_case('s3', 'MID_B005')


def persist(root, cfg, step=1000, h=.959, e=2.2):
    ev = evaluation(cfg, step, h, e)
    path = Path(root)/'runs'/cfg['run_id']/'attempt_01'
    path.mkdir(parents=True, exist_ok=True)
    R.atomic_json(path/'config.json', cfg)
    R.atomic_json(path/f'evaluation_{step:07d}.json', ev)
    return R.emit_evaluation(root, cfg, ev, timestamp='2026-10-19T00:00:00+00:00')


class PlanTests(unittest.TestCase):
    def test_default_is_exactly_three_cases_one_paired_seed(self):
        cfg = plan.default_config('s3')
        plan.validate_config(cfg)
        rows = list(csv.DictReader(io.StringIO(plan.catalog_csv('s3'))))
        self.assertEqual(len(rows), 3)
        self.assertEqual([row['case'] for row in rows], ['MID_B005', 'MID_B010', 'MID_B020'])
        self.assertEqual({row['seed'] for row in rows}, {'261018101'})
        self.assertEqual([float(row['beta']) for row in rows], [.05, .1, .2])
        self.assertNotIn('budget', cfg)
        self.assertFalse(cfg['allow_fresh_donor_fallback'])
        self.assertFalse(cfg['allow_cross_server_donor'])

    def test_interp23_version_identity_and_source_pins(self):
        from panda_midpoint import calibration, runtime
        from panda_midpoint.contract import UPSAMPLING
        from panda_midpoint.resampling import UPSAMPLER_VERSION
        cfg = config()
        self.assertTrue(cfg['campaign_id'].endswith('_v2'))
        self.assertTrue(cfg['run_id'].startswith('PMID_U23_'))
        self.assertIn('U23', cfg['run_display'])
        self.assertEqual(cfg['geometry']['student_upsampling'], UPSAMPLING)
        self.assertEqual(UPSAMPLING['schema'], UPSAMPLER_VERSION)
        for pins in (runtime.code_pin(), calibration.code_pin()):
            self.assertIn('align/resample.py', pins)
            self.assertIn('panda_midpoint/resampling.py', pins)

    def test_coefficients_are_only_loss_difference(self):
        cases = [plan.resolve_case('s3', name) for name in plan.CASES]
        for cfg in cases:
            self.assertEqual(cfg['student_architecture'], {'bands':8, 'width':112, 'depth':[1,2,3]})
            self.assertEqual(cfg['stage2_spec']['alpha'], .25)
            self.assertEqual(cfg['stage2_spec']['gamma'], .5)
            self.assertEqual(cfg['planned_horizon'], 50000)
            self.assertEqual(cfg['evaluation_steps'][-1], 50000)
            self.assertEqual(cfg['augmentation'], cases[0]['augmentation'])
            self.assertEqual(cfg['training_spec'], cases[0]['training_spec'])
            self.assertEqual(cfg['stream_seeds'], cases[0]['stream_seeds'])
            self.assertEqual(cfg['geometry']['pan_shift_factor'], .5)
            self.assertEqual(cfg['geometry']['ms_shift_factor'], -.5)
        self.assertIn('interpolation', cases[1]['question'])

    def test_optional_second_seed_requires_explicit_request(self):
        self.assertEqual(sum(len(g['runs']) for g in plan.groups('s3')), 3)
        self.assertEqual(sum(len(g['runs']) for g in plan.groups('s3', ('R1','R2'))), 6)
        self.assertEqual(plan.resolve_case('s3','MID_B005','R2',replicas=('R1','R2'))['seed'],261018102)
        with self.assertRaises(ValueError): plan.resolve_case('s3','MID_B005','R2')
        for replicas in ((), ('R1','R1'), ('R3',)):
            with self.assertRaises(ValueError): plan.groups('s3', replicas)
        for server in ('s1','s2','s4','s5'):
            with self.assertRaises(ValueError): plan.default_config(server)

    def test_explicit_replicas_bind_actual_campaign_digest_and_evidence(self):
        for replicas in (('R1',), ('R2',), ('R1','R2')):
            campaign=plan.default_config('s3',replicas)
            expected=plan.canonical_sha256(campaign)
            for replica in replicas:
                cfg=plan.resolve_case('s3','MID_B005',replica,replicas=replicas)
                self.assertEqual(cfg['config_digest'],expected)
                self.assertEqual(cfg['evidence_level'],
                    'PAIRED_ONE_SEED_SCREEN' if len(replicas)==1 else 'PAIRED_TWO_SEEDS')
            rows=list(csv.DictReader(io.StringIO(plan.catalog_csv('s3',replicas))))
            self.assertEqual(len(rows),3*len(replicas))
            self.assertEqual({row['replica'] for row in rows},set(replicas))
        one=plan.resolve_case('s3','MID_B005')
        two=plan.resolve_case('s3','MID_B005',replicas=('R1','R2'))
        self.assertNotEqual(one['config_digest'],two['config_digest'])

    def test_original_checkpoint_policy_is_preserved(self):
        a = {'HQNR':.9591,'ERGAS':2.10,'completed_update':1000}
        b = {'HQNR':.961,'ERGAS':2.20,'completed_update':2000}
        self.assertGreater(plan.rank(a), plan.rank(b))
        a['HQNR']=.9589
        self.assertLess(plan.rank(a), plan.rank(b))


class ReportingTests(unittest.TestCase):
    def test_namespaced_immutable_rows_all_metrics_and_geometry(self):
        cfg = config()
        record = R.make_record(cfg,evaluation(cfg))
        R.validate_record(record)
        self.assertTrue(record['record_id'].startswith('PMID-'))
        self.assertEqual(len(record['row']),64)
        self.assertEqual(record['destination']['sheet_title'],'WV3-ablations')
        self.assertEqual(record['row'][R.INDEX['Server']],'s3')
        for col in ('RMSE↓','CC↑','JQM↑'):
            self.assertIsInstance(record['row'][R.INDEX[col]],(int,float))
        note=json.loads(record['row'][R.INDEX['Original description / notes']])
        self.assertTrue(note['official_references_unchanged'])
        self.assertEqual(note['output_coordinate_system'],'MIDPOINT_NO_OUTPUT_RETURN_WARP')
        for key,val in (('geometry',{}),('engine_kind','stage2_condq'),('server','s1')):
            bad=deepcopy(cfg);bad[key]=val
            with self.assertRaises(ValueError): R.make_record(bad,evaluation(bad))

    def test_projection_keeps_old_display_gate_but_all_raw_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=config()
            record=persist(root,cfg,1000,.9591,2.1)
            R.LocalOutbox(root).enqueue(record)
            persist(root,cfg,2000,.961,2.4)
            value=P.project_campaign(root,'s3')
            self.assertFalse(value['conflicts'])
            self.assertEqual(value['selected'][0]['step'],2000)
            persist(root,cfg,3000,.960,2.045)
            persist(root,cfg,4000,.961,2.04)
            value=P.project_campaign(root,'s3')
            self.assertFalse(value['conflicts'])
            self.assertEqual([r['step'] for r in value['selected']],[3000,4000])
            self.assertEqual(len(list((Path(root)/'outbox').glob('*.json'))),4)
            self.assertEqual(len((Path(root)/'records.jsonl').read_text().splitlines()),4)

    def test_tampered_partial_or_synthetic_evidence_cannot_publish(self):
        for kind in ('partial','synthetic','reference_identity'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                persist(root,config())
                path=next(Path(root).glob('runs/*/attempt_01/evaluation_*.json'))
                ev=json.loads(path.read_text())
                if kind=='partial': ev['fr']['per_scene'].pop()
                elif kind=='synthetic': ev['metric_function_injection']=True
                else: ev['state_after']['inference_policy_digest']='b'*64
                R.atomic_json(path,ev)
                self.assertTrue(P.project_campaign(root,'s3')['conflicts'])

    def test_cost_missing_blocks_only_cloud_not_local_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root,config())
            self.assertTrue(P.project(root,'s3')['conflicts'])
            self.assertFalse(P.project(root,'s3',require_cost=False)['conflicts'])

    def test_safe_stop_is_terminal_only_after_evidence_readback(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root,config())
            R.atomic_json(Path(root)/'campaign_state.json',{'status':'STOPPED_SAFE'})
            before=P._terminal_state(root)
            value=P.project_campaign(root,'s3')
            self.assertIsNotNone(before)
            self.assertFalse(P.terminal_upload_complete(root,value,{'readback_verified':False},before))
            self.assertTrue(P.terminal_upload_complete(root,value,{'readback_verified':True},before))
            R.atomic_json(Path(root)/'campaign_state.json',{'status':'RUNNING'})
            self.assertFalse(P.terminal_upload_complete(root,value,{'readback_verified':True},before))


class CLITests(unittest.TestCase):
    def test_catalog_and_wrong_server(self):
        output=io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(['catalog','--server','s3','--csv']),0)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(output.getvalue())))),3)
        with patch('sys.stderr',new=io.StringIO()):
            with self.assertRaises(SystemExit): main(['run','--server','s1'])

    def test_mount_helper_uses_wv3_lane(self):
        with patch('edgeres48.docker_mounts.targets',return_value=['/data/wv3.h5']) as target:
            with redirect_stdout(io.StringIO()): self.assertEqual(main(['mounts']),0)
        target.assert_called_once_with(ROOT,'s1')

    def test_publication_import_needs_no_torch(self):
        code='''
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','numpy','h5py','google'):
            raise AssertionError('Unexpected dependency: '+fullname)
sys.meta_path.insert(0,Reject())
import panda_midpoint.publication, panda_midpoint.cost
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,text=True,capture_output=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_shell_syntax_no_destructive_commands_and_s3_identity(self):
        for name in ('run_panda_midpoint.sh','run_panda_midpoint_docker.sh'):
            path=ROOT/'scripts'/name
            subprocess.run(['bash','-n',str(path)],check=True)
            source=path.read_text()
            for forbidden in ('docker stop','docker rm','pkill','killall','--replace'):
                self.assertNotIn(forbidden,source)
            result=subprocess.run(['bash',str(path),'s1'],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,2)
            # The checkout is s1: an explicit s3 argument still must not launch.
            if (ROOT/'gspread/server.txt').read_text().strip()!='s3':
                result=subprocess.run(['bash',str(path),'s3'],capture_output=True,text=True,timeout=10)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('server marker',result.stderr)


class CostTests(unittest.TestCase):
    def test_cost_signature_is_new_geometry_not_loss_or_seed(self):
        from panda_midpoint import cost
        base=cost.signature(config())
        for case in plan.CASES:
            self.assertEqual(base,cost.signature(plan.resolve_case('s3',case,'R2',replicas=('R1','R2'))))
        self.assertIn('geometry',base)
        self.assertIn('panda_midpoint/method.py',base['source_sha256'])
        self.assertIn('align/resample.py',base['source_sha256'])
        self.assertIn('panda_midpoint/resampling.py',base['source_sha256'])
        self.assertEqual(base['base'], 'INTERP23_MS_POST_AUG_PHASE_CORRECTED')
        self.assertEqual(base['reconstructor'],{'bands':8,'width':112,'depth':[1,2,3]})

    def test_deployment_has_no_kd_teacher_and_runs_two_warps(self):
        import torch
        from panda_midpoint import cost, method
        from panda_stage2_condq.tests.test_training_evaluation import EvalSources, rr_batch
        graph=cost.build_deployment(config(),EvalSources())
        self.assertFalse(hasattr(graph,'student'))
        self.assertEqual(sum(p.numel() for p in graph.parameters()),
            sum(p.numel() for p in graph.aligner.parameters())+
            sum(p.numel() for p in graph.teacher.parameters()))
        with torch.no_grad():
            graph.aligner.fc2.bias.copy_(torch.tensor([.1,-.2]))
            # Avoid the zero-initialized final-layer equivalence trap.
            for p in graph.teacher.parameters():
                if not bool(torch.any(p)): p.normal_(std=.001)
        batch=rr_batch()
        with (patch.object(method,'warp_pan',wraps=method.warp_pan) as warp,
              patch.object(method, 'upsample_lrms_after_augmentation',
                           wraps=method.upsample_lrms_after_augmentation) as up):
            result=graph.native_prediction(batch)
        self.assertEqual(warp.call_count,2)
        self.assertEqual(up.call_count,1)
        expected_base = method.upsample_lrms_after_augmentation(batch.ms, batch.lineage.augmentations)
        torch.testing.assert_close(result['ms_base'], method.warp_pan(expected_base, result['u']*-.5))
        torch.testing.assert_close(warp.call_args_list[0].args[1],result['u']*.5)
        torch.testing.assert_close(warp.call_args_list[1].args[1],result['u']*-.5)


class ShellLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.bin=self.root/'bin';self.bin.mkdir()
        (self.root/'scripts').mkdir();(self.root/'gspread').mkdir();(self.root/'work_dir').mkdir()
        (self.root/'gspread/server.txt').write_text('s3\n')
        for name in ('run_panda_midpoint.sh','run_panda_midpoint_docker.sh'):
            shutil.copyfile(ROOT/'scripts'/name,self.root/'scripts'/name)
        self.capture=self.root/'calls.jsonl'
        self.data=self.root/'registered.h5';self.data.touch()
        dlpan=self.root/'DLPan';wald=dlpan/'01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py'
        wald.parent.mkdir(parents=True);wald.touch()
        self.env={**os.environ,'PATH':str(self.bin)+':'+os.environ['PATH'],
            'MID_CAPTURE':str(self.capture),'MID_MOUNT':str(self.data),'MID_BUSY':'0',
            'MID_EXISTS':'0','MID_PREFLIGHT_FAIL':'0','MID_START_FAIL':'0',
            'MID_CONTRACT_BAD':'0',
            'PAN_MIDPOINT_IMAGE':'local:s3-unchanged','PANCRAFTER_DLPAN':str(dlpan)}
        self.executable('python-stub','''import json,os,sys
args=sys.argv[1:]
with open(os.environ['MID_CAPTURE'],'a') as f: f.write(json.dumps(['python']+args)+'\\n')
if args[:3]==['-m','panda_midpoint','mounts']: print(os.environ['MID_MOUNT'])
if args and args[0]=='-c' and 'require_idle_gpu' in args[1]: sys.exit(int(os.environ['MID_BUSY']))
if args and args[0]=='-c' and 'from panda_midpoint.contract import CAMPAIGN_ID' in args[1]:
    print('' if os.environ['MID_CONTRACT_BAD']=='1' else 'PANDA_S3_MIDPOINT_STAGE2_20261019_v2')
''')
        self.env['PAN_HOST_PYTHON']=self.env['PAN_PYTHON']=str(self.bin/'python-stub')
        self.executable('docker','''import json,os,sys
args=sys.argv[1:]
with open(os.environ['MID_CAPTURE'],'a') as f: f.write(json.dumps(['docker']+args)+'\\n')
if args[:2]==['container','inspect']: sys.exit(0 if os.environ['MID_EXISTS']=='1' else 1)
if args[:2]==['run','--rm']: sys.exit(int(os.environ['MID_PREFLIGHT_FAIL']))
if args[:2]==['run','--detach']: sys.exit(int(os.environ['MID_START_FAIL']))
''')
        self.executable('nohup','''import json,os,sys
with open(os.environ['MID_CAPTURE'],'a') as f: f.write(json.dumps(['nohup']+sys.argv[1:])+'\\n')
''')

    def executable(self,name,source):
        path=self.bin/name;path.write_text('#!'+sys.executable+'\n'+source);path.chmod(0o700)

    def calls(self):
        return [json.loads(x) for x in self.capture.read_text().splitlines()] if self.capture.exists() else []

    def launch(self):
        return subprocess.run(['bash',str(self.root/'scripts/run_panda_midpoint_docker.sh'),'s3'],
            env=self.env,text=True,capture_output=True,timeout=20)

    def test_exact_image_mounts_and_preflight_before_training(self):
        result=self.launch();self.assertEqual(result.returncode,0,result.stderr)
        # The launcher intentionally detaches the publisher. Wait for the stub's
        # receipt rather than racing its process startup against shell exit.
        for _ in range(100):
            if any(c[0] == 'nohup' for c in self.calls()):
                break
            time.sleep(.01)
        calls=self.calls();runs=[c for c in calls if c[:2]==['docker','run']]
        self.assertEqual(len(runs),2)
        preflight,launch=runs
        self.assertIn('--rm',preflight);self.assertNotIn('--gpus',preflight)
        self.assertIn('--detach',launch);self.assertIn('--gpus',launch)
        self.assertIn('local:s3-unchanged',launch)
        self.assertIn('panda-midpoint-wv3-s3-v2',launch)
        self.assertIn(f'type=bind,src={self.root},dst=/workspace,readonly',launch)
        self.assertIn(f'type=bind,src={self.root}/work_dir,dst=/workspace/work_dir',launch)
        self.assertEqual(launch[-3:],['bash','scripts/run_panda_midpoint.sh','s3'])
        publisher = next(c for c in calls if c[0] == 'nohup')
        self.assertEqual(publisher[publisher.index('--root')+1],
            str(self.root/'work_dir/panda_midpoint'/plan.CAMPAIGN_ID/'s3/WV3'))
        gpu=next(i for i,c in enumerate(calls) if 'require_idle_gpu' in str(c))
        self.assertLess(calls.index(preflight),gpu);self.assertLess(gpu,calls.index(launch))

    def test_existing_or_busy_or_failed_preflight_never_interrupts_or_starts(self):
        for key in ('MID_EXISTS','MID_BUSY','MID_PREFLIGHT_FAIL','MID_CONTRACT_BAD'):
            self.env[key]='1';before=len(self.calls());result=self.launch()
            self.assertNotEqual(result.returncode,0,key)
            self.assertFalse(any(c[:3]==['docker','run','--detach'] for c in self.calls()[before:]))
            self.assertFalse(any(c[0]=='nohup' for c in self.calls()[before:]))
            self.env[key]='0'

    def test_failed_detached_start_does_not_spawn_watcher(self):
        self.env['MID_START_FAIL']='1';result=self.launch()
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(any(c[0]=='nohup' for c in self.calls()))


if __name__=='__main__': unittest.main()
