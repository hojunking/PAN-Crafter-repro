"""Stubbed launch tests: no Docker, GPU, credential read or cloud writes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from panda_warp30 import plan
ROOT=Path(__file__).resolve().parents[2]

class ShellLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.bin=self.root/'bin';self.bin.mkdir()
        (self.root/'scripts').mkdir();(self.root/'gspread').mkdir();(self.root/'work_dir').mkdir()
        (self.root/'gspread/server.txt').write_text('s3\n')
        for name in ('run_panda_warp30.sh','run_panda_warp30_docker.sh'):
            shutil.copyfile(ROOT/'scripts'/name,self.root/'scripts'/name)
        self.capture=self.root/'calls.jsonl'
        self.data=self.root/'registered.h5';self.data.touch()
        dlpan=self.root/'DLPan';wald=dlpan/'01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py'
        wald.parent.mkdir(parents=True);wald.touch()
        self.env={**os.environ,'PATH':str(self.bin)+':'+os.environ['PATH'],
            'MID_CAPTURE':str(self.capture),'MID_MOUNT':str(self.data),'MID_BUSY':'0',
            'MID_EXISTS':'0','MID_PREFLIGHT_FAIL':'0','MID_START_FAIL':'0',
            'MID_CONTRACT_BAD':'0',
            'PAN_WARP30_IMAGE':'local:s3-unchanged','PANCRAFTER_DLPAN':str(dlpan)}
        self.executable('python-stub','''import json,os,sys
args=sys.argv[1:]
with open(os.environ['MID_CAPTURE'],'a') as f: f.write(json.dumps(['python']+args)+'\\n')
if args[:3]==['-m','panda_warp30','mounts']: print(os.environ['MID_MOUNT'])
if args and args[0]=='-c' and 'require_idle_gpu' in args[1]: sys.exit(int(os.environ['MID_BUSY']))
if args and args[0]=='-c' and 'from panda_warp30.contract import CAMPAIGN_ID' in args[1]:
    print('' if os.environ['MID_CONTRACT_BAD']=='1' else 'PANDA_S3_WARP_STAGE12_30H_20261019_v1')
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
        return subprocess.run(['bash',str(self.root/'scripts/run_panda_warp30_docker.sh'),'s3'],
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
        self.assertIn('panda-warp30-wv3-s3-v1',launch)
        self.assertIn(f'type=bind,src={self.root},dst=/workspace,readonly',launch)
        self.assertIn(f'type=bind,src={self.root}/work_dir,dst=/workspace/work_dir',launch)
        self.assertEqual(launch[-3:],['bash','scripts/run_panda_warp30.sh','s3'])
        publisher = next(c for c in calls if c[0] == 'nohup')
        self.assertEqual(publisher[publisher.index('--root')+1],
            str(self.root/'work_dir/panda_warp30'/plan.CAMPAIGN_ID/'s3/WV3'))
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
