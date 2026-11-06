from copy import deepcopy
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from panda_gf2_sharpness import plan as P, audit
from panda_gf2_sharpness.replay import matching_predictions,resume_point
from panda_sensor_visuals.common import write


class ReviewTests(unittest.TestCase):
    def test_full_original_configs_schedule_and_frozen_training_resume(self):
        runner=Path(__file__).with_name('training_probe.py')
        for family in ('focus','psnr'):
            with self.subTest(family=family):
                result=subprocess.run([sys.executable,str(runner),family],text=True,capture_output=True,
                    env={**os.environ,'PYTHONPATH':str(P.ROOT)},timeout=180)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)

    def test_raw_replay_must_match_both_splits_and_scene_order(self):
        original={s:{'per_scene':[dict(scene_id=f'GF2:{s}:{i}',raw_prediction_sha256=str(i)) for i in range(20)]} for s in ('rr','fr')}
        self.assertTrue(matching_predictions(deepcopy(original),original))
        for split in ('rr','fr'):
            bad=deepcopy(original);bad[split]['per_scene'][7]['raw_prediction_sha256']='changed'
            self.assertFalse(matching_predictions(bad,original))
        bad=deepcopy(original);bad['fr']['per_scene'].reverse()
        self.assertFalse(matching_predictions(bad,original))

    def test_inventory_does_not_retrain_invalid_existing_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);case=P.cases()[0];folder=root/case['original_directory'];folder.mkdir(parents=True)
            write(folder/'config.json',{})
            for step in P.EARLY:write(folder/f'evaluation_{step:07d}.json',{})
            with patch.object(P,'cases',return_value=[case]),patch.object(P,'validate_case'):
                value=audit.inspect(root);self.assertTrue(value['all_sources_ready'])
                self.assertEqual(value['rows'][0]['missing_early'],list(P.EARLY))
                self.assertTrue(value['rows'][0]['replay_required'])
                (folder/'models').mkdir();(folder/'models/step_0001000.pt').write_bytes(b'invalid')
                with patch.object(audit,'checkpoint_evidence',side_effect=ValueError('bad saved identity')):
                    with self.assertRaisesRegex(ValueError,'bad saved identity'):audit.inspect(root)

    def test_orphan_checkpoint_is_discoverable_without_latest(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'models').mkdir()
            self.assertIsNone(resume_point(folder))
            for step in (500,1000):(folder/'models'/f'step_{step:07d}.pt').write_bytes(str(step).encode())
            saved=resume_point(folder);self.assertEqual(saved['completed_update'],1000)
            write(folder/'models/step_0001000.json',{'artifact':{'file_sha256':'different'}})
            with self.assertRaisesRegex(ValueError,'changed since saving'):resume_point(folder)

    def test_launcher_mounts_originals_read_only_and_uses_separate_output(self):
        from panda_gf2_sharpness import enqueue
        from panda_gf2_w128 import enqueue as base
        def inspect(kind,name,**kwargs):
            return None if kind=='container' else {'Id':'sha256:testimage','Config':{'Env':[]}}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'gspread').mkdir();(root/'gspread/server.txt').write_text('s3')
            for family in ('panda_sensor5','panda_sensor_focus','panda_gf2_psnr'):
                (root/'work_dir'/family).mkdir(parents=True)
            with patch.object(enqueue,'ROOT',root),patch.object(base,'ROOT',root),patch.object(base,'inspect',side_effect=inspect),\
                 patch.object(enqueue,'verify',return_value={'source_sha256':'a'*64}),patch.object(base,'wait_for_gpu') as wait,\
                 patch('edgeres48.docker_mounts.targets',return_value=[]),\
                 patch.object(enqueue.subprocess,'run',return_value=SimpleNamespace(stdout='container-id\n')) as run:
                value=enqueue.launch('image',wait=True)
                cmd=run.call_args.args[0]
                self.assertEqual(cmd[-6:],['sha256:testimage','python','-u','-m','panda_gf2_sharpness','run'])
                mounts=[cmd[i+1] for i,token in enumerate(cmd) if token=='--mount']
                writable=[m for m in mounts if not m.endswith(',readonly')]
                self.assertEqual(len(writable),1);self.assertIn(P.OUTPUT,writable[0])
                for family in ('panda_sensor5','panda_sensor_focus','panda_gf2_psnr'):
                    self.assertTrue(any(f'/work_dir/{family},readonly' in m for m in mounts))
                wait.assert_called_once_with(True);self.assertEqual(value['status'],'STARTED')


if __name__=='__main__':unittest.main()
