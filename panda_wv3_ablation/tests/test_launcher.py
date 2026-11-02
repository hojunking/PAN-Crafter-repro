import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from panda_wv3_ablation import enqueue,plan


class LauncherTests(unittest.TestCase):
    def test_s3_uses_wv3_mounts_and_does_not_start_a_predecessor(self):
        image={'Id':'sha256:fake','Config':{'Env':[]}}
        with patch('panda_wv3_ablation.enqueue.subprocess.run',return_value=SimpleNamespace(stdout=json.dumps([image]))), \
             patch('edgeres48.docker_mounts.targets',return_value=[Path('/data/WV3/train.h5')]) as mounts, \
             patch('panda_wv3_ablation.enqueue.release_check',return_value='a'*64):
            cmd=enqueue.command('s3','local:test',name='synthetic-check',gpu=False)
        mounts.assert_called_once_with(plan.ROOT,'s1')
        self.assertIn('panda.server=s3',cmd)
        self.assertNotIn('--gpus',cmd)
        self.assertTrue(any('WV3/train.h5' in s for s in cmd))
        self.assertFalse(any('GF2' in s for s in cmd))
        self.assertTrue(any('panda_wv3_ablation' in s for s in cmd))
