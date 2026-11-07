"""Portable launch isolation and ownership tests; no Docker or cloud side effects."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from panda_wv3_hyper import enqueue,plan


class LaunchTests(unittest.TestCase):
    def test_both_lanes_mount_wv3_and_only_new_output_is_writable(self):
        image={'Id':'sha256:example','Config':{'Env':['PATH=/opt/conda/bin']}}
        result=SimpleNamespace(stdout=json.dumps([image]),returncode=0)
        for server in ('s1','s3'):
            with patch.object(enqueue.subprocess,'run',return_value=result), \
                    patch('edgeres48.docker_mounts.targets',return_value=[Path('/data/WV3.h5')]) as targets, \
                    patch.object(enqueue,'verify',return_value={'source_sha256':'proof'}):
                command=enqueue.command(server,'test',detached=True)
            targets.assert_called_once_with(plan.ROOT,'s1')
            mounts=[command[i+1] for i,v in enumerate(command) if v=='--mount']
            self.assertEqual([m for m in mounts if not m.endswith(',readonly')],
                [f'type=bind,src={plan.ROOT}/work_dir/panda_wv3_hyper,dst=/workspace/work_dir/panda_wv3_hyper'])
            self.assertIn('panda.server='+server,command)
            self.assertIn(enqueue.container_name(server),command)
            self.assertEqual(command[-1],'sha256:example')

    def test_existing_container_cannot_switch_server_or_checkout(self):
        info={'Config':{'Labels':{'panda.campaign':plan.CAMPAIGN_ID,
            'panda.server':'s1','panda.source':'proof'}},
            'Mounts':[{'Destination':'/workspace','Source':str(plan.ROOT)}]}
        with patch.object(enqueue,'verify',return_value={'source_sha256':'proof'}):
            enqueue.verify_container(info,'s1')
            with self.assertRaises(ValueError):enqueue.verify_container(info,'s3')
            info['Mounts'][0]['Source']='/another/checkout'
            with self.assertRaises(ValueError):enqueue.verify_container(info,'s1')


if __name__=='__main__':unittest.main()
