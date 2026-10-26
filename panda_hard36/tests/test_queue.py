"""The queued successor cannot overlap or silently replace TUNE30."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from panda_hard36 import queue as Q


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.state=self.root/'work_dir/panda_tune30'/Q.PREVIOUS_CAMPAIGN/'s1/WV3/campaign_state.json'
        self.state.parent.mkdir(parents=True)

    def test_live_predecessor_waits_without_docker_or_gpu_or_state_mutation(self):
        self.state.write_text(json.dumps({'status':'RUNNING'}))
        before=self.state.read_bytes()
        with patch.object(Q.subprocess,'run') as run:
            self.assertFalse(Q.readiness(self.root,'s1')['ready'])
            run.assert_not_called()
        self.assertEqual(self.state.read_bytes(),before)

    def test_failed_or_stopped_predecessor_blocks(self):
        for status in ('FAILED','STOPPED_SAFE','FINISHED_WITH_EXPLICIT_SKIPS'):
            self.state.write_text(json.dumps({'status':status}))
            with self.assertRaisesRegex(ValueError,'requires inspection'):
                Q.readiness(self.root,'s1')

    def test_terminal_evidence_still_waits_for_container_and_idle_gpu(self):
        self.state.write_text(json.dumps({'status':'FINISHED_REGISTERED_QUEUE'}))
        with patch.object(Q,'predecessor_receipt',return_value={'verified':True}), \
             patch.object(Q,'container_gate') as gate, patch.object(Q.subprocess,'run') as run:
            run.return_value=SimpleNamespace(stdout=json.dumps({'Running':True}))
            self.assertEqual(Q.readiness(self.root,'s1')['status'],'WAITING_FOR_CONTAINER_EXIT')
            gate.assert_not_called()
            run.side_effect=[SimpleNamespace(stdout=json.dumps({'Running':False})),
                             SimpleNamespace(stdout='12345\n')]
            self.assertEqual(Q.readiness(self.root,'s1')['status'],'WAITING_FOR_IDLE_GPU')
            run.side_effect=[SimpleNamespace(stdout=json.dumps({'Running':False})),
                             SimpleNamespace(stdout='')]
            self.assertTrue(Q.readiness(self.root,'s1')['ready'])
