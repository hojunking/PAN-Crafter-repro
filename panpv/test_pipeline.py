"""Finite donor-to-campaign orchestration; no training, GPU or remote writes."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from panpv.common import atomic_json, file_sha, load_json
from panpv.pipeline import immutable_json, run_pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.inputs = self.repo / 'inputs'
        self.inputs.mkdir()
        self.work = self.repo / 'work_dir' / 'campaign'
        self.preparation = self.repo / 'preparation.json'
        self.entries = [dict(scene_id='s' + str(i), sha256=str(i) * 64,
                             patch_index_sha256='a' * 64) for i in range(5)]
        atomic_json(self.preparation, dict(state='NEW_DATA_VERIFIED_FORMAL_CAMPAIGN_NOT_STARTED',
                                          input_root=str(self.inputs), scenes=self.entries))
        self.args = SimpleNamespace(repo_root=self.repo, work_root=self.work,
                                    preparation=self.preparation, device='cuda')
        weights = self.repo / 'donor.safetensors'
        weights.write_bytes(b'test mock weights, never admissible in production')
        provenance = self.repo / 'donor.json'
        atomic_json(provenance, dict(status='READY', completed_step=50000))
        self.donor = dict(status='READY', path=str(weights), sha256=file_sha(weights),
                          provenance_path=str(provenance), provenance_sha256=file_sha(provenance))

    def mocks(self, donor=None, hashes=None):
        stack = ExitStack()
        stack.enter_context(patch.dict('os.environ', PANPV_HOSTNAME='knuvi-hojun-local'))
        stack.enter_context(patch('panpv.pipeline.configure_torch'))
        stack.enter_context(patch('panpv.pipeline.shutil.disk_usage',
                                  return_value=SimpleNamespace(free=100 * 1024**3)))
        stack.enter_context(patch('panpv.pipeline.source_hashes', side_effect=hashes or
                                  [dict(source='sha')] * 2))
        scene = stack.enter_context(patch('panpv.v3_data.SceneV3'))
        scene.return_value.__enter__.return_value.summary.return_value = dict(patch_index_sha256='a' * 64)
        stack.enter_context(patch('panpv.campaign_eval.freeze_protocol', return_value=dict(frozen=True)))
        self.pretrain = stack.enter_context(patch('panpv.pretrain.run_pretrain',
                                                  return_value=donor if donor is not None else self.donor))
        def proof(scene, donor, output, device):
            atomic_json(output, dict(status='PASS', donor_sha256=donor['sha256']))
        stack.enter_context(patch('panpv.pipeline.adaptation_preflight', side_effect=proof))
        self.campaign = stack.enter_context(patch('panpv.campaign.run_campaign', return_value=0))
        return stack

    def test_complete_donor_chains_exact_registry(self):
        with self.mocks():
            self.assertEqual(run_pipeline(self.args), 0)
        self.pretrain.assert_called_once_with(self.repo, self.work / 'donor_pretrain',
                                             device='cuda', micro_batch=12)
        registry = load_json(self.work / 'registry.json')
        self.assertEqual(registry['donor']['sha256'], self.donor['sha256'])
        self.assertEqual(len(registry['scenes']), 5)
        self.assertEqual(registry['seed'], 261002)
        self.assertEqual(registry['micro_batch'], 12)
        self.assertEqual(load_json(self.work / 'pipeline_status.json')['campaign_status_path'],
                         str(self.work / 'campaign_status.json'))
        self.campaign.assert_called_once_with(self.work / 'registry.json')

    def test_paused_donor_cannot_start_adaptation(self):
        with self.mocks(donor=dict(status='PAUSED_SAFE', completed_step=100)):
            self.assertEqual(run_pipeline(self.args), 130)
        self.campaign.assert_not_called()
        self.assertFalse((self.work / 'registry.json').exists())
        self.assertEqual(load_json(self.work / 'pipeline_status.json')['stage'], 'PAUSED_DONOR_PRETRAINING')

    def test_unfinished_donor_fails_closed(self):
        with self.mocks(donor=dict(status='DIVERGED', completed_step=9)):
            with self.assertRaisesRegex(RuntimeError, 'did not finish safely'):
                run_pipeline(self.args)
        self.campaign.assert_not_called()
        self.assertEqual(load_json(self.work / 'pipeline_status.json')['stage'], 'FAILED')

    def test_source_change_blocks_chain(self):
        with self.mocks(hashes=[dict(source='before'), dict(source='after')]):
            with self.assertRaisesRegex(RuntimeError, 'Source changed'):
                run_pipeline(self.args)
        self.campaign.assert_not_called()

    def test_cpu_production_fallback_rejected(self):
        self.args.device = 'cpu'
        with self.mocks():
            with self.assertRaisesRegex(ValueError, 'requires.*CUDA'):
                run_pipeline(self.args)
        self.pretrain.assert_not_called()

    def test_immutable_artifact_conflict(self):
        path = self.repo / 'immutable.json'
        immutable_json(path, dict(a=1))
        immutable_json(path, dict(a=1))
        with self.assertRaisesRegex(ValueError, 'differs'):
            immutable_json(path, dict(a=2))


if __name__ == '__main__':
    unittest.main()
