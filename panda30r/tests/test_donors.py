"""Synthetic local checkpoint inventory; no CUDA, real data or evaluation runs."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from panda30r import donors as D


class DonorInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        registry = self.root / 'ablr2/sensor_sources.json'
        registry.parent.mkdir()
        self.dump(registry, {'schema': 'ABLR2_SENSOR_SOURCES_v1', 'sensors': {'WV3': {
            'band_order': ['Coastal', 'B', 'G', 'Y', 'R', 'RE', 'NIR1', 'NIR2'],
            'splits': {'train': {'path': 'fixture-never-read.h5', 'sha256': 'a' * 64,
                                 'source_identity': 'SYNTHETIC-INVENTORY-ONLY'}}}}})

    def dump(self, path, value):
        path.write_text(json.dumps(value))

    def fixture(self, server='s1', *, final_without_sidecar=False):
        package, campaign, run_id = D.source_run(server)
        directory = self.root / 'work_dir' / package / campaign / server / 'WV3/runs' / run_id / 'attempt_01'
        models = directory / 'models'
        models.mkdir(parents=True)
        cfg = {'campaign_id': campaign, 'run_id': run_id, 'server': server, 'dataset': 'WV3',
               'phase': 'donor', 'replica': 'R1', 'seed': D.SEED, 'attempt': 1,
               'case': 'D_A1' if server == 's4' else 'A1', 'planned_horizon': 50000,
               'loss_mask_policy': D.MASK2, 'lambda_abs': 2., 'lambda_epsilon': .003,
               'lr_A': 1e-5, 'lr_U': 1e-4, 'baseline_resolved': {
                   'source_registry_sha256': D._sha(self.root / 'ablr2/sensor_sources.json'),
                   'aligner_architecture': {'name': 'DirectPANAligner', 'view_margin': 8},
                   'teacher_architecture': {'bands': 8, 'width': 112, 'depth': [1, 2, 3]},
                   'teacher_input_layout': 'P0', 'teacher_base': 'BICUBIC_MS'}}
        self.dump(directory / 'config.json', cfg)
        self.dump(directory / 'run_state.json', {'run_id': run_id, 'attempt': 1, 'phase': 'donor',
                                                'status': 'COMPLETED', 'step': 50000})
        manifest = D._expected_train_manifest(self.root, cfg)
        rows = {}
        for step in (7000, 50000):
            stage = {'aligner.weight': torch.tensor([1., 2.]),
                     'teacher.weight': torch.tensor([3., step / 10000.]),
                     '_extra_state': {'sources': {'sensor': 'WV3', 'rr_manifest_sha256': manifest}}}
            state_hash = D._tensor_hash(stage)[0]
            identity = {key: cfg[key] for key in ('campaign_id', 'server', 'dataset', 'phase', 'run_id', 'attempt', 'seed')}
            identity.update(completed_update=step, model_state_sha256=state_hash)
            metrics = {'HQNR': .960 if step == 7000 else .915, 'ERGAS': 2.37 if step == 7000 else 2.07,
                       'SCC': .98}
            evaluation = {key: deepcopy(identity) for key in ('state_before', 'rr_state', 'fr_state', 'state_after')}
            evaluation.update(full_method_eligible=True, metric_function_injection=False,
                              measurement_domain='OFFICIAL_FULL20_NATIVE',
                              official_reference='original_PAN/native_LMS; FR full512 unmasked; RR all20 dim_cut21',
                              qualification={'same_state_verified': True, 'metrics_complete': True, 'metrics_finite': True})
            for split in ('rr', 'fr'):
                evaluation[split] = {'metrics': ({'HQNR': metrics['HQNR']} if split == 'fr' else
                                                {'ERGAS': metrics['ERGAS'], 'SCC': metrics['SCC']}),
                                     'per_scene': [{'scene_id': f'WV3:{split}:{i}', 'raw_prediction_sha256': 'b' * 64}
                                                   for i in range(20)]}
            self.dump(directory / f'evaluation_{step:07d}.json', evaluation)
            payload = {'schema': 'DONOR30_CHECKPOINT_V1', 'config': deepcopy(cfg),
                       'identity': deepcopy(identity), 'state_hash': state_hash,
                       'trainer': {'updates': step, 'stage': stage},
                       'evaluation': deepcopy(evaluation) if step == 7000 else None}
            name = ('step_0007000.pt' if step == 7000 else
                    'final_0050000.pt' if final_without_sidecar else 'resume_step_0050000_123456.pt')
            torch.save(payload, models / name)
            row = {**metrics, 'completed_update': step, 'file_sha256': D._sha(models / name),
                   'state_hash': state_hash, 'run_id': run_id, 'attempt': 1, 'seed': D.SEED,
                   'checkpoint_path': '/workspace/old-container-prefix/models/' + name,
                   'checkpoint_verified': True, 'same_state_verified': True,
                   'metrics_complete': True, 'metrics_finite': True}
            rows[step] = row
        self.dump(models / 'selection.json', {'saved': [rows[7000]],
                  'best_selected' if server == 's4' else 'best_gate_ergas': rows[7000],
                  'best_hqnr': rows[7000]})
        if not final_without_sidecar:
            self.dump(models / 'latest_resume.json', rows[50000])
        return directory

    def test_absent_local_run_is_only_normal_fallback(self):
        self.assertIsNone(D.discover_pair('s1', root=self.root))
        self.fixture('s3')
        self.assertIsNone(D.discover_pair('s1', root=self.root))
        self.assertIsNone(D.discover_pair('s4', root=self.root))

    def test_legacy_gate_pointer_and_resume_final_are_verified(self):
        directory = self.fixture()
        pair = D.discover_pair('s1', root=self.root)
        self.assertEqual(pair['early']['completed_update'], 7000)
        self.assertEqual(pair['late']['completed_update'], 50000)
        self.assertEqual(Path(pair['early']['checkpoint_path']).parent, directory / 'models')
        self.assertEqual(pair['late']['sha_provenance'], 'SAVED_SHA_AND_OFFICIAL_TENSOR_STATE_VERIFIED')
        self.assertTrue(pair['early']['requires_runtime_full40_replay'])
        self.assertFalse(pair['early']['reloaded_full40_verified'])

    def test_s4_final_without_sidecar_requires_tensor_proof_and_says_so(self):
        self.fixture('s4', final_without_sidecar=True)
        pair = D.discover_pair('s4', root=self.root)
        self.assertEqual(pair['late']['sha_provenance'], 'COMPUTED_AFTER_OFFICIAL_TENSOR_STATE_VERIFIED')
        self.assertEqual(pair['early']['sha_provenance'], 'SAVED_SHA_AND_OFFICIAL_TENSOR_STATE_VERIFIED')

    def test_corrupt_existing_file_does_not_trigger_fresh_fallback(self):
        directory = self.fixture()
        path = directory / 'models/step_0007000.pt'
        with path.open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaisesRegex(D.DonorInventoryError, 'SHA proof'):
            D.discover_pair('s1', root=self.root)

    def test_late_file_missing_is_error_not_fallback(self):
        directory = self.fixture()
        (directory / 'models/resume_step_0050000_123456.pt').unlink()
        with self.assertRaisesRegex(D.DonorInventoryError, 'Missing, linked or foreign'):
            D.discover_pair('s1', root=self.root)

    def test_tensor_edit_with_fresh_file_hash_is_not_same_official_state(self):
        directory = self.fixture('s4', final_without_sidecar=True)
        path = directory / 'models/final_0050000.pt'
        payload = torch.load(path, weights_only=False)
        payload['trainer']['stage']['teacher.weight'][0] += 1
        torch.save(payload, path)
        with self.assertRaisesRegex(D.DonorInventoryError, 'A/T tensors differ'):
            D.discover_pair('s4', root=self.root)

    def test_changed_training_manifest_rejected_even_when_tensor_hash_matches(self):
        directory = self.fixture('s4', final_without_sidecar=True)
        path = directory / 'models/final_0050000.pt'
        payload = torch.load(path, weights_only=False)
        payload['trainer']['stage']['_extra_state']['sources']['rr_manifest_sha256'] = 'c' * 64
        torch.save(payload, path)
        with self.assertRaisesRegex(D.DonorInventoryError, 'training source binding'):
            D.discover_pair('s4', root=self.root)

    def test_incomplete_full40_or_mixed_states_rejected(self):
        directory = self.fixture()
        path = directory / 'evaluation_0050000.json'
        original = json.loads(path.read_text())
        for mode in ('scene', 'state'):
            evaluation = deepcopy(original)
            if mode == 'scene':
                evaluation['fr']['per_scene'].pop()
            else:
                evaluation['rr_state']['completed_update'] = 49000
            self.dump(path, evaluation)
            with self.subTest(mode=mode), self.assertRaises(D.DonorInventoryError):
                D.discover_pair('s1', root=self.root)

    def test_foreign_server_and_architecture_not_accepted(self):
        directory = self.fixture()
        path = directory / 'config.json'
        original = json.loads(path.read_text())
        for mode in ('server', 'architecture'):
            cfg = deepcopy(original)
            if mode == 'server':
                cfg['server'] = 's3'
            else:
                cfg['baseline_resolved']['teacher_architecture']['width'] = 96
            self.dump(path, cfg)
            with self.subTest(mode=mode), self.assertRaises(D.DonorInventoryError):
                D.discover_pair('s1', root=self.root)

    def test_source_registry_changes_invalidate_inventory(self):
        self.fixture()
        path = self.root / 'ablr2/sensor_sources.json'
        data = json.loads(path.read_text())
        data['sensors']['WV3']['splits']['train']['sha256'] = 'c' * 64
        self.dump(path, data)
        with self.assertRaisesRegex(D.DonorInventoryError, 'Source registry SHA'):
            D.discover_pair('s1', root=self.root)

    def test_symlink_cannot_escape_local_checkpoint_directory(self):
        directory = self.fixture()
        path = directory / 'models/step_0007000.pt'
        relocated = self.root / 'foreign.pt'
        path.rename(relocated)
        path.symlink_to(relocated)
        with self.assertRaisesRegex(D.DonorInventoryError, 'Missing, linked or foreign'):
            D.discover_pair('s1', root=self.root)

    def test_selection_cannot_override_the_gate(self):
        directory = self.fixture()
        path = directory / 'models/selection.json'
        registry = json.loads(path.read_text())
        row = deepcopy(registry['saved'][0])
        row.update(HQNR=.961, ERGAS=2.1, completed_update=8000)
        registry['saved'].append(row)
        self.dump(path, registry)
        with self.assertRaisesRegex(D.DonorInventoryError, 'selection pointer differs'):
            D.discover_pair('s1', root=self.root)


if __name__ == '__main__':
    unittest.main()
