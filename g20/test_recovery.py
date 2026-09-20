from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from g20.common import atomic_json, read_json, sha256
from g20.test_postrun import fixture
from g20.training import recover_exact50k_status


def recovery_fixture(root):
    case, cfg, data, ref, grid = fixture(root)
    wd = root / 'work_dir' / case.run_id
    folder = wd / 'candidates/50000'
    state = torch.load(folder / 'training_state.pt', weights_only=False)
    save_file(state['model_state'], str(folder / 'model.safetensors'))
    state.update(model_sha256=sha256(folder / 'model.safetensors'), precision='fp32',
                 scheduler={'last_epoch': 50000}, training_seconds=120., evaluation_seconds=30.,
                 io_seconds=2., diagnostic_seconds=5., peak_memory_bytes=4096)
    torch.save(state, folder / 'training_state.pt')
    identity = read_json(folder / 'identity.json')
    identity.update(model_sha256=state['model_sha256'], training_state_sha256=sha256(folder / 'training_state.pt'),
                    full_state=True, role=case.role, sensor='GF2', num_bands=4, input_layout=case.input_layout)
    atomic_json(folder / 'identity.json', identity)
    grid['records'][-1]['checkpoint_identity'] = identity
    atomic_json(wd / 'official/raw_grid.json', grid)
    atomic_json(wd / 'meta/training_status.json', dict(status='RUNNING', actual_updates=49490,
                training_complete=False, deadline_utc='2026-09-20T18:00:00+00:00'))
    return wd, cfg, grid


class RecoveryTests(unittest.TestCase):
    def test_exact50k_status_recovers_after_training_cutoff_without_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wd, cfg, grid = recovery_fixture(root)
            with patch('g20.training.source_identity', return_value=grid['source_identity']), \
                 patch('g20.training.runtime_context', side_effect=AssertionError('No training admission')), \
                 patch('torch.optim.AdamW', side_effect=AssertionError('No optimizer')):
                recovered = recover_exact50k_status(wd, cfg, root)
            self.assertEqual(recovered['status'], 'TRAIN50K_COMPLETE')
            self.assertTrue(recovered['training_complete'])
            self.assertTrue(recovered['recovered_from_exact50k'])
            self.assertEqual(recovered['actual_updates'], 50000)
            self.assertEqual(recovered['sample_exposure']['total_sample_presentations'], 50000 * 48)
            self.assertEqual(recovered['training_seconds'], 120.)
            self.assertEqual(recovered['n_evaluated'], 50)
            self.assertEqual(recovered, read_json(wd / 'meta/training_status.json'))
            self.assertFalse(recover_exact50k_status(wd, cfg, root))
            self.assertEqual(recovered, read_json(wd / 'meta/training_status.json'))

    def test_missing_final_candidate_is_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(recover_exact50k_status(root / 'missing', {}, root))
            self.assertFalse((root / 'missing').exists())

    def test_wrong_existing_bytes_scheduler_exposure_and_source_fail_closed(self):
        for fault in ('bytes', 'scheduler', 'exposure', 'source'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                wd, cfg, grid = recovery_fixture(root)
                folder = wd / 'candidates/50000'
                before = read_json(wd / 'meta/training_status.json')
                identity = read_json(folder / 'identity.json')
                if fault == 'bytes':
                    (folder / 'model.safetensors').write_bytes(b'corrupt')
                elif fault in ('scheduler', 'exposure'):
                    state = torch.load(folder / 'training_state.pt', weights_only=False)
                    if fault == 'scheduler':
                        state['scheduler']['last_epoch'] = 49999
                    else:
                        state['exposure_counts'][0, 0] -= 1
                    torch.save(state, folder / 'training_state.pt')
                    identity['training_state_sha256'] = sha256(folder / 'training_state.pt')
                    atomic_json(folder / 'identity.json', identity)
                release = {'content_sha256': '0' * 64} if fault == 'source' else grid['source_identity']
                with patch('g20.training.source_identity', return_value=release), self.assertRaises(ValueError):
                    recover_exact50k_status(wd, cfg, root)
                self.assertEqual(before, read_json(wd / 'meta/training_status.json'))


if __name__ == '__main__':
    unittest.main()
