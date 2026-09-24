"""Synthetic state-schema acceptance, NOT evidence of50K actual training.

All artifacts live in an automatically removed temporary folder. A tiny model
and one real AdamW step provide real tensor/moment structures; update counters
are deliberately set to50000 to exercise receipt validation without training.
The CUDA Philox byte-vector shape16 is from the original F1 saved checkpoint;
no CUDA APIs or production artifacts are written or evaluated by these tests.
"""
from pathlib import Path
import copy
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from safetensors.torch import save_file

from fh12.model import state_hash
from fh12.training import BatchStream
from maina_hqnr.common import atomic_json, object_sha, sha256
from maina_hqnr.data import stream_manifest
from maina_hqnr.plan import DATA_SHA, TEACHER_SHA, make_case
from maina_hqnr.postrun import GRID, _training_evidence
from maina_hqnr.training import make_optimizer, make_scheduler


class TinyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 2)
        self.aligner = torch.nn.Linear(2, 1)


class CompletionAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.wd = Path(self.temp.name)
        case = make_case('s4', 1, 'BASE')
        self.cfg = dict(seed=case['seed'], optimizer='AdamW', learning_rate=1e-4,
            weight_decay=.01, betas=[.9, .999], eps=1e-8, num_warmup=100, num_iter=50000,
            lr_scheduler='cosine', fh20r1=dict(aligner_lr=3e-6),
            maina_hqnr=dict(case=case, attempt=0, source_identity=dict(content_sha256='s'*64), binding_sha256='b'*64))
        self.model = TinyStudent()
        opt = make_optimizer(self.model, self.cfg)
        for parameter in self.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        opt.step()
        # Synthetic counters: these tests never claim50K optimizer executions.
        for value in opt.state.values():
            value['step'].fill_(50000)
        scheduler = make_scheduler(opt, self.cfg).state_dict()
        scheduler.update(last_epoch=50000, _step_count=50001)
        stream = BatchStream(9714, 48, case['seed'])
        while stream.epoch*stream.count+stream.count < 50000:
            stream.new_epoch()
        stream.cursor = 50000-stream.epoch*stream.count
        sm = stream_manifest(9714, 48, case['seed'])
        self.context = dict(run_id=case['run_id'], attempt=0, config_sha256=object_sha(self.cfg),
            source_identity=self.cfg['maina_hqnr']['source_identity'], bindings_sha256='b'*64,
            data_sha256=DATA_SHA, reference_sha256='r'*64, teacher_sha256=TEACHER_SHA,
            init_U_sha256=state_hash(self.model.backbone.state_dict()),
            init_A_sha256=state_hash(self.model.aligner.state_dict()), stream_sha256=object_sha(sm),
            candidate_grid_sha256=object_sha(list(GRID)))
        self.state = dict(full_state=True, precision='fp32', update=50000,
            model_state=copy.deepcopy(self.model.state_dict()), optimizer=copy.deepcopy(opt.state_dict()),
            scheduler=scheduler, sampler=stream.state_dict(),
            rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                     cuda=[torch.zeros(16, dtype=torch.uint8)]), **self.context)
        atomic_json(self.wd/'stream_manifest.json', sm)
        for path in ('meta/training_start_manifest.json', 'init_manifest.json'):
            atomic_json(self.wd/path, self.context)
        atomic_json(self.wd/'meta/training_status.json', dict(self.context, actual_updates=50000,
            training_complete=True, status='TRAIN_COMPLETE_EVAL_PENDING'))
        self.identities = {}
        for step in GRID:
            folder = self.wd/'candidates'/str(step); folder.mkdir(parents=True)
            save_file({k: v.detach().contiguous() for k, v in self.model.state_dict().items()}, str(folder/'model.safetensors'))
            identity = dict(self.context, update=step, full_state=step == 50000,
                model_sha256=sha256(folder/'model.safetensors'), state_hash=state_hash(self.model.state_dict()))
            self.identities[step] = identity
            atomic_json(folder/'identity.json', identity)
        self._publish_state()
        atomic_json(self.wd/'candidate_ledger.json', dict(expected_steps=list(GRID),
            records=[dict(update=step, model_sha256=self.identities[step]['model_sha256'],
                          state_hash=self.identities[step]['state_hash']) for step in GRID]))

    def _publish_state(self):
        path = self.wd/'candidates/50000/training_state.pt'
        torch.save(self.state, path)
        self.identities[50000]['training_state_sha256'] = sha256(path)
        atomic_json(path.parent/'identity.json', self.identities[50000])

    def verify(self):
        # Only expensive architecture construction is mocked. File hashes,
        # state tensors, sampler replay, actual AdamW loader and schema checks run.
        def loader(*args, **kwargs):
            model = TinyStudent(); model.load_state_dict(self.model.state_dict())
            return model, self.identities[50000]
        with mock.patch('fh20r1.common.load_checkpoint_model', side_effect=loader):
            return _training_evidence(self.cfg, self.wd)

    def test_positive_original_state_schema_with_16_byte_cuda_rng(self):
        status, identities = self.verify()
        self.assertEqual(status['actual_updates'], 50000)
        self.assertEqual(list(identities), list(GRID))

    def test_rejects_false_full_state_missing_model_and_fake_rng(self):
        for mutation in ('full_state', 'model', 'rng', 'cuda'):
            with self.subTest(mutation=mutation):
                saved = copy.deepcopy(self.state)
                if mutation == 'full_state': self.state['full_state'] = False
                if mutation == 'model': self.state.pop('model_state')
                if mutation == 'rng': self.state['rng'] = dict(fake=True)
                if mutation == 'cuda': self.state['rng']['cuda'] = [torch.zeros(3, dtype=torch.uint8)]
                self._publish_state()
                with self.assertRaises(ValueError): self.verify()
                self.state = saved; self._publish_state()

    def test_rejects_wrong_sampler_cursor_and_replayed_order(self):
        saved = copy.deepcopy(self.state)
        self.state['sampler']['cursor'] -= 1; self._publish_state()
        with self.assertRaises(ValueError): self.verify()
        self.state = saved
        self.state['sampler']['order'] = self.state['sampler']['order'].roll(1)
        self._publish_state()
        with self.assertRaises(ValueError): self.verify()

    def test_rejects_wrong_tensor_and_optimizer_coverage(self):
        saved = copy.deepcopy(self.state)
        next(iter(self.state['model_state'].values())).add_(1); self._publish_state()
        with self.assertRaises(ValueError): self.verify()
        self.state = saved
        self.state['optimizer']['state'].pop(next(iter(self.state['optimizer']['state'])))
        self._publish_state()
        with self.assertRaises(ValueError): self.verify()

    def test_rejects_moment_shape_wrong_teacher_and_changed_checkpoint(self):
        first = next(iter(self.state['optimizer']['state'].values()))
        first['exp_avg'] = torch.zeros(99); self._publish_state()
        with self.assertRaises(ValueError): self.verify()
        self.identities[GRID[0]]['teacher_sha256'] = 'x'*64
        atomic_json(self.wd/'candidates'/str(GRID[0])/'identity.json', self.identities[GRID[0]])
        with self.assertRaises(ValueError): self.verify()


if __name__ == '__main__':
    unittest.main()
