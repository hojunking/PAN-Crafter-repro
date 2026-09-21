"""CPU numerical checks use explicit synthetic scope, not production PASS assets."""
import random
import unittest

import numpy as np
import torch
from torch import nn

from fh12.calibration import axis16_q
from ablr2.reference_parity import (isolated_teacher, parity_indices,
                                   validate_receipt, verify_q_cache)


class Dataset:
    has_gt, base_count = True, 20

    def __len__(self):
        return self.base_count

    def base(self, index):
        gen = torch.Generator().manual_seed(int(index) + 62)
        pan = torch.randn(1, 16, 16, generator=gen)
        ms = torch.randn(8, 4, 4, generator=gen)
        gt = torch.full((8, 16, 16), float(index)/100)
        return gt, gt, ms, pan[:, 2::4, 2::4], pan, torch.tensor([index, 0, 0, 0])

    def get_view(self, index, rot, augment):
        row = self.base(index)
        return (*[torch.rot90(x.flip((-2, -1)), rot, (-2, -1)) for x in row[:5]],
                torch.tensor([index, rot, 1, 1]))


class Teacher(nn.Module):
    bands = 8

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.2))
        self.child = nn.Dropout()

    def predict_delta(self, pan, base):
        # Deliberately consume all CPU RNGs, without changing the numerical output.
        noise = (random.random() + np.random.rand() + torch.rand(())) * 0
        return torch.stack((pan[:, 0, 3, 4], pan[:, 0, 6, 8]), 1) * self.scale + noise

    def forward(self, pan, ms, lp):
        return {'y': self.scale * pan.repeat(1, 8, 1, 1)}


class ReferenceParityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.dataset, self.model = Dataset(), Teacher()
        self.identity = dict(reference_id='TEST', teacher_checkpoint_sha256='a'*64, q_cache_sha256='b'*64)
        self.q = np.empty((20, 4), np.float32)
        with isolated_teacher(self.model):
            for view in range(4):
                rows = [self.dataset.get_view(i, view, True) for i in range(20)]
                self.q[:, view] = axis16_q(self.model, torch.stack([r[4] for r in rows]),
                                          torch.stack([r[2] for r in rows]))[0].numpy()

    def verify(self, **kwargs):
        return verify_q_cache(self.model, self.dataset, self.q, device='cpu', synthetic_test=True,
                              reference_identity=self.identity, data_identity='c'*64, **kwargs)

    def test_actual_online_four_view_q_and_rng_modes_are_restored(self):
        self.model.child.eval()
        self.model.scale.grad = torch.tensor(7.)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        receipt = self.verify()
        self.assertTrue(receipt['passed'])
        self.assertEqual(len(receipt['indices']), 16)
        self.assertEqual(receipt['views'], [0, 1, 2, 3])
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.child.training)
        self.assertTrue(self.model.scale.requires_grad)
        self.assertEqual(self.model.scale.grad.item(), 7.)
        self.assertEqual(before[0], random.getstate())
        np.testing.assert_array_equal(before[1][1], np.random.get_state()[1])
        self.assertTrue(torch.equal(before[2], torch.get_rng_state()))

    def test_tampered_cache_rejected_and_model_restored_on_failure(self):
        self.q[parity_indices(20)[0], 2] += .1
        with self.assertRaisesRegex(ValueError, 'q-cache/online'):
            self.verify()
        self.assertTrue(self.model.training)
        self.assertTrue(self.model.scale.requires_grad)

    def test_incorrect_view_id_metadata_is_not_accepted(self):
        original = self.dataset.get_view
        self.dataset.get_view = lambda i, r, augment: (*original(i, r, augment)[:5], torch.tensor([i, r, 0, 0]))
        with self.assertRaisesRegex(ValueError, 'metadata'):
            self.verify()

    def test_synthetic_receipt_cannot_certify_production(self):
        receipt = self.verify()
        with self.assertRaisesRegex(ValueError, 'protocol'):
            validate_receipt(receipt, reference_identity=self.identity, data_identity='c'*64, q=self.q)

    def test_expired_deadline_does_not_run_model(self):
        with self.assertRaises(TimeoutError):
            self.verify(deadline='2000-01-01T00:00:00Z')
        self.assertTrue(self.model.training)


if __name__ == '__main__':
    unittest.main()
