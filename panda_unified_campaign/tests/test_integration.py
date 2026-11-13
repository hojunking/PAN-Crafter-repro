"""Cross-stage contracts on real serialized four/eight-band synthetic observations."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch
from donor30.masking import masked_mean
from panda_method.calibration import state_dict_sha256
from panda_stage2_condq.sources import RegisteredRRTrain
from ta2.evaluation import file_sha
from panda_unified import core
from panda_unified_campaign import plan, model, training


def synthetic_source(root, sensor, count=2):
    root = Path(root)
    bands = 8 if sensor == 'WV3' else 4
    rng = np.random.default_rng(47)
    p = root / 'train.h5'
    with h5py.File(p, 'w') as h:
        for name, ch, side in [('pan', 1, 64), ('ms', bands, 16),
                               ('lms', bands, 64), ('gt', bands, 64)]:
            h[name] = rng.uniform(300, 1500, (count, ch, side, side)).astype('float32')
    catalog = root / 'catalog.json'
    order = ['Coastal', 'B', 'G', 'Y', 'R', 'RE', 'NIR1', 'NIR2'] if bands == 8 else ['B', 'G', 'R', 'NIR']
    catalog.write_text(json.dumps({'schema': 'ABLR2_SENSOR_SOURCES_v1', 'sensors': {
        sensor: {'band_order': order, 'splits': {'train': {'path': 'train.h5', 'sha256': file_sha(p)}}}}}))
    overrides = ({s: {'path': 'train.h5', 'sha256': file_sha(p),
                     'reason': 'Synthetic test fixture with identical PAN/GT row identity'}
                  for s in ('train', 'val')} if sensor == 'QB' else None)
    return RegisteredRRTrain.from_catalog(catalog, root=root, sensor=sensor, lms_overrides=overrides)


class Stats:
    tau_e = .1
    artifact_sha256 = 'synthetic-statistics'
    def validate_binding(self, binding): pass
    def assert_integrity(self, **kwargs): pass
    def lookup_exact(self, g, q):
        self.looked_up = len(g)
        return torch.full_like(q, .5)


class UnifiedIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_paired_queue_and_local_dependencies(self):
        s1, s2 = plan.queue('s1'), plan.queue('s2')
        self.assertEqual([c['stage'] for c in s1], ['stage1', 'stage2'] * 2)
        self.assertEqual(sum(c['stage'] == 'stage1' for c in s2), 2)
        self.assertEqual(sum(c['stage'] == 'stage2' for c in s2), 10)
        donors = {c['run_id']: c for c in s2 if c['stage'] == 'stage1'}
        for c in s2:
            self.assertEqual(c['width'], 128)
            self.assertEqual(c['architecture'], 'W128_D123_UBIL')
            self.assertEqual(c['teacher_warp'], 'LANCZOS4_REFLECTION')
            if c['stage'] == 'stage2':
                d = donors[c['donor_run_id']]
                self.assertEqual(d['reference_mode'], c['reference_mode'])
                self.assertEqual(d['server'], 's2')
                self.assertEqual(c['beta'], 0)
                self.assertEqual(c['scale'], 1)
        groups = [[c for c in s2 if c['stage'] == 'stage2' and c['reference_mode'] == m] for m in ('RGB', 'RGBN')]
        for a, b in zip(*groups):
            for key in ('alpha', 'gamma', 'lr', 'seed', 'stream_seeds', 'horizon'):
                self.assertEqual(a[key], b[key])

    def test_stage1_and_frozen_stage2_share_actual_pan_teacher_and_mask(self):
        for server, mode in [('s1', 'RGBN'), ('s2', 'RGB'), ('s2', 'RGBN')]:
            with self.subTest(server=server, mode=mode), tempfile.TemporaryDirectory() as tmp:
                q = plan.queue(server)
                cfg = next(c for c in q if c['stage'] == 'stage1' and c['reference_mode'] == mode)
                scfg = next(c for c in q if c['stage'] == 'stage2' and c['reference_mode'] == mode)
                source = synthetic_source(tmp, cfg['sensor'])
                batch = source.batch([0])
                net = model.Stage1(cfg, source.inventory.manifest())
                with torch.no_grad(): net.aligner.fc2.bias.copy_(torch.tensor([.625, -.375]))
                native = net.native_prediction(batch)
                mask = core.support(batch.pan, native['u'])
                with patch('pa.warp.warp_pan', side_effect=AssertionError('bicubic PAN used')):
                    loss = net(batch)
                    loss['loss'].backward()
                self.assertTrue(torch.isfinite(net.aligner.fc2.bias.grad).all())
                self.assertGreater(float(net.aligner.fc2.bias.grad.abs().sum()), 0)
                torch.testing.assert_close(loss['rec'], masked_mean((native['y'] - batch.gt).abs(), mask))
                net.zero_grad(set_to_none=True)
                net.aligner.requires_grad_(False).eval()
                net.teacher.requires_grad_(False).eval()
                frozen = core.frozen_predictions(net.aligner, net.teacher, batch)
                self.assertTrue(torch.equal(native['pan_aligned'], frozen['pan_input']))
                self.assertTrue(torch.equal(native['y'], frozen['teacher_y']))
                self.assertTrue(torch.equal(mask, frozen['loss_mask']))
                binding = source.inventory.manifest()
                binding.update(sensor=cfg['sensor'], reference_mode=mode,
                    alignment_target=cfg['alignment_target'], warp=core.WARP_CONTRACT,
                    scharr={'reference_indices': cfg['reference_bands']})
                donor = SimpleNamespace(aligner=net.aligner, teacher=net.teacher,
                    binding=binding, selection_receipt={}, assert_frozen=lambda: None)
                stats = Stats()
                student = model.Stage2(scfg, donor, stats)
                before = {r: state_dict_sha256(getattr(student, r)) for r in ('aligner', 'teacher')}
                with patch('pa.warp.warp_pan', side_effect=AssertionError('bicubic PAN used')):
                    result = student(batch)
                    result['loss'].backward()
                self.assertEqual(stats.looked_up, int(mask.sum()))
                self.assertEqual(int(mask.sum()), 3025)
                self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in student.student.parameters()))
                self.assertTrue(all(p.grad is None for r in before for p in getattr(student, r).parameters()))
                self.assertEqual(before, {r: state_dict_sha256(getattr(student, r)) for r in before})
                self.assertEqual(student.teacher.sampling_variant, student.student.sampling_variant)

    def test_complete_stage1_resume_includes_epsilon_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = synthetic_source(tmp, 'QB')
            cfg = deepcopy(plan.resolve('s2', 'S1_RGB_FULL', 1))
            cfg.update(batch=1, microbatch=1)
            cfg['config_sha256'] = plan.digest({k: v for k, v in cfg.items() if k != 'config_sha256'})
            with patch.object(plan, 'validate'):
                trainer = training.Trainer(cfg, model.Stage1(cfg, source.inventory.manifest()))
            stream = training.stream_for(cfg, source, 'cpu')
            trainer.step(stream.next_update())
            path = Path(tmp) / 'resume.pt'
            training.save(path, trainer, stream, {'synthetic': True})
            expected = trainer.step(stream.next_update())
            model_hash = training.model_hash(trainer.model)
            training.load(path, trainer, stream, {'synthetic': True})
            actual = trainer.step(stream.next_update())
            self.assertEqual(expected, actual)
            self.assertEqual(model_hash, training.model_hash(trainer.model))


if __name__ == '__main__': unittest.main()
