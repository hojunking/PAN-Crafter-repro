import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file

from fh12.common import atomic_json, object_sha, sha256
from g20.model import state_hash
from gfb20 import assets


class TinyPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.aligner = nn.Linear(2, 2)


def endpoint(root, step=100000):
    model = TinyPair()
    weights = root / 'model.safetensors'
    state = model.state_dict()
    save_file(state, str(weights))
    identity = dict(update=step, full_state=True, state_hash=state_hash(state),
        model_sha256=sha256(weights), config_sha256='c' * 64, data_sha256='d' * 64,
        reference_sha256='r' * 64, source_identity={'fixture': True})
    full = dict(full_state=True, precision='fp32', scheduler={'last_epoch': step}, model_state=state,
                training_seconds=100, **{k: identity[k] for k in (
                    'update', 'config_sha256', 'data_sha256', 'reference_sha256',
                    'source_identity', 'model_sha256')})
    torch.save(full, root / 'training_state.pt')
    identity['training_state_sha256'] = sha256(root / 'training_state.pt')
    return model, identity, full


def progress(full, cfg, count=3072):
    from fh12.training import BatchStream
    stream = BatchStream(count, cfg['batch_size'], cfg['seed'])
    stream.epoch, stream.cursor = divmod(full['update'], stream.count)
    total = full['update'] * cfg['batch_size']
    counts = torch.full((count * 4,), total // (count * 4), dtype=torch.int64)
    counts[:total % (count * 4)] += 1
    full.update(sampler=stream.state_dict(), exposure_counts=counts.reshape(count, 4),
        optimizer={'param_groups': [dict(name='U', lr=0.), dict(name='A', lr=0.)]})
    full['scheduler']['_last_lr'] = [0., 0.]
    return full


class AssetTests(unittest.TestCase):
    def test_tolerated_online_measurements_do_not_change_resume_identity(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            atomic_json(root / 'reference.json', {'fixture': True})
            atomic_json(root / 'data.json', {'fixture': 'data'})
            cfg = {'gfb20': dict(reference_manifest=str(root / 'reference.json'),
                dataset_manifest=str(root / 'data.json'), server='s4', reference_key='R4_100')}
            base = dict(native_binding={'server': 's4'}, native_reference_id='R4_100',
                        q_cache_sha256='a' * 64, teacher_checkpoint_sha256='b' * 64)
            results = []
            for error in (0., 1e-7):
                info = dict(base, native_online_parity={'passed': True, 'max_abs_error': error})
                with patch.object(assets, 'load_reference', return_value=(None, np.ones((1, 4)), info)):
                    results.append(assets.load_training_assets(cfg, root))
            self.assertEqual(object_sha(results[0]['reference']), object_sha(results[1]['reference']))
            self.assertNotEqual(results[0]['native_online_parity'], results[1]['native_online_parity'])
            self.assertNotIn('native_online_parity', results[0]['reference'])

    def test_historical_qg40_native_parity_dispatch_preserves_manifest(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            model = TinyPair()
            model.bands = 4
            save_file(model.state_dict(), str(root / 'teacher.safetensors'))
            np.savez(root / 'q.npz', q=np.full((3072, 4), .5, np.float32))
            data = dict(schema='QG40_DATA_v1', sensor='GF2', num_bands=4, max_pixel=1023,
                mtf_sensor='GF2', band_order=['B', 'G', 'R', 'NIR'], recipe={}, augmentation={},
                splits={'train': {'sha256': 'd', 'lpan_sha256': 'l'}})
            original = dict(reference_id='GF2_TA', teacher_run_id='original', teacher_checkpoint_sha256='t',
                q_cache_sha256='q', source_identity={'content_sha256': 's'}, tau_R=.01, q_ref=.5)
            evidence = dict(native_manifest=original, teacher_config={'seed': 91001},
                resolved_artifacts=dict(teacher_checkpoint=str(root / 'teacher.safetensors'),
                    q_cache_path=str(root / 'q.npz'), dataset_manifest_path='original-data'),
                data=data, origin_root=str(root), native_reference_id='R3_TA50', binding_sha256='b')
            with patch.object(assets, 'verify_reference', return_value=evidence), patch.object(
                    assets, 'build_model', return_value=(model, {})), patch(
                    'qg40.data.build_dataset', return_value='qg40-dataset') as old, patch(
                    'g20.data.build_dataset') as newer, patch('qg40.reference_parity.verify_q_cache',
                    return_value={'passed': True}) as parity:
                loaded, q, info = assets.load_reference(evidence)
            self.assertEqual(q.shape, (3072, 4))
            old.assert_called_once_with(data, 'train', root=str(root))
            newer.assert_not_called()
            self.assertEqual(parity.call_args.args[1], 'qg40-dataset')
            self.assertEqual(info['native_manifest'], original)
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))

    def test_native_reference_exact_scales_and_historical_seed_binding(self):
        from qg40.plan import build_config, teacher_for
        from qg40.calibration import select_calibration_indices
        from fh12.data import RECIPE, AUGMENTATION
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cfg = build_config(teacher_for('GF2_TA'))
            cfg_path = root / 'teacher.json'
            atomic_json(cfg_path, cfg)
            raw, lp = root / 'raw.bin', root / 'lp.bin'
            raw.write_bytes(b'fixture ordered DN1023 GF2 data')
            lp.write_bytes(b'fixture native LP')
            data = dict(sensor='GF2', num_bands=4, max_pixel=1023, mtf_sensor='GF2',
                band_order=['B', 'G', 'R', 'NIR'], recipe=RECIPE, augmentation=AUGMENTATION,
                splits={s: dict(dataroot=str(raw), lpan_path=str(lp), sha256=sha256(raw),
                    lpan_sha256=sha256(lp), count=3072 if s == 'train' else 20)
                    for s in ('train', 'val', 'rr', 'fr')})
            atomic_json(root / 'data.json', data)
            _model, identity, full = endpoint(root, 50000)
            identity.update(role='T', sensor='GF2', num_bands=4, input_layout='P0',
                config_sha256=object_sha(cfg), data_sha256=object_sha(data), reference_sha256=None)
            identity['source_identity']['content_sha256'] = 'fixture'
            full.update({k: identity[k] for k in ('config_sha256', 'data_sha256', 'reference_sha256')})
            progress(full, cfg)
            torch.save(full, root / 'training_state.pt')
            identity['training_state_sha256'] = sha256(root / 'training_state.pt')
            atomic_json(root / 'identity.json', identity)
            q = np.full((3072, 4), .5, np.float32)
            indices = select_calibration_indices(len(q))
            np.savez(root / 'q.npz', q=q, calibration_indices=indices)
            manifest = dict(schema='QG40_REFERENCE_v1', server='s3', sensor='GF2', num_bands=4,
                max_pixel=1023, reference_id='GF2_TA', teacher_layout='P0', teacher_update=50000,
                teacher_run_id=Path(cfg['work_dir']).name, source_identity=identity['source_identity'],
                teacher_checkpoint=str(root / 'model.safetensors'), teacher_checkpoint_sha256=identity['model_sha256'],
                teacher_training_state=str(root / 'training_state.pt'), teacher_training_state_sha256=identity['training_state_sha256'],
                teacher_config=str(cfg_path), teacher_config_sha256=object_sha(cfg), teacher_config_file_sha256=sha256(cfg_path),
                teacher_checkpoint_identity=str(root / 'identity.json'), teacher_checkpoint_identity_sha256=sha256(root / 'identity.json'),
                dataset_manifest_path=str(root / 'data.json'), dataset_manifest_sha256=sha256(root / 'data.json'),
                q_cache_path=str(root / 'q.npz'), q_cache_sha256=sha256(root / 'q.npz'), q_shape=list(q.shape),
                tau_R=.01, q_ref=.5, calibration_indices_sha256=object_sha(indices.tolist()),
                LP_recipe=RECIPE, augmentation=AUGMENTATION)
            cal = {k: manifest[k] for k in ('tau_R', 'q_ref', 'q_shape', 'teacher_checkpoint_sha256',
                                           'source_identity', 'calibration_indices_sha256')}
            cal.update(schema='QG40_CALIBRATION_v1', synthetic_test=False)
            atomic_json(root / 'cal.json', cal)
            atomic_json(root / 'parity.json', {'fixture': True})
            manifest.update(calibration_path=str(root / 'cal.json'), calibration_sha256=sha256(root / 'cal.json'),
                q_cache_parity_path=str(root / 'parity.json'), q_cache_parity_sha256=sha256(root / 'parity.json'))
            atomic_json(root / 'reference.json', manifest)
            spec = dict(teacher_seed=91001, teacher_updates=50000, teacher_run_id=None,
                        published_teacher_sha256=identity['model_sha256'])
            with patch.object(assets, 'source_compatibility', return_value={'passed': True}), patch(
                    'qg40.reference_parity.validate_receipt') as parity:
                bound = assets._reference_binding(root / 'reference.json', 'R3_TA50', 's3', spec, root, root)
                self.assertEqual(bound['teacher_config']['seed'], 91001)
                self.assertNotIn('teacher_seed', bound['native_manifest'])
                self.assertEqual(bound['native_manifest'], manifest)
                self.assertTrue(parity.called)
                # Changing both calibration metadata and manifest cannot repair
                # a false median: actual full native q is checked independently.
                manifest['q_ref'] = cal['q_ref'] = .4
                atomic_json(root / 'cal.json', cal)
                manifest['calibration_sha256'] = sha256(root / 'cal.json')
                atomic_json(root / 'reference.json', manifest)
                with self.assertRaisesRegex(ValueError, 'q/scales'):
                    assets._reference_binding(root / 'reference.json', 'R3_TA50', 's3', spec, root, root)

    def test_original_endpoint_u_and_a_required(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            model, identity, _ = endpoint(root)
            values = assets._finite_state(root / 'model.safetensors', identity)
            self.assertEqual(state_hash(values), state_hash(model.state_dict()))
            bad = {k: v.clone() for k, v in values.items() if k.startswith('backbone.')}
            save_file(bad, str(root / 'model.safetensors'))
            identity.update(model_sha256=sha256(root / 'model.safetensors'), state_hash=state_hash(bad))
            with self.assertRaisesRegex(ValueError, 'Both original U and A'):
                assets._finite_state(root / 'model.safetensors', identity)

    def test_original_exact_sampler_and_zero_lr_are_verified(self):
        cfg = dict(seed=91001, batch_size=48, num_iter=50000)
        full = progress(dict(update=50000, scheduler={'last_epoch': 50000}), cfg)
        assets._original_progress(full, cfg, 3072)
        full['sampler']['cursor'] += 1
        with self.assertRaisesRegex(ValueError, 'sampler progress'):
            assets._original_progress(full, cfg, 3072)
        full['sampler']['cursor'] -= 1
        full['optimizer']['param_groups'][1]['lr'] = 3e-7
        with self.assertRaisesRegex(ValueError, 'cosine'):
            assets._original_progress(full, cfg, 3072)

    def test_fullstate_cannot_substitute_best_or_parent_optimizer_progress(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            model, identity, full = endpoint(root)
            self.assertEqual(assets._fullstate(root / 'training_state.pt', identity,
                              model.state_dict())['update'], 100000)
            for mutation in ('scheduler', 'model', 'update'):
                bad = copy.deepcopy(full)
                if mutation == 'scheduler': bad['scheduler']['last_epoch'] = 50000
                if mutation == 'model': bad['model_state']['aligner.bias'].add_(1)
                if mutation == 'update': bad['update'] = 45450
                torch.save(bad, root / 'training_state.pt')
                identity['training_state_sha256'] = sha256(root / 'training_state.pt')
                with self.assertRaises(ValueError):
                    assets._fullstate(root / 'training_state.pt', identity, model.state_dict())

    def test_nonfinite_or_altered_bytes_never_accepted(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            model, identity, _ = endpoint(root)
            model.aligner.bias.data[0] = float('nan')
            save_file(model.state_dict(), str(root / 'model.safetensors'))
            with self.assertRaisesRegex(ValueError, 'bytes'):
                assets._finite_state(root / 'model.safetensors', identity)
            identity.update(model_sha256=sha256(root / 'model.safetensors'), state_hash=state_hash(model.state_dict()))
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                assets._finite_state(root / 'model.safetensors', identity)

    def test_binding_content_is_not_a_pass_only_claim(self):
        good = assets._signed(dict(schema=assets.SCHEMA, parent_id='P3HI'))
        assets._verify_signature(good, assets.SCHEMA)
        good['parent_id'] = 'P3LO'
        with self.assertRaises(ValueError): assets._verify_signature(good, assets.SCHEMA)

    def test_shared_source_and_runtime_no_head_waiver(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / 'shared.py'
            source.write_text('# fixture\n')
            origin = dict(files={'shared.py': sha256(source)}, torch='same', numpy='same')
            origin['content_sha256'] = object_sha(origin['files'])
            with patch.object(assets, 'NUMERICAL_PATHS', ('shared.py',)), patch.object(
                    assets, 'RUNTIME_KEYS', ('torch', 'numpy')), patch(
                    'gfb20.common.source_identity', return_value=origin):
                self.assertTrue(assets.source_compatibility(origin, root)['passed'])
                changed = dict(origin, torch='other')
                with self.assertRaisesRegex(ValueError, 'torch'):
                    assets.source_compatibility(changed, root)
                source.write_text('# changed\n')
                with self.assertRaisesRegex(ValueError, 'shared.py'):
                    assets.source_compatibility(origin, root)

    def test_discovery_does_not_invent_missing_paths(self):
        doc = dict(parents={'P3HI': dict(owner_server='s3', source_run_id='EXACT_ORIGINAL', parent_step=100000)})
        with tempfile.TemporaryDirectory() as name, patch.object(assets, 'registry', return_value=doc):
            root = Path(name)
            self.assertIsNone(assets.discover_bindings('s3', root)['P3HI'])
            run = root / 'work_dir/EXACT_ORIGINAL/meta'
            run.mkdir(parents=True)
            (run / 'config.resolved.yaml').write_text('{}')
            binding = assets.discover_bindings('s3', root)['P3HI']
            self.assertTrue(binding['parent_checkpoint'].endswith('/candidates/100000/model.safetensors'))
            # Discovery is only a proposed input map, not a verified parent.
            self.assertNotIn('passed', binding)
            with self.assertRaises(ValueError): assets.discover_bindings('s1', root)

    def test_unbound_and_nonlocal_parent_fail_closed(self):
        with self.assertRaises(ValueError): assets.bind_parent('P5A', 's3', {})
        with self.assertRaises(FileNotFoundError): assets.bind_parent('P3OLD', 's3', None)
        with self.assertRaises(ValueError): assets.bind_fresh_parent('P3NEW1', 's4', 'missing')
        with self.assertRaises(ValueError): assets.bind_fresh_parent('P3HI', 's3', 'missing')

    def test_reference_bridge_exception_is_not_inherited(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'ref.json'
            atomic_json(path, dict(schema='G20_REFERENCE_BRIDGE_v1', complete=True,
                output_parity={'passed': False}, acceptance_status='ACCEPTED_WITH_PARITY_EXCEPTION'))
            with self.assertRaisesRegex(ValueError, 'passed measured parity'):
                assets._reference_binding(path, 'R3_TA50', 's3', {}, name, name)

    def test_load_parent_preserves_own_aligner_not_teacher(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            model, identity, _ = endpoint(root)
            evidence = dict(parent_checkpoint=str(root / 'model.safetensors'), parent_seed=95002)
            captured = {}
            def builder(*args, **kwargs):
                captured.update(kwargs)
                return TinyPair(), {}
            with patch.object(assets, 'verify_parent', return_value=evidence), patch.object(assets, 'build_model', side_effect=builder):
                loaded, actual = assets.load_parent(evidence)
            self.assertEqual(state_hash(loaded.state_dict()), state_hash(model.state_dict()))
            self.assertEqual(state_hash(captured['teacher_aligner_state']), state_hash(model.aligner.state_dict()))
            self.assertIs(actual, evidence)


if __name__ == '__main__':
    unittest.main()
