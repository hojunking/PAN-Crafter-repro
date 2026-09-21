"""Local-owner and endpoint contract rejection tests, with no GPU or real assets."""
import tempfile
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from l100.common import atomic_json
from fh12.training import BatchStream
from l100.plan import REFERENCES, build_config, case_for
from l100.references import reference_path, validate_reference, data_signature, _validate_endpoint_progress


class ReferenceTests(unittest.TestCase):
    def test_no_cross_server_or_legacy_reference_paths(self):
        for reference_id, ref in REFERENCES.items():
            self.assertIn(reference_id, str(reference_path(reference_id, ref.server_id)))
            for server in ('s1', 's2', 's3', 's4', 's5'):
                if server != ref.server_id:
                    with self.assertRaisesRegex(ValueError, 'local Teacher'):
                        reference_path(reference_id, server)
        for legacy in ('R0', 'R1', 'GF2_TA', '../reference'):
            with self.assertRaises(ValueError):
                reference_path(legacy, 's4')

    def test_bridge_or_relabelled_manifest_rejected_before_artifact_load(self):
        reference_id = next(key for key, ref in REFERENCES.items() if ref.server_id == 's4')
        with tempfile.TemporaryDirectory() as temp:
            path = reference_path(reference_id, 's4', temp)
            atomic_json(path, dict(schema='G20_REFERENCE_BRIDGE_v1', complete=True, owner_server='s4'))
            with patch('l100.references.source_identity', return_value={'fixture': 'source'}):
                with self.assertRaisesRegex(ValueError, 'owner/endpoint'):
                    validate_reference(reference_id, 's4', temp)

    def test_path_independent_signature_keeps_data_order_identity(self):
        data = dict(sensor='GF2', num_bands=4, splits={'train': dict(count=19809, sha256='a',
                    lpan_sha256='b', sample_order_sha256='c', dataroot='/original/path')})
        other = dict(data, splits={'train': dict(data['splits']['train'], dataroot='/same/bytes/elsewhere')})
        self.assertEqual(data_signature(data), data_signature(other))
        other['splits']['train']['sample_order_sha256'] = 'different'
        self.assertNotEqual(data_signature(data), data_signature(other))

    def test_direct_calibration_endpoint_rejects_sampler_or_optimizer_tamper(self):
        case = case_for('L100I1-T03')
        cfg = build_config(case)
        stream = BatchStream(48, 48, case.seed)
        stream.epoch, stream.cursor = case.updates - 1, 1
        state = dict(sampler=stream.state_dict(), scheduler={'_last_lr': [0., 0.]},
                     optimizer={'param_groups': [dict(name='U', lr=0.), dict(name='A', lr=0.)]})
        _validate_endpoint_progress(state, cfg, case, 48)
        wrong = copy.deepcopy(state)
        wrong['sampler']['cursor'] = 0
        with self.assertRaisesRegex(ValueError, 'sampler'):
            _validate_endpoint_progress(wrong, cfg, case, 48)
        for field in ('optimizer', 'scheduler'):
            wrong = copy.deepcopy(state)
            if field == 'optimizer':
                wrong[field]['param_groups'][1]['lr'] = 1e-6
            else:
                wrong[field]['_last_lr'][0] = float('nan')
            with self.assertRaisesRegex(ValueError, 'optimizer LR'):
                _validate_endpoint_progress(wrong, cfg, case, 48)


if __name__ == '__main__':
    unittest.main()
