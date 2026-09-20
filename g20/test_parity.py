from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fh12.common import atomic_json
from g20.parity import compare_metrics, verify_r0_parity, discover_parity_evidence, _metric_pair


class EvaluatorParityTests(unittest.TestCase):
    def test_both_thresholds_are_required(self):
        baseline = dict(hqnr=.96, ergas=.55)
        self.assertTrue(compare_metrics(baseline, dict(hqnr=.960009, ergas=.55009))['passed'])
        self.assertFalse(compare_metrics(baseline, dict(hqnr=.960011, ergas=.55))['passed'])
        self.assertFalse(compare_metrics(baseline, dict(hqnr=.96, ergas=.55011))['passed'])

    def test_rounded_spreadsheet_metrics_are_not_evidence(self):
        with self.assertRaisesRegex(ValueError, 'official GF2'):
            _metric_pair(dict(rr={'ergas': .55}, fr={'hqnr': .96}))

    def test_absent_evidence_reports_wait_not_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            result = verify_r0_parity(folder, 's3')
            self.assertEqual(result['status'], 'WAIT_PARITY')
            self.assertFalse(result['passed'])

    def test_single_asset_cannot_certify_two_asset_parity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'evidence.json'
            atomic_json(path, dict(schema='G20_R0_EVALUATOR_EVIDENCE_v1', assets={'TA': {}}))
            with self.assertRaisesRegex(ValueError, 'two explicit'):
                verify_r0_parity(folder, 's3', evidence_path=path)

    def test_default_discovery_binds_evidence_but_never_fabricates_pass(self):
        from g20.common import camp, read_json
        with tempfile.TemporaryDirectory() as folder:
            atomic_json(camp(folder, 's4') / 'dataset_manifest.json', {'local': True})
            evidence = dict(schema='G20_R0_EVALUATOR_EVIDENCE_v1', assets={'TA': {}, 'S92001': {}})
            with patch('g20.parity.discover_parity_evidence', return_value=dict(
                    status='EVIDENCE_READY_NOT_REPLAYED', evidence=evidence)), \
                 patch('g20.parity._prior', side_effect=FileNotFoundError('Asset disappeared after discovery')):
                result = verify_r0_parity(folder, 's4')
            self.assertEqual(result['status'], 'WAIT_PARITY')
            self.assertEqual(read_json(camp(folder, 's4') / 'r0_parity_evidence.json'), evidence)
            with patch('g20.parity.discover_parity_evidence') as discovery:
                verify_r0_parity(folder, 's4', evidence_path=Path(folder) / 'explicit-missing.json')
            discovery.assert_not_called()

    def test_discovery_is_readonly_and_binds_actual_asset_files(self):
        import yaml
        from g20.common import camp
        with tempfile.TemporaryDirectory() as folder:
            atomic_json(camp(folder, 's4') / 'dataset_manifest.json', {'local': True})
            self.assertEqual(discover_parity_evidence(folder, 's4')['status'], 'WAIT_PARITY')
            for label, role, seed in [('TA', 'T', 91001), ('S92001', 'S', 92001)]:
                run = (Path(folder) / 'work_dir/QGBASE_GF2_TA_P0_W112_D123_TS91001_FRESH50_v1'
                       if label == 'TA' else camp(folder, 's4') / 'parity_inputs' / label)
                (run / 'meta').mkdir(parents=True)
                (run / 'meta/config.resolved.yaml').write_text(yaml.safe_dump(dict(seed=seed,
                    qg40=dict(sensor='GF2', server_id='s3', role=role))))
                atomic_json(run / 'dataset_manifest.json', {'original': True})
                atomic_json(run / 'candidates/50000/identity.json', {'identity': label})
                (run / 'candidates/50000/model.safetensors').write_bytes(label.encode())
                atomic_json(run / 'official/exact50k.json', {'report': label})
            before = sorted(str(path) for path in Path(folder).rglob('*'))
            with patch('g20.parity._prior', return_value=None) as check:
                result = discover_parity_evidence(folder, 's4')
            self.assertEqual(result['status'], 'EVIDENCE_READY_NOT_REPLAYED')
            self.assertEqual(set(result['evidence']['assets']), {'TA', 'S92001'})
            self.assertEqual(check.call_count, 2)
            self.assertEqual(before, sorted(str(path) for path in Path(folder).rglob('*')))

    def test_both_actual_evaluations_are_required_and_one_mismatch_blocks(self):
        from g20.common import camp
        rr = dict(ergas=.55, q4=.99, n_scenes=20, crop='20:-21', q_block=32,
                  official_complete=True, sensor='GF2', num_bands=4, max_dn=1023)
        fr = dict(hqnr=.96, n_scenes=20, reference='native_PAN', support='full512', masking=False,
                  aggregation='mean_per_scene_HQNR', hqnr_variant='raw-original', official_complete=True,
                  sensor='GF2', num_bands=4, max_dn=1023)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'evidence.json'
            entry = dict(checkpoint_sha256='checkpoint', prior_report_sha256='report')
            atomic_json(path, dict(schema='G20_R0_EVALUATOR_EVIDENCE_v1', assets={'TA': entry, 'S92001': entry}))
            atomic_json(camp(folder, 's3') / 'dataset_manifest.json', {'bound': True})
            identity = {'identity': True}
            prior = ({}, {'checkpoint': Path(folder) / 'model.safetensors'}, dict(hqnr=.96, ergas=.55), {}, identity)
            with patch('g20.parity._prior', return_value=prior), \
                 patch('g20.parity._path_file'), patch('g20.parity.source_identity', return_value={'source': True}), \
                 patch('g20.data.build_dataset', return_value=object()), \
                 patch('qg40.common.load_checkpoint_model', return_value=(object(), identity)), \
                 patch('g20.evaluation.evaluate_model', side_effect=[dict(rr=rr, fr=fr), dict(rr=rr, fr=dict(fr, hqnr=.959))]) as evaluate:
                result = verify_r0_parity(folder, 's3', evidence_path=path)
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(result['status'], 'BLOCKED_PARITY')
            self.assertTrue(result['assets']['TA']['passed'])
            self.assertFalse(result['assets']['S92001']['passed'])

    def test_sealed_pass_rechecks_inputs_without_repeating_inference(self):
        from g20.common import camp
        rr = dict(ergas=.55, q4=.99, n_scenes=20, crop='20:-21', q_block=32,
                  official_complete=True, sensor='GF2', num_bands=4, max_dn=1023)
        fr = dict(hqnr=.96, n_scenes=20, reference='native_PAN', support='full512', masking=False,
                  aggregation='mean_per_scene_HQNR', hqnr_variant='raw-original', official_complete=True,
                  sensor='GF2', num_bands=4, max_dn=1023)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'evidence.json'
            entry = dict(checkpoint_sha256='checkpoint', prior_report_sha256='report')
            atomic_json(path, dict(schema='G20_R0_EVALUATOR_EVIDENCE_v1', assets={'TA': entry, 'S92001': entry}))
            atomic_json(camp(folder, 's3') / 'dataset_manifest.json', {'bound': True})
            identity = {'identity': True}
            prior = ({}, {'checkpoint': Path(folder) / 'model.safetensors'}, dict(hqnr=.96, ergas=.55), {}, identity)
            with patch('g20.parity._prior', return_value=prior) as validate, \
                 patch('g20.parity._path_file'), patch('g20.parity.source_identity', return_value={'source': True}) as source, \
                 patch('g20.data.build_dataset', return_value=object()), \
                 patch('qg40.common.load_checkpoint_model', return_value=(object(), identity)), \
                 patch('g20.evaluation.evaluate_model', return_value=dict(rr=rr, fr=fr)) as evaluate:
                first = verify_r0_parity(folder, 's3', evidence_path=path)
                self.assertTrue(first['passed'])
                calls = validate.call_count
                self.assertEqual(verify_r0_parity(folder, 's3', evidence_path=path), first)
                self.assertEqual(validate.call_count, calls + 2)
                self.assertEqual(evaluate.call_count, 2)
                source.return_value = {'source': 'changed'}
                with self.assertRaisesRegex(ValueError, 'Cached evaluator parity identity'):
                    verify_r0_parity(folder, 's3', evidence_path=path)
                target = camp(folder, 's3') / 'diagnostics/r0_evaluator_parity.json'
                atomic_json(target, dict(first, passed=False))
                with self.assertRaisesRegex(ValueError, 'PASS receipt changed'):
                    verify_r0_parity(folder, 's3', evidence_path=path)


if __name__ == '__main__':
    unittest.main()
