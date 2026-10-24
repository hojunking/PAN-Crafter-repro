"""Publication retains exact continuation evidence and excludes weight averaging."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from panda_follow30 import plan, reporting as R, publication as P, best_publication as B
from panda_follow30.contract import SOURCE_FIELDS, TRAINING_SCHEDULE_FIELDS


class ReportingContractTests(unittest.TestCase):
    def test_all_registered_individual_checkpoints_are_accepted(self):
        for server in ('s1', 's3'):
            for group in plan.groups(server):
                for case, replica in group['runs']:
                    with self.subTest(server=server, case=case, replica=replica):
                        cfg = plan.resolve_case(server, case, replica)
                        R._config(cfg)
                        self.assertEqual(cfg['model_variant'], 'RAW')
                        self.assertEqual(cfg['weight_averaging'], 'EXCLUDED_USER_DECISION_20261024')

    def test_each_scheduler_and_source_field_is_registry_bound(self):
        original = plan.resolve_case('s3', 'S100DIFFF15C30', 'R1')
        for key in TRAINING_SCHEDULE_FIELDS:
            field = 'lr' if key == 'lr_S' else key
            with self.subTest(training_field=field):
                cfg = deepcopy(original)
                value = cfg['training_spec'][field]
                cfg['training_spec'][field] = value+'_CHANGED' if isinstance(value, str) else value+1
                with self.assertRaises(R.ReportingError):
                    R._config(cfg)
        for key in SOURCE_FIELDS:
            with self.subTest(source_field=key):
                cfg = deepcopy(original)
                cfg[key] = 'DIFFERENT_SOURCE'
                with self.assertRaises(R.ReportingError):
                    R._config(cfg)

    def test_continuation_schedule_cannot_be_relabelled(self):
        cfg = plan.resolve_case('s3', 'S100DIFFF15C30', 'R1')
        cfg['schedule'] = 'SOURCE_THEN_CONSTANT'
        with self.assertRaises(R.ReportingError):
            R._config(cfg)

    def test_model_averaging_is_rejected_in_configuration(self):
        original = plan.resolve_case('s3', 'S100DIFFF15C30', 'R1')
        for key, value in (('model_variant', 'SWA'), ('model_variant', 'EMA'),
                           ('swa', {'enabled': True}), ('swa', {}),
                           ('averaging_manifest', {}), ('averaged_from_run_id', 'OTHER'),
                           ('averaging', {}), ('ema', {}), ('model_soup', {}),
                           ('weight_averaging', 'ENABLED')):
            with self.subTest(field=key, value=value):
                cfg = deepcopy(original)
                cfg[key] = value
                with self.assertRaises(R.ReportingError):
                    R._config(cfg)

    def test_raw_label_cannot_hide_averaged_artifact(self):
        cfg = plan.resolve_case('s1', 'S100DIFFUBILC30', 'R1')
        for key, value in (('model_variant', 'SWA'), ('averaging', {}),
                           ('averaging_manifest', {}), ('averaged_from_run_id', 'OTHER')):
            with self.subTest(field=key):
                candidate = {'record': {'source': {'extra': {
                    'configuration': cfg, 'artifact': {key: value}}}}}
                with self.assertRaisesRegex(ValueError, 'WEIGHT_AVERAGED_MODEL_EXCLUDED'):
                    P._artifact(candidate, '/tmp')

    def test_source_identity_and_path_cannot_cross_seed_or_step(self):
        original = plan.resolve_case('s1', 'S100DIFFF15C30', 'R2')
        for key, value in (('source_run_id', 'PNX30_OTHER'), ('source_replica', 'R1'),
                           ('source_attempt', 2), ('source_completed_update', 25000),
                           ('source_checkpoint_relative_path', 'work_dir/other.pt'),
                           ('initialization', 'FRESH')):
            with self.subTest(identity_field=key):
                cfg = deepcopy(original)
                cfg[key] = value
                with self.assertRaises(R.ReportingError):
                    R._config(cfg)

    def test_fresh_runs_cannot_claim_a_source(self):
        original = plan.resolve_case('s1', 'S100DIFFUBILC30', 'R1')
        for key, value in (('source_case', 'S100DIFFC30'), ('source_run_id', 'PNX30_OTHER'),
                           ('source_completed_update', 15000)):
            with self.subTest(field=key):
                cfg = deepcopy(original)
                cfg[key] = value
                with self.assertRaises(R.ReportingError):
                    R._config(cfg)

    def test_continuation_timing_counts_only_new_updates(self):
        cfg = plan.resolve_case('s1', 'S100DIFFF15C30', 'R1')
        lineage = R.update_lineage(cfg, 30000)
        self.assertEqual(lineage['local_updates'], 15000)
        self.assertEqual(lineage['inherited_updates'], 15000)
        self.assertEqual(lineage['lifetime_updates'], 30000)
        self.assertNotIn('shared_training_run_id', lineage)

    def test_all_joint_passes_and_qualified_or_fallback_representative_remain(self):
        cfg = plan.resolve_case('s1', 'S100DIFFUBILC30', 'R1')
        key = (cfg['campaign_id'], cfg['dataset'], cfg['server'], cfg['run_id'], cfg['attempt'])
        cases = [
            ([(.9585, 2.040), (.959, 2.039), (.960, 2.08)], [1000, 2000], 2000, B.SELECTOR),
            ([(.9585, 2.08), (.959, 2.05), (.9584, 2.01)], [2000], 2000, B.SELECTOR),
            ([(.957, 2.03), (.958, 2.10), (.9579, 2.00)], [2000], 2000, B.FALLBACK_SELECTOR),
        ]
        for metrics, expected_steps, representative_step, selector in cases:
            with self.subTest(metrics=metrics):
                sources = {}
                for i, (hqnr, ergas) in enumerate(metrics, 1):
                    ident = f'source-{i}'
                    candidate = {'key': key, 'step': i*1000, 'hqnr': hqnr, 'ergas': ergas,
                                 'state': 'a'*64, 'protocol': 'test', 'source_result_id': ident}
                    row = ['']*len(R.HEADERS)
                    row[R.INDEX['Original description / notes']] = R.canonical_json(
                        {'extra_parameters': {'configuration': cfg}})
                    sources[ident] = {'candidate': candidate, 'row': row, 'evaluation_evidence': {}}
                value = {'policy': P.POLICY, 'conflicts': [], 'rows': [], 'notes': [], 'selected': [],
                         'source_evidence': sources,
                         'runs': [{'run_key': key, 'last_logged_status': 'COMPLETED',
                                   'last_observed_at': '2026-10-24T10:00:00+00:00'}]}
                with patch.object(P, 'project_campaign', return_value=value), \
                     patch.object(P, '_artifact', return_value={'file_sha256': 'a'*64}), \
                     patch('panda_follow30.cost.attach_cost'):
                    result = B.project('/tmp', 's1')
                self.assertFalse(result['conflicts'])
                self.assertEqual(sorted(x['step'] for x in result['selected']), expected_steps)
                notes = [json.loads(row[R.INDEX['Original description / notes']])['publication']
                         for row in result['rows']]
                representatives = [n for n in notes if n['is_case_representative']]
                self.assertEqual(len(representatives), 1)
                self.assertEqual(representatives[0]['evaluation_step'], representative_step)
                self.assertEqual(representatives[0]['selection'], selector)
                self.assertEqual(result['publication_thresholds']['hqnr_min'], .9585)
                self.assertEqual(result['publication_thresholds']['ergas_max'], 2.040)


if __name__ == '__main__':
    unittest.main()
