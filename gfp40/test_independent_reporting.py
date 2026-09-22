"""Fixed G025 confirmation aggregates independent server-local clocks."""
import copy
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest

from gfp40.common import atomic_json, object_sha, read_json, camp, run_dir
from gfp40.plan import (CAMPAIGN_ID, SERVERS, EXECUTION_POLICY, EXECUTION_POLICY_SHA256,
                       CONFIRMATION_FAMILY, registry_sha256, case_for)
from gfp40.policy import CampaignWindow
from gfp40.reporting import aggregate, confirmation_pairs
from gfp40.upload import flatten_summary


def case_row(identifier):
    case = case_for(identifier)
    metric = dict(rr={'ergas': .55, 'scc': .98, 'rmse': 4.123456789, 'cc': .99},
        fr={'hqnr': .953 if case.arm == 'MIX' else .95,
            'd_s': .027 if case.arm == 'MIX' else .03, 'd_lambda': .02, 'jqm': .88},
        update=case.updates, val_ergas=.55)
    return dict(campaign_id=CAMPAIGN_ID, complete=True, integrity_verified=True,
        run_id=case.run_id, case_id=case.case_id, server=case.server, stage=case.phase,
        profile=case.arm, family=case.calibration_family, stream_seed=case.stream_seed,
        local_updates=case.updates, parent_step=case.parent_step,
        lifetime_updates=case.lifetime_student_updates,
        parent_run_id=case.parent_run_id,reference_key=case.reference_key,
        execution_policy=copy.deepcopy(EXECUTION_POLICY),
        execution_policy_sha256=EXECUTION_POLICY_SHA256,
        registry_sha256=registry_sha256(), confirmation_family=CONFIRMATION_FAMILY,
        selections={name: copy.deepcopy(metric) for name in ('EXACT_FINAL', 'RR_VAL_SELECTED', 'RAW_AUX')})


def fixture_reports(root, pairs=2):
    paths = []
    for offset, server in enumerate(SERVERS):
        t0 = dt.datetime(2026, 9, 22, tzinfo=dt.timezone.utc) + dt.timedelta(hours=offset * 3)
        window = CampaignWindow(t0, server=server).to_dict()
        rows = {identifier: case_row(identifier) for pair in confirmation_pairs(server)[:pairs]
                for identifier in pair}
        body = dict(campaign_id=CAMPAIGN_ID, server=server, window=window,
            execution_policy=copy.deepcopy(EXECUTION_POLICY),
            execution_policy_sha256=EXECUTION_POLICY_SHA256, registry_sha256=registry_sha256(),
            confirmation_family=CONFIRMATION_FAMILY, rows=rows)
        path = Path(root) / (server + '.json')
        atomic_json(path, dict(body, evidence_sha256=object_sha(body)))
        paths.append(path)
    return paths


def rewrite(path, value):
    body = {key: val for key, val in value.items() if key != 'evidence_sha256'}
    atomic_json(path, dict(body, evidence_sha256=object_sha(body)))


class IndependentReportingTests(unittest.TestCase):
    def test_three_different_t0s_same_fixed_recipe_succeed(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = fixture_reports(folder)
            result = aggregate(paths, Path(folder) / 'aggregate.json')
            self.assertEqual(result['family'], 'G025')
            self.assertTrue(result['independent_server_clocks'])
            self.assertEqual(len({v['t0_utc'] for v in result['server_windows'].values()}), 3)
            self.assertTrue(result['primary_confirmation']['operational_success'])
            self.assertEqual(result['primary_confirmation']['completed_pairs'], 6)
            self.assertFalse(any(k.startswith('selection_lock') for k in result))

    def test_partial_primary_pairs_cannot_be_replaced_by_optional_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = fixture_reports(folder, pairs=3)
            report = read_json(paths[0])
            for identifier in confirmation_pairs('s3')[1]:
                del report['rows'][identifier]
            rewrite(paths[0], report)
            result = aggregate(paths, Path(folder) / 'aggregate.json')
            self.assertEqual(result['primary_confirmation']['completed_pairs'], 5)
            self.assertFalse(result['primary_confirmation']['operational_success'])

    def test_registry_policy_family_and_case_mismatch_rejected(self):
        for field, value in (('registry_sha256', '0' * 64),
                ('execution_policy_sha256', '0' * 64), ('confirmation_family', 'G050')):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                paths = fixture_reports(folder)
                report = read_json(paths[0]); report[field] = value; rewrite(paths[0], report)
                with self.assertRaises(ValueError): aggregate(paths, Path(folder) / 'bad.json')
        for field, value in (('family', 'G050'), ('run_id', 'OLD_MSTAR_RUN'),
                ('stream_seed', 1234), ('stage', 'LOCKED_CONFIRM'),
                ('registry_sha256', '0' * 64), ('execution_policy', {'clock': 'SHARED'})):
            with self.subTest(row_field=field), tempfile.TemporaryDirectory() as folder:
                paths = fixture_reports(folder); report = read_json(paths[0])
                report['rows']['A12'][field] = value; rewrite(paths[0], report)
                with self.assertRaises(ValueError): aggregate(paths, Path(folder) / 'bad.json')

    def test_cross_server_window_or_changed_horizon_is_rejected(self):
        for mutation in ('server', 'duration'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                paths = fixture_reports(folder); report = read_json(paths[0])
                if mutation == 'server': report['window'] = read_json(paths[1])['window']
                else: report['window']['deadline_utc'] = '2026-09-30T00:00:00+00:00'
                rewrite(paths[0], report)
                with self.assertRaises(ValueError): aggregate(paths, Path(folder) / 'bad.json')

    def test_shared_lock_evidence_is_rejected_not_silently_reclassified(self):
        for where in ('report', 'row'):
            with self.subTest(where=where), tempfile.TemporaryDirectory() as folder:
                paths = fixture_reports(folder); report = read_json(paths[0])
                target = report if where == 'report' else report['rows']['A12']
                target['selection_lock_sha256'] = 'a' * 64; rewrite(paths[0], report)
                with self.assertRaises(ValueError): aggregate(paths, Path(folder) / 'bad.json')

    def test_upload_fields_preserve_metrics_and_report_local_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); case = case_for('A12')
            window = CampaignWindow('2026-09-22T02:00:00+00:00', server='s3').to_dict()
            atomic_json(camp(root, 's3') / 'campaign_window.json', window)
            values = flatten_summary(case_row('A12'), case, root)
            self.assertEqual(values['confirmation_family'], 'G025')
            self.assertEqual(values['execution_policy_sha256'], EXECUTION_POLICY_SHA256)
            self.assertEqual(json.loads(values['local_window']), window)
            self.assertEqual(values['local_window_sha256'], object_sha(window))
            self.assertTrue(values['independent_server_clock'])
            self.assertEqual(values['EXACT_FINAL rmse'], 4.123456789)
            self.assertEqual(values['EXACT_FINAL cc'], .99)
            self.assertEqual(values['EXACT_FINAL jqm'], .88)
            self.assertFalse(any(k.startswith('selection_lock') for k in values))
            unchanged = flatten_summary(case_row('A07'), case_for('A07'), root)
            self.assertEqual(unchanged['source_design_run_id'], case_for('A07').run_id)
            self.assertEqual(unchanged['source_design_phase'], case_for('A07').phase)


if __name__ == '__main__':
    unittest.main()
