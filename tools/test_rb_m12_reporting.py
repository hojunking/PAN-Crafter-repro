"""CPU-only M12 reporting/Sheet transport regressions; never authenticates."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from reporting_bridge import rb_m12_contract as c
from reporting_bridge import rb_m12_sheets as s
from panda_rb_m12.analysis import analyze, holm_family, paired_statistics, radius_summary, paired_modes, stats

FORMULA = ('=LET(allrows,legacy_parser(),LET(rbload,LAMBDA(tab,IFNA(FILTER(INDIRECT("\'"&tab&"\'!A2:BL"),'
           'INDIRECT("\'"&tab&"\'!BH2:BH")<>""),MAKEARRAY(1,64,LAMBDA(rr,cc,"")))),' + s.BASE_UNION + ','
           'LET(kept,FILTER(joined,CHOOSECOLS(joined,1)<>""),routebucket,IF(REGEXMATCH(TO_TEXT(CHOOSECOLS(kept,28)),"^(02|03|04|06|10) [|] "),"Ablations",CHOOSECOLS(kept,27)),'
           'HSTACK(CHOOSECOLS(kept,SEQUENCE(1,26)),routebucket,CHOOSECOLS(kept,SEQUENCE(1,37,28)))))))')


def record(case=None, selector='EXACT_50000'):
    case = case or c.registry()[0]
    return dict(case, selection_id=selector, checkpoint_sha256='a'*64, protocol=c.NATIVE_PROTOCOL,
                update=50000, alias_of='EXACT_50000' if selector != 'EXACT_50000' else None,
                metrics={key: .5 for key in c.METRICS},
                provenance={key: 'b'*64 for key in (*c.COHORT_KEYS, 'initialization_sha256', 'consumed_stream_sha256')}, completed_at_utc='2026-10-01T00:00:00Z')


class Fake:
    def __init__(self):
        self.props = {name: {'title': name, 'sheetId': sid, 'gridProperties': {'rowCount': 12000 if name == '_records' else 3000, 'columnCount': 64}}
                      for name, sid in s.FIXED_IDS.items()}
        self.meta = []; self.cells = {}; self.live_formula = FORMULA; self.writes = []
        self.fail_after_write = False; self.main_leak = False; self.no_view = False
        for name, index in (('_records', 1), ('WV3-ablations', 5), ('WV3-main', 5)):
            self.cells[name, index] = list(c.NATIVE_HEADERS)

    def metadata(self):
        return {'spreadsheetId': c.SPREADSHEET_ID, 'sheets': [{'properties': copy.deepcopy(p)} for p in self.props.values()], 'developerMetadata': copy.deepcopy(self.meta)}

    def formula(self):
        return self.live_formula

    def formulas(self, name):
        return self.values(name)

    def values(self, name):
        title, span = name.split('!'); title = title.strip("'")
        bounds = re.fullmatch('A([0-9]+):[A-Z]+([0-9]+)', span)
        start, end = map(int, bounds.groups())
        cells = {index: value for (tab, index), value in self.cells.items() if tab == title}
        if title in s.FIXED_IDS and start == 2:
            if title != 'WV3-main' or self.main_leak:
                if not self.no_view:
                    native = [v for (tab, index), v in sorted(self.cells.items()) if tab in ('_rb_m12_s1', '_rb_m12_s3', '_rb_m12_s5') and index >= 2]
                    cells.update({i+6: v for i, v in enumerate(native)})
        rows = [copy.deepcopy(cells.get(i, [])) for i in range(start, end+1)]
        while rows and not rows[-1]:
            rows.pop()
        return rows

    def structural(self, requests):
        self.writes.extend(copy.deepcopy(requests))
        for request in requests:
            if 'addSheet' in request:
                p = request['addSheet']['properties']; self.props[p['title']] = copy.deepcopy(p)
            elif 'createDeveloperMetadata' in request:
                self.meta.append(copy.deepcopy(request['createDeveloperMetadata']['developerMetadata']))
            else:
                body = request['appendDimension']
                p = next(p for p in self.props.values() if p['sheetId'] == body['sheetId'])
                p['gridProperties']['rowCount'] += body['length']

    def write_raw(self, entries):
        for name, rows in entries:
            title, span = name.split('!'); title = title.strip("'")
            assert title in s.SPECS
            start = int(re.match('A([0-9]+)', span)[1])
            for index, row in enumerate(rows, start):
                self.cells[title, index] = list(row)
            self.writes.append((name, copy.deepcopy(rows)))
        if self.fail_after_write:
            self.fail_after_write = False
            raise TimeoutError('Network timed out after server committed RAW')

    def write_formula(self, before, after):
        assert self.live_formula == before
        self.live_formula = after; self.writes.append(('_records!A2', after))


class ContractTests(unittest.TestCase):
    def test_registry_120_fresh(self):
        self.assertEqual(len(c.registry()), 120)
        self.assertEqual(len({r['run_id'] for r in c.registry()}), 120)
        self.assertFalse(any(r['run_id'].startswith('RB01') for r in c.registry()))

    def test_native_slots_80_each(self):
        for server in c.SERVERS:
            slots = [c.source_row(record(r, sel)) for r in c.registry() if r['server'] == server for sel in c.SELECTIONS]
            self.assertEqual(sorted(slots), list(range(2, 82)))

    def test_points_14112_no_fitting(self):
        slots = [c.point_row(dict(r, mode=m, shift_id=f'D{i:03d}')) for r in c.registry() if r['case_id'] in c.CASES[:6] for m in c.MODES for i in range(49)]
        self.assertEqual(sorted(slots), list(range(2, 14114)))
        with self.assertRaises(ValueError):
            c.point_row(dict(c.registry()[6], mode='A_ON', shift_id='D000'))

    def test_native_mapping_alias_and_sheet_metadata(self):
        row = c.build_native_row(record(), 271001100)
        self.assertEqual(len(row), 64)
        self.assertEqual(row[25:28], ['WV3', 'Ablations', '06 | Rebuttal M12 | 01 q cue and routing'])
        self.assertIn('gid=271001100&range=A2:BL2', row[58])
        self.assertEqual(row[42], 'EXACT_50000')
        fitting = c.build_native_row(record(c.registry()[6]), 271001100)
        self.assertEqual(fitting[27], '06 | Rebuttal M12 | 02 hard-soft fitting')

    def test_actual_checkpoint_and_selector_in_id(self):
        a = record(); b = record(selector='RR_VAL_ERGAS_MIN')
        self.assertNotEqual(c.result_id(a), c.result_id(b))
        b = dict(a, checkpoint_sha256='c'*64)
        self.assertNotEqual(c.result_id(a), c.result_id(b))

    def test_wrong_old_campaign_unknownseed_forbidden(self):
        for update in ({'campaign_id': 'oldB01'}, {'seed': 1}, {'run_id': 'RB01_fake'}, {'repeat': True}):
            with self.assertRaises(ValueError):
                c.validate_case(dict(record(), **update))

    def test_stress_protocol_never_native(self):
        with self.assertRaises(ValueError):
            c.result_id(dict(record(), protocol=c.STRESS_PROTOCOL))


class StatisticalTests(unittest.TestCase):
    def test_sample_sd_and_nonfinite(self):
        self.assertEqual(stats([1, 3])['sample_std'], math.sqrt(2))
        with self.assertRaises(ValueError):
            stats([float('nan')])

    def test_paired_t_and_bootstrap_deterministic(self):
        items = [dict(server=server, repeat=i+1, seed=i, delta=float(i+1)) for server in c.SERVERS for i in range(4)]
        a = paired_statistics(items, 'hqnr'); b = paired_statistics(items, 'hqnr')
        self.assertEqual(a, b)
        self.assertEqual(a['n'], 12)
        self.assertEqual(a['improved'], 12)
        self.assertEqual(a['bootstrap_stratum_sizes'], {'s1': 4, 's3': 4, 's5': 4})
        self.assertAlmostEqual(sum(a['ci95'])/2, 2.5)

    def test_holm_preserves_family_missing(self):
        tests = [dict(p_two_sided=p) for p in [.01, .04, .03, .001, .1, .5, None, None]]
        holm_family(tests)
        self.assertEqual(tests[3]['p_holm_primary8'], .008)
        self.assertEqual(tests[0]['p_holm_primary8'], .07)
        self.assertIsNone(tests[-1]['p_holm_primary8'])

    def test_aliases_do_not_double_students_and_factorial(self):
        rows = []
        values = {'QMEAN': 1., 'QEDGE': 3., 'QALIGN': 4., 'QFULL': 10.}
        for case in c.registry():
            for selector in c.SELECTIONS:
                r = record(case, selector); r['metrics'] = {'hqnr': values.get(case['case_id'], 0.), 'ergas': 2.}
                rows.append(r)
        report = analyze(rows)
        full = next(r for r in report['native_summary'] if r['case_id'] == 'QFULL' and r['server'] == 'ALL')
        self.assertEqual(full['metrics']['hqnr']['n'], 12)
        effects = {r['contrast']: r for r in report['contrasts'] if r['selection_id'] == 'EXACT_50000'}
        self.assertEqual(effects['EDGE_MAIN_EFFECT']['metrics']['hqnr']['mean'], 4.)
        self.assertEqual(effects['ALIGNER_MAIN_EFFECT']['metrics']['hqnr']['mean'], 5.)
        self.assertEqual(effects['GEOMETRY_INTERACTION']['metrics']['hqnr']['mean'], 4.)
        self.assertEqual(len(report['primary_endpoint_tests']), 8)

    def test_cohort_and_duplicate_guards(self):
        a = record(); b = record(c.registry()[1]); b['provenance']['teacher_sha256'] = 'c'*64
        with self.assertRaises(ValueError):
            analyze([a, b])
        with self.assertRaises(ValueError):
            analyze([a, a])
        b = record(c.registry()[1]); b['provenance']['consumed_stream_sha256'] = 'e'*64
        with self.assertRaisesRegex(ValueError, 'same initial states'):
            analyze([a, b])

    def test_coverage_requires_all_scenes_all_directions(self):
        points = [dict(radius_hr=.25, n_scenes=20, n_valid=20, clean=True, ergas=1.) for _ in range(8)]
        # radius_summary expects a complete fixed grid even when a direction is flagged.
        grid = [dict(radius_hr=0., n_scenes=20, n_valid=20, clean=True, ergas=1.)]
        for radius in (.25, .5, 1., 2., 3., 4.):
            grid.extend([dict(p, radius_hr=radius) for p in points])
        grid[1]['n_valid'] = 19; grid[1]['clean'] = False
        summary = radius_summary([dict(case_id='QFULL', mode='A_ON', points=grid)], ('s1',))
        radius = next(r for r in summary if r['radius_hr'] == .25)
        self.assertEqual(radius['clean']['ergas']['n'], 0)
        self.assertEqual(radius['flagged_all']['ergas']['n'], 1)
        self.assertEqual(radius['invalid_scene_directions'], 1)

    def test_modes_paired_within_student_and_checkpoint(self):
        grid = [dict(id=f'D{i:03d}', radius_hr=0. if i == 0 else (.25, .5, 1., 2., 3., 4.)[(i-1)//8],
                     n_scenes=20, n_valid=20, clean=True, ergas=3.) for i in range(49)]
        a = dict(case_id='QFULL', server='s1', repeat=1, seed=261001101, mode='A_ON', points=grid, checkpoint_sha256='a'*64)
        b = dict(a, mode='A_NATIVE_FIXED', points=[dict(p, ergas=4.) for p in grid])
        rows = paired_modes([a, b], ('s1',))
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[0]['metrics']['ergas']['clean']['mean'], -1.)
        self.assertEqual(rows[0]['n_paired_students'], 1)
        with self.assertRaises(ValueError):
            paired_modes([a, dict(b, checkpoint_sha256='b'*64)], ('s1',))


class SheetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.fake = Fake(); self.uploader = s.Uploader(self.fake, self.tmp.name)

    def test_patch_minimal_and_idempotent(self):
        after = s.patch_formula(FORMULA)
        self.assertEqual(after.replace(s.M12_UNION, s.BASE_UNION), FORMULA)
        self.assertEqual(s.patch_formula(after), after)

    def test_unknown_formula_and_old_no_routing_rejected(self):
        for formula in ('=old()', FORMULA.replace('routebucket', 'bad'), FORMULA.replace(s.BASE_UNION, s.BASE_UNION + s.BASE_UNION)):
            with self.assertRaises(ValueError):
                s.patch_formula(formula)

    def test_setup_dry_then_activate_then_noop(self):
        dry = self.uploader.setup()
        self.assertTrue(dry['formula_changed']); self.assertFalse(self.fake.writes)
        self.uploader.setup(activate=True)
        self.assertEqual(len(self.fake.props), 8)
        self.assertEqual(self.fake.props['_rb_m12_points']['gridProperties']['rowCount'], 14113)
        noop = self.uploader.setup()
        self.assertEqual(noop['structural_requests'], [])
        self.assertEqual(noop['header_writes'], 0)
        self.assertFalse(noop['formula_changed'])

    def test_foreign_header_and_ownership_rejected(self):
        self.uploader.setup(activate=True)
        self.fake.cells['_rb_m12_s1', 1][0] = 'unknown'
        with self.assertRaises(ValueError):
            self.uploader.setup(activate=True)

    def test_sheet_nested_ownership_metadata_is_accepted(self):
        self.uploader.setup(activate=True)
        original = self.fake.metadata
        def nested():
            value = original(); entries = value.pop('developerMetadata')
            for sheet in value['sheets']:
                sheet['developerMetadata'] = [m for m in entries if m['location']['sheetId'] == sheet['properties']['sheetId']]
            return value
        self.fake.metadata = nested
        self.assertFalse(self.uploader.setup()['formula_changed'])
        self.assertEqual(self.uploader.sync([record()], activate=True)['status'], 'VERIFIED')

    def test_native_retry_no_new_rows_and_manual_review(self):
        self.uploader.setup(activate=True)
        first = self.uploader.sync([record()], activate=True)
        self.assertEqual(first['changed_rows'], 1)
        self.fake.cells['_rb_m12_s1', 2][61] = 'review kept'
        second = self.uploader.sync([record()], activate=True)
        self.assertEqual(second['changed_rows'], 0)
        self.assertEqual(second['status'], 'VERIFIED')
        self.assertEqual(self.fake.cells['_rb_m12_s1', 2][61], 'review kept')

    def test_post_commit_network_failure_retries_without_duplicate(self):
        self.uploader.setup(activate=True)
        self.fake.fail_after_write = True
        with self.assertRaises(TimeoutError):
            self.uploader.sync([record()], activate=True)
        result = self.uploader.sync([record()], activate=True)
        self.assertEqual(result['changed_rows'], 0)
        self.assertEqual(result['status'], 'VERIFIED')

    def test_changed_checkpoint_same_slot_conflicts(self):
        self.uploader.setup(activate=True)
        self.uploader.sync([record()], activate=True)
        with self.assertRaisesRegex(ValueError, 'EVIDENCE_CONFLICT'):
            self.uploader.sync([dict(record(), checkpoint_sha256='d'*64)], activate=True)

    def test_main_leak_not_verified(self):
        self.uploader.setup(activate=True); self.fake.main_leak = True
        with self.assertRaisesRegex(ValueError, 'WV3-main'):
            self.uploader.sync([record()], activate=True)

    def test_source_formula_cannot_be_overwritten(self):
        self.uploader.setup(activate=True)
        self.fake.cells['_rb_m12_s1', 2] = ['=ARRAYFORMULA(foo)']
        with self.assertRaisesRegex(ValueError, 'formula/spill'):
            self.uploader.sync([record()], activate=True)

    def test_unread_view_does_not_claim_verified(self):
        self.uploader.setup(activate=True); self.fake.no_view = True
        with self.assertRaisesRegex(ValueError, 'readback'):
            self.uploader.sync([record()], activate=True)
        self.assertEqual(self.fake.cells['_rb_m12_s1', 2][51], 'UPLOAD_PENDING')

    def test_capacity_stress_pending_does_not_block_native(self):
        self.fake.props['unrelated'] = {'title': 'unrelated', 'sheetId': 99,
                                        'gridProperties': {'rowCount': 83000, 'columnCount': 100}}
        prepared = self.uploader.setup(activate=True)
        self.assertTrue(prepared['stress_source_pending_capacity'])
        self.assertNotIn('_rb_m12_points', self.fake.props)
        pending = dict(c.registry()[0], mode='A_ON', shift_id='D000')
        result = self.uploader.sync([record()], [pending], activate=True)
        self.assertEqual(result['status'], 'VERIFIED')
        self.assertEqual(result['stress_source_pending_points'], 1)

    def test_s3_and_s5_writer_forbidden(self):
        for server in ('s3', 's5'):
            with self.assertRaises(ValueError):
                s.Uploader(self.fake, self.tmp.name, writer_server=server)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        from panda_rb_m12 import reporting as r
        self.r = r
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.wd = self.root / 'run'; self.wd.mkdir()
        self.case = c.registry()[0]
        self.selection = dict(complete=True, actual_updates=50000,
                              primary=dict(selection_id='EXACT_50000', update=50000, checkpoint_sha256='a'*64))
        (self.wd / 'checkpoints').mkdir()
        self.r.atomic_json(self.wd / 'checkpoints/selection_manifest.json', self.selection)
        self.cfg = {'panda_rb_m12': {'alpha': 1., 'beta': .1, 'lambda_E': .002, 'aligner_lr': 3e-6}, 'learning_rate': 1e-4}
        self.binding = {'q_ref': .1, 'tau_R': .2}
        self.context = dict(case=self.case, config_sha256=r.object_sha(self.cfg),
                            binding_sha256=r.object_sha(self.binding), binding_common_sha256='b'*64,
                            data_content_identity={'rr': {'count': 20}, 'fr': {'count': 20}},
                            evaluator_identity={'content_sha256': 'c'*64}, source_identity={'content_sha256': 'd'*64},
                            teacher_sha256='e'*64, raw_map_identity_sha256='f'*64,
                            initialization_sha256='1'*64, consumed_stream_sha256='2'*64,
                            selection_manifest_sha256=r.sha256(self.wd / 'checkpoints/selection_manifest.json'))
        for name, replacement in [('validate_plan', lambda root: True), ('training_runs', lambda root: [self.case]),
                                  ('run_dir', lambda run_id, root: self.wd)]:
            p = patch.object(r, name, replacement); p.start(); self.addCleanup(p.stop)

    def seal(self, payload):
        return dict(payload, payload_sha256=self.r.object_sha(payload))

    def native(self):
        import yaml
        (self.wd / 'native').mkdir(); (self.wd / 'meta').mkdir()
        (self.wd / 'native/per_scene.csv').write_text('scene,metric\n0,1\n')
        (self.wd / 'meta/config.resolved.yaml').write_text(yaml.safe_dump(self.cfg))
        self.r.atomic_json(self.wd / 'bindings.json', self.binding)
        selected = dict(selection_id='EXACT_50000', update=50000, checkpoint_sha256='a'*64, alias_of=None,
                        rr={k: .5 for k in ('ergas', 'scc', 'sam', 'psnr', 'ssim', 'q8')},
                        fr={k: .5 for k in ('hqnr', 'd_s', 'd_lambda')})
        return self.seal(dict(schema='PANDA_M12_NATIVE_v1', complete=True, context=self.context,
                             selections={'EXACT_50000': selected, 'RR_VAL_ERGAS_MIN': dict(selected, selection_id='RR_VAL_ERGAS_MIN', update=1010, alias_of='EXACT_50000')},
                             file_hashes={'native/per_scene.csv': self.r.sha256(self.wd / 'native/per_scene.csv')},
                             elapsed_seconds=1., completed_at_utc='2026-10-01T00:00:00Z'))

    def test_same_weights_different_selected_step_preserved(self):
        report = self.native()
        rows = self.r.native_records(self.case, report)
        self.assertEqual(rows[1]['update'], 1010)
        self.assertEqual(rows[1]['alias_of'], 'EXACT_50000')
        self.r.atomic_json(self.wd / 'native/metrics.json', report)
        result = self.r.summarize(self.root, ('s1',))
        self.assertEqual(result['completed_students'], 1)
        self.assertEqual(result['completed_native'], 2)
        self.assertEqual(result['rows'][0]['config_fields']['q_ref'], .1)
        self.assertEqual(result['rows'][0]['config_fields']['teacher_reference'], 'F1 exact50K / ' + 'e'*12)

    def test_changed_config_or_binding_rejected(self):
        report = self.native(); self.r.atomic_json(self.wd / 'native/metrics.json', report)
        self.r.atomic_json(self.wd / 'bindings.json', dict(self.binding, q_ref=.3))
        with self.assertRaisesRegex(ValueError, 'binding hash'):
            self.r.summarize(self.root, ('s1',))

    def test_render_and_package_measured_evidence_only(self):
        report = self.native(); self.r.atomic_json(self.wd / 'native/metrics.json', report)
        for name in ('init_manifest.json', 'stream_manifest.json', 'case.json'):
            self.r.atomic_json(self.wd / name, {'fixture': True})
        gathered = self.r.summarize(self.root, ('s1',))
        rendered = self.r.export_tables(gathered, self.root / 'review')
        self.assertIn('review_tables.md', rendered['files'])
        self.assertIn('n=1', (self.root / 'review/review_tables.md').read_text())
        self.assertIn('stress_fixed192_auxiliary.csv', rendered['files'])
        result = self.r.package(self.root, ('s1',), self.root / 'return.zip')
        self.assertEqual(result['measured_students'], 1)
        with zipfile.ZipFile(self.root / 'return.zip') as archive:
            manifest = json.loads(archive.read('manifest.json'))
            self.assertIn('run/bindings.json', manifest['files'])
            self.assertIn('run/init_manifest.json', manifest['files'])
            self.assertFalse(any(name.endswith('.safetensors') for name in archive.namelist()))
        with self.assertRaises(FileExistsError):
            self.r.package(self.root, ('s1',), self.root / 'return.zip')

    def test_artifact_hash_and_escape_rejected(self):
        report = self.native(); (self.wd / 'native/per_scene.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            self.r.verify_artifacts(report, self.wd)
        bad = {k: v for k, v in report.items() if k != 'payload_sha256'}
        bad['file_hashes'] = {'../escape': 'a'*64}
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.r.verify_artifacts(self.seal(bad), self.wd)

    def test_stress_only_pending_native_does_not_block_summary(self):
        folder = self.wd / 'stress/A_ON'; folder.mkdir(parents=True)
        scalar = folder / 'per_scene.csv'; scalar.write_text('scene,metric\n0,1\n')
        points = []
        for i in range(49):
            points.append(dict(id=f'D{i:03d}', radius_hr=0. if i == 0 else (.25, .5, 1., 2., 3., 4.)[(i-1)//8],
                               dy=0., dx=0., angle_deg=None if i == 0 else ((i-1)%8)*45,
                               n_scenes=20, n_expected=20, n_valid=20, n_failures=0, clean=True, ergas=1.))
        curve = self.seal(dict(schema='PANDA_M12_CURVE_v1', complete=True,
            identity=dict(context=self.context, checkpoint_sha256='a'*64, mode='A_ON', roi_primary='fixed160', grid_sha256='c'*64),
            n_observations=980, n_numerical_failures=0, n_invalid_geometry=0, n_auxiliary_failures=1,
            curve=points, auxiliary_curve=points, file_hashes={'stress/A_ON/per_scene.csv': self.r.sha256(scalar)}))
        self.r.atomic_json(folder / 'completion.json', curve)
        result = self.r.summarize(self.root, ('s1',))
        self.assertEqual(result['completed_students'], 0)
        self.assertEqual(result['completed_curves'], 1)
        self.assertEqual(result['outcome'], 'PROCESSED_WITH_FLAGS')
        self.assertEqual(result['auxiliary_flagged_curves'], 1)
        self.assertFalse(result['complete'])


if __name__ == '__main__':
    unittest.main()
