#!/usr/bin/env python3
"""FH20R1 upload integrity tests. Temporary JSON and fake Sheets only."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fh20r1 import upload as U
from fh20r1.common import atomic_json, object_sha, sha256, read_json
from fh20r1.plan import CAMPAIGN_ID, active_cases, build_config
from tools.fh12_upload_tests import FakeSheet


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fh20r1-upload-')
        self.root = Path(self.temp.name)
        self.case = active_cases('s3', include_reserve=False)[0]
        self.run = self.case.run_id; self.cfg = build_config(self.case)
        self.wd = self.root / 'work_dir' / self.run
        (self.wd / 'meta').mkdir(parents=True)
        (self.wd / 'meta/config.resolved.yaml').write_text(yaml.safe_dump(self.cfg))
        server = self.root / 'gspread/server.txt'
        server.parent.mkdir(); server.write_text('s3(5090)\n')
        self.source = dict(git_release='fixture', content_sha256='fixture-content')
        self.data = dict(recipe=dict(phase_id='offset2'),
                         splits={name: dict(lpan_sha256='LP-' + name) for name in ('train', 'val', 'rr', 'fr')})
        atomic_json(self.root / self.cfg['fh20r1']['dataset_manifest'], self.data)
        self.bridge_path = self.root / self.cfg['fh20r1']['reference_bridge']
        self.bridge = dict(alias=self.case.teacher_alias, server='s3', status='PASS', complete=True,
                           consumer_source_identity=self.source, teacher_checkpoint_sha256='original-exact50k',
                           tau_R=.02, q_ref=.5, origin_reference=dict(calibration_sha256='calibration-fixture'))
        atomic_json(self.bridge_path, self.bridge)
        self.context = dict(run_id=self.run, campaign_id=CAMPAIGN_ID, config_sha256=object_sha(self.cfg),
                            source_identity=self.source, data_sha256=object_sha(self.data),
                            official_complete=True, normal_same_step_A_U=True)
        for name, label, step in [('raw_max', 'RAW_MAX', 1010), ('target_selection', 'TARGET', 2020),
                                  ('exact50k', 'EXACT50K', 50000), ('rr_val_selected', 'RR_VAL_SELECTED', 3030),
                                  ('e_min_diag', 'E_MIN_DIAG50', 2020)]:
            folder = self.wd / 'candidates' / str(step); folder.mkdir(parents=True, exist_ok=True)
            (folder / 'model.safetensors').write_bytes(f'fixture-{step}'.encode())
            identity = dict(update=step, model_sha256=sha256(folder / 'model.safetensors'),
                            config_sha256=object_sha(self.cfg), data_sha256=object_sha(self.data),
                            source_identity=self.source, reference_sha256=object_sha(self.bridge))
            atomic_json(folder / 'identity.json', identity)
            rr = dict(ergas=2.05 if step == 1010 else 2.035, scc=.988, psnr=38., sam=2.7,
                      q8=.922, ssim=.976, official_complete=True)
            fr = dict(hqnr=.959 if step == 1010 else .9586, d_lambda=.016, d_s=.025)
            report = dict(self.context, selection_id=label, step=step, checkpoint_sha256=identity['model_sha256'],
                          checkpoint_identity=identity, rr=rr, fr=fr, val_ergas=2.1)
            if name == 'target_selection':
                report.update(n_evaluated=50, n_eligible=3, target_status='official', joint_pass=True)
            if name == 'e_min_diag': report.update(n_evaluated=50)
            atomic_json(self.wd / 'official' / f'{name}.json', report)
        atomic_json(self.wd / 'official/postrun_status.json', dict(self.context, actual_updates=50000))
        atomic_json(self.wd / 'official/profile.json', dict(config_sha256=object_sha(self.cfg),
                    source_identity=self.source, params_m=3., flops_g=100., infer_ms=9., mem_mb=200.,
                    scope='A+frontend+U', flops_convention='fixture MAC'))
        self.gu = types.SimpleNamespace(CRED='account.json', SHEET='fake-only', ORIGIN_ROW=2)
        self.requested = []

    def tearDown(self): self.temp.cleanup()

    def invoke(self, sheet):
        def worksheet(name): self.requested.append(name); return sheet
        book = types.SimpleNamespace(worksheet=worksheet)
        api = types.SimpleNamespace(service_account=lambda **_: types.SimpleNamespace(open=lambda _: book))
        with patch.object(U, 'legacy_constants', return_value=self.gu), \
             patch.object(U, 'source_identity', return_value=self.source), patch.dict(sys.modules, {'gspread': api}):
            return U.upload_run(self.run, self.root)

    def prepare_reuse(self):
        from fh12.plan import cases_for as old_cases, build_config as old_config, CAMPAIGN_ID as OLD_ID
        from fh12.postrun import select_records, report_selection
        from fh20r1.preflight import find_reusable_runs
        rows = self.prepare_grid()
        # No current shipped old/new case pair is accidentally duplicated. This
        # fixture simulates an additional legitimate historical same-seed case.
        old_case = replace(next(c for c in old_cases('s3') if c.role == 'S' and c.input_layout == self.case.input_layout
                                and c.width == self.case.width and c.depth == self.case.depth), student_seed=self.case.seed)
        import fh12.upload as legacy_upload
        mock_registry=patch.object(legacy_upload,'case_for',return_value=old_case)
        mock_registry.start(); self.addCleanup(mock_registry.stop)
        source = self.root / 'work_dir' / old_case.run_id
        self.wd.rename(source); (self.wd / 'meta').mkdir(parents=True)
        cfg = old_config(old_case)
        (source / 'meta/config.resolved.yaml').write_text(yaml.safe_dump(cfg))
        numeric = self.root / 'fh12/model.py'; numeric.parent.mkdir(); numeric.write_text('# fixture numerical source\n')
        self.source['files'] = {'fh12/model.py': sha256(numeric)}
        import fh20r1.preflight as preflight
        mock_source=patch.object(preflight,'source_identity',return_value=self.source)
        mock_source.start(); self.addCleanup(mock_source.stop)
        lp = self.root / 'work_dir/_fh12/s3/lpan_manifest.json'
        atomic_json(lp, dict(recipe=dict(id='fixture-LP', phase_id='offset2')))
        ref = dict(teacher_run_id=old_case.teacher_run_id, teacher_checkpoint_sha256=self.bridge['teacher_checkpoint_sha256'],
                   lpan_manifest_sha256=sha256(lp), tau_R=.02, q_ref=.5, calibration_sha256='fixture-cal', q_cache_sha256='fixture-q')
        atomic_json(self.root / 'work_dir/_fh12/s3/references' / old_case.teacher_run_id / 'reference_manifest.json', ref)
        self.bridge['origin_reference'] = ref; self.bridge['consumer_source_identity'] = self.source
        atomic_json(self.bridge_path, self.bridge)
        atomic_json(source / 'init_manifest.json', dict(teacher_ref_sha=self.bridge['teacher_checkpoint_sha256']))
        context = dict(campaign_id=OLD_ID, run_id=old_case.run_id, official_complete=True,
                       config_sha256=object_sha(cfg), data_sha256=object_sha(self.data), source_identity=self.source)
        for record in rows:
            record['checkpoint_identity'].update(config_sha256=object_sha(cfg), source_identity=self.source,
                                                 reference_sha256=object_sha(ref))
            atomic_json(source / 'candidates' / str(record['update']) / 'identity.json', record['checkpoint_identity'])
        grid = dict(context, records=rows, complete=True)
        atomic_json(source / 'official/raw_grid.json', grid)
        selected = select_records(rows)
        for name, key, label in [('raw_max','raw_max','RAW_MAX'), ('target_selection','target','TARGET'),
                                 ('exact50k','exact50k','EXACT50K'), ('rr_val_selected','rr_val_selected','RR_VAL_SELECTED'),
                                 ('e_min_diag','e_min_diag','E_MIN_DIAG50')]:
            extra = dict(n_evaluated=50)
            if key == 'target': extra.update(n_eligible=selected['n_eligible'], target_status=selected['target_status'])
            atomic_json(source / 'official' / f'{name}.json', report_selection(label, selected[key], context, **extra))
        atomic_json(source / 'official/postrun_status.json', dict(context, actual_updates=50000))
        profile = read_json(source / 'official/profile.json')
        profile.update(config_sha256=object_sha(cfg), source_identity=self.source)
        atomic_json(source / 'official/profile.json', profile)
        atomic_json(source / 'official/upload_receipt.json', dict(gid=U.GIDS['s3'], row=37, readback_verified=True))
        proof = find_reusable_runs(self.root, 's3', self.bridge)['runs'][self.run]
        atomic_json(self.wd / 'meta/reuse_reference.json', dict(campaign_id=CAMPAIGN_ID, run_id=self.run, **proof))
        return source

    def test_reuse_source_row_link_has_zero_new_updates_time_and_original_metrics(self):
        source = self.prepare_reuse()
        with patch.object(U, 'source_identity', return_value=self.source): values = U.row_values(self.run, self.root)
        self.assertTrue(values['FH20R1 reused']); self.assertEqual(values['FH20R1 source run'], source.name)
        self.assertEqual(values['FH20R1 source sheet row'], 37)
        self.assertEqual(values['FH20R1 actual updates'], 0); self.assertFalse(values['FH20R1 official complete'])
        self.assertEqual(values['FH20R1 source actual updates'], 50000)
        self.assertEqual(values['Train(h)'], 0); self.assertEqual(values['FH20R1 run effective seconds'], 0)
        self.assertEqual(values['RAW_MAX step'], 1010); self.assertEqual(values['Exact50K step'], 50000)
        self.assertEqual(values['FH20R1 q cache SHA256'], 'fixture-q')
        sheet = FakeSheet(['Run', 'Notes'], U.GIDS['s3']); sheet.cells[4] = [source.name, 'preserve original']
        self.assertEqual(self.invoke(sheet)['row'], 5); self.assertEqual(sheet.cells[4], [source.name, 'preserve original'])

    def test_reuse_upload_only_creates_link_receipt_without_training_or_inference(self):
        from fh20r1 import postrun as P
        source = self.prepare_reuse()
        with patch.object(U, 'source_identity', return_value=self.source), \
             patch.object(U, 'upload_run', return_value=dict(readback_verified=True, row=50)), \
             patch.object(P, 'load_checkpoint_model', side_effect=AssertionError('no repeated evaluation')):
            self.assertEqual(P.process(self.run, upload=True, upload_only=True, root=self.root), 0)
        status = read_json(self.wd / 'official/postrun_status.json')
        self.assertEqual(status['status'], 'REUSED_LINK_COMPLETE'); self.assertTrue(status['sheet_uploaded'])
        self.assertEqual(status['source_run_id'], source.name); self.assertEqual(status['actual_updates'], 0)
        self.assertFalse(status['official_complete']); self.assertEqual(status['credited_new_seconds'], 0)
        self.assertFalse((self.wd / 'official/raw_grid.json').exists())
        with self.assertRaisesRegex(ValueError, 'upload-only'): P.process(self.run, root=self.root)

    def test_reuse_rejects_changed_source_weights_proof_or_metrics(self):
        source = self.prepare_reuse()
        path = self.wd / 'meta/reuse_reference.json'; saved = read_json(path)
        with patch.object(U, 'source_identity', return_value=self.source):
            changed = dict(saved); changed['identity_sha256'] = 'wrong'; atomic_json(path, changed)
            with self.assertRaisesRegex(ValueError, 'proof'): U.row_values(self.run, self.root)
            atomic_json(path, saved)
            metric = source / 'official/raw_max.json'; report = read_json(metric); report['rr']['ergas'] = 9.
            atomic_json(metric, report)
            with self.assertRaisesRegex(ValueError, 'selected metrics'): U.row_values(self.run, self.root)
            (source / 'candidates/1010/model.safetensors').write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError, 'bytes'): U.row_values(self.run, self.root)

    def test_reuse_rejects_changed_numeric_source_or_calibration(self):
        self.prepare_reuse()
        with patch.object(U, 'source_identity', return_value=self.source):
            self.bridge['origin_reference']['q_ref'] = .9; atomic_json(self.bridge_path, self.bridge)
            with self.assertRaisesRegex(ValueError, 'calibration'): U.row_values(self.run, self.root)
            (self.root / 'fh12/model.py').write_text('# changed\n')
            with self.assertRaisesRegex(ValueError, 'numerical source'): U.row_values(self.run, self.root)

    def test_reuse_detection_rejects_missing_numerical_coverage_before_done_reused(self):
        from fh20r1.preflight import find_reusable_runs
        source = self.prepare_reuse()
        grid = read_json(source / 'official/raw_grid.json')
        grid['source_identity'] = dict(grid['source_identity'], files={})
        atomic_json(source / 'official/raw_grid.json', grid)
        status = read_json(source / 'official/postrun_status.json'); status['source_identity'] = grid['source_identity']
        atomic_json(source / 'official/postrun_status.json', status)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            find_reusable_runs(self.root, 's3', self.bridge)

    def test_runner_condition_is_used_for_branch_provenance(self):
        atomic_json(self.root / 'work_dir/_fh20r1/s3/branch_record.json',
                    dict(condition='PRIMARY', reason='fixture decision'))
        self.assertEqual(U.row_values(self.run, self.root)['FH20R1 branch'], 'PRIMARY')

    def test_all_nine_metrics_exact_stage_and_reference_provenance(self):
        values = U.row_values(self.run, self.root)
        for prefix in ('RAW_MAX', 'Target', 'Exact50K', 'RR_VAL_SELECTED', 'E_MIN_DIAG50'):
            for label in list(U.RR_LABELS) + list(U.FR_LABELS): self.assertIn(prefix + ' ' + label, values)
        self.assertEqual(values['ERGAS↓'], values['RAW_MAX ERGAS↓'])
        self.assertNotEqual(values['ERGAS↓'], values['Target ERGAS↓'])
        self.assertEqual(values['HQNR↑'], values['RAW_MAX HQNR(raw)↑'])
        self.assertEqual(values['FH20R1 Teacher SHA256'], 'original-exact50k')
        self.assertEqual(values['FH20R1 calibration SHA256'], 'calibration-fixture')
        self.assertEqual(values['FH20R1 tau_R'], .02)
        self.assertFalse(any('NOA' in key or 'JQM' in key or 'Q4' in key for key in values))

    def test_effective_time_breakdown_is_run_segment_union_not_category_sum(self):
        atomic_json(self.wd / 'meta/timing_committed.json', dict(segments=[
            dict(kind='train', start_utc='2026-09-19T00:00:00+00:00', end_utc='2026-09-19T00:00:20+00:00'),
            dict(kind='eval', start_utc='2026-09-19T00:00:10+00:00', end_utc='2026-09-19T00:00:30+00:00'),
            dict(kind='diagnostic', start_utc='2026-09-19T00:00:30+00:00', end_utc='2026-09-19T00:00:40+00:00')]))
        values = U.row_values(self.run, self.root)
        self.assertEqual(values['FH20R1 run effective seconds'], 40.)
        self.assertEqual(values['FH20R1 committed train seconds'], 20.)
        self.assertEqual(values['FH20R1 committed eval seconds'], 20.)
        self.assertEqual(values['FH20R1 committed diagnostic seconds'], 10.)

    def prepare_grid(self):
        """Fifty fixture candidates; reuse already-profiled cost, so never a model/GPU."""
        rows = []
        for step in U.GRID_STEPS:
            folder = self.wd / 'candidates' / str(step); folder.mkdir(parents=True, exist_ok=True)
            (folder / 'model.safetensors').write_bytes(f'fixture-{step}'.encode())
            identity = dict(update=step, model_sha256=sha256(folder / 'model.safetensors'),
                            config_sha256=object_sha(self.cfg), data_sha256=object_sha(self.data),
                            source_identity=self.source, reference_sha256=object_sha(self.bridge))
            atomic_json(folder / 'identity.json', identity)
            rows.append(dict(update=step, checkpoint_identity=identity,
                             rr=dict(ergas=2.05 - step / 1e7, scc=.988, psnr=38., sam=2.7,
                                     q8=.922, ssim=.976, official_complete=True),
                             fr=dict(hqnr=.959 - step / 1e8, d_lambda=.016, d_s=.025),
                             val_ergas=2.1 - step / 1e7))
        atomic_json(self.wd / 'official/raw_grid.json', dict(self.context, records=rows, complete=True))
        atomic_json(self.wd / 'meta/training_status.json', dict(actual_updates=50000, training_complete=True))
        return rows

    def test_postrun_fifty_grid_json_to_upload_rows_without_new_inference(self):
        from fh20r1 import postrun as P
        self.prepare_grid()
        with patch.object(P, 'source_identity', return_value=self.source), \
             patch.object(P, 'load_checkpoint_model', side_effect=AssertionError('no new inference')):
            self.assertEqual(P.process(self.run, device='cpu', root=self.root), 0)
        values = U.row_values(self.run, self.root)
        self.assertEqual(values['RAW_MAX step'], 1010)
        self.assertEqual(values['Exact50K step'], 50000)
        self.assertEqual(values['Target step'], 50000)
        self.assertEqual(values['Target n eligible'], 50)

    def test_postrun_wrong_checkpoint_stage_context_fails_before_reporting(self):
        from fh20r1 import postrun as P
        rows = self.prepare_grid(); rows[0]['checkpoint_identity']['data_sha256'] = 'wrong'
        atomic_json(self.wd / 'candidates/1010/identity.json', rows[0]['checkpoint_identity'])
        atomic_json(self.wd / 'official/raw_grid.json', dict(self.context, records=rows, complete=True))
        with patch.object(P, 'source_identity', return_value=self.source):
            with self.assertRaisesRegex(ValueError, 'identity'): P.process(self.run, device='cpu', root=self.root)

    def test_postrun_upload_failure_retains_official_outputs_and_returns_success(self):
        from fh20r1 import postrun as P
        self.prepare_grid()
        with patch.object(P, 'source_identity', return_value=self.source), \
             patch.object(U, 'upload_run', side_effect=RuntimeError('fake offline')):
            self.assertEqual(P.process(self.run, device='cpu', root=self.root, upload=True), 0)
        state = read_json(self.wd / 'official/postrun_status.json')
        self.assertTrue(state['official_complete']); self.assertFalse(state['sheet_uploaded'])
        self.assertEqual(state['status'], 'EVAL_COMPLETE_UPLOAD_PENDING')
        self.assertIn('fake offline', state['upload_error'])

    def test_no_eligible_target_is_explicit_blank_and_other_stages_preserved(self):
        path = self.wd / 'official/target_selection.json'
        atomic_json(path, dict(self.context, selection_id='TARGET', selection=None, n_eligible=0,
                              n_evaluated=50, target_status='no_eligible', joint_pass=False))
        values = U.row_values(self.run, self.root)
        self.assertEqual(values['Target ERGAS↓'], '')
        self.assertEqual(values['Target checkpoint SHA256'], '')
        self.assertNotEqual(values['Exact50K ERGAS↓'], '')

    def test_changed_weights_rejected(self):
        (self.wd / 'candidates/1010/model.safetensors').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'checkpoint'): U.row_values(self.run, self.root)

    def test_identity_context_even_if_report_copies_bad_identity_is_rejected(self):
        for key in ('config_sha256', 'data_sha256', 'source_identity'):
            with self.subTest(key=key):
                path = self.wd / 'official/raw_max.json'; saved = read_json(path)
                changed = copy.deepcopy(saved); changed['checkpoint_identity'][key] = 'wrong'
                atomic_json(path, changed)
                atomic_json(self.wd / 'candidates/1010/identity.json', changed['checkpoint_identity'])
                with self.assertRaisesRegex(ValueError, 'checkpoint'): U.row_values(self.run, self.root)
                atomic_json(path, saved)
                atomic_json(self.wd / 'candidates/1010/identity.json', saved['checkpoint_identity'])

    def test_stage_label_and_cross_a_u_rejected(self):
        path = self.wd / 'official/raw_max.json'; saved = read_json(path)
        for key, value in [('selection_id', 'TARGET'), ('normal_same_step_A_U', False)]:
            changed = dict(saved); changed[key] = value; atomic_json(path, changed)
            with self.assertRaisesRegex(ValueError, 'context'): U.row_values(self.run, self.root)
            atomic_json(path, saved)

    def test_status_config_and_incomplete_run_rejected(self):
        path = self.wd / 'official/postrun_status.json'; saved = read_json(path)
        for key, value in [('config_sha256', 'wrong'), ('official_complete', False), ('actual_updates', 49999)]:
            changed = dict(saved); changed[key] = value; atomic_json(path, changed)
            with self.assertRaisesRegex(ValueError, '50K'): U.row_values(self.run, self.root)
            atomic_json(path, saved)

    def test_reference_wrong_server_incomplete_release_or_alias_rejected(self):
        for key, value in [('server', 's1'), ('complete', False), ('status', 'PENDING'),
                           ('consumer_source_identity', {}), ('alias', 'F1')]:
            with self.subTest(key=key):
                changed = dict(self.bridge); changed[key] = value; atomic_json(self.bridge_path, changed)
                with self.assertRaisesRegex(ValueError, 'bridge context'): U.row_values(self.run, self.root)
        atomic_json(self.bridge_path, self.bridge)

    def test_reference_contents_changed_after_training_rejected(self):
        changed = dict(self.bridge); changed['q_ref'] = .8; atomic_json(self.bridge_path, changed)
        with self.assertRaisesRegex(ValueError, 'reference bridge identity'): U.row_values(self.run, self.root)

    def test_nonfinite_or_unofficial_metric_rejected(self):
        path = self.wd / 'official/raw_max.json'; saved = read_json(path)
        for key, value in [('hqnr', 'NaN'), ('d_s', 'Infinity')]:
            changed = copy.deepcopy(saved); changed['fr'][key] = value; atomic_json(path, changed)
            with self.assertRaisesRegex(ValueError, 'Nonfinite'): U.row_values(self.run, self.root)
        changed = copy.deepcopy(saved); changed['rr']['official_complete'] = False; atomic_json(path, changed)
        with self.assertRaisesRegex(ValueError, 'Unofficial'): U.row_values(self.run, self.root)

    def test_compound_upsert_preserves_historical_rows_noa_jqm_headers(self):
        sheet = FakeSheet(['Run', 'Notes', 'NOA metric', 'JQM↑', 'Q4↑', 'Target ERGAS↓'], U.GIDS['s3'])
        before = copy.deepcopy(sheet.cells[4]); first = self.invoke(sheet); second = self.invoke(sheet)
        self.assertEqual(first['row'], 5); self.assertEqual(second['row'], 5)
        self.assertTrue(second['readback_verified']); self.assertEqual(sheet.cells[4], before)
        self.assertEqual(self.requested, ['WV3-s3(5090)'] * 2)
        labels = U.label_map(sheet.cells[3]); self.assertEqual(labels['Target ERGAS↓'], 6)
        self.assertEqual(sheet.cells[5][labels['FH20R1 upload status']-1],'READBACK_VERIFIED')
        self.assertEqual(sheet.cells[5][labels['Target ERGAS↓'] - 1], 2.035)
        for label in ('NOA metric', 'JQM↑', 'Q4↑'): self.assertEqual(sheet.cells[5][labels[label] - 1], '')

    def test_same_run_other_campaign_and_annotation_rows_are_preserved(self):
        sheet = FakeSheet(['Run', 'FH20R1 campaign', 'FH20R1 run id'], U.GIDS['s3'])
        sheet.cells[4] = [self.run, 'OTHER_CAMPAIGN', self.run]
        sheet.cells[5] = ['', 'human annotation']; before = copy.deepcopy(sheet.cells)
        self.assertEqual(self.invoke(sheet)['row'], 6)
        self.assertEqual(sheet.cells[4], before[4]); self.assertEqual(sheet.cells[5], before[5])

    def test_duplicate_key_or_header_and_wrong_gid_never_write(self):
        cases = [FakeSheet(['Run', 'Run'], U.GIDS['s3']), FakeSheet(['Run'], -1)]
        duplicate = FakeSheet(['Run', 'FH20R1 campaign', 'FH20R1 run id'], U.GIDS['s3'])
        duplicate.cells[4] = [self.run, CAMPAIGN_ID, self.run]; duplicate.cells[5] = duplicate.cells[4].copy()
        cases.append(duplicate)
        for sheet in cases:
            with self.assertRaises(ValueError): self.invoke(sheet)
            self.assertEqual(sheet.writes, [])

    def test_wrong_server_rejected_before_network(self):
        (self.root / 'gspread/server.txt').write_text('s1\n')
        with patch.object(U, 'legacy_constants', side_effect=AssertionError('no network')):
            with self.assertRaises(ValueError): U.upload_run(self.run, self.root)

    def test_failed_readback_has_no_success_receipt(self):
        sheet = FakeSheet(['Run'], U.GIDS['s3']); sheet.corrupt = True
        with self.assertRaisesRegex(ValueError, 'readback'): self.invoke(sheet)
        labels=U.label_map(sheet.cells[3])
        self.assertEqual(sheet.cells[5][labels['FH20R1 upload status']-1],'READBACK_PENDING')
        self.assertFalse((self.wd / 'official/upload_receipt.json').exists())


if __name__ == '__main__': unittest.main(verbosity=2)
