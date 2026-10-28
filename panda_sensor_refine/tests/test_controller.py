import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_sensor_refine import queue as Q, launch as L, orchestration as O


class ControllerContracts(unittest.TestCase):
    def test_watcher_identity_excludes_trainers_publishers_other_servers(self):
        base = ['python', '-m', 'panda_s1loss9.queue', '--server', 's1']
        self.assertTrue(Q.watcher_matches(base, Q.ROOT, 's1'))
        self.assertFalse(Q.watcher_matches(base, Q.ROOT, 's3'))
        self.assertFalse(Q.watcher_matches(base, '/tmp/other-checkout', 's1'))
        for module in ('panda_s1loss9', 'panda_s1loss9.best_publication', 'panda_sensor_refine.queue'):
            self.assertFalse(Q.watcher_matches([module if v == 'panda_s1loss9.queue' else v for v in base], Q.ROOT, 's1'))
        recovery = ['python', str(Q.ROOT / 'tools/s1loss9_recovery.py'), 'watch', '--server', 's1']
        self.assertTrue(Q.watcher_matches(recovery, Q.ROOT, 's1'))
        self.assertFalse(Q.watcher_matches(['run' if v == 'watch' else v for v in recovery], Q.ROOT, 's1'))

    def test_container_identity_must_match_before_stop(self):
        saved = {'name': 'trainer', 'id': 'old', 'image': 'same', 'argv': ['python', 'train']}
        current = {'Id': 'replacement', 'Image': 'same', 'Config': {'Cmd': saved['argv']}, 'State': {'Running': True}}
        with patch.object(L, 'inspect_optional', return_value=current), patch.object(Q.subprocess, 'run') as signal:
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                Q.stop_trainers({'containers': [saved]})
            signal.assert_not_called()

    def test_started_pair_is_kept_even_when_new_queue_excludes_it(self):
        configs = []
        for pair in ('A050_E0003', *O.RETAINED_PAIRS['s3']):
            configs += [dict(run_id=pair + '_d', loss_pair=pair, phase='donor'),
                        dict(run_id=pair + '_s', loss_pair=pair, phase='stage2')]
        state = {'runs': [{'run_id': 'A050_E0003_s', 'status': 'STOPPED'}]}
        selected = O.selected_configs('s3', state, configs)
        self.assertEqual([c['run_id'] for c in selected[:2]], ['A050_E0003_d', 'A050_E0003_s'])
        self.assertEqual([c['loss_pair'] for c in selected[2::2]], list(O.RETAINED_PAIRS['s3']))
        state['runs'].append({'run_id': configs[-1]['run_id'], 'status': 'RUNNING'})
        with self.assertRaises(ValueError): O.selected_configs('s3', state, configs)

    def test_only_exact_preoptimizer_support_failure_can_be_isolated(self):
        row = dict(status='FAILED', error=O.SUPPORT_ERROR, poisoned=False, run_id='d', step=10, directory='/x')
        cfg = dict(phase='donor')
        with patch.object(O, 'authenticate_stop', return_value={}), patch.object(O, 'local', return_value=Path('/x')), \
                patch.object(O, 'sha', return_value='sha'):
            O.support_failure(row, cfg)
            for changes in ({'poisoned': True}, {'error': "ValueError('CUDA OOM')"},
                            {'stop_checkpoint_error': 'disk'}, {'status': 'COMPLETED'}):
                with self.assertRaises(ValueError): O.support_failure(dict(row, **changes), cfg)
            with self.assertRaises(ValueError): O.support_failure(row, {'phase': 'stage2'})

    def test_legacy_accounting_retains_completed_and_never_hides_started_child(self):
        configs = [dict(run_id='good', planned_horizon=30, phase='stage2', loss_pair='good'),
                   dict(run_id='bad', planned_horizon=50, phase='donor', loss_pair='bad'),
                   dict(run_id='child', planned_horizon=30, phase='stage2', loss_pair='bad'),
                   dict(run_id='removed', planned_horizon=50, phase='donor', loss_pair='removed')]
        state = {'runs': [dict(run_id='good', status='COMPLETED', step=30),
                          dict(run_id='bad', status='FAILED', step=10)]}
        with patch.object(O, 'legacy_configs', return_value=configs), patch.object(O, 'legacy_root', return_value=Path('/tmp')), \
                patch.object(O, 'read', return_value=state), patch.object(O, 'support_failure', return_value={'run_id': 'bad'}), \
                patch.object(O, 'final_evidence', return_value={'run_id': 'good'}):
            result = O.build_completion('/tmp', 's3', ['good', 'bad', 'child'])
            self.assertFalse(result['all_original_runs_completed'])
            self.assertEqual([v['run_id'] for v in result['held_students']], ['child'])
            self.assertEqual([v['run_id'] for v in result['excluded_unstarted']], ['removed'])
            state['runs'].append(dict(run_id='child', status='STOPPED', step=10))
            with self.assertRaisesRegex(ValueError, 'cannot be concealed'):
                O.build_completion('/tmp', 's3', ['good', 'bad', 'child'])

    def test_existing_launch_receipt_does_not_launch_duplicate_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            receipt = {'kind': 'legacy', 'target': None, 'container_id': 'existing'}
            (d / 'legacy.json').write_text(json.dumps(receipt))
            with patch.object(L, 'launcher_root', return_value=d), patch.object(L.subprocess, 'run') as run:
                self.assertEqual(L.launch_job('s3', 'legacy'), receipt)
                run.assert_not_called()
                with self.assertRaises(ValueError): L.launch_job('s3', 'legacy', target='different')

    def test_s3_completed_sensor5_enters_legacy_without_old_takeover_stage_gate(self):
        state = {'status': 'FINISHED_REGISTERED_QUEUE', 'runs': []}
        with patch.object(Q, 'state_of', side_effect=lambda p, s: state if p == 'panda_sensor5' else None), \
                patch.object(Q, 'configs_of', return_value=[]), \
                patch('panda_s1loss9.orchestration.initial_complete', return_value=True):
            snapshot, boundaries = Q.validate_inventory('s3')
            self.assertEqual(boundaries, [])
            self.assertIn('panda_sensor5', snapshot)

    def test_refinement_goes_directly_to_original_wv3_and_never_old_sensor_tuning(self):
        with tempfile.TemporaryDirectory() as tmp:
            chain = {'stage': 'WAIT_REFINE'}
            with patch.object(L, 'wait_job', return_value=True), patch.object(Q, 'core_complete', return_value=True), \
                    patch.object(Q, 'reconcile_wv3_receipts'):
                Q.advance('s1', Path(tmp), chain, {})
            self.assertEqual(chain['stage'], 'RESUME_TUNE30')
            with patch('panda_sensor5.queue.advance') as original:
                Q.advance('s1', Path(tmp), chain, {})
                self.assertEqual(original.call_args.args[2], chain)
                self.assertEqual(original.call_args.args[1], Q.ROOT / 'work_dir/panda_sensor5_launchers/s1')

    def test_terminal_core_failure_requires_matching_failed_donor_for_hold(self):
        from panda_sensor_refine import plan
        cfgs = [dict(run_id='d', case='D', phase='donor', planned_horizon=50),
                dict(run_id='s', case='S', phase='stage2', donor_version='D', planned_horizon=30)]
        row = {'run_id': 'd', 'status': 'FAILED'}
        state = {'status': 'FINISHED_WITH_EXPLICIT_FAILURES', 'runs': [row],
                 'isolated_support_failures': {'d': {'checkpoint_sha256': 'same', 'step': 10}},
                 'held_runs': {'s': {'status': 'HELD_FAILED_DONOR', 'donor_case': 'D'}}}
        with patch.object(Q, 'read', return_value=state), patch.object(plan, 'groups', return_value=[{'runs': [('D','R1'), ('S','R1')]}]), \
                patch.object(plan, 'resolve_case', side_effect=lambda srv, c, r: cfgs[0 if c == 'D' else 1]), \
                patch.object(Q, 'predecessor_receipt', return_value={'external_donor': {'status': 'COMPLETED'}}), \
                patch.object(Q, 'verify_core_failure'):
            self.assertTrue(Q.core_complete('s1'))
            state['held_runs']['s']['donor_case'] = 'unrelated'
            with self.assertRaises(ValueError): Q.core_complete('s1')

    def test_preflight_failure_is_authenticated_without_fake_production_checkpoint(self):
        from panda_sensor_refine import plan, runtime
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'preflight').mkdir()
            cfg = dict(run_id='donor', case='D', phase='donor', replica='R1')
            code = {'source.py': 'expected'}
            proof = dict(schema='SENSOR_REFINE_PREFLIGHT_SUPPORT_FAILURE_V1',
                evidence_kind='FAILED_PREFLIGHT_NO_PRODUCTION_UPDATES',
                run_id='donor', case='D', step=0, error='SUPPORT_VIOLATION: preflight',
                requested_config=cfg, code_pin=code, production_run_absent=True,
                production_updates=0, production_weights_started=False, disposable_preflight_only=True, donor=None)
            path = root / 'preflight/D_R1_support_failure.json'
            path.write_text(json.dumps(proof))
            record = dict(run_id='donor', case='D', step=0, error=proof['error'],
                evidence_kind=proof['evidence_kind'], receipt_path=str(path), receipt_sha256=O.sha(path),
                code_pin_sha256=plan.canonical_sha256(code), config_sha256=plan.canonical_sha256(cfg),
                donor_checkpoint_sha256=None)
            with patch.object(runtime, 'code_pin', return_value=code):
                Q.verify_core_failure(root, cfg, record, {}, {})
                with self.assertRaises(ValueError): Q.verify_core_failure(root, cfg, record, {'donor': {}}, {})
                (root / 'runs/donor').mkdir(parents=True)
                with self.assertRaises(ValueError): Q.verify_core_failure(root, cfg, record, {}, {})

    def test_final_takeover_refresh_occurs_after_controller_lock_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            proposed = dict(token='preserved', controller_pin={'file': 'same'}, containers=['old'])
            current = dict(token='new', controller_pin={'file': 'same'}, containers=['new'])
            with patch.object(Q, 'capture_takeover', return_value=current):
                finalized = Q.finalize_takeover('s1', d, proposed)
                self.assertEqual(finalized['containers'], ['new'])
                self.assertEqual(finalized['token'], 'preserved')
                self.assertTrue(finalized['finalized_under_old_watcher_locks'])
                self.assertEqual(Q.finalize_takeover('s1', d, finalized), finalized)

    def test_student_preflight_uses_scientific_selector_digest_not_json_file_hash(self):
        from panda_sensor_refine import plan, runtime
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'preflight').mkdir()
            checkpoint = root / 'donor.pt'; checkpoint.write_bytes(b'authentic donor')
            selector = root / 'selection.json'; selector.write_text('{"receipt_sha256": "scientific"}\n')
            self.assertNotEqual(O.sha(selector), 'scientific')
            donor = dict(checkpoint_path=str(checkpoint), checkpoint_sha256=O.sha(checkpoint),
                         selector_receipt_path=str(selector), selector_sha256='scientific', source_config={'case': 'source'})
            cfg = dict(run_id='student', case='S', phase='stage2', replica='R1', donor_version='S1_A200_E030')
            code = {'source.py': 'expected'}
            proof = dict(schema='SENSOR_REFINE_PREFLIGHT_SUPPORT_FAILURE_V1',
                evidence_kind='FAILED_PREFLIGHT_NO_PRODUCTION_UPDATES', run_id='student', case='S', step=0,
                error='SUPPORT_VIOLATION: calibration', requested_config=cfg, code_pin=code,
                production_run_absent=True, production_updates=0, production_weights_started=False,
                disposable_preflight_only=True, donor=donor)
            path = root / 'preflight/S_R1_support_failure.json'; path.write_text(json.dumps(proof))
            record = dict(run_id='student', case='S', step=0, error=proof['error'], evidence_kind=proof['evidence_kind'],
                receipt_path=str(path), receipt_sha256=O.sha(path), code_pin_sha256=plan.canonical_sha256(code),
                config_sha256=plan.canonical_sha256(cfg), donor_checkpoint_sha256=donor['checkpoint_sha256'])
            state = {'donors': {'S1_A200_E030': donor}}
            with patch.object(runtime, 'code_pin', return_value=code), \
                    patch('panda_s1loss9.selection.validate_receipt', return_value={'receipt_sha256': 'scientific'}):
                Q.verify_core_failure(root, cfg, record, {}, state)
                checkpoint.write_bytes(b'changed donor')
                with self.assertRaises(ValueError): Q.verify_core_failure(root, cfg, record, {}, state)

    def test_stopped_boundary_follows_actual_run_when_trainer_advanced_before_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            chain = {'stage': 'WAIT_SAFE_STOP'}
            actual = [{'kind': 'tune-boundary', 'package': 'panda_tune30', 'target': 'newly_started'}]
            request = {'boundaries': [{'target': 'already_finished'}]}
            state = {'panda_tune30': {'runs': [{'run_id': 'newly_started', 'status': 'STOPPED'}]}}
            with patch.object(Q, 'stopped_and_verified', return_value=True), \
                    patch.object(Q, 'validate_inventory', return_value=(state, actual)):
                Q.advance('s1', d, chain, request)
            self.assertEqual(chain['boundaries'], actual)
            self.assertEqual(request['boundaries'][0]['target'], 'already_finished')
            self.assertTrue((d / 'stopped_boundaries.json').exists())

    def test_legacy_preflight_failure_never_requires_or_creates_a_production_run(self):
        from panda_s1loss9 import runtime
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); d = root / 'launchers'; d.mkdir()
            campaign = root / 'campaign'; campaign.mkdir()
            cfg = dict(run_id='d', case='D', phase='donor')
            code = {'original.py': 'frozen'}
            proof = dict(schema='SENSOR_REFINE_LEGACY_PREFLIGHT_SUPPORT_FAILURE_V1', run_id='d', case='D', step=0,
                error='SUPPORT_VIOLATION: preflight', requested_config=cfg, legacy_code_pin=code,
                production_updates=0, production_run_absent=True, production_weights_started=False, donor=None)
            path = d / 'legacy_preflight_D.json'; path.write_text(json.dumps(proof))
            record = dict(run_id='d', step=0, error=proof['error'], receipt_path='launchers/legacy_preflight_D.json',
                          receipt_sha256=O.sha(path))
            with patch.object(O, 'legacy_root', return_value=campaign), patch.object(O, 'launcher_root', return_value=d), \
                    patch.object(runtime, 'code_pin', return_value=code):
                self.assertEqual(O.validate_preflight_failure(root, 's3', cfg, record, {'runs': []}), record)
                with self.assertRaises(ValueError):
                    O.validate_preflight_failure(root, 's3', cfg, record, {'runs': [{'run_id': 'd'}]})
                (campaign / 'runs/d').mkdir(parents=True)
                with self.assertRaises(ValueError): O.validate_preflight_failure(root, 's3', cfg, record, {'runs': []})

    def test_interrupted_wv3_control_receipt_archived_idempotently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); d = root / 'new'; d.mkdir()
            path = root / 'old.json'
            receipt = {'package': 'panda_tune30', 'container': 'old', 'container_id': 'id'}
            path.write_text(json.dumps(receipt))
            request = {'wv3_resume_receipts': {'old.json': {'value': receipt, 'sha256': O.sha(path)}}}
            with patch.object(Q, 'ROOT', root), patch.object(Q, 'state_of', return_value={'status': 'PAUSED_FOR_SENSOR5'}), \
                    patch.object(L, 'inspect_optional', return_value={'Id': 'id', 'State': {'Running': False}}):
                Q.reconcile_wv3_receipts('s1', d, request)
                self.assertFalse(path.exists())
                self.assertTrue((d / 'prior_old.json').exists())
                Q.reconcile_wv3_receipts('s1', d, request)


if __name__ == '__main__':
    unittest.main()
