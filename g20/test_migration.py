from copy import deepcopy
import datetime as dt
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from g20.common import CAMPAIGN_ID, atomic_json, object_sha, read_json, sha256
from g20 import controller, migration
from g20.plan import R0_PINS, registry_sha256
from g20.policy import CampaignWindow


class SourceAmendmentTests(unittest.TestCase):
    def setUp(self):
        from g20.gpu_exception import AUTH_SCHEMA, GPU_NAME, POLICY, authorization_path
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder = self.root / 'work_dir/_g20/s1'
        self.old = dict(files={'g20/controller.py': 'a' * 64, 'g20/training.py': 'b' * 64,
                               'qg40/model.py': 'c' * 64, 'external/DLPan/wald_utilities.py': 'd' * 64},
                        git_release='old', torch='2.4.0', numpy='1.26.4', scipy='1.13.1',
                        skimage='0.24.0', cuda='11.8', cudnn=90100, tf32_matmul=False,
                        tf32_cudnn=True, numeric_method_revision='QG40_SYNC_FREQ_C4_v1',
                        consumer_campaign=CAMPAIGN_ID)
        self.old['content_sha256'] = object_sha(self.old['files'])
        self.new = deepcopy(self.old)
        self.new['files'].update({'g20/controller.py': 'e' * 64,
                                  'g20/gpu_exception.py': 'f' * 64, 'g20/migration.py': '1' * 64})
        self.new['content_sha256'] = object_sha(self.new['files'])
        self.new['git_release'] = 'new'
        self.spec = dict(sensor='GF2', test='bound-original')
        self.window = CampaignWindow('2050-01-01T00:00:00Z').to_dict()
        self.registration = dict(identity=dict(campaign_id=CAMPAIGN_ID, server_id='s1', device='cuda',
            source_identity=self.old, window=self.window, registry_sha256=registry_sha256(),
            sensor_spec_sha256=object_sha(self.spec)), registered_at_utc='2050-01-01T00:00:00Z')
        self.state = dict(campaign_id=CAMPAIGN_ID, server_id='s1', status='BLOCKED_INTEGRITY',
                          reason=migration.BLOCKED_REASON, runs={}, blocks={}, observations=[],
                          active_processes=[], gpu_processes=[])
        self.data = dict(original_dataset='fixed')
        self.preflight = dict(complete=True, status='LOCAL_READY', server_id='s1', device='cuda',
            source_identity=self.old, sensor_spec_sha256=object_sha(self.spec),
            dataset_manifest_sha256=object_sha(self.data))
        for name, value in [('registration', self.registration), ('status', self.state),
                            ('sensor_spec', self.spec), ('dataset_manifest', self.data),
                            ('preflight', self.preflight), ('method_regression', {'complete': True}),
                            ('phase_coordinates', {'complete': True})]:
            atomic_json(self.folder / (name + '.json'), value)
        atomic_json(self.root / 'work_dir/_g20/campaign_window.json', self.window)
        self.auth_path = authorization_path(self.root, 's1')
        self.auth = dict(schema=AUTH_SCHEMA, campaign_id=CAMPAIGN_ID, server_id='s1', reference_id='R0',
            original_registration_sha256=sha256(self.folder / 'registration.json'),
            consumer_source_identity=self.new, gpu_name=GPU_NAME,
            teacher_checkpoint_sha256=R0_PINS['teacher_sha256'],
            q_cache_sha256=R0_PINS['q_cache_sha256'], origin_manifest_sha256=R0_PINS['calibration_id'],
            policy=deepcopy(POLICY), user_approved=True, approval_text='승인한다')
        atomic_json(self.auth_path, self.auth)

    def publish(self):
        return migration.prepare_amendment(self.root, 's1', self.new)

    def verify(self, current=None):
        return migration.validate_amendment(self.root, 's1', self.registration, current or self.new)

    def test_original_registration_without_amendment_has_unchanged_behavior(self):
        with patch('g20.controller.source_identity', return_value=self.old):
            self.assertEqual(controller.verify_registration(self.root, 's1'), self.registration)
        self.assertFalse(self.verify())
        with patch('g20.controller.source_identity', return_value=self.new), self.assertRaises(ValueError):
            controller.verify_registration(self.root, 's1')

    def test_publication_preserves_original_bytes_and_is_idempotent(self):
        names = ['registration.json', 'status.json', 'preflight.json', 'dataset_manifest.json']
        before = {name: (self.folder / name).read_bytes() for name in names}
        receipt = self.publish()
        self.assertEqual(receipt, self.publish())
        self.assertTrue(self.verify())
        for name, payload in before.items():
            self.assertEqual(payload, (self.folder / name).read_bytes())
        for name in receipt['archives']:
            self.assertEqual((self.folder / name).read_bytes(),
                             (self.folder / 'source_amendment/archive' / name).read_bytes())
        with patch('g20.controller.source_identity', return_value=self.new):
            self.assertEqual(controller.verify_registration(self.root, 's1'), self.registration)
        self.assertEqual(set(receipt['changed_files']),
                         {'g20/controller.py', 'g20/gpu_exception.py', 'g20/migration.py'})

    def test_replay_allows_progress_after_pinned_unstarted_publication(self):
        self.publish()
        state = deepcopy(self.state)
        state.update(status='RUNNING', runs={'G20-T01': {'training_started': True}},
                     blocks={'block': {'status': 'ADMITTED'}})
        atomic_json(self.folder / 'status.json', state)
        atomic_json(self.root / 'work_dir/G20-T01/meta/training_start_manifest.json', {'started': True})
        self.assertTrue(self.verify())
        with self.assertRaises(ValueError):
            migration.archive_preflight_for_amendment(self.root, 's1', self.preflight, self.new)

    def test_missing_authorization_is_not_implicit_approval(self):
        self.auth_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.publish()

    def test_authorization_scope_policy_and_user_approval_are_exact(self):
        for key, value in [('server_id', 's2'), ('reference_id', 'R1'), ('user_approved', False),
                           ('approval_text', ''), ('original_registration_sha256', '0' * 64),
                           ('q_cache_sha256', '0' * 64), ('consumer_source_identity', self.old),
                           ('policy', dict(self.auth['policy'], clock_changed=True))]:
            with self.subTest(key=key):
                invalid = deepcopy(self.auth)
                invalid[key] = value
                atomic_json(self.auth_path, invalid)
                with self.assertRaises(ValueError):
                    self.publish()
        atomic_json(self.auth_path, self.auth)
        with self.assertRaises(ValueError):
            migration.prepare_amendment(self.root, 's2', self.new)

    def test_core_source_and_runtime_changes_cannot_be_authorized(self):
        variants = []
        for filename in ('qg40/model.py', 'g20/training.py', 'g20/model.py', 'fh12/common.py'):
            source = deepcopy(self.new)
            source['files'][filename] = '9' * 64
            source['content_sha256'] = object_sha(source['files'])
            variants.append(source)
        for key, value in [('cuda', '12.6'), ('torch', '2.5.0'), ('tf32_matmul', True),
                           ('numeric_method_revision', 'changed')]:
            source = deepcopy(self.new)
            source[key] = value
            variants.append(source)
        for source in variants:
            with self.subTest(source=source):
                atomic_json(self.auth_path, dict(self.auth, consumer_source_identity=source))
                with self.assertRaises(ValueError):
                    migration.prepare_amendment(self.root, 's1', source)

    def test_both_content_maps_and_no_source_deletion_are_validated(self):
        for target in ('original', 'current', 'deleted'):
            with self.subTest(target=target):
                registration, source = deepcopy(self.registration), deepcopy(self.new)
                if target == 'original':
                    registration['identity']['source_identity']['content_sha256'] = '0' * 64
                elif target == 'current':
                    source['content_sha256'] = '0' * 64
                else:
                    del source['files']['g20/controller.py']
                    source['content_sha256'] = object_sha(source['files'])
                atomic_json(self.folder / 'registration.json', registration)
                atomic_json(self.auth_path, dict(self.auth, consumer_source_identity=source,
                    original_registration_sha256=sha256(self.folder / 'registration.json')))
                with self.assertRaises(ValueError):
                    migration.prepare_amendment(self.root, 's1', source)
        atomic_json(self.folder / 'registration.json', self.registration)

    def test_clock_sensor_registry_and_dataset_cannot_change(self):
        cases = [('sensor_spec.json', {'sensor': 'OTHER'}), ('dataset_manifest.json', {'data': 'changed'})]
        for name, value in cases:
            with self.subTest(name=name):
                original = read_json(self.folder / name)
                atomic_json(self.folder / name, value)
                with self.assertRaises(ValueError):
                    self.publish()
                atomic_json(self.folder / name, original)
        atomic_json(self.root / 'work_dir/_g20/campaign_window.json',
                    CampaignWindow('2050-01-01T01:00:00Z').to_dict())
        with self.assertRaises(ValueError):
            self.publish()
        atomic_json(self.root / 'work_dir/_g20/campaign_window.json', self.window)
        with patch('g20.plan.registry_sha256', return_value='0' * 64), self.assertRaises(ValueError):
            self.publish()

    def test_status_requires_exact_empty_reference_import_block(self):
        for key, value in [('status', 'LOCAL_READY'), ('reason', 'different integrity error'),
                           ('runs', {'G20-T01': {'status': 'RUNNING'}}),
                           ('blocks', {'block': {'status': 'ADMITTED'}}),
                           ('active_processes', [123]), ('observations', [{'hours': 1}])]:
            with self.subTest(key=key):
                atomic_json(self.folder / 'status.json', dict(self.state, **{key: value}))
                with self.assertRaises(ValueError):
                    self.publish()

    def test_training_manifest_and_checkpoint_block_new_publication(self):
        for filename in ('meta/training_start_manifest.json', 'last/model.safetensors', 'last/training_state.pt'):
            with self.subTest(filename=filename):
                path = self.root / 'work_dir/G20_GF2_T_S1_TA' / filename
                atomic_json(path, {'training': True})
                with self.assertRaises(ValueError):
                    self.publish()
                path.unlink()

    def test_start_resume_uses_amendment_without_rewriting_registration(self):
        window = CampaignWindow((dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).isoformat()).to_dict()
        self.registration['identity']['window'] = window
        atomic_json(self.root / 'work_dir/_g20/campaign_window.json', window)
        atomic_json(self.folder / 'registration.json', self.registration)
        self.auth['original_registration_sha256'] = sha256(self.folder / 'registration.json')
        atomic_json(self.auth_path, self.auth)
        spec = SimpleNamespace(sensor='GF2', is_bound=True, to_dict=lambda: self.spec)
        with patch('g20.controller.validate_definition'), \
             patch('g20.controller.SensorSpec.from_dict', return_value=spec), \
             patch('g20.controller.source_identity', return_value=self.new), \
             patch('g20.deployment.install_recovery', return_value={}), \
             patch('g20.controller.run', return_value=0) as run:
            with self.assertRaisesRegex(ValueError, 'registration differs'):
                controller.start(self.root, 's1', spec_path=self.folder / 'sensor_spec.json', foreground=True)
            run.assert_not_called()
            self.publish()
            original_sha = sha256(self.folder / 'registration.json')
            self.assertEqual(controller.start(self.root, 's1',
                spec_path=self.folder / 'sensor_spec.json', foreground=True), 0)
            run.assert_called_once_with(self.root, 's1')
            self.assertEqual(sha256(self.folder / 'registration.json'), original_sha)

    def test_active_runner_lock_blocks_publication(self):
        with migration.locked(self.folder / '.runner.lock'), self.assertRaises(BlockingIOError):
            self.publish()

    def test_receipt_archive_auth_and_registration_tamper_fail_closed(self):
        receipt = self.publish()
        receipt_path = migration.amendment_path(self.root)
        atomic_json(receipt_path, dict(receipt, changed_files={}))
        with self.assertRaises(ValueError):
            self.verify()
        atomic_json(receipt_path, receipt)
        archive = receipt_path.parent / 'archive/preflight.json'
        atomic_json(archive, dict(self.preflight, complete=False))
        with self.assertRaises(ValueError):
            self.verify()
        atomic_json(archive, self.preflight)
        atomic_json(self.auth_path, dict(self.auth, added='tamper'))
        with self.assertRaises(ValueError):
            self.verify()
        atomic_json(self.auth_path, self.auth)
        atomic_json(self.folder / 'registration.json', dict(self.registration, registered_at_utc='later'))
        with self.assertRaises(ValueError):
            self.verify()

    def test_new_source_must_match_receipt_exactly_on_replay(self):
        self.publish()
        changed = deepcopy(self.new)
        changed['files']['g20/controller.py'] = '8' * 64
        changed['content_sha256'] = object_sha(changed['files'])
        with self.assertRaises(ValueError):
            self.verify(changed)

    def test_readiness_regeneration_requires_exact_archived_original(self):
        self.publish()
        self.assertTrue(migration.archive_preflight_for_amendment(
            self.root, 's1', self.preflight, self.new))
        changed = dict(self.preflight, complete=False)
        atomic_json(self.folder / 'preflight.json', changed)
        with self.assertRaises(ValueError):
            migration.archive_preflight_for_amendment(self.root, 's1', changed, self.new)

    def test_preflight_runs_new_checks_instead_of_reusing_old_complete(self):
        from g20 import preflight
        self.publish()
        with patch('g20.preflight.source_identity', return_value=self.new), \
             patch('g20.preflight.check_deadline'), \
             patch('g20.plan.SensorSpec.from_dict'), \
             patch('g20.data.prepare_data', return_value=self.data), \
             patch('g20.data.build_dataset'), patch('g20.evaluation.FRMetrics'), \
             patch('g20.preflight.method_checks', return_value={'complete': False}) as check:
            with self.assertRaisesRegex(ValueError, 'regression failed'):
                preflight.execute(self.root, 's1', self.folder / 'sensor_spec.json', None, device='cpu')
        check.assert_called_once_with(self.root)
        self.assertEqual(read_json(self.folder / 'preflight.json'), self.preflight)
        self.assertTrue(self.verify())


class SecondSourceAmendmentTests(unittest.TestCase):
    """The second grant is a linked receipt, not an overwrite of the first."""

    def setUp(self):
        SourceAmendmentTests.setUp(self)
        from g20.gpu_exception import authorization_path, policy_for
        self.first = self.publish()
        self.first_source = deepcopy(self.new)
        self.first_receipt_path = migration.amendment_path(self.root, version=1)
        self.first_auth_path = authorization_path(self.root, version=1)
        self.preserved = {path: path.read_bytes() for path in (
            self.first_receipt_path, self.first_auth_path,
            self.first_receipt_path.parent / 'archive/preflight.json',
            self.folder / 'registration.json')}
        self.new['files']['g20/metric_exception.py'] = '2' * 64
        self.new['files']['g20/migration.py'] = '3' * 64
        self.new['content_sha256'] = object_sha(self.new['files'])
        self.metric_path = self.folder / 'metric_exception_authorization.json'
        self.metric = dict(consumer_source_identity=self.new, user_approved=True)
        atomic_json(self.metric_path, self.metric)
        mocked = patch('g20.metric_exception.authorization', return_value=self.metric)
        mocked.start()
        self.addCleanup(mocked.stop)
        self.auth = dict(self.auth, consumer_source_identity=self.new,
            previous_authorization_sha256=sha256(self.first_auth_path),
            metric_authorization_sha256=sha256(self.metric_path), policy=policy_for(self.root))
        self.auth_path = authorization_path(self.root)
        atomic_json(self.auth_path, self.auth)
        self.preflight = dict(self.preflight, source_identity=self.first_source)
        atomic_json(self.folder / 'preflight.json', self.preflight)

    publish = SourceAmendmentTests.publish
    verify = SourceAmendmentTests.verify

    def test_second_preserves_first_and_archives_exact_prior_live_readiness(self):
        receipt = self.publish()
        self.assertEqual(receipt['schema'], migration.SCHEMA_V2)
        self.assertEqual(receipt['previous_amendment_sha256'], sha256(self.first_receipt_path))
        self.assertEqual(receipt['previous_consumer_source_identity'], self.first_source)
        self.assertTrue(self.verify())
        self.assertEqual(receipt, self.publish())
        for path, payload in self.preserved.items():
            self.assertEqual(path.read_bytes(), payload)
        self.assertTrue(migration.archive_preflight_for_amendment(
            self.root, 's1', self.preflight, self.new))

    def test_second_rejects_mutated_first_auth_receipt_and_archive(self):
        self.publish()
        for path, payload in self.preserved.items():
            with self.subTest(path=path):
                atomic_json(path, dict(read_json(path), tampered=True))
                with self.assertRaises(ValueError):
                    self.verify()
                path.write_bytes(payload)

    def test_second_metric_and_original_authorization_hashes_are_bound(self):
        for key in ('metric_authorization_sha256', 'previous_authorization_sha256'):
            atomic_json(self.auth_path, dict(self.auth, **{key: '0' * 64}))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'supplemental'):
                self.publish()
        atomic_json(self.auth_path, self.auth)
        self.publish()
        atomic_json(self.metric_path, dict(self.metric, tampered=True))
        with self.assertRaises(ValueError):
            self.verify()

    def test_second_requires_exact_first_readiness_and_no_training(self):
        original_preflight = read_json(self.first_receipt_path.parent / 'archive/preflight.json')
        atomic_json(self.folder / 'preflight.json', original_preflight)
        with self.assertRaisesRegex(ValueError, 'readiness'):
            self.publish()
        atomic_json(self.folder / 'preflight.json', self.preflight)
        atomic_json(self.root / 'work_dir/G20_GF2_S/meta/training_status.json', {'step': 1})
        with self.assertRaisesRegex(ValueError, 'training/checkpoint'):
            self.publish()

    def test_second_cannot_select_first_receipt_when_metric_grant_disappears(self):
        self.publish()
        self.metric_path.unlink()
        self.assertEqual(migration.amendment_path(self.root).parent.name, 'source_amendment_v2')
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_second_cannot_change_runtime_or_numerical_code(self):
        for key, value in [('torch', '2.8.0'), ('cuda', '12.8')]:
            changed = dict(self.new, **{key: value})
            atomic_json(self.auth_path, dict(self.auth, consumer_source_identity=changed))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'runtime'):
                migration.prepare_amendment(self.root, 's1', changed)



if __name__ == '__main__':
    unittest.main()
