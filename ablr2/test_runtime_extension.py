"""Offline extension authorization, immutable handover and local resource gates."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock

from ablr2 import common,migration,handoff,resources
from ablr2.common import atomic_json,read_json,object_sha,camp,run_dir,RuntimePaused
from ablr2.plan import cases_for,campaign_id


class ReleaseManifestTests(unittest.TestCase):
    def test_import_time_manifest_and_dynamic_dependencies_are_pinned(self):
        from ablr2.plan import EXTENSION_BUNDLE,SOURCE_SHAS,source_path
        identity=common.source_identity(common.ROOT)
        expected=[EXTENSION_BUNDLE+'/SHA256SUMS.txt','gspread/sheet_categories.py','model/se.py']
        expected += [str(source_path(name,common.ROOT).relative_to(common.ROOT)) for name in SOURCE_SHAS]
        for name in expected:
            self.assertEqual(identity['files'].get(name),common.sha256(common.ROOT/name),name)


class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.folder=camp(self.root,'s1');self.source=dict(content_sha256='a'*64,files={'a':'b'*64})
        patcher=patch.object(common,'source_identity',return_value=self.source);patcher.start();self.addCleanup(patcher.stop)
        self.lease=dict(campaign_id=campaign_id('s1'),server='s1',sensor='WV3',expires_utc='2000-01-01T00:00:00+00:00')
        atomic_json(self.folder/'lease.json',self.lease)

    def test_no_design_or_expired_lease_implicitly_authorizes_forever(self):
        with self.assertRaises(PermissionError):common.grant_until_stop(self.root,'s1')
        with self.assertRaises(RuntimePaused):common.authorization_context(self.root,'s1')
        self.assertFalse((self.folder/'authorization.json').exists())

    def test_explicit_new_receipt_preserves_old_lease_and_is_idempotent(self):
        before=(self.folder/'lease.json').read_bytes()
        first=common.grant_until_stop(self.root,'s1',operator_authorized=True)
        self.assertEqual(common.authorization_context(self.root,'s1'),(first,None))
        self.assertEqual(first['original_lease'],self.lease)
        self.assertEqual(common.grant_until_stop(self.root,'s1',operator_authorized=True),first)
        self.assertEqual((self.folder/'lease.json').read_bytes(),before)
        self.assertFalse(first['automatic_renewal'])

    def test_immutable_receipt_and_runtime_binding_cannot_be_changed(self):
        doc=common.grant_until_stop(self.root,'s1',operator_authorized=True)
        atomic_json(self.folder/'registration_ablr2x.json',dict(source_identity={'changed':True}))
        with self.assertRaisesRegex(RuntimePaused,'source'):common.authorization_context(self.root,'s1')
        atomic_json(self.folder/'registration_ablr2x.json',dict(source_identity=self.source))
        doc['sequence']=2;atomic_json(self.folder/'authorization.json',doc)
        with self.assertRaisesRegex(RuntimePaused,'receipt'):common.authorization_context(self.root,'s1')

    def test_s3_campaign_id_and_s4_s5_protection(self):
        doc=common.grant_until_stop(self.root,'s3',operator_authorized=True)
        self.assertEqual(doc['campaign_id'],campaign_id('GF2'))
        for server in ('s4','s5'):
            with self.assertRaises(ValueError):common.grant_until_stop(self.root,server,operator_authorized=True)


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.root=self.base/'repo';self.root.mkdir();(self.root/'work_dir').mkdir()
        self.old=self.base/'repo-runtime-ablr2-s1-aaaaaaaaaaaa';self.new=self.base/'repo-runtime-ablr2x-s1-bbbbbbbbbbbb'
        for directory in (self.old,self.new):
            directory.mkdir();(directory/'work_dir').symlink_to(self.root/'work_dir',target_is_directory=True)
        self.folder=camp(self.root,'s1');self.case=cases_for('s1')[0]
        self.source=dict(content_sha256='a'*64,files={'old.py':'a'*64})
        self.origin=dict(path=str(self.old),git_commit='a'*40,files=self.source['files'])
        atomic_json(self.folder/'runtime_release.json',self.origin)
        atomic_json(self.folder/'runtime_release_ablr2x.json',dict(path=str(self.new),git_commit='b'*40,files={}))
        atomic_json(self.folder/'registration.json',dict(source_identity=self.source))
        atomic_json(self.folder/'preflight.json',dict(source_identity=self.source,complete=True))
        atomic_json(self.folder/'seed_ledger.json',[dict(seed=2025,key='original')])
        atomic_json(self.folder/'lease.json',dict(expires_utc='2000-01-01T00:00:00+00:00'))
        self.state=dict(cycle=7,incumbent='R04',active_stage='OLD_P02',active_run=self.case.run_id,
            runs={self.case.run_id:dict(complete=True,train_hours=12.3)},observations=[dict(hours=9.8)],setup_hours=2.5)
        atomic_json(self.folder/'state.json',self.state)
        atomic_json(run_dir(self.case.run_id,self.root)/'meta/training_start_manifest.json',dict(source_identity=self.source))
        for patcher in (patch('ablr2.deployment.verify_release',side_effect=lambda receipt,*a,**k:Path(receipt['path'])),
                patch.object(migration,'_live_controller',return_value=False),
                patch.object(resources,'idle_evidence',return_value=dict(idle=True,gpu_pids=[],other_processes=[]))):
            patcher.start();self.addCleanup(patcher.stop)

    def test_readonly_plan_never_touches_original_state_or_lease(self):
        before=(self.folder/'state.json').read_bytes();report=migration.plan_migration(self.root,'s1')
        self.assertEqual(report['cycle'],7);self.assertEqual((self.folder/'state.json').read_bytes(),before)
        self.assertFalse((self.folder/'migration/request.json').exists())

    def test_request_and_boundary_preserve_cursors_seeds_costs_and_original_receipts(self):
        names=('state.json','seed_ledger.json','lease.json','registration.json','preflight.json','runtime_release.json')
        before={name:(self.folder/name).read_bytes() for name in names}
        request=migration.request_migration(self.root,'s1',operator_authorized=True)
        self.assertEqual(read_json(self.folder/'control.json')['command'],'STOP_AFTER_RUN')
        value=migration.finalize_migration(self.root,'s1',self.new)
        self.assertTrue(value['complete']);self.assertEqual(value['original_state'],self.state)
        self.assertEqual(read_json(self.folder/'control.json')['command'],'CONTINUE')
        self.assertEqual(migration.execution_root(self.new,self.case),self.old)
        self.assertEqual({name:(self.folder/name).read_bytes() for name in names},before)
        self.assertEqual(migration.request_migration(self.root,'s1',operator_authorized=True),request)

    def test_manual_stop_is_never_discharged_by_handover(self):
        migration.request_migration(self.root,'s1',operator_authorized=True)
        manual=dict(command='STOP_NOW_SAFE',operator_action=True);atomic_json(self.folder/'control.json',manual)
        migration.finalize_migration(self.root,'s1',self.new)
        self.assertEqual(read_json(self.folder/'control.json'),manual)

    def test_live_or_incomplete_original_run_blocks_new_owner(self):
        migration.request_migration(self.root,'s1',operator_authorized=True)
        with patch.object(migration,'_live_controller',return_value=True):
            with self.assertRaisesRegex(RuntimePaused,'overlap'):migration.finalize_migration(self.root,'s1',self.new)
        self.state['runs'][self.case.run_id]['complete']=False;atomic_json(self.folder/'state.json',self.state)
        with self.assertRaisesRegex(RuntimePaused,'COMPLETION'):migration.finalize_migration(self.root,'s1',self.new)
        self.assertFalse((self.folder/'migration/transition.json').exists())

    def test_waiter_never_converts_expired_original_lease(self):
        migration.request_migration(self.root,'s1',operator_authorized=True)
        self.state['runs'][self.case.run_id]['complete']=False;atomic_json(self.folder/'state.json',self.state)
        before=(self.folder/'lease.json').read_bytes()
        with patch.object(common,'authorization_context',return_value=({'service_mode':'UNTIL_OPERATOR_STOP'},None)):
            self.assertEqual(migration.await_handover(self.new,'s1'),75)
        self.assertEqual((self.folder/'lease.json').read_bytes(),before)
        self.assertIn('ORIGINAL_FINITE_LEASE',read_json(self.folder/'migration/waiter_status.json')['reason'])

    def test_nonactive_started_job_cannot_recover_under_new_source(self):
        other=cases_for('s1')[1]
        atomic_json(run_dir(other.run_id,self.root)/'meta/training_start_manifest.json',dict(source_identity=self.source))
        migration.request_migration(self.root,'s1',operator_authorized=True)
        with self.assertRaisesRegex(RuntimePaused,'ORIGINAL_STARTED_JOBS'):
            migration.finalize_migration(self.root,'s1',self.new)
        self.assertFalse((self.folder/'migration/transition.json').exists())

    def test_different_sources_require_measured_proof_not_pass_only(self):
        with self.assertRaisesRegex(ValueError,'measured'):common.assert_compatible_source(self.source,{'new':True},self.root,'s1')
        self.assertIsNone(common.assert_compatible_source(self.source,self.source,self.root,'s1'))
        path=self.root/'fake_parity.json';atomic_json(path,dict(passed=True))
        with patch.object(migration,'source_identity',return_value={'new':True}):
            with self.assertRaises(ValueError):migration.install_source_bridge(self.root,'s1',path,operator_authorized=True)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        patcher=patch.object(handoff,'inventory',return_value=[]);patcher.start();self.addCleanup(patcher.stop)

    def test_original_p40_has_no_run_boundary_and_is_not_silently_drained_by_block(self):
        old=self.root/'work_dir/_gfp40/s3';atomic_json(old/'status.json',dict(active_run='OLD',runs={}))
        with self.assertRaisesRegex(RuntimePaused,'UNSUPPORTED'):handoff.request(self.root,operator_authorized=True)
        self.assertFalse((old/'control.json').exists())

    def test_explicit_safe_request_only_changes_s3_control(self):
        old=self.root/'work_dir/_gfp40/s3';protected=self.root/'work_dir/_gfp40/s4'
        atomic_json(old/'status.json',dict(active_run='OLD',runs={}))
        atomic_json(protected/'control.json',dict(command='CONTINUE',protected=True));before=(protected/'control.json').read_bytes()
        doc=handoff.request(self.root,operator_authorized=True,boundary='SAFE')
        self.assertEqual(read_json(old/'control.json')['command'],'STOP_NOW_SAFE')
        self.assertFalse(doc['signals_sent']);self.assertFalse(doc['cron_modified'])
        self.assertEqual((protected/'control.json').read_bytes(),before)
        for server in ('s1','s2','s4','s5'):
            with self.assertRaises(ValueError):handoff.request(self.root,server,operator_authorized=True)

    def test_pcrepro_after_current_run_does_not_rewrite_queue(self):
        old=self.root/'work_dir/_pcrepro/s3';state=dict(cycle=3,index=2,active_run='OLD',runs={})
        atomic_json(old/'state.json',state);before=(old/'state.json').read_bytes()
        handoff.request(self.root,operator_authorized=True)
        self.assertEqual(read_json(old/'control.json')['command'],'STOP_AFTER_CURRENT_RUN')
        self.assertEqual((old/'state.json').read_bytes(),before)


class ResourceTests(unittest.TestCase):
    def test_next_block_uses_100gib_floor_and_double_projected_write(self):
        case=cases_for('s1')[0];gib=1024**3
        measured=dict(disk={'free_bytes':150*gib},gpu={'selected_device':0})
        with patch.object(resources,'projected_case_write',return_value=60*gib),patch.object(resources,'inventory',return_value=measured):
            value=resources.assess_block('/unused',[case,case])
        self.assertEqual(value['required_disk_bytes'],240*gib);self.assertFalse(value['allowed'])
        self.assertFalse(value['automatic_pruning'])
        with patch.object(resources,'projected_case_write',return_value=1),patch.object(resources,'inventory',return_value=measured):
            value=resources.assess_case('/unused',case)
        self.assertEqual(value['required_disk_bytes'],100*gib);self.assertTrue(value['allowed'])


if __name__=='__main__':unittest.main()
