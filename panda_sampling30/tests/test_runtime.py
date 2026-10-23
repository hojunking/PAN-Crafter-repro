"""Unlimited fresh runtime and wait-only predecessor handoff guards."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from panda_sampling30 import plan, runtime as R, transition as T
from panda_schedule30 import plan as predecessor


class Clock:
    def __init__(self):
        self.wall, self.monotonic = 200000., 50000.

    def __call__(self):
        return self.wall


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.clock = Clock()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(R, 'code_pin', return_value={'synthetic.py':'d'*64}))
        self.stack.enter_context(patch.object(R, 'evaluator_identity', return_value={'protocol':'SYNTHETIC'}))
        self.stack.enter_context(patch.object(R.time, 'monotonic', side_effect=lambda:self.clock.monotonic))
        self.stack.enter_context(patch.object(R.shutil, 'disk_usage', return_value=SimpleNamespace(free=10*1024**3)))
        self.campaign = R.Campaign('s1', output=self.temp.name, device='cpu', clock=self.clock)

    def test_no_wall_deadline_even_after_long_pause_and_restart(self):
        c = self.campaign
        started = c.state['started_unix']
        self.clock.wall += 365*24*3600
        self.clock.monotonic += 365*24*3600
        c.check()
        c.persist()
        restarted = R.Campaign('s1', output=self.temp.name, device='cpu', clock=self.clock)
        restarted.check()
        self.assertEqual(restarted.state['started_unix'], started)
        self.assertIsNone(restarted.deadline_unix)
        self.assertGreaterEqual(restarted.elapsed_seconds(), 365*24*3600)

    def test_stop_and_disk_guards_remain(self):
        self.campaign.stop_requested = True
        with self.assertRaisesRegex(R.BudgetExpired, 'STOP_REQUESTED'):
            self.campaign.check()
        self.campaign.stop_requested = False
        with patch.object(R.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(R.BudgetExpired, 'DISK_CLOSEOUT'):
                self.campaign.check()

    def test_architecture_has_distinct_throughput_key(self):
        cfgs = [plan.resolve_case('s1', case, replica)
                for group in plan.groups('s1') for case,replica in group['runs']]
        key_by_variant = {}
        for cfg in cfgs:
            key = self.campaign.timing_key(cfg)
            variant = cfg['sampling_variant']
            if variant in key_by_variant:
                self.assertEqual(key_by_variant[variant], key)
            key_by_variant[variant] = key
        self.assertEqual(len(set(key_by_variant.values())), 3)

    def test_fresh_only_build_rejects_source_import(self):
        case,replica = plan.groups('s1')[0]['runs'][0]
        cfg = plan.resolve_case('s1', case, replica)
        cfg['inherited_updates'] = 25000
        with self.assertRaisesRegex(ValueError, 'fresh Student'):
            self.campaign.build(cfg)

    def test_all_finite_groups_admitted_without_time_budget(self):
        for group in self.campaign.config['groups']:
            self.assertTrue(self.campaign.admit_group(group))
            self.assertIsNone(self.campaign.state['groups'][-1]['remaining_seconds'])

    def test_trace_preserves_uncheckpointed_tail_as_evidence(self):
        directory = Path(self.temp.name)/'trace'
        directory.mkdir()
        rows = [{'step':step, 'local_update':step, 'seconds_including_IO':.1} for step in (1,2,3)]
        path = directory/'train.jsonl'
        original = ''.join(json.dumps(row)+'\n' for row in rows)
        path.write_text(original)
        R.recover_trace(directory, 2)
        self.assertEqual([json.loads(line)['step'] for line in path.read_text().splitlines()], [1,2])
        self.assertEqual(next(directory.glob('train_before_resume_*.jsonl')).read_text(), original)


class TransitionTests(unittest.TestCase):
    @staticmethod
    def completed(server):
        return {'status':'FINISHED_REGISTERED_QUEUE', 'runs':[
            {'case':case, 'replica':replica, 'status':'COMPLETED'}
            for group in predecessor.groups(server) for case,replica in group['runs']]}

    def test_every_running_case_waits_without_stop(self):
        for server in ('s1','s3'):
            for case,replica in [(c,r) for g in predecessor.groups(server) for c,r in g['runs']]:
                campaign = {'status':'RUNNING', 'runs':[{'case':case, 'replica':replica, 'status':'RUNNING'}]}
                original = deepcopy(campaign)
                self.assertEqual(T.transition_decision(server, {'Running':True}, campaign),
                                 'WAITING_PREVIOUS_QUEUE_COMPLETION')
                self.assertEqual(campaign, original)

    def test_safe_stop_waits_for_operator_resume(self):
        self.assertEqual(T.transition_decision('s1', {'Running':False,'Status':'exited','ExitCode':0},
            {'status':'STOPPED_SAFE', 'runs':[]}), 'WAITING_PREVIOUS_OPERATOR_RESUME')

    def test_complete_declared_queue_admits_receipt_verification(self):
        for server in ('s1','s3'):
            self.assertEqual(T.transition_decision(server, {'Running':False,'Status':'exited','ExitCode':0},
                             self.completed(server)), 'VERIFY_BASELINE_RECEIPTS')

    def test_abnormal_or_incomplete_predecessor_fails_closed(self):
        for error in ('exit', 'missing', 'failed_run', 'duplicate'):
            campaign = self.completed('s1')
            container = {'Running':False,'Status':'exited','ExitCode':0}
            if error == 'exit': container['ExitCode'] = 1
            elif error == 'missing': campaign['runs'].pop()
            elif error == 'failed_run': campaign['runs'][0]['status'] = 'FAILED'
            else: campaign['runs'].append(deepcopy(campaign['runs'][0]))
            with self.assertRaises(RuntimeError):
                T.transition_decision('s1', container, campaign)

    def test_explicitly_skipped_control_still_blocks_sampling(self):
        campaign = self.completed('s1')
        group = next(g for g in predecessor.groups('s1') if ('S020C30','R2') in g['runs'])
        campaign['status'] = 'FINISHED_WITH_EXPLICIT_SKIPS'
        campaign['runs'] = [row for row in campaign['runs'] if (row['case'],row['replica']) != ('S020C30','R2')]
        campaign['pending_amendments'] = [{'group':group['name'], 'skip_groups':[group['name']],
            'request_id':'explicit-user-change', 'reason':'Synthetic test only'}]
        with self.assertRaisesRegex(RuntimeError, 'Both completed'):
            T.transition_decision('s1', {'Running':False,'Status':'exited','ExitCode':0}, campaign)

    def test_baseline_receipts_bind_same_server_same_seed_without_weight_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = {'runs':[]}
            donor = root/'synthetic_donor.pt'
            donor.write_bytes(b'SYNTHETIC DONOR FOR HASH TEST')
            selected = {'checkpoint_path':str(donor), 'file_sha256':T.file_sha(donor),
                        'source_run_id':'SYNTHETIC_LOCAL_DONOR', 'completed_update':50000,
                        'source_campaign_id':'SYNTHETIC_DONOR_CAMPAIGN'}
            selector = {'selected':selected, 'receipt_sha256':'b'*64,
                        'validation_inventory':{'inventory_sha256':'c'*64}}
            for replica in ('R1','R2'):
                cfg = predecessor.resolve_case('s1','S020C30',replica)
                binding = {key:'d'*64 for key in T.DONOR_MATCH_KEYS}
                binding.update(sensor='WV3', source_split='RR_train_only', sample_count=100,
                    donor_checkpoint_sha256=selected['file_sha256'], donor_run_id=selected['source_run_id'],
                    donor_step=50000, source_campaign_id=selected['source_campaign_id'],
                    mature30_selector_sha256=selector['receipt_sha256'],
                    mature30_validation_inventory_sha256='c'*64,
                    catalog_sha256=cfg['baseline_resolved']['source_registry_sha256'])
                cfg.update(donor_checkpoint=str(donor), donor_checkpoint_sha256=selected['file_sha256'],
                    selector_receipt_sha256=selector['receipt_sha256'], parent_run=selected['source_run_id'],
                    donor_binding=binding)
                directory = root/'work_dir/panda_schedule30'/T.PREVIOUS_CAMPAIGN/'s1/WV3/runs'/cfg['run_id']/'attempt_01'
                (directory/'models').mkdir(parents=True)
                checkpoint = directory/'models/step_0030000.pt'
                checkpoint.write_bytes(b'SYNTHETIC CHECKPOINT CONTENT FOR HASH-ONLY RECEIPT TEST')
                identity = {'run_id':cfg['run_id'], 'seed':cfg['seed'], 'completed_update':30000,
                            'model_state_sha256':'a'*64}
                evaluation = {'state_before':identity, 'state_after':identity}
                artifact = {'completed_update':30000, 'file_sha256':T.file_sha(checkpoint),
                            'checkpoint_path':str(checkpoint), 'identity':identity, 'model_state_sha256':'a'*64}
                checkpoint.with_suffix('.json').write_text(json.dumps(
                    {'artifact':artifact,'config':cfg,'evaluation':evaluation}))
                (directory/'evaluation_0030000.json').write_text(json.dumps(evaluation))
                campaign['runs'].append({'run_id':cfg['run_id'],'status':'COMPLETED',
                                         'step':30000,'directory':str(directory)})
            donor_patch = patch('panda_sampling30.donor.source_receipt',
                                return_value=(root/'synthetic_selector.json', selector))
            with donor_patch:
                receipts = T.baseline_receipts(root, campaign, 's1')
            self.assertEqual([r['replica'] for r in receipts], ['R1','R2'])
            self.assertTrue(all(r['reference_only_no_weights_imported'] and not r['reposted_to_sheet']
                                for r in receipts))
            T.require_matched_donor_binding(binding, receipts)
            for key in ('source_h5_sha256', 'teacher_state_sha256', 'sample_ids_sha256'):
                wrong = dict(binding, **{key:'e'*64})
                with self.assertRaisesRegex(ValueError, 'Loaded sampling donor/data'):
                    T.require_matched_donor_binding(wrong, receipts)
            first_path = Path(receipts[0]['checkpoint_path'])
            meta_path = first_path.with_suffix('.json')
            original = json.loads(meta_path.read_text())
            for key,new_value in (('dataset','GF2'), ('stream_seeds',{}), ('augmentation',{}),
                                  ('selector_receipt_sha256','e'*64), ('donor_checkpoint_sha256','e'*64)):
                wrong = deepcopy(original)
                wrong['config'][key] = new_value
                meta_path.write_text(json.dumps(wrong))
                with patch('panda_sampling30.donor.source_receipt',
                           return_value=(root/'synthetic_selector.json',selector)):
                    with self.assertRaises(ValueError):
                        T.baseline_receipts(root, campaign, 's1')
            meta_path.write_text(json.dumps(original))
            first_path.write_bytes(b'CORRUPTED')
            with patch('panda_sampling30.donor.source_receipt',
                       return_value=(root/'synthetic_selector.json',selector)):
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    T.baseline_receipts(root, campaign, 's1')

    def test_baseline_receipts_reject_cross_server_or_missing_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'did not complete'):
                T.baseline_receipts(root, {'runs':[]}, 's3')
            cfg = predecessor.resolve_case('s1','S020C30','R1')
            with self.assertRaisesRegex(ValueError, 'did not complete'):
                T.baseline_receipts(root, {'runs':[{'run_id':cfg['run_id'],'status':'COMPLETED','step':30000}]}, 's3')


if __name__ == '__main__':
    unittest.main()
