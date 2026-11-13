"""Reference switching must affect selection/stopping, never numerical evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from directshift48.reporting import INDEX
from panda_eq40_zeroq import plan, publication as pub
from . import policy


def record(cfg, step, original, corrected, ergas, psnr=38.1):
    return {'step': step, 'run_id': cfg['run_id'],
        'fr': {'metrics': {'HQNR': original, 'D_s': 1-original, 'D_lambda': .01, 'JQM': .95}},
        'fr_corrected_pan': {'metrics': {'HQNR': corrected, 'D_s': 1-corrected, 'D_lambda': .01, 'JQM': .97}},
        'rr': {'metrics': {'ERGAS': ergas, 'PSNR': psnr}},
        'evidence_path': '/synthetic/eval.json', 'evidence_sha256': 'e'*64,
        'model_sha256': 'm'*64, 'checkpoint': {'file_sha256': 'f'*64},
        'recorded_at': '2026-11-14T00:00:00+00:00', 'elapsed_seconds': 1.,
        'jqm_variant': 'test', 'donor_selection': {'donor_run_id': cfg['donor_run_id']}}


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        policy.install()

    def test_corrected_gate_and_ergas_priority(self):
        cfg = plan.queue('s1')[0]
        rows = [record(cfg, 1000, .99, .94, 2.01), record(cfg, 38000, .91, .960, 2.0217),
                record(cfg, 40000, .92, .9601, 2.0222)]
        before = deepcopy(rows)
        selected = policy.choose(rows, 'stage2', cfg)
        self.assertEqual([r[0]['step'] for r in selected], [38000, 40000])
        self.assertTrue(policy.joint_pass(rows[1], cfg))
        self.assertFalse(policy.joint_pass(rows[0], cfg))
        self.assertEqual(rows, before)

    def test_fallback_missing_reference_and_GF2_psnr(self):
        cfg = plan.queue('s3')[0]
        a, b = record(cfg, 1000, .99, .94, .50, 45.1), record(cfg, 2000, .9, .95, .52, 45.1)
        self.assertEqual(policy.choose([a,b], 'stage2', cfg)[0][0]['step'], 2000)
        b['fr_corrected_pan']['metrics']['HQNR'] = .97
        b['rr']['metrics']['PSNR'] = 45.07
        self.assertTrue(policy.joint_pass(b, cfg)); self.assertFalse(pub.all_targets_pass(b, cfg))
        del a['fr_corrected_pan']
        with self.assertRaises(KeyError): policy.choose([a], 'stage2', cfg)

    def test_legacy_row_migration_and_new_reference_provenance(self):
        cfg = plan.queue('s1')[0]; e = record(cfg, 38000, .919, .960, 2.0217)
        with policy.legacy_context():
            old, _ = policy._legacy_make(cfg, 'COMPLETED', e, 'BEST_HQNR_FALLBACK', 'ROW0_ORIGINAL')
        value = {'configs': {cfg['run_id']: cfg}, 'proof': {(cfg['run_id'], 38000): e}}
        pub.verify_owned(old, value, 's1')
        new, note = pub.make_row(cfg, 'COMPLETED', e, 'HQNR_QUALIFIED_MIN_ERGAS', 'ROW0_ORIGINAL')
        pub.verify_owned(new, value, 's1')
        self.assertEqual(old[3:15], new[3:15])
        self.assertEqual(old[INDEX['Result_ID']], new[INDEX['Result_ID']])
        self.assertEqual(json.loads(note)['checkpoint_selection_reference'], 'corrected PAN')
        altered = list(old); altered[INDEX['HQNR↑']] += .1
        with self.assertRaises(ValueError): pub.verify_owned(altered, value, 's1')

    def test_QB_stop_uses_corrected_all_three_and_strict_threshold(self):
        cfg = plan.queue('s2')[0]
        with tempfile.TemporaryDirectory() as tmp, patch('panda_eq40_zeroq.evaluation.verify_record'):
            directory = Path(tmp)
            for step in (8000,9000,10000):
                (directory/f'evaluation_{step:07d}.json').write_text(json.dumps(record(cfg, step, .75, .89, 3.6)))
            stopped = policy.early_stop_evidence(cfg, directory, 10000)
            self.assertEqual(stopped['policy']['reference'], 'CORRECTED_PAN')
            (directory/'evaluation_0010000.json').write_text(json.dumps(record(cfg, 10000, .75, .90, 3.6)))
            self.assertIsNone(policy.early_stop_evidence(cfg, directory, 10000))
            self.assertIsNone(policy.early_stop_evidence(cfg, directory, 9000))

    def test_reopen_only_when_corrected_stop_no_longer_applies(self):
        from panda_eq40_zeroq.runtime import initial_state
        with tempfile.TemporaryDirectory() as tmp, patch('panda_eq40_zeroq.evaluation.verify_record'):
            root = Path(tmp); state = initial_state('s2')
            for i, (job, cfg) in enumerate(zip(state['runs'], plan.queue('s2'))):
                job.update(status='EARLY_STOPPED', step=10000)
                d = root/'runs'/job['run_id']; (d/'resume').mkdir(parents=True)
                for step in (8000,9000,10000):
                    (d/f'evaluation_{step:07d}.json').write_text(json.dumps(record(cfg, step, .75, .91 if i==0 else .89, 3.6)))
                (d/'resume/step_0010000_test.pt').write_bytes(b'fixture')
                (d/'resume/step_0010000_test.json').write_text(json.dumps({'step':10000}))
                (d/'early_stop.json').write_text(json.dumps({'reason':'ORIGINAL_REFERENCE'}))
            (root/'campaign_state.json').write_text(json.dumps(state))
            with patch.object(plan, 'campaign_root', return_value=root):
                review = policy.adopt_runtime_policy('s2')
            after = json.loads((root/'campaign_state.json').read_text())
            self.assertEqual([j['status'] for j in after['runs']], ['PAUSED', 'EARLY_STOPPED'])
            self.assertEqual([j['step'] for j in after['runs']], [10000, 10000])
            self.assertEqual(review[0]['previous_stop_evidence'], {'reason':'ORIGINAL_REFERENCE'})


if __name__ == '__main__': unittest.main()
