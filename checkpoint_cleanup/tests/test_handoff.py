import json
from pathlib import Path
import tempfile
import unittest

from checkpoint_cleanup.handoff import category, inventory, joint_pass, rule, s2_pins
from checkpoint_cleanup.ledger import sha


class HandoffTests(unittest.TestCase):
    def test_main_tuning_is_not_component_ablation(self):
        self.assertEqual(category('panda_qb_s2_frozen'), 'MAIN_TUNING')
        self.assertEqual(rule('panda_qb_s2_frozen', 'stage2', 'COMPLETED', 30000),
                         {'action': 'REVIEW_NO_JOINT_PASS_RUNS'})
        self.assertEqual(rule('panda_qb_s2', 'donor', 'COMPLETED', 50000)['action'],
                         'KEEP_STAGE1_DONOR')

    def test_stage1_ablation_retains_validation_selected_exception(self):
        result = rule('panda_wv3_ablation_rgb', 'stage1', 'COMPLETED', 50000)
        self.assertEqual(result['cutoff_inclusive'], 30000)
        self.assertTrue(result['protect_validation_selected'])
        result = rule('panda_wv3_ablation', 'stage2', 'COMPLETED', 30000)
        self.assertEqual(result['cutoff_inclusive'], 20000)
        self.assertFalse(result['protect_validation_selected'])
        for status in ['RUNNING', 'FAILED', 'STOPPED', 'QUEUED']:
            self.assertEqual(rule('panda_wv3_ablation_rgb', 'stage1', status, 50000)['action'],
                             'KEEP_INCOMPLETE')

    def test_unrounded_joint_target_and_gf2_psnr(self):
        self.assertTrue(joint_pass('QB', {'HQNR': .9315, 'ERGAS': 3.5}))
        self.assertFalse(joint_pass('QB', {'HQNR': .9314999, 'ERGAS': 3.49}))
        self.assertFalse(joint_pass('QB', {'HQNR': .95, 'ERGAS': 3.50001}))
        self.assertFalse(joint_pass('QB', {'HQNR': .93, 'ERGAS': 3.48}))
        self.assertTrue(joint_pass('GF2', {'HQNR': .9645, 'ERGAS': .522, 'PSNR': 45.08}))
        self.assertFalse(joint_pass('GF2', {'HQNR': .97, 'ERGAS': .50, 'PSNR': 45.079}))
        with self.assertRaises(KeyError):
            joint_pass('GF2', {'HQNR': .97, 'ERGAS': .50})
        with self.assertRaises(ValueError):
            joint_pass('QB', {'HQNR': float('nan'), 'ERGAS': 3.4})

    def test_s2_pins_include_required_stage2_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'work_dir/source/model.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'fixture')
            plan = {'donors': {'D4': {
                'checkpoint_path': '/workspace/work_dir/source/model.pt',
                'checkpoint_sha256': sha(checkpoint),
                'baseline_R1': {'artifact': {'checkpoint_path': '/workspace/work_dir/source/baseline.pt',
                                             'file_sha256': 'a' * 64}}}}}
            p = root / 'config/panda_qb_s2_frozen/plan.json'
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps(plan))
            pins = s2_pins(root)
            self.assertEqual(pins[0]['status'], 'VERIFIED')
            self.assertEqual(pins[1]['role'], 'REQUIRED_STAGE2_BASELINE')
            self.assertEqual(pins[1]['status'], 'MISSING_OR_CHANGED')

    def test_inventory_only_named_server_and_no_delete_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for server in ['s1', 's3']:
                campaign = root / f'work_dir/panda_wv3_ablation_rgb/campaign/{server}/WV3'
                run = campaign / 'runs/fixture'
                run.mkdir(parents=True)
                (campaign / 'campaign_state.json').write_text(json.dumps({
                    'status': 'COMPLETED', 'runs': [{'run_id': 'fixture', 'status': 'COMPLETED'}]}))
                (run / 'config.json').write_text(json.dumps({'run_id': 'fixture', 'server': server,
                                                            'stage': 'stage1', 'horizon': 50000}))
            report = inventory('s3', root)
            self.assertEqual(len(report['campaigns']), 1)
            self.assertIn('/s3/', report['campaigns'][0]['root'])
            self.assertEqual(report['campaigns'][0]['runs'][0]['cutoff_inclusive'], 30000)
            self.assertFalse(report['deletion_ready'])
            self.assertEqual(report['delete_paths'], [])
            self.assertEqual(report['errors'], [])


if __name__ == '__main__':
    unittest.main()
