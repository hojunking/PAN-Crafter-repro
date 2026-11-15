from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import recipes, transition


def state(server='s1'):
    return {'runs': [dict(run_id=c['run_id'], case=c['case_id'], status='QUEUED', step=0)
                    for c in recipes.ORIGINAL_CONFIGS[server].values()]}


class TransitionTests(unittest.TestCase):
    def test_actual_progress_drives_server_specific_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = state(); s['runs'][0].update(status='COMPLETED', step=40000)
            s['runs'][1].update(status='PAUSED', step=28450)
            result = transition.classify(s, tmp)
            self.assertEqual(result['c30_order'], ['C_CORRECTED_DIFF', 'A_RAW', 'B_RAW_CORRECTED'])
            self.assertEqual(result['finish_original_run_ids'], [s['runs'][1]['run_id']])
            self.assertEqual(transition.next_original(s, result), s['runs'][1])
            s['runs'][1].update(status='COMPLETED', step=40000)
            self.assertIsNone(transition.next_original(s, result))
            # A still-active QB yields B,C replacements before the A rerun.
            s = state('s2'); s['runs'][0].update(status='RUNNING', step=25000)
            self.assertEqual(transition.classify(s, tmp)['c30_order'],
                             ['B_RAW_CORRECTED', 'C_CORRECTED_DIFF', 'A_RAW'])

    def test_stale_zero_step_never_overwrites_a_started_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = state(); folder = Path(tmp) / 'runs' / s['runs'][0]['run_id']
            folder.mkdir(parents=True); (folder / 'train.jsonl').write_text('{"step":1}\n')
            result = transition.classify(s, tmp)
            self.assertEqual(result['rerun_cases'], ['A_RAW'])
            self.assertEqual(result['finish_original_run_ids'], [s['runs'][0]['run_id']])

    def test_idempotent_manifest_and_no_original_pending_execution(self):
        from panda_input3 import runtime
        with tempfile.TemporaryDirectory() as tmp:
            root, new = Path(tmp) / 'old', Path(tmp) / 'new'
            root.mkdir(); s = state(); s['runs'][0].update(status='PAUSED', step=250)
            path = root / 'campaign_state.json'; path.write_text(json.dumps(s))
            with patch.object(recipes, 'ORIGINAL_ROOT', return_value=root), \
                 patch.object(recipes, 'campaign_root', return_value=new), \
                 patch.object(runtime, 'read_state', side_effect=lambda *_: json.loads(path.read_text())):
                a = transition.prepare('s1'); b = transition.prepare('s1')
                self.assertEqual(a, b)
                actual = json.loads(path.read_text())
                self.assertEqual([r['status'] for r in actual['runs']],
                                 ['PAUSED', transition.REPLACED, transition.REPLACED])
                actual['runs'][0].update(status='COMPLETED', step=40000)
                self.assertIsNone(transition.next_original(actual, a))
                # A stale original worker resuming a replaced job is detected.
                directory = root / 'runs' / s['runs'][1]['run_id']; directory.mkdir(parents=True)
                (directory / 'config.json').write_text('{}')
                with self.assertRaises(ValueError): transition.prepare('s1')


class RecipeTests(unittest.TestCase):
    def test_all_nine_configs_only_change_horizon_and_identity(self):
        allowed = {'campaign_id', 'method_revision', 'run_id', 'experiment', 'horizon',
                   'primary_step', 'common_comparison_step', 'evaluation_steps', 'schedule',
                   'schedule_comparison', 'config_sha256'}
        from panda_input3.training import CosineHold
        for server in ('s1', 's2', 's3'):
            for case in recipes.CASES:
                old = recipes.ORIGINAL_CONFIGS[server][case]; new = recipes.resolve(server, case)
                self.assertEqual({k: v for k, v in old.items() if k not in allowed},
                                 {k: v for k, v in new.items() if k not in allowed})
                self.assertEqual(new['horizon'], 30000)
                self.assertEqual(new['evaluation_steps'], list(range(1000, 30001, 1000)))
                schedule = CosineHold(new)
                self.assertEqual(schedule.rate(30000), 0.)
                self.assertGreater(schedule.rate(29000), 0.)
                self.assertAlmostEqual(schedule.rate(new['warmup']), old['lr'])
                self.assertEqual(new['schedule_comparison']['parent_config_sha256'], old['config_sha256'])
                self.assertNotEqual(old['run_id'], new['run_id'])
                with self.assertRaises(ValueError): schedule.rate(30001)

    def test_container_uses_correct_mounts_and_module(self):
        from .enqueue import command
        cmd = command('s1', 'hojunqueen/pancrafter-env:torch2.4.0-cu118', detached=True)
        self.assertEqual(cmd[-3:], ['python', '-m', 'panda_input3_c30'])
        mounts = [cmd[i + 1] for i, v in enumerate(cmd) if v == '--mount']
        self.assertTrue(any(m.endswith('dst=/workspace/work_dir/panda_input3_c30') for m in mounts))
        self.assertTrue(any(m.endswith('dst=/workspace/work_dir/panda_input3') for m in mounts))
        self.assertIn('panda.campaign=' + recipes.CAMPAIGN_ID, cmd)


class ZInstalledWorkerTests(unittest.TestCase):
    def test_shared_engine_uses_c30_identity_and_original_target_policy(self):
        # Keep the process-local adapter last; original training runs in a
        # separate child process, never in a process with this adapter installed.
        from .install import install
        install()
        from panda_input3 import plan, publication as pub, release
        from panda_input3_targets.test_reporting import record
        from panda_input3_targets.policy import choose
        from directshift48.reporting import INDEX
        cfg = plan.resolve('s1', 'A_RAW')
        self.assertEqual(cfg['horizon'], 30000)
        self.assertIn(recipes.ORIGINAL_CAMPAIGN, str(recipes.ORIGINAL_ROOT('s1')))
        proof = release.verify()
        self.assertNotEqual(proof['source_sha256'], proof['training_release'])
        r = record(cfg, 30000, e=2.05)
        selected = choose([r], 'stage2', cfg)
        self.assertEqual(len(selected), 1)  # HQNR-qualified representative, but no endpoint detail
        row, note = pub.make_row(cfg, 'COMPLETED', r, selected[0][1], 'ROW0_ORIGINAL')
        self.assertEqual(row[INDEX['Campaign_ID']], recipes.CAMPAIGN_ID)
        self.assertIn('INPUT3-C30', row[INDEX['Experiment']])
        self.assertIn('ERGAS 미달', row[INDEX['Selection@Step']])
        self.assertEqual(json.loads(note)['configuration']['schedule']['decay_end'], 30000)


if __name__ == '__main__': unittest.main()
