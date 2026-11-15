from copy import deepcopy
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import recipes, predecessor


class RepeatTests(unittest.TestCase):
    def test_same_recipes_new_paired_randomness_and_distinct_identity(self):
        allowed = {'campaign_id', 'repeat', 'replica', 'seed', 'stream_seeds', 'reference_repeats',
            'run_id', 'experiment', 'official_selection_reference', 'checkpoint_selection',
            'repetition_comparison', 'schedule_comparison', 'config_sha256'}
        ids = set()
        for server in ('s1', 's2', 's3'):
            configs = recipes.queue(server)
            self.assertEqual([c['case_id'] for c in configs], list(recipes.CASES))
            for c in configs:
                old = recipes.PARENTS[server][c['case_id']]
                self.assertEqual({k: v for k, v in old.items() if k not in allowed},
                                 {k: v for k, v in c.items() if k not in allowed})
                self.assertEqual(c['horizon'], 30000)
                self.assertEqual(c['seed'], 261018102)
                self.assertNotEqual(c['stream_seeds']['S_init'], old['stream_seeds']['S_init'])
                self.assertNotEqual(c['stream_seeds']['rr_data'], old['stream_seeds']['rr_data'])
                self.assertEqual(c['stream_seeds'], configs[0]['stream_seeds'])
                self.assertEqual(c['fixed_donor'], old['fixed_donor'])
                self.assertNotIn(c['run_id'], ids); ids.add(c['run_id'])
                recipes.validate(c)
                changed = deepcopy(c); changed['gamma'] = .5
                with self.assertRaises(ValueError): recipes.validate(changed)
            with self.assertRaises(ValueError): recipes.resolve(server, 'A_RAW', 1)

    def test_waits_for_all_prior_jobs_and_final_publication(self):
        expected = list(recipes.PARENTS['s1'].values())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = root / 'campaign_state.json'
            lockpath = root / 'runner.lock'; lockpath.touch()
            s = dict(campaign_id=recipes.previous.CAMPAIGN_ID, server='s1', status='RUNNING',
                queue_sha256=recipes.previous.original.digest(expected),
                runs=[dict(run_id=c['run_id'], status='COMPLETED', step=30000) for c in expected])
            with patch.object(recipes.previous, 'campaign_root', return_value=root), \
                 patch.object(recipes.previous, 'queue', return_value=expected):
                self.assertFalse(predecessor.finished('s1'))
                p.write_text(json.dumps(s)); self.assertFalse(predecessor.finished('s1'))
                s['status'] = 'COMPLETED'; p.write_text(json.dumps(s))
                with lockpath.open('r') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    self.assertFalse(predecessor.finished('s1'))
                self.assertTrue(predecessor.finished('s1'))
                s['runs'][-1]['step'] = 29000; p.write_text(json.dumps(s))
                with self.assertRaises(ValueError): predecessor.finished('s1')

    def test_container_preserves_earlier_roots_and_owns_only_new_outputs(self):
        from .enqueue import command
        cmd = command('s1', 'hojunqueen/pancrafter-env:torch2.4.0-cu118', detached=True)
        self.assertEqual(cmd[-3:], ['python', '-m', 'panda_input3_repeat'])
        self.assertIn('panda.campaign=' + recipes.CAMPAIGN_ID, cmd)
        mounts = [cmd[i + 1] for i, v in enumerate(cmd) if v == '--mount']
        self.assertTrue(any(m.endswith('dst=/workspace/work_dir/panda_input3_repeat') for m in mounts))
        self.assertTrue(any(m.endswith('dst=/workspace,readonly') for m in mounts))
        self.assertFalse(any(m.endswith('dst=/workspace/work_dir/panda_input3_c30') for m in mounts))


if __name__ == '__main__': unittest.main()
