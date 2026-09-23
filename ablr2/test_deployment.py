"""Frozen launch construction is tested without Git mutations or processes."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ablr2.deployment import frozen_checkout
from ablr2.common import camp,read_json
from fh12.data import write_immutable_json  # Load native dependencies before subprocess mocks.


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'repo';self.root.mkdir()
        (self.root/'data').mkdir();(self.root/'work_dir').mkdir()
        self.head='a'*40
        self.identity=patch('ablr2.deployment.source_identity',return_value={'files':{}})
        self.identity.start();self.addCleanup(self.identity.stop)

    def git(self,args,**kwargs):
        if args[1:3]==['rev-parse','HEAD']: return self.head+'\n'
        if args[1]=='ls-files': return b'tools/ablr2_start.sh\0'
        if args[1]=='diff': return ''
        raise AssertionError(args)

    def test_untracked_code_cannot_be_started(self):
        def untracked(args,**kwargs):
            return b'' if args[1]=='ls-files' else self.git(args,**kwargs)
        with patch('ablr2.deployment.subprocess.check_output',side_effect=untracked), \
             patch('ablr2.deployment.subprocess.run') as mutate:
            with self.assertRaisesRegex(ValueError,'Commit'): frozen_checkout(self.root,'s1')
            mutate.assert_not_called()

    def test_dirty_numerical_release_does_not_stash_or_reset(self):
        def dirty(args,**kwargs):
            return 'tools/ablr2_start.sh\n' if args[1]=='diff' else self.git(args,**kwargs)
        with patch('ablr2.deployment.subprocess.check_output',side_effect=dirty), \
             patch('ablr2.deployment.subprocess.run') as mutate:
            with self.assertRaisesRegex(ValueError,'Uncommitted'): frozen_checkout(self.root,'s1')
            mutate.assert_not_called()

    def test_frozen_checkout_reuses_original_release_and_shared_assets(self):
        def create(args,**kwargs):
            self.assertEqual(args[:4],['git','worktree','add','--detach'])
            Path(args[4]).mkdir()
        with patch('ablr2.deployment.subprocess.check_output',side_effect=self.git), \
             patch('ablr2.deployment.subprocess.run',side_effect=create) as mutate:
            first=frozen_checkout(self.root,'s1')
            self.assertEqual((first/'data').resolve(),(self.root/'data').resolve())
            self.assertEqual((first/'work_dir').resolve(),(self.root/'work_dir').resolve())
            self.assertEqual(frozen_checkout(self.root,'s1'),first)
            self.assertEqual(mutate.call_count,1)
            self.assertEqual(read_json(camp(self.root,'s1')/'runtime_release.json')['git_commit'],self.head)
            self.assertFalse(camp(self.root,'s2').exists())

    def test_other_server_lanes_are_rejected_before_git(self):
        with patch('ablr2.deployment.subprocess.check_output') as git:
            with self.assertRaises(ValueError): frozen_checkout(self.root,'s4')
            git.assert_not_called()


if __name__=='__main__': unittest.main()
