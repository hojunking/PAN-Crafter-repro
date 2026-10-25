"""Read-only FOLLOW30 gate rejects incomplete or damaged final artifacts.

Tiny synthetic bytes stand in for checkpoint files. No model, GPU, production
state, or predecessor file is modified by these tests.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_tune30 import transition as T


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.directory=self.root/'work_dir/panda_follow30'/T.PREVIOUS_CAMPAIGN/'s1/WV3'
        self.stack=ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(T.previous,'baseline',return_value={}))
        self.stack.enter_context(patch.object(T.previous,'groups',return_value=[
            {'runs':[('SYNTHETIC','R1'),('SYNTHETIC','R2')]}]))
        self.stack.enter_context(patch.object(T.previous,'resolve_case',side_effect=lambda server,case,replica:{
            'run_id':'SYNTHETIC_'+replica,'planned_horizon':1000}))
        self.state={'status':'FINISHED_REGISTERED_QUEUE','runs':[]}
        self.artifacts=[]
        for replica in ('R1','R2'):
            run_id='SYNTHETIC_'+replica
            attempt=self.directory/'runs'/run_id/'attempt_01'
            models=attempt/'models'
            models.mkdir(parents=True)
            checkpoint=models/'step_0001000.pt'
            checkpoint.write_bytes(('synthetic checkpoint '+replica).encode())
            evaluation=attempt/'evaluation_0001000.json'
            evaluation.write_text(json.dumps({
                'qualification':{'same_state_verified':True,'metrics_complete':True,'metrics_finite':True},
                'state_before':{'model_state_sha256':'a'*64}}))
            metadata=checkpoint.with_suffix('.json')
            metadata.write_text(json.dumps({
                'artifact':{'checkpoint_verified':True,'completed_update':1000,
                            'model_state_sha256':'a'*64,
                            'file_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
                'config':{'run_id':run_id}}))
            self.artifacts.append((checkpoint,evaluation,metadata))
            # Exercise the actual container-to-host path translation.
            container_path=Path('/workspace')/attempt.relative_to(self.root)
            self.state['runs'].append({'run_id':run_id,'case':'SYNTHETIC','replica':replica,
                'status':'COMPLETED','step':1000,'directory':str(container_path)})
        self.write_state()

    def write_state(self, state=None):
        (self.directory/'campaign_state.json').write_text(json.dumps(self.state if state is None else state))

    def snapshot(self):
        return {str(path.relative_to(self.root)):path.read_bytes()
                for path in self.root.rglob('*') if path.is_file()}

    def test_running_predecessor_is_rejected_without_mutation(self):
        self.state['status']='RUNNING'
        self.write_state()
        before=self.snapshot()
        with self.assertRaisesRegex(ValueError,'WAITING_FOLLOW30_COMPLETE: RUNNING'):
            T.predecessor_receipt(self.root,'s1')
        self.assertEqual(self.snapshot(),before)

    def test_complete_final_artifacts_produce_stable_read_only_receipt(self):
        before=self.snapshot()
        receipt=T.predecessor_receipt(self.root,'s1')
        self.assertEqual(receipt['schema'],'PANDA_TUNE30_COMPLETE_FOLLOW30_GATE_V1')
        self.assertEqual(receipt['server'],'s1')
        self.assertEqual(receipt['predecessor_campaign'],T.PREVIOUS_CAMPAIGN)
        self.assertEqual(receipt['successor_skips'],[])
        self.assertFalse(receipt['predecessor_mutated'])
        self.assertEqual([r['run_id'] for r in receipt['runs']],['SYNTHETIC_R1','SYNTHETIC_R2'])
        for row,(checkpoint,evaluation,_) in zip(receipt['runs'],self.artifacts):
            self.assertEqual(row['step'],1000)
            self.assertEqual(row['checkpoint_sha256'],hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(row['evaluation_sha256'],hashlib.sha256(evaluation.read_bytes()).hexdigest())
        self.assertEqual(T.predecessor_receipt(self.root,'s1'),receipt)
        self.assertEqual(self.snapshot(),before)

    def test_checkpoint_byte_tamper_is_rejected_even_with_valid_sidecar(self):
        self.artifacts[0][0].write_bytes(b'tampered checkpoint bytes')
        before=self.snapshot()
        with self.assertRaisesRegex(ValueError,'Invalid same-state final FOLLOW30 evidence'):
            T.predecessor_receipt(self.root,'s1')
        self.assertEqual(self.snapshot(),before)

    def test_missing_checkpoint_evaluation_or_metadata_is_rejected(self):
        for path in self.artifacts[0]:
            with self.subTest(artifact=path.name):
                content=path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaisesRegex(ValueError,'Missing final FOLLOW30'):
                        T.predecessor_receipt(self.root,'s1')
                    self.assertFalse(path.exists())
                finally:
                    path.write_bytes(content)

    def test_terminal_controller_cannot_hide_incomplete_run(self):
        for changed in ({'status':'RUNNING'},{'status':'STOPPED'},{'step':999}):
            with self.subTest(changed=changed):
                state=deepcopy(self.state)
                state['runs'][0].update(changed)
                self.write_state(state)
                with self.assertRaisesRegex(ValueError,'did not reach its registered horizon'):
                    T.predecessor_receipt(self.root,'s1')

    def test_missing_or_duplicate_registered_run_is_rejected(self):
        state=deepcopy(self.state)
        state['runs'].pop()
        self.write_state(state)
        with self.assertRaisesRegex(ValueError,'queue incomplete'):
            T.predecessor_receipt(self.root,'s1')
        state=deepcopy(self.state)
        state['runs'].append(deepcopy(state['runs'][0]))
        self.write_state(state)
        with self.assertRaisesRegex(ValueError,'Duplicate predecessor runs'):
            T.predecessor_receipt(self.root,'s1')


if __name__=='__main__':
    unittest.main()
