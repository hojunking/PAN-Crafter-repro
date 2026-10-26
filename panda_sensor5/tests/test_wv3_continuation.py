from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_sensor5.wv3_continue import bind_entrypoint, install_bridge, reconcile_predecessor
from panda_sensor5 import queue


def proofs():
    a={'run_id':'A','step':30000,'checkpoint_sha256':'a'*64,'evaluation_sha256':'b'*64}
    b={'run_id':'B','step':30000,'checkpoint_sha256':'c'*64,'evaluation_sha256':'d'*64}
    old={'schema':'PANDA_HARD36_COMPLETE_TUNE30_GATE_V1','server':'s3',
         'predecessor_campaign':'TUNE','state_sha256':'old','runs':[a],
         'predecessor_budget_skips':['B'],'successor_skips':[],
         'predecessor_mutated':False}
    current=deepcopy(old)
    current.update(state_sha256='new',runs=deepcopy([a,b]),predecessor_budget_skips=[])
    return old,current


class TestContinuationProof(unittest.TestCase):
    def test_entrypoint_pin_is_durable_and_rejects_change(self):
        with tempfile.TemporaryDirectory() as directory:
            proof=bind_entrypoint(directory)
            self.assertEqual(bind_entrypoint(directory),proof)
            path=Path(directory)/'sensor5_continuation_entrypoint.json'
            changed=dict(proof,bridge_source_sha256='other')
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError): bind_entrypoint(directory)

    def test_exact_original_budget_skip_completion_allowed(self):
        old,current=proofs()
        self.assertEqual(reconcile_predecessor(old,current),['B'])
        self.assertEqual(old['predecessor_budget_skips'],['B'])

    def test_changed_old_checkpoint_or_evaluation_rejected(self):
        for field in ('checkpoint_sha256','evaluation_sha256','step'):
            old,current=proofs()
            current['runs'][0][field]='changed'
            with self.assertRaises(ValueError): reconcile_predecessor(old,current)

    def test_identity_or_method_policy_change_rejected(self):
        for field,value in [('server','s1'),('predecessor_campaign','other'),
                            ('successor_skips',['skip']),('predecessor_mutated',True)]:
            old,current=proofs(); current[field]=value
            with self.assertRaises(ValueError): reconcile_predecessor(old,current)

    def test_partial_extra_or_unregistered_completion_rejected(self):
        old,current=proofs(); old['predecessor_budget_skips']=['B','C']
        with self.assertRaises(ValueError): reconcile_predecessor(old,current)
        old,current=proofs(); old['predecessor_budget_skips']=[]
        with self.assertRaises(ValueError): reconcile_predecessor(old,current)
        old,current=proofs(); current['predecessor_budget_skips']=['C']
        with self.assertRaises(ValueError): reconcile_predecessor(old,current)

    def test_bridge_preserves_admission_and_keeps_revalidating(self):
        from panda_hard36 import transition
        old,current=proofs()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            admission={'immutable_predecessor_migration':old,'config':{'fixed':1},
                       'code':{'unchanged':True},'sources':{'unchanged':True}}
            path=root/'admission.json'; path.write_text(json.dumps(admission))
            original=path.read_bytes()
            (root/'campaign_state.json').write_text(json.dumps({'migration_receipt':old}))
            with patch('panda_sensor5.wv3_continue.campaign_root',return_value=root), \
                 patch.object(transition,'predecessor_receipt',return_value=current) as official:
                proof=install_bridge('s3',root=root)
                self.assertEqual(proof['additional_completed_tune_runs'],['B'])
                self.assertEqual(path.read_bytes(),original)
                self.assertEqual(transition.predecessor_receipt(root,'s3'),old)
                changed=deepcopy(current); changed['state_sha256']='changed_again'
                official.return_value=changed
                with self.assertRaises(ValueError): transition.predecessor_receipt(root,'s3')
            receipt=json.loads((root/'sensor5_predecessor_continuation.json').read_text())
            self.assertTrue(receipt['all_other_admission_checks_unchanged'])

    def test_original_state_must_match_admission(self):
        from panda_hard36 import transition
        old,current=proofs()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'admission.json').write_text(json.dumps({'immutable_predecessor_migration':old}))
            (root/'campaign_state.json').write_text(json.dumps({'migration_receipt':current}))
            with patch('panda_sensor5.wv3_continue.campaign_root',return_value=root), \
                 patch.object(transition,'predecessor_receipt',return_value=current):
                with self.assertRaises(ValueError): install_bridge('s3',root=root)
            self.assertFalse((root/'sensor5_predecessor_continuation.json').exists())

    def test_cutover_intent_before_signal_is_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            initial={'package':'panda_tune30','target_run_id':'A'}
            with patch.object(queue,'initial_cutover',return_value=initial), \
                 patch.object(queue,'inspect_optional',return_value={'State':{'Running':True}}), \
                 patch.object(queue,'state_of',return_value={'runs':[{'run_id':'A','status':'RUNNING'}]}), \
                 patch.object(queue.subprocess,'run') as signal:
                self.assertFalse(queue.ensure_cutover('s3',Path(directory)))
                signal.assert_called_once_with(['docker','kill','--signal=SIGTERM',
                    'panda-tune30-hardonly-wv3-s3-v1'],check=True)

    def test_changed_target_is_never_signalled(self):
        with tempfile.TemporaryDirectory() as directory:
            initial={'package':'panda_tune30','target_run_id':'A'}
            with patch.object(queue,'initial_cutover',return_value=initial), \
                 patch.object(queue,'inspect_optional',return_value={'State':{'Running':True}}), \
                 patch.object(queue,'state_of',return_value={'runs':[{'run_id':'B','status':'RUNNING'}]}), \
                 patch.object(queue.subprocess,'run') as signal:
                with self.assertRaises(ValueError): queue.ensure_cutover('s3',Path(directory))
                signal.assert_not_called()
