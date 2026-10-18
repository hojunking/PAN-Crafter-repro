"""No post-training code-pin override, no fresh 30h clock on prep recovery."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from panda30r import plan
from panda30r.recover_preflight import recover


class RecoverTests(unittest.TestCase):
    def test_only_empty_failed_preparation_can_change_isolated_code(self):
        for mode in ('okay','trained','timed','foreign_code','changed_config'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                state={'status':'FAILED','runs':[],'timing':{},'donors':{},'started_unix':100.,'elapsed_seconds':3.}
                code={'panda30r/engine.py':'a'*64,'shared.py':'b'*64}
                admission={'config':plan.default_config('s1'),'code':deepcopy(code),'evaluator':{'test':1},'environment':{'test':2}}
                code['panda30r/engine.py']='c'*64
                if mode=='trained': state['runs']=[{'status':'FAILED','step':1}]
                if mode=='timed': state['timing']={'mature_teacher':{}}
                if mode=='foreign_code': code['shared.py']='d'*64
                if mode=='changed_config': admission['config']['budget']['total_seconds']=40*3600
                (root/'campaign_state.json').write_text(json.dumps(state))
                (root/'admission.json').write_text(json.dumps(admission))
                with patch('panda30r.recover_preflight.code_pin',return_value=code), \
                     patch('panda30r.recover_preflight.configure_precision'), \
                     patch('panda30r.recover_preflight.evaluator_identity',return_value={'test':1}), \
                     patch('panda30r.recover_preflight.environment_manifest',return_value={'test':2}), \
                     patch('panda30r.recover_preflight.time.time',return_value=130.):
                    if mode!='okay':
                        with self.assertRaises(ValueError): recover(root,reason='synthetic regression')
                        self.assertEqual(json.loads((root/'campaign_state.json').read_text()),state)
                    else:
                        result=recover(root,reason='synthetic regression')
                        after=json.loads((root/'campaign_state.json').read_text())
                        self.assertEqual(after['started_unix'],100.)
                        self.assertEqual(after['elapsed_seconds'],30.)
                        self.assertTrue(Path(result['audit_path']).exists())
