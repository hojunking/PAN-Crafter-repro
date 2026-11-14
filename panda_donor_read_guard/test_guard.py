import hashlib
import os
from pathlib import Path
import tempfile
import time
import unittest
from .guard import protect,sha,fingerprint

class GuardTests(unittest.TestCase):
    def fixture(self,root):
        path=Path(root)/'donor';path.write_bytes(b'verified donor bytes')
        receipt={'selected':{'checkpoint_path':'donor','file_sha256':sha(path)}}
        return path,receipt

    def test_real_relatime_transition_legacy_fails_guard_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,receipt=self.fixture(tmp)
            def age():os.utime(path,ns=(time.time_ns()-3*86400*10**9,path.stat().st_mtime_ns))
            def legacy(*args,**kwargs):
                before=path.stat();content=path.read_bytes()
                if path.stat()!=before:raise ValueError('Donor checkpoint changed while loading')
                return content
            age()
            with self.assertRaisesRegex(ValueError,'changed while loading'):legacy()
            age();before=fingerprint(path)
            result=protect(legacy)({},receipt,None,tmp,None,device='cpu')
            self.assertEqual(result,b'verified donor bytes');self.assertEqual(before,fingerprint(path))

    def test_corrupt_file_is_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,receipt=self.fixture(tmp);path.write_bytes(b'corrupted')
            def loader(*a,**kw):self.fail('Corrupt donor must not be deserialized')
            with self.assertRaisesRegex(ValueError,'before guarded load'):
                protect(loader)({},receipt,None,tmp,None,device='cpu')

    def test_change_during_load_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,receipt=self.fixture(tmp)
            def loader(*a,**kw):path.write_bytes(b'changed');return object()
            with self.assertRaisesRegex(ValueError,'during guarded load'):
                protect(loader)({},receipt,None,tmp,None,device='cpu')

    def test_unrelated_failure_is_not_suppressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,receipt=self.fixture(tmp)
            def loader(*a,**kw):raise ValueError('Loaded donor A/T state differs')
            with self.assertRaisesRegex(ValueError,'A/T state differs'):
                protect(loader)({},receipt,None,tmp,None,device='cpu')

if __name__=='__main__':unittest.main()
