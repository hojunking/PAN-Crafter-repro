"""Fail-closed asset boundary tests, all writes restricted to temporary fixtures."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from maina_hqnr import assets
from maina_hqnr.common import object_sha,sha256


class AssetTests(unittest.TestCase):
    def test_path_only_overrides(self):
        good={'bridge_path':'assets/F1/bridge.json','data':{'rr':{'dataroot':'rr.h5'}},
              'artifacts':{'teacher_checkpoint':'teacher.safetensors'}}
        self.assertEqual(assets._validate_overrides(good),good)
        for bad in ({'teacher':'F5'},{'data':{'rr':{'sha256':'replacement'}}},
                    {'data':{'unknown':{}}},{'data':[]},{'regenerate':True}):
            with self.assertRaises(ValueError): assets._validate_overrides(bad)

    def test_relative_paths_are_portable_not_guessed(self):
        self.assertEqual(assets._path('/home/knuvi/Desktop/song/PAN-Crafter/work_dir/F1','/srv/release'),
                         Path('/srv/release/work_dir/F1'))
        self.assertEqual(assets._path('/external/native.h5','/srv/release'),Path('/external/native.h5'))
        with self.assertRaises(ValueError): assets._path('../outside','/srv/release')

    def test_full_file_sha_required(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'asset';p.write_bytes(b'original')
            self.assertEqual(assets._checked_file(p,sha256(p)),str(p.resolve()))
            with self.assertRaises(ValueError): assets._checked_file(p,sha256(p)[:16])
            with self.assertRaises(ValueError): assets._checked_file(p,'0'*64)
            with self.assertRaises(FileNotFoundError): assets._checked_file(Path(temp)/'missing','0'*64)

    def test_original_q_and_actual_indices_are_frozen(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'q.npz';idx=np.arange(3072,dtype=np.int64)
            q=np.full((9714,4),assets.Q_REF,dtype=np.float64)
            np.savez(path,q=q,calibration_indices=idx)
            origin=dict(calibration_indices_sha256=object_sha(idx.tolist()),q_ref=assets.Q_REF,tau_R=assets.TAU_R)
            self.assertEqual(assets._q_values(origin,path).shape,(9714,4))
            bad=dict(origin,q_ref=assets.Q_REF*2)
            with self.assertRaises(ValueError): assets._q_values(bad,path)
            q[1,1]=np.nan;np.savez(path,q=q,calibration_indices=idx)
            with self.assertRaises(ValueError): assets._q_values(origin,path)

    def test_missing_f1_cannot_substitute_or_write(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(assets,'verify_sources',return_value=True):
            with self.assertRaises(FileNotFoundError): assets.verify_assets('s4',temp)
            self.assertFalse((Path(temp)/'work_dir/maina_hqnr').exists())
            with self.assertRaises(ValueError): assets.verify_assets('s1',temp)

    def test_forged_binding_rejected(self):
        with self.assertRaises(ValueError):
            assets.validate_bindings({'server':'s4','schema':'G23SENS_ASSET_BINDING_v1'},rehash=False)


if __name__=='__main__': unittest.main()
