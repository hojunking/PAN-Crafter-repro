"""Offline asset provenance failures; no campaign binding publication."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from g23sens import assets
from g23sens.test_data import fixture


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_reference_config_is_exact_original_git_blob(self):
        cfg=assets.reference_config(assets.ROOT)
        self.assertEqual(cfg['model_args']['depth'],[1,2,1]);self.assertEqual(cfg['kdv']['rec']['tau'],assets.TAU0)
        path=self.root/assets.BASE_CONFIG;path.parent.mkdir(parents=True);path.write_bytes((assets.ROOT/assets.BASE_CONFIG).read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'blob'):assets.reference_config(self.root)

    def test_native_h5_hashes_all_original_arrays_and_keeps_scene_population(self):
        _,path=fixture(self.root,count=11)
        with patch.dict(assets.COUNTS,train=11):identity=assets.h5_identity(path,'train')
        self.assertEqual(identity['count'],11);self.assertEqual(set(identity['arrays']),{'gt','lms','ms','pan'})
        self.assertEqual(identity['arrays']['gt']['dtype'],'float64')
        with self.assertRaisesRegex(ValueError,'population'):assets.h5_identity(path,'train')
        with h5py.File(path,'a') as h:h['ms'][0,0,0,0]=-1
        with patch.dict(assets.COUNTS,train=11):
            with self.assertRaisesRegex(ValueError,'range'):assets.h5_identity(path,'train')

    def test_stopcheck_interrupts_long_asset_scan(self):
        _,path=fixture(self.root)
        with self.assertRaises(InterruptedError):assets.h5_identity(path,'train',stopcheck=lambda:True)

    def _cue_inputs(self):
        path=assets.ROOT/'assets/qedge9/cue_T0_AXIS16_v1.json';man=json.loads(path.read_text())
        teacher=dict(aligner_state_hash=man['teacher']['aligner_state_hash'],tensors_sha256_16=man['teacher']['tensors_sha256_16'])
        train=dict(count=man['dataset']['n_train'],file_sha256=man['dataset']['train_sha256'])
        lp=dict(file_sha256=man['dataset']['train_pan_sha256'])
        return path,assets.ROOT/man['npz'],teacher,train,lp

    def test_actual_fixed_cue_complete_coverage_raw_q_and_teacher_identity(self):
        args=self._cue_inputs();identity=assets.cue_identity(*args)
        self.assertEqual(identity['n_views'],9714*4);self.assertEqual(identity['q_ref_expected'],assets.Q0)
        self.assertTrue(identity['teacher_identity_verified']);self.assertEqual(len(identity['raw_q_sha256']),64)
        bad=deepcopy(args[2]);bad['aligner_state_hash']='f'*16
        with self.assertRaisesRegex(ValueError,'Teacher'):assets.cue_identity(args[0],args[1],bad,args[3],args[4])
        with self.assertRaisesRegex(ValueError,'data/LP'):
            assets.cue_identity(*args[:4],dict(file_sha256='f'*64))

    def test_cue_negative_or_incomplete_sample_rotation_coverage_is_rejected(self):
        args=self._cue_inputs()
        with np.load(args[1],allow_pickle=False) as z:values={k:z[k] for k in z.files}
        values['q'][0]=-1;npz=self.root/'negative.npz';np.savez(npz,**values)
        man=json.loads(args[0].read_text());man['npz_sha256']=assets.sha256(npz)
        path=self.root/'cue.json';path.write_text(json.dumps(man))
        with patch('kdv.edge_gate.check_asset',return_value=[]):
            with self.assertRaisesRegex(ValueError,'nonnegative'):
                assets.cue_identity(path,npz,*args[2:])
        values['q'][0]=.5;values['index'][0]=values['index'][1];values['rot'][0]=values['rot'][1]
        np.savez(npz,**values);man['npz_sha256']=assets.sha256(npz);path.write_text(json.dumps(man))
        with patch('kdv.edge_gate.check_asset',return_value=[]):
            with self.assertRaisesRegex(ValueError,'coverage'):
                assets.cue_identity(path,npz,*args[2:])

    def test_calibration_wrong_teacher_or_tau_scale_cannot_become_new_origin(self):
        args=self._cue_inputs();cue=assets.cue_identity(*args)
        path=assets.ROOT/'work_dir/_pakd50/calibration_resolved.json'
        doc=json.loads(path.read_text());local=self.root/'calibration.json';local.write_text(json.dumps(doc))
        identity=assets.calibration_identity(local,args[2],cue,args[3])
        self.assertEqual(identity['tau_R_expected'],assets.TAU0)
        doc['tau_R']*=2;local.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError,'fixed T0'):
            assets.calibration_identity(local,args[2],cue,args[3])

    def test_protected_other_lanes_cannot_bind(self):
        for server in ('s1','s2','s3'):
            with self.assertRaises(ValueError):assets.resolve_bindings(self.root,server)

    def test_distributable_calibration_is_canonical_and_never_falls_back_over_mismatch(self):
        original=(assets.ROOT/'assets/pakd50/calibration_resolved.json').read_bytes()
        packed=self.root/'assets/pakd50/calibration_resolved.json'
        legacy=self.root/'work_dir/_pakd50/calibration_resolved.json'
        legacy.parent.mkdir(parents=True);legacy.write_bytes(original)
        self.assertEqual(assets.canonical_calibration(self.root),legacy)
        packed.parent.mkdir(parents=True);packed.write_bytes(original)
        self.assertEqual(assets.canonical_calibration(self.root),packed)
        packed.write_bytes(original+b'\n')
        with self.assertRaisesRegex(ValueError,'bytes changed'):assets.canonical_calibration(self.root)


if __name__=='__main__':unittest.main()
