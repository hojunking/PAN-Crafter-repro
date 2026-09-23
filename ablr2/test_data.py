import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

import h5py
import numpy as np
import torch

from ablr2.common import atomic_json,read_json,sha256,camp
from ablr2.data import ABLR2Dataset,prepare_data,validate_manifest
from ablr2.plan import SensorSpec
from fh12.data import RECIPE,AUGMENTATION,canonical_sha


class DatasetTests(unittest.TestCase):
    def test_parent_constructor_and_native_fixed_views_c4_c8(self):
        for sensor,bands,order in (('QB',4,('B','G','R','NIR')),('GF2',4,('B','G','R','NIR')),
                                  ('WV3',8,('Coastal','B','G','Y','R','RE','NIR1','NIR2'))):
            with tempfile.TemporaryDirectory() as directory:
                source,lp=Path(directory)/'source.h5',Path(directory)/'lp.h5'
                pan=np.arange(2*16*16,dtype=np.float32).reshape(2,1,16,16)
                with h5py.File(source,'w') as out:
                    out['pan']=pan;out['ms']=np.ones((2,bands,4,4),np.float32)*100
                    out['lms']=np.ones((2,bands,16,16),np.float32)*200
                    out['gt']=np.ones((2,bands,16,16),np.float32)*300
                with h5py.File(lp,'w') as out:
                    out['lpan']=pan[:,:,2::4,2::4]
                    out.attrs['source_sha256']=sha256(source);out.attrs['recipe_sha256']=canonical_sha(RECIPE)
                dataset=ABLR2Dataset(source,lp,spec=SensorSpec(sensor,order),split='train')
                self.assertEqual(len(dataset),2)
                self.assertEqual(dataset.bands,bands)
                base=dataset.base(1)
                self.assertTrue(torch.equal(base[4],torch.from_numpy(pan[1])* (2/dataset.spec.max_dn)-1))
                self.assertEqual(base[-1].tolist(),[1,0,0,0])
                actual=dataset[(1,2)]
                self.assertEqual(actual[-1].tolist(),[1,2,1,1])
                self.assertTrue(torch.equal(actual[4],torch.rot90(base[4].flip((-2,-1)),2,(-2,-1))))
                with self.assertRaises(IndexError):dataset.base(2)


class PreparationTests(unittest.TestCase):
    def fixture(self,root):
        spec=SensorSpec('QB',('B','G','R','NIR'))
        bindings={};items={}
        counts=dict(train=17139,val=1905,rr=20,fr=20)
        scans={}
        for split in ('train','val','rr','fr'):
            source=root/(split+'.h5');source.write_bytes(('source-'+split).encode())
            lp=root/(split+'-lp.h5');lp.write_bytes(('lp-'+split).encode())
            bindings[split]=dict(path=str(source),sha256=sha256(source),source_identity='fixed-'+split)
            scans[split]=dict(count=counts[split],shapes={},statistics={})
            items[split]=dict(dataroot=str(source),sha256=sha256(source),source_identity='fixed-'+split,
                lpan_path=str(lp),lpan_sha256=sha256(lp),sample_order_sha256='d'*64,lpan_canonical_sha256='e'*64,**scans[split])
        proof={}
        for split in ('train','val'):
            raw=root/('raw-'+split);raw.write_bytes(('raw-'+split).encode())
            proof['raw_'+split+'_path']=str(raw);proof['raw_'+split+'_sha256']=sha256(raw)
            items[split]['msfix_audit']={'status':'PASS','scope':'synthetic mocked audit'}
        entry=dict(splits=bindings,source_provenance=proof)
        data=dict(schema='ABLR2_DATA_v1',server='s2',sensor='QB',num_bands=4,max_pixel=2047,mtf_sensor='QB',
            band_order=list(spec.band_order),source_provenance=proof,recipe=RECIPE,augmentation=AUGMENTATION,
            augmentation_sha256=canonical_sha(AUGMENTATION),splits=items)
        for name in ('ablr2/sensor_sources.json','ablr2/data.py','qg40/data.py','fh12/data.py','tools/repair_lpan.py','tools/repair_qb_ms.py'):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
        external=root/'external.json';atomic_json(external,data)
        return spec,entry,data,external,scans

    def test_external_manifest_requires_whole_lp_and_raw_qb_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);spec,entry,data,external,scans=self.fixture(root)
            with patch('ablr2.data.source_catalog',return_value=(spec,entry)), \
                 patch('ablr2.data.scan_source',side_effect=lambda p,s,*a:scans[s]) as scan, \
                 patch('ablr2.data.verify_lp_cache',return_value=dict(sample_order_sha256='d'*64,lpan_canonical_sha256='e'*64)) as lp, \
                 patch('ablr2.data.verify_qb_msfix',return_value={'status':'PASS','scope':'synthetic mocked audit'}) as qb:
                target=prepare_data(root,'s2',manifest_path=external)
                self.assertEqual(scan.call_count,4);self.assertEqual(lp.call_count,4);self.assertEqual(qb.call_count,2)
                self.assertEqual(read_json(target),data)
                proof=read_json(camp(root,'s2')/'data_verification_ablr2x.json')
                self.assertEqual(proof['dataset_manifest_sha256'],canonical_sha(data))
                self.assertTrue(proof['raw_qb_msfix_verified'])
                # Only the exact locally published receipt enables reuse.
                prepare_data(root,'s2')
                self.assertEqual(lp.call_count,4);self.assertEqual(qb.call_count,2)
                proof['dataset_manifest_sha256']='f'*64
                atomic_json(camp(root,'s2')/'data_verification_ablr2x.json',proof)
                prepare_data(root,'s2')
                self.assertEqual(lp.call_count,8);self.assertEqual(qb.call_count,4)

    def test_external_bad_lp_is_not_trusted_even_with_claimed_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);spec,entry,data,external,scans=self.fixture(root)
            with patch('ablr2.data.source_catalog',return_value=(spec,entry)), \
                 patch('ablr2.data.scan_source',side_effect=lambda p,s,*a:scans[s]), \
                 patch('ablr2.data.verify_lp_cache',side_effect=ValueError('PAN/LP mismatch')):
                with self.assertRaisesRegex(ValueError,'PAN/LP'):
                    prepare_data(root,'s2',manifest_path=external)
            self.assertFalse((camp(root,'s2')/'dataset_manifest.json').exists())

    def test_changed_raw_qb_fails_before_receipt_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);spec,entry,data,external,scans=self.fixture(root)
            Path(entry['source_provenance']['raw_train_path']).write_bytes(b'changed raw')
            with patch('ablr2.data.source_catalog',return_value=(spec,entry)):
                with self.assertRaisesRegex(ValueError,'raw QB'):
                    prepare_data(root,'s2',manifest_path=external)

    def test_wrong_sensor_or_sample_count_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _,_,data,_,_=self.fixture(Path(directory))
            with self.assertRaises(ValueError):validate_manifest(data,'s1')
            data['splits']['rr']['count']=19
            with self.assertRaisesRegex(ValueError,'count'):validate_manifest(data,'s2')


if __name__=='__main__':unittest.main()
