"""CPU-only legacy-method and exact full-stream replay checks."""
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from feeders.feeder import PanFeeder
from g23sens import data


def collate_metadata(batch):
    # CPU sandbox disallows torch shared-storage AF_UNIX socket creation.
    # NumPy transport still runs the actual four Feeder worker processes.
    return np.stack([row[-1].numpy() for row in batch])


def serial_transport(*args,**kwargs):
    kwargs['num_workers']=0
    return DataLoader(*args,**kwargs)


def fixture(root,count=11,split='train'):
    path=Path(root)/(('train' if split=='train' else 'valid')+'_wv3.h5')
    shapes=dict(gt=(count,8,64,64),lms=(count,8,64,64),ms=(count,8,16,16),pan=(count,1,64,64))
    with h5py.File(path,'w') as h:
        for k,shape in shapes.items():h[k]=(np.arange(np.prod(shape)).reshape(shape)%1999).astype(np.float64)+.125
    lp=path.with_name(path.stem+'_pan.h5');shape=(count,1,16,16)
    with h5py.File(lp,'w') as h:h['lpan']=(np.arange(np.prod(shape)).reshape(shape)%1999).astype(np.float32)+.25
    binding=dict(data={split:dict(path=str(path),count=count,arrays={k:dict(shape=list(v)) for k,v in shapes.items()}),
        split+'_lp':dict(path=str(lp),count=count,arrays={'lpan':dict(shape=list(shape))})})
    return binding,path


class LegacyDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.binding,self.path=fixture(self.root)
        self.dataset=data.NativeDataset(self.binding,'train',self.root);self.addCleanup(self.dataset.close)

    def test_lazy_access_is_bit_exact_actual_original_feeder_all_rotations(self):
        original=PanFeeder(str(self.path),crop=False,hflip=True,vflip=True,rot=True,return_meta=True)
        for rotation in range(4):
            with patch('feeders.feeder.random.randint',return_value=rotation):wanted=original[3]
            actual=self.dataset[(3,rotation,1,1)]
            for key,value in zip(('gt','lms','ms','lpan','pan','meta'),wanted):
                self.assertTrue(torch.equal(actual[key],value),(rotation,key))
        self.assertTrue(self.dataset.hflip and self.dataset.vflip)

    def test_fixed_view_never_consumes_global_randomness(self):
        random.seed(8754);before=random.getstate();torch_before=torch.get_rng_state().clone()
        self.dataset[(0,3,1,1)]
        self.assertEqual(before,random.getstate());self.assertTrue(torch.equal(torch_before,torch.get_rng_state()))
        with self.assertRaises(ValueError):self.dataset[(0,1,0,1)]

    def test_schedule_matches_actual_legacy_four_worker_dataloader_across_epochs(self):
        original=PanFeeder(str(self.path),crop=False,hflip=True,vflip=True,rot=True,return_meta=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(76123);state=torch.get_rng_state().clone()
            actual=[]
            for _ in range(3):
                loader=DataLoader(original,batch_size=3,shuffle=True,num_workers=4,drop_last=True,
                                  collate_fn=collate_metadata,timeout=15)
                actual.extend(batch for batch in loader)
        expected,manifest=data.stream_schedule(len(original),3,state,workers=4,total_updates=len(actual))
        self.assertTrue(np.array_equal(expected,np.asarray(actual)))
        self.assertEqual(manifest['epochs'][0]['discarded_per_full_epoch'],2)

    def test_resume_reproduces_pending_batch_without_counting_prefetch(self):
        transport=patch.object(data,'DataLoader',side_effect=serial_transport)
        transport.start();self.addCleanup(transport.stop)
        state=torch.Generator().manual_seed(492).get_state()
        left=data.BatchStream(self.dataset,3,seed=9,initial_torch_rng_state=state,workers=4,total_updates=12)
        self.addCleanup(left.close)
        first=left.next_batch();self.assertEqual(left.completed_updates,0)
        self.assertIs(first,left.next_batch());left.advance()
        saved=left.state_dict();expected=left.next_batch()
        right=data.BatchStream(self.dataset,3,seed=9,initial_torch_rng_state=state,workers=4,total_updates=12)
        self.addCleanup(right.close);right.load_state_dict(saved)
        actual=right.next_batch()
        for key in expected:self.assertTrue(torch.equal(expected[key],actual[key]),key)
        right.advance();self.assertEqual(right.completed_updates,2)
        saved['consumed_prefix_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'prefix'):right.load_state_dict(saved)

    def test_full50k_manifest_is_paired_and_seed_identity_sensitive(self):
        state=torch.Generator().manual_seed(192).get_state()
        a,ma=data.stream_schedule(101,2,state,workers=4,total_updates=50000,seed=1234)
        b,mb=data.stream_schedule(101,2,state,workers=4,total_updates=50000,seed=1234)
        self.assertEqual(a.shape,(50000,2,4));self.assertTrue(np.array_equal(a,b));self.assertEqual(ma,mb)
        self.assertTrue(np.all(a[:,:,2:]==1));self.assertEqual(set(a[:,:,1].ravel()),{0,1,2,3})
        self.assertFalse(ma['exact_historical_whole_training_stream_claim'])

    def test_validation_is_native_unaugmented_and_raw_dn_kept(self):
        binding,path=fixture(self.root,split='val');dataset=data.NativeDataset(binding,'val',self.root)
        self.addCleanup(dataset.close);original=PanFeeder(str(path),return_meta=True)
        for key,value in zip(('gt','lms','ms','lpan','pan','meta'),original[0]):
            self.assertTrue(torch.equal(dataset[0][key],value))
        self.assertFalse(dataset.augment);self.assertEqual(dataset.raw(0)['gt'].dtype,np.float64)


if __name__=='__main__':unittest.main()
