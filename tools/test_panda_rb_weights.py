import json
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from panda_rb import weights as w


class WeightTests(unittest.TestCase):
    def fixture(self,n=37):
        rng=np.random.default_rng(2)
        return rng.random((n,4),dtype=np.float32),np.round(rng.random((n,4)),1).astype(np.float32)

    def test_qfull_matches_old_torch_raw_conversion(self):
        q,e=self.fixture();arrays,_=w.build_weights(q,e,.4532603621482849,9281101)
        expected=(.4532603621482849/(.4532603621482849+torch.from_numpy(q))).numpy()
        np.testing.assert_array_equal(arrays['QFULL'],expected)

    def test_mean_full_population_not_calibration_or_point_five(self):
        q,e=self.fixture();arrays,report=w.build_weights(q,e,.4532603621482849,9281101)
        mean=arrays['QFULL'].mean(dtype=np.float64)
        self.assertEqual(report['training_population_mean_s'],mean)
        np.testing.assert_array_equal(arrays['QMEAN'],np.full_like(q,mean))
        self.assertNotEqual(mean,.5)

    def test_shuffle_stratum_multiset_derangement(self):
        q,e=self.fixture();a,_=w.build_weights(q,e,.45,9281101)
        for v in range(4):
            for g in range(5):
                ids=np.flatnonzero(a['e_stratum'][:,v]==g)
                np.testing.assert_array_equal(np.sort(a['QFULL'][ids,v]),np.sort(a['QSHUF'][ids,v]))
                self.assertTrue(np.all(a['shuffle_source'][ids,v]!=ids))
                self.assertTrue(np.all(a['e_stratum'][a['shuffle_source'][ids,v],v]==g))

    def test_surrogate_same_multiset_inverse_error_rank(self):
        q,e=self.fixture();a,_=w.build_weights(q,e,.45,9281101)
        for v in range(4):
            order=np.lexsort((np.arange(len(q)),e[:,v]))
            self.assertTrue(np.all(np.diff(a['QESUR'][order,v])<=0))
            np.testing.assert_array_equal(np.sort(a['QESUR'][:,v]),np.sort(a['QFULL'][:,v]))

    def test_rng_isolated(self):
        q,e=self.fixture();np.random.seed(2);torch.manual_seed(3);random.seed(4)
        before=(np.random.get_state(),torch.get_rng_state().clone(),random.getstate())
        w.build_weights(q,e,.45,9281101)
        after=(np.random.get_state(),torch.get_rng_state(),random.getstate())
        np.testing.assert_array_equal(before[0][1],after[0][1]);self.assertEqual(before[0][2:],after[0][2:])
        self.assertTrue(torch.equal(before[1],after[1]));self.assertEqual(before[2],after[2])

    def test_deterministic_six_distinct_permutations(self):
        q,e=self.fixture();hashes=[]
        for seed in [9281101,9281102,9281301,9281302,9281501,9281502]:
            a,r=w.build_weights(q,e,.45,seed);b,s=w.build_weights(q,e,.45,seed)
            self.assertEqual(r,s);np.testing.assert_array_equal(a['QSHUF'],b['QSHUF'])
            hashes.append(r['arrays_sha256']['shuffle_source'])
        self.assertEqual(len(set(hashes)),6)

    def test_singleton_recorded_and_fixed(self):
        q,e=self.fixture(3);a,r=w.build_weights(q,e,.45,7)
        self.assertEqual(len(r['singleton_groups']),12)
        np.testing.assert_array_equal(a['QFULL'],a['QSHUF'])

    def test_ties_follow_source_index(self):
        q=np.ones((20,4),np.float32);e=np.zeros_like(q)
        a,_=w.build_weights(q,e,.45,7)
        np.testing.assert_array_equal(a['e_stratum'][:,0],np.repeat(np.arange(5),4))
        np.testing.assert_array_equal(a['surrogate_source'][:,0],np.arange(20))

    def test_invalid_population_rejected(self):
        q,e=self.fixture()
        for bad in (-1,float('nan'),float('inf')):
            broken=q.copy();broken[0,0]=bad
            with self.assertRaises(ValueError):w.build_weights(broken,e,.45,2)
        with self.assertRaises(ValueError):w.build_weights(q,e[:2],.45,2)
        with self.assertRaises(ValueError):w.build_weights(q,e,0,2)

    def test_cache_rederive_and_tamper_detection(self):
        q,e=self.fixture();a,r=w.build_weights(q,e,.45,7)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);w._save_npz(p/'seed_7.npz',a)
            r.update(cache_sha256=w.sha256_file(p/'seed_7.npz'),binding_common_sha256='b')
            (p/'seed_7.json').write_text(json.dumps(r))
            loaded,_=w.load_weights(p,7,{'common_sha256':'b'})
            np.testing.assert_array_equal(loaded['QFULL'],a['QFULL'])
            with self.assertRaises(ValueError):w.load_weights(p,7,{'common_sha256':'other'})
            r['q_ref']=.7;(p/'seed_7.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError):w.load_weights(p,7)

    def test_teacher_error_unclipped_exact_views_and_no_state_rng_mutation(self):
        class Teacher(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer('scale',torch.tensor(2.))
            def forward(self,pan,ms,lp):return {'y':pan.repeat(1,8,1,1)*self.scale}
        class Data:
            def __init__(self):self.seen=[]
            def __len__(self):return 3
            def __getitem__(self,key):
                i,v=key;self.seen.append(key)
                return (torch.zeros(8,4,4),torch.zeros(8,4,4),torch.zeros(8,1,1),
                        torch.zeros(1,1,1),torch.full((1,4,4),float(i+v+1)),torch.tensor([i,v,1,1]))
        t=Teacher();t.train();ds=Data();rng=torch.get_rng_state().clone()
        actual=w.compute_e_bar(t,ds,'cpu',2)
        np.testing.assert_array_equal(actual,2*(np.arange(3)[:,None]+np.arange(4)[None,:]+1))
        self.assertEqual(set(ds.seen),{(i,v) for i in range(3) for v in range(4)})
        self.assertEqual(len(ds.seen),12);self.assertTrue(t.training)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(t.scale.item(),2.)

if __name__ == '__main__':unittest.main()
