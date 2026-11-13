"""Exact valid-domain statistics, mode-separated validation and donor loading."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import h5py
import numpy as np
import torch

from panda_method.edge_data import EdgeBatchLineage, EdgeObservationBatch
from panda_unified import core
from panda_unified_campaign import assets, calibration, model, plan, selection, training
from ta2.evaluation import file_sha


def binding(sensor='QB', mode='RGB', count=3):
    ids = [f'{sensor}:train:{i}' for i in range(count)]
    value = dict(sensor=sensor, source_split='RR_train_only', sample_ids=ids,
        sample_ids_sha256=calibration.sample_ids_sha256(ids), sample_count=count,
        source_manifest_sha256='1'*64, aligner_state_sha256='2'*64,
        teacher_state_sha256='3'*64, donor_checkpoint_id='4'*64,
        reference_mode=mode, mask_policy=core.MASK_POLICY,
        reconstruction_spec=dict(bands=8 if sensor=='WV3' else 4,
            width=112 if sensor=='WV3' else 128, depth=[1,2,3],
            sampling_variant='UBIL', input_layout='P0', base_reference='INTERP23_MS'),
        scharr=dict(kernel='absolute_xy_/32', padding='reflect1',
            reference_indices=plan.reference_bands(sensor, mode),
            comparison_domain=core.MASK_POLICY), warp=core.WARP_CONTRACT)
    return ids, value


class ConstantAligner(torch.nn.Module):
    """Deliberately exposes fc2.out_features, without a synthetic outputs field."""
    def __init__(self):
        super().__init__()
        self.fc2 = torch.nn.Linear(1,2)

    def forward(self, pan, lms):
        return torch.zeros((len(pan),2), device=pan.device)


class ValidationSources:
    sensor, bands, peak, catalog_sha256 = 'QB', 4, 2047., '8'*64

    def __init__(self, path):
        self.path = path
        self.gt = np.random.default_rng(8).uniform(200,1700,(2,4,64,64)).astype('float32')
        with h5py.File(path,'w') as handle:
            handle['pan'] = self.gt.mean(1,keepdims=True)
            handle['gt'] = self.gt
        self.entry = {'splits': {'val': {'sha256':file_sha(path)}}}

    def _verify(self, split):
        if split!='val':
            raise AssertionError('Validation must not read train/test')
        return self.path

    def manifest_sha(self, split):
        self._verify(split)
        return '9'*64

    def batch(self, split, indexes, device):
        self._verify(split)
        gt = torch.from_numpy(self.gt[indexes]/self.peak*2-1).to(device)
        ms = torch.nn.functional.avg_pool2d(gt,4)
        lineage = EdgeBatchLineage('validation','RR','9'*64,
            tuple(f'QB:val:{i}' for i in indexes), 'QB','PAN_RR',
            tuple((False,False,0) for _ in indexes))
        return EdgeObservationBatch(gt.mean(1,keepdim=True),ms,gt,lineage,gt)


class CalibrationSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_exact_variable_support_population_and_immutable_archive(self):
        for sensor, mode in [('WV3','RGBN'),('QB','RGB'),('QB','RGBN')]:
            with self.subTest(sensor=sensor, mode=mode), tempfile.TemporaryDirectory() as tmp:
                ids, bound = binding(sensor,mode)
                mask = core.support(torch.zeros(3,1,64,64),
                    torch.tensor([[0.,0.],[.5,.5],[6.,-8.]]))
                self.assertEqual(mask.flatten(1).sum(1).tolist(),[3025,3025,2652])
                e = torch.arange(3*4096,dtype=torch.float32).reshape(3,1,64,64)/1000
                g, q = e*.03, e*.07
                builder = calibration.CalibrationBuilder(ids,bound,
                    memory_budget_bytes=32*1024**2, mask_policy=core.MASK_POLICY)
                builder.add(ids,e,g,q,sensor=sensor,loss_mask=mask)
                stats = builder.finalize()
                self.assertEqual(stats.metadata['n_e'],8702)
                self.assertEqual(stats.metadata['n_q'],8702)
                self.assertNotEqual(stats.metadata['n_q'],3*48*48)
                self.assertAlmostEqual(stats.tau_e,float(torch.quantile(e[mask].double(),.5)),places=10)
                self.assertEqual(stats.lookup_exact(g[mask],q[mask]).shape,q[mask].shape)
                path = Path(tmp)/'calibration.npz'
                receipt = stats.save(path)
                loaded = calibration.CalibrationStats.load(path,expected_binding=bound,
                    expected_file_sha256=receipt['file_sha256'],expected_mask_policy=core.MASK_POLICY)
                self.assertEqual(loaded.artifact_sha256,stats.artifact_sha256)
                with self.assertRaises(FileExistsError):
                    stats.save(path)
                changed = deepcopy(bound)
                changed['reference_mode'] = 'RGB' if mode=='RGBN' else 'RGBN'
                changed['scharr']['reference_indices'] = plan.reference_bands(sensor,changed['reference_mode'])
                with self.assertRaisesRegex(ValueError,'binding mismatch'):
                    calibration.CalibrationStats.load(path,expected_binding=changed)
                loaded._cuts.add_(1)
                with self.assertRaisesRegex(ValueError,'immutable calibration tensor changed'):
                    loaded.assert_integrity()

    def test_legacy_architecture_or_domain_cannot_bind(self):
        _, good = binding()
        for field, value in [('width',112),('sampling_variant','BASE'),('base_reference','BICUBIC_MS')]:
            bad = deepcopy(good)
            bad['reconstruction_spec'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,'shared UBIL backbone'):
                calibration.validate_binding(bad)
        bad = deepcopy(good)
        bad['scharr']['roi_margin'] = 8
        with self.assertRaisesRegex(ValueError,'operator binding mismatch'):
            calibration.validate_binding(bad)

    def test_validation_inventory_and_rre_follow_reference_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = ValidationSources(Path(tmp)/'validation.h5')
            inventories = {m:selection.validation_inventory(source,m) for m in ('RGB','RGBN')}
            self.assertNotEqual(inventories['RGB']['inventory_sha256'],inventories['RGBN']['inventory_sha256'])
            teacher = core.build_reconstructor('QB',128,1,role='T')
            errors = []
            for mode, inventory in inventories.items():
                other = 'RGBN' if mode=='RGB' else 'RGB'
                with self.assertRaisesRegex(ValueError,'protocol mismatch'):
                    selection.validate_inventory(inventory,source,other)
                value = SimpleNamespace(cfg={'reference_mode':mode},aligner=ConstantAligner(),teacher=teacher)
                result = selection.collect_validation(value,source,inventory=inventory,
                    include_ergas=True,epsilon=False,batch_size=2)
                self.assertEqual(result['sample_count'],2)
                self.assertEqual(result['protocol']['reference_band_indices'],plan.reference_bands('QB',mode))
                self.assertEqual(result['protocol']['val_ergas_domain'],'COMMON_LANCZOS4_SUPPORT_ERODED_SCHARR1')
                self.assertTrue(np.isfinite(result['Val_ERGAS']))
                errors.append(result['RRE50'])
            # PAN is constructed from all four bands: the four-band reference
            # has lower registration residual than RGB on these fixed inputs.
            self.assertLess(errors[1],errors[0])

    def test_selected_donor_loads_only_new_exact_schema_and_mode(self):
        cfg = plan.resolve('s2','S1_RGB_FULL',1)
        _, inventory = binding(count=2)
        registered = SimpleNamespace(inventory=SimpleNamespace(manifest=lambda:deepcopy(inventory)),
            verify=lambda:SimpleNamespace(manifest=lambda:deepcopy(inventory)))
        stage = model.Stage1(cfg,inventory)
        sha = training.model_hash(stage)
        pins = {'source_release':'test-source-release'}
        payload = dict(schema='PANDA_UNIFIED_EVALUATED_MODEL_V1',config=cfg,pins=pins,
            updates=1000,model=stage.state_dict(),model_sha256=sha)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'selected.pt'
            torch.save(payload,path)
            receipt = {'receipt_sha256':'a'*64,'selected':dict(checkpoint_path='selected.pt',
                file_sha256=file_sha(path),model_sha256=sha,step=1000)}
            donor = assets._load_selected(cfg,receipt,registered,Path(tmp),pins,device='cpu')
            donor.assert_frozen()
            self.assertEqual(donor.binding['reference_mode'],'RGB')
            self.assertEqual(donor.teacher.spec.width,128)
            self.assertEqual(donor.binding['reconstruction_spec']['sampling_variant'],'UBIL')
            wrong = plan.resolve('s2','S1_RGBN_FULL',1)
            with self.assertRaisesRegex(ValueError,'same-source unified donor'):
                assets._load_selected(wrong,receipt,registered,Path(tmp),pins,device='cpu')
            payload['schema'] = 'WV3_MAIN_RGB_EVALUATED_MODEL_V1'
            torch.save(payload,path)
            receipt['selected']['file_sha256'] = file_sha(path)
            with self.assertRaisesRegex(ValueError,'same-source unified donor'):
                assets._load_selected(cfg,receipt,registered,Path(tmp),pins,device='cpu')


if __name__=='__main__':
    unittest.main()
