import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch
from pan_shared.model import SharedPLHUNet, SingleSensorPLHUNet, to_reference_state
from pan_shared.frontend import make_lpan, normalize_dn, plh_inputs, up4, fixed_view
from pan_shared.data import scan_source, ensure_lp_cache, SensorDataset, prepare_catalog
from pan_shared.common import file_sha

PROJECT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def fixture(sensor='WV3', h=16, w=20):
    bands = 8 if sensor == 'WV3' else 4
    generator = torch.Generator().manual_seed(29)
    return tuple(torch.randn(shape, generator=generator) for shape in
                 [(1, 1, h, w), (1, bands, h // 4, w // 4), (1, 1, h // 4, w // 4)])


class ModelContractTests(unittest.TestCase):
    def test_zero_head_and_plh_channels(self):
        model = SharedPLHUNet(width=8)
        for sensor in model.sensors:
            pan, ms, lp = fixture(sensor)
            x, base = plh_inputs(sensor, pan, ms, lp)
            self.assertTrue(torch.equal(x[:, 0:1], pan))
            self.assertTrue(torch.equal(x[:, 1:2], up4(lp)))
            self.assertTrue(torch.equal(x[:, 2:3], pan - up4(lp)))
            self.assertGreater(float(x[:, 1:3].abs().sum()), 0)
            self.assertTrue(torch.equal(model(sensor, pan, ms, lp), base))
            self.assertEqual(torch.count_nonzero(model.stems[sensor].weight[:, 1:3]).item(), 0)
            self.assertTrue(model.stems[sensor].weight.requires_grad)

    def test_canonical_single_shared_and_rng(self):
        before = torch.get_rng_state().clone()
        shared = SharedPLHUNet(width=8, seed=271001)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for sensor in shared.sensors:
            single = SingleSensorPLHUNet(sensor, width=8, seed=271001)
            for name, tensor in single.state_dict().items():
                self.assertTrue(torch.equal(tensor, shared.state_dict()[name]), name)
        self.assertNotEqual(shared.stems['GF2'].weight.data_ptr(), shared.stems['QB'].weight.data_ptr())
        self.assertFalse(torch.equal(shared.stems['GF2'].weight, shared.stems['QB'].weight))
        other = SharedPLHUNet(width=8, seed=271002)
        self.assertFalse(torch.equal(shared.stems['WV3'].weight, other.stems['WV3'].weight))

    def test_inactive_branch_optimizer_moments_preserved(self):
        model = SharedPLHUNet(width=8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        # First activate all branches so inactive optimizer moments exist.
        for sensor in model.sensors:
            optimizer.zero_grad(set_to_none=True)
            model(sensor, *fixture(sensor)).abs().mean().backward()
            optimizer.step()
        inactive = {n: p for n, p in model.named_parameters() if n.startswith(('stems.QB', 'heads.QB', 'stems.GF2', 'heads.GF2'))}
        before = {n: p.clone() for n, p in inactive.items()}
        moments = {n: {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in optimizer.state[p].items()} for n, p in inactive.items()}
        optimizer.zero_grad(set_to_none=True)
        model('WV3', *fixture()).abs().mean().backward()
        self.assertTrue(all(p.grad is None for p in inactive.values()))
        optimizer.step()
        for name, p in inactive.items():
            self.assertTrue(torch.equal(p, before[name]))
            for key, value in moments[name].items():
                self.assertTrue(torch.equal(optimizer.state[p][key], value))
        pointers = [p.data_ptr() for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(set(pointers)), len(pointers))

    def test_no_warp_and_no_conditional_modules(self):
        model = SharedPLHUNet(width=8)
        with patch('torch.nn.functional.grid_sample', side_effect=AssertionError('warp')), patch('torch.nn.functional.affine_grid', side_effect=AssertionError('warp')):
            model('WV3', *fixture())
        for name, module in model.named_modules():
            text = name.lower() + type(module).__name__.lower()
            self.assertFalse(any(x in text for x in ('attention', 'swin', 'aligner', 'teacher', 'modulation')))

    def test_active_gradient_after_zero_head_step(self):
        model = SharedPLHUNet(width=8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            model('WV3', *fixture()).square().mean().backward()
            optimizer.step()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.trunk.parameters() if p.grad is not None), 0)
        self.assertGreater(float(model.stems['WV3'].weight.grad[:, 1:3].abs().sum()), 0)

    def test_geometry_rejected(self):
        model = SharedPLHUNet(width=8)
        pan, ms, lp = fixture()
        with self.assertRaises(ValueError):
            model('GF2', pan, ms, lp)
        with self.assertRaises(ValueError):
            model('WV3', pan[..., :-1], ms, lp)

    def test_fixed_source_parity_in_isolated_process(self):
        """AST copies only definitions from pinned files; no legacy runtime imports."""
        code = r'''
import ast,sys,types,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pan_shared.model import SharedPLHUNet,SingleSensorPLHUNet,to_reference_state
from pan_shared.frontend import up4
torch.set_num_threads(1)
root=Path(sys.argv[1])/'vendor_reference/model_source'
namespace={'torch':torch,'nn':nn,'F':F}
def extract(path,names):
 tree=ast.parse(path.read_text())
 chosen=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 exec(compile(ast.Module(body=chosen,type_ignores=[]),str(path),'exec'),namespace)
extract(root/'model/pancrafter.py',{'DownConv','UpConv','GroupNorm32','zero_module'})
namespace['CMAAA']=type('UnusedCMAAA',(nn.Module,),{})
stub=types.ModuleType('model.swin')
stub.swin_blocks=lambda depth,*args: nn.ModuleList([]) if depth==0 else (_ for _ in ()).throw(ValueError('Nonzero Swin'))
sys.modules['model.swin']=stub
extract(root/'model/pancrafter_paper.py',{'ChannelLayerNorm','_norm','ResBlock','PANCrafterPaper'})
rows=[]
for width,sensor,shapes in [(8,'WV3',[(64,64),(256,256),(512,512),(64,80)]),(8,'GF2',[(64,64)]),(8,'QB',[(64,64)]),(104,'WV3',[(64,64),(256,256),(512,512),(64,80)]),(128,'WV3',[(64,64)])]:
 shared=SharedPLHUNet(width=width,seed=271001).eval()
 single=SingleSensorPLHUNet(sensor,width=width,seed=271001).eval()
 # Nontrivial synthetic state: zero-head identity alone cannot prove topology parity.
 g=torch.Generator().manual_seed(125)
 with torch.no_grad():
  for name,p in shared.named_parameters():
   if 'heads.' in name or 'out_layers.3.' in name:
    p.copy_(torch.randn(p.shape,generator=g)*0.02)
 single.load_state_dict({n:shared.state_dict()[n] for n in single.state_dict()})
 bands=8 if sensor=='WV3' else 4
 reference=namespace['PANCrafterPaper'](out_channels=bands,hidden_size=width,depth=(1,2,2),n_attn=0,attn_locations=[],norm='ln',in_mode='released',mode_modulation=False).eval()
 reference.load_state_dict(to_reference_state(shared,sensor),strict=True)
 with torch.inference_mode():
  for h,w in shapes:
   pan=torch.randn((1,1,h,w),generator=g);ms=torch.randn((1,bands,h//4,w//4),generator=g);lp=torch.randn((1,1,h//4,w//4),generator=g)
   expected=up4(ms)+reference(pan,lp,ms,torch.ones(1))
   actual=shared(sensor,pan,ms,lp);independent=single(sensor,pan,ms,lp)
   delta=(actual-expected).abs()
   assert torch.equal(actual,independent)
   assert delta.max().item()<=1e-5 and delta.mean().item()<=1e-6
   rows.append({'sensor':sensor,'width':width,'shape':[h,w],'bitwise':torch.equal(actual,expected),'max_abs':delta.max().item(),'mean_abs':delta.mean().item()})
print(json.dumps(rows))
'''
        result = subprocess.run([sys.executable, '-c', textwrap.dedent(code), str(PROJECT)], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + result.stdout)
        rows = json.loads(result.stdout)
        self.assertEqual(len(rows), 11)
        self.assertTrue(all(row['bitwise'] for row in rows))


class FrontendDataTests(unittest.TestCase):
    def test_lp_exact_reference_and_phase(self):
        import cv2
        pan = np.arange(2 * 64 * 80, dtype=np.float64).reshape(2, 1, 64, 80)
        kernel = cv2.getGaussianKernel(41, 1.98)
        expected = np.stack([cv2.filter2D(x[0], -1, kernel @ kernel.T, borderType=cv2.BORDER_REPLICATE)[2::4, 2::4][None] for x in pan]).astype(np.float32)
        self.assertTrue(np.array_equal(make_lpan(pan), expected))
        self.assertEqual(make_lpan(pan).dtype, np.float32)

    def test_normalization_matches_frozen_torch_order_and_preserves_ringing(self):
        array = np.array([-5, 0, 1, 1023, 2047, 2050], dtype=np.float32)
        for sensor, maximum in [('WV3', 2047), ('QB', 2047), ('GF2', 1023)]:
            expected = torch.from_numpy(array.copy()).mul_(2 / maximum).sub_(1).numpy()
            self.assertTrue(np.array_equal(normalize_dn(array, sensor), expected))
            self.assertLess(normalize_dn(array, sensor)[0], -1)

    def test_fixed_four_view_contract(self):
        array = np.arange(64).reshape(1, 8, 8)
        for rotation in range(4):
            self.assertTrue(np.array_equal(fixed_view(array, rotation), np.rot90(array[:, ::-1, ::-1], rotation, axes=(1, 2))))
        with self.assertRaises(ValueError):
            fixed_view(array, 4)

    def make_source(self, root, count=2):
        source = root / 'native.h5'
        with h5py.File(source, 'w') as handle:
            for key, shape in {'pan': (count, 1, 64, 64), 'ms': (count, 4, 16, 16), 'lms': (count, 4, 64, 64), 'gt': (count, 4, 64, 64)}.items():
                handle.create_dataset(key, data=np.arange(np.prod(shape), dtype=np.float32).reshape(shape) % 1000)
        return source

    def test_fullscan_lp_and_loader_are_readonly_and_reproducible(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = self.make_source(root)
            initial = file_sha(source)
            scan = scan_source(source, 'GF2', 'train', expected_count=2)
            lp = ensure_lp_cache(source, root / 'lp.h5', initial)
            self.assertEqual(initial, file_sha(source))
            self.assertEqual(lp, ensure_lp_cache(source, root / 'lp.h5', initial))
            manifest = {'sensors': {'GF2': {'bands': 4, 'max_dn': 1023, 'splits': {'train': dict(dataroot=str(source), sha256=initial, **scan, **lp)}}}}
            dataset = SensorDataset(manifest, 'GF2', 'train')
            actual = dataset.fetch([1, 0, 1], rotations=[2, 1, 2])
            self.assertTrue(torch.equal(actual['pan'][0], actual['pan'][2]))
            self.assertTrue(torch.equal(actual['pan'][1], torch.from_numpy(normalize_dn(fixed_view(dataset.raw([0])['pan'][0], 1), 'GF2'))))
            self.assertEqual(initial, file_sha(source))

    def test_reject_bad_scan_nonfinite_or_count(self):
        with tempfile.TemporaryDirectory() as name:
            source = self.make_source(Path(name))
            with self.assertRaisesRegex(ValueError, 'GEOMETRY'):
                scan_source(source, 'GF2', 'train')
            with h5py.File(source, 'r+') as handle:
                handle['ms'][1, 2, 4, 9] = np.nan
            with self.assertRaisesRegex(ValueError, 'NONFINITE'):
                scan_source(source, 'GF2', 'train', expected_count=2)

    def test_bad_lp_cache_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = self.make_source(root)
            cache = root / 'lp.h5'
            ensure_lp_cache(source, cache, file_sha(source))
            cache.chmod(0o644)
            with h5py.File(cache, 'r+') as handle:
                handle['lpan'][1, 0, 3, 2] += 1
            with self.assertRaisesRegex(ValueError, 'correspondence'):
                ensure_lp_cache(source, cache, file_sha(source))

    def test_production_scan_shortcut_forbidden(self):
        with self.assertRaisesRegex(ValueError, 'mandatory'):
            prepare_catalog('unread', 'unread', 'unread', full_scan=False)

    def test_cache_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = self.make_source(root)
            owned, outside = root / 'owned', root / 'outside'
            owned.mkdir()
            outside.mkdir()
            (owned / 'sensor').symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                ensure_lp_cache(source, owned / 'sensor' / 'cache.h5', file_sha(source), allowed_root=owned)
            self.assertEqual(list(outside.iterdir()), [])

    def test_vendored_source_hashes(self):
        folder = PROJECT / 'vendor_reference/model_source'
        manifest = json.loads((folder / 'manifest.json').read_text())
        for relative, identity in manifest['files'].items():
            self.assertEqual(file_sha(folder / relative), identity['vendored_sha256'])

    def test_operator_cancel_interrupts_scan_and_lp_without_source_write(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = self.make_source(root)
            initial = file_sha(source)
            def stop():
                raise RuntimeError('STOPPED_BY_OPERATOR')
            with self.assertRaisesRegex(RuntimeError, 'STOPPED_BY_OPERATOR'):
                scan_source(source, 'GF2', 'train', expected_count=2, check_cancel=stop)
            calls = []
            def stop_on_chunk():
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError('STOPPED_BY_OPERATOR')
            with self.assertRaisesRegex(RuntimeError, 'STOPPED_BY_OPERATOR'):
                ensure_lp_cache(source, root / 'cache.h5', initial, check_cancel=stop_on_chunk)
            self.assertEqual(file_sha(source), initial)
            self.assertFalse((root / 'cache.h5').exists())
            self.assertEqual(list(root.glob('.lpan-*')), [])
