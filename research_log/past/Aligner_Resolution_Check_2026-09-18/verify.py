"""Read existing probe records and check coordinate/shape contracts on CPU only.

No training, full reconstruction evaluation, GPU use, or operational artifact writes.
Random inputs check tensor shapes, not learned registration accuracy.
"""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path('/home/knuvi/Desktop/song/PAN-Crafter')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pa.aligner import PANGlobalAligner
from pa.model import PAModel
from pa.warp import warp_pan

torch.set_num_threads(2)
torch.manual_seed(20260918)
MANIFEST = {}
N2 = 'PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT'
T0 = 'PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1'
G23 = 'PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S1234_FRESH50_v1'


def read(rel, kind='json'):
    data = (ROOT / rel).read_bytes()
    MANIFEST[rel] = hashlib.sha256(data).hexdigest()
    p = OUT / 'inputs' / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    if kind == 'json':
        return json.loads(data)
    if kind == 'csv':
        return list(csv.DictReader(data.decode().splitlines()))
    return data.decode()


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


fits = []
for name, run, ck, tag in [('N2_legacy', N2, 'last', ''),
                          ('N2_PALSV18', N2, 'last', '_palsv18'),
                          ('T0_PALSV18', T0, 'best_hqnr', '_palsv18')]:
    j = read(f'work_dir/{run}/results/po10_diag_{ck}{tag}.json')
    for scale in ('native64', 'rr256', 'fr512'):
        suffix = f'{tag}_{ck}' if tag else ''
        rel = f'work_dir/{run}/offset_response_{scale}{suffix}.csv'
        all_rows = read(rel, 'csv')
        rows = [r for r in all_rows if r['kind'] != 'stress']
        e = np.array([[float(r['ey']), float(r['ex']), 1] for r in rows])
        q = np.array([[float(r['q_dy']), float(r['q_dx'])] for r in rows])
        coef = np.linalg.lstsq(e, q, rcond=None)[0]
        B = coef[:2].T
        reported = np.asarray(j['response'][scale]['B'])
        diff = float(np.max(np.abs(B-reported)))
        assert diff < 1e-12, (name, scale, diff)
        pr = [r for r in rows if r['kind'] == 'probe' and (float(r['ey']) != 0 or float(r['ex']) != 0)]
        res = np.array([[float(r['q_dy'])+float(r['ey']), float(r['q_dx'])+float(r['ex'])] for r in pr])
        fits.append(dict(name=name, run=run, checkpoint=ck, scale=scale, n_samples=len({r['sample'] for r in rows}),
                         n_rows=len(rows), B_yy=float(B[0, 0]), B_yx=float(B[0, 1]), B_xy=float(B[1, 0]), B_xx=float(B[1, 1]),
                         max_fit_difference=diff, fixed_probe_epe=float(np.linalg.norm(res, axis=1).mean()),
                         fixed_probe_component_mae=float(np.abs(res).mean()),
                         checkpoint_hash_present=bool(j.get('ckpt_sha256')),
                         checkpoint_hash_matches=(j.get('ckpt_sha256') == digest(ROOT/f'work_dir/{run}/{ck}/model.safetensors')) if j.get('ckpt_sha256') else None))
legacy_best = read(f'work_dir/{N2}/results/po10_diag_best_hqnr.json')
best_vs_last = {scale: {'best_hqnr': legacy_best['response'][scale]['B_diag'],
                       'last': [r['B_yy'], r['B_xx']]}
                for scale in ('native64', 'rr256', 'fr512')
                for r in fits if r['name'] == 'N2_legacy' and r['scale'] == scale}
with (OUT/'response_refits.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(fits[0])); w.writeheader(); w.writerows(fits)

sizes = [(64, 64), (128, 128), (256, 256), (512, 512), (256, 320)]
coords = []
with torch.inference_mode():
    for h, w in sizes:
        pan = torch.zeros(1, 1, h, w)
        pan[0, 0, h//2, w//2] = 1.0
        captured = []
        original = F.grid_sample

        def recording_grid(p, grid, **kwargs):
            captured.append(grid.clone())
            return original(p, grid, **kwargs)

        delta = torch.tensor([[.5, -.3]])
        with patch('pa.warp.F.grid_sample', side_effect=recording_grid):
            warp_pan(pan, delta)
        grid = captured[0]
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        recovered_dy = (grid[0, :, :, 1]+1)*h/2-.5-yy
        recovered_dx = (grid[0, :, :, 0]+1)*w/2-.5-xx
        err = max(float((recovered_dy-.5).abs().max()), float((recovered_dx+.3).abs().max()))
        assert err < 1e-4, (h, w, err)
        shifted = warp_pan(pan, torch.tensor([[1., -2.]]))
        arg = int(shifted[0, 0].argmax()); peak = [arg//w, arg % w]
        expected_peak = [h//2-1, w//2+2]
        assert peak == expected_peak
        coords.append(dict(H=h, W=w, input_delta=[.5, -.3], max_recovered_pixel_error=err,
                           integer_delta=[1., -2.], observed_peak=peak, expected_peak=expected_peak, passed=True))

shapes = []
checkpoint_hashes = {}
for name, run, ck in [('N2', N2, 'last'), ('T0', T0, 'best_hqnr'), ('G23', G23, 'best_hqnr')]:
    path = ROOT/f'work_dir/{run}/{ck}/model.safetensors'
    checkpoint_hashes[name] = dict(path=str(path.relative_to(ROOT)), sha256=digest(path))
    state = load_file(str(path), device='cpu')
    astate = {k[len('aligner.'):]: v for k, v in state.items() if k.startswith('aligner.')}
    aligner = PANGlobalAligner(8)
    aligner.load_state_dict(astate, strict=True)
    model = PAModel(torch.nn.Identity(), aligner, aligner_margin=4).eval()
    stages = {}
    handles = [aligner.res.register_forward_hook(lambda _m, _i, out: stages.update(feature_map=list(out.shape))),
               aligner.fc1.register_forward_pre_hook(lambda _m, inp: stages.update(gap=list(inp[0].shape)))]
    with torch.inference_mode():
        for h, w in sizes:
            p = torch.randn(1, 1, h, w)
            ms = torch.randn(1, 8, h//4, w//4)
            m = F.interpolate(ms, scale_factor=4, mode='bicubic')
            d = model.predict_delta(p, m)
            assert d.shape == (1, 2) and torch.isfinite(d).all()
            assert stages['gap'] == [1, 64]
            shapes.append(dict(model=name, H=h, W=w, aligner_view=[h-8, w-8], **stages,
                               output=list(d.shape), finite=True))
    for handle in handles:
        handle.remove()
    after = {k: v for k, v in aligner.state_dict().items()}
    assert all(torch.equal(v, after[k]) for k, v in astate.items())

for rel in ('pa/warp.py', 'pa/aligner.py', 'pa/model.py', 'pa/offset.py', 'tools/po10_diag.py', 'tools/s1_aligner_analysis.py'):
    read(rel, 'text')
result = dict(captured_at=datetime.datetime.now().isoformat(), head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              torch_version=torch.__version__, device='cpu', response_refits=fits, n2_best_vs_last=best_vs_last,
              coordinate_checks=coords, shape_checks=shapes, checkpoint_hashes=checkpoint_hashes,
              limitations=['Random-input forwards verify shape support only, not absolute alignment accuracy',
                           'Existing 64/256/512 probe samples come from different splits; not a same-scene size-only intervention',
                           'Legacy quoted N2 JSON has no checkpoint hash; later PALSV18 JSON includes a matching hash',
                           'No new real-image probe evaluation or training was performed'])
(OUT/'verification.json').write_text(json.dumps(result, indent=2))
(OUT/'input_manifest.json').write_text(json.dumps(MANIFEST, indent=2))
print(json.dumps(dict(response_refits=fits, n2_best_vs_last=best_vs_last, coordinate_checks=coords,
                     n_shape_checks=len(shapes), checkpoint_hashes=checkpoint_hashes), indent=2))
