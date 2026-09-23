"""ABLR2 sensor-bound official metrics; no masked/shifted evaluation references.

The established MATLAB-port metric primitives are reused without modification.
WV3 uses eight-band Q8; QB/GF2 use genuine four-band Q4 and their own DN scale. JQM remains an
explicitly labelled SRF substitute, not a claim of SIPSA equivalence.
"""
from pathlib import Path
import os
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from fh12.common import check_deadline, sha256
from tools.eval_dlpan import scc_dlpan, psnr_global, ssim_skimage
from tools.metrics.eval_rr import sam, ergas
from tools.metrics.q2n import q2n

RR_COMMON = ('ergas', 'scc', 'psnr', 'sam', 'ssim', 'rmse', 'cc')
FR_KEYS = ('hqnr', 'd_lambda', 'd_s')
JQM_VARIANT = 'SRF-substitute (NNLS-normalized), MTF41, phase2, global CMSC, v1=0.5; not SIPSA-equivalent'


def _spec(value):
    if isinstance(value, str):
        from ablr2.plan import sensor_spec
        value = sensor_spec(value)
    if (value.sensor, value.num_bands, value.max_dn) not in (('WV3', 8, 2047), ('QB', 4, 2047), ('GF2', 4, 1023)):
        raise ValueError('ABLR2 permits only WV3/C8/DN2047, QB/C4/DN2047 or GF2/C4/DN1023')
    return value


def canonical_band_indices(spec):
    spec = _spec(spec)
    if spec.sensor in ('QB', 'GF2'):
        from qg40.data import canonical_band_indices as c4_order
        return c4_order(spec.band_order)
    aliases = dict(c='coastal', cb='coastal', coastal='coastal', coastalblue='coastal',
                   b='blue', blue='blue', g='green', green='green', y='yellow', yellow='yellow',
                   r='red', red='red', re='rededge', rededge='rededge',
                   n1='nir1', nir1='nir1', nearinfrared1='nir1', n2='nir2', nir2='nir2', nearinfrared2='nir2')
    names = [aliases.get(str(x).lower().replace('_', '').replace('-', '').replace(' ', ''))
             for x in (spec.band_order or ())]
    expected = ('coastal', 'blue', 'green', 'yellow', 'red', 'rededge', 'nir1', 'nir2')
    if len(names) != 8 or set(names) != set(expected):
        raise ValueError('WV3 source band order requires explicit coastal/B/G/Y/R/RE/NIR1/NIR2 evidence')
    return tuple(names.index(name) for name in expected)


def rr_keys(spec):
    return RR_COMMON + ('q' + str(_spec(spec).num_bands),)


def _summary(rows, keys):
    result = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    if not rows or not all(np.isfinite(v) for v in result.values()):
        raise FloatingPointError('Missing/nonfinite official metrics')
    return dict(result, n_scenes=len(rows), per_scene=rows,
                standard_deviation={k: float(np.std([r[k] for r in rows], ddof=1)) for k in keys})


def rr_metrics(sr, gt, spec, include_q=True):
    spec = _spec(spec)
    sr, gt = np.asarray(sr, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if sr.shape != (20, spec.num_bands, 256, 256) or gt.shape != sr.shape:
        raise ValueError('Official RR requires corresponding 20 x sensor bands x 256 x 256')
    if not np.isfinite(sr).all() or not np.isfinite(gt).all():
        raise FloatingPointError('Nonfinite RR inputs, including excluded borders')
    rows, qkey = [], 'q' + str(spec.num_bands)
    for i, (pred, truth) in enumerate(zip(sr.transpose(0, 2, 3, 1)[:, 20:-21, 20:-21],
                                          gt.transpose(0, 2, 3, 1)[:, 20:-21, 20:-21])):
        if pred.std() == 0 or truth.std() == 0:
            raise ValueError(f'Undefined flattened Pearson CC at scene {i}')
        row = dict(ergas=ergas(pred, truth), sam=sam(pred, truth), scc=scc_dlpan(pred, truth),
                   psnr=psnr_global(pred, truth, spec.max_dn), ssim=ssim_skimage(pred, truth, spec.max_dn),
                   rmse=float(np.sqrt(np.mean((pred - truth) ** 2))),
                   cc=float(np.corrcoef(pred.ravel(), truth.ravel())[0, 1]))
        if include_q:
            row[qkey] = float(q2n(truth, pred, 32, 32)[0])
        row['band_relative_mse'] = (np.mean((pred - truth) ** 2, axis=(0, 1)) / truth.mean((0, 1)) ** 2).tolist()
        rows.append(row)
    result = _summary(rows, rr_keys(spec) if include_q else RR_COMMON)
    result.update(sensor=spec.sensor, num_bands=spec.num_bands, max_dn=spec.max_dn, crop='20:-21', q_block=32,
                  official_complete=bool(include_q), protocol=f'DLPan_RR_dim21_Q{spec.num_bands}_Q32_SCC2Dzero_PSNRglobal_SSIMgauss11; DN RMSE/flattened Pearson CC')
    return result


def signed_ds_details(high, low, official_per_scene):
    high, low = np.asarray(high, dtype=np.float64), np.asarray(low, dtype=np.float64)
    reported = np.asarray(official_per_scene, dtype=np.float64)
    if high.shape not in ((20, 4), (20, 8)) or low.shape != high.shape or reported.shape != (20,):
        raise ValueError('Signed Ds requires actual scene20 x sensor-band operands')
    if not all(np.isfinite(x).all() for x in (high, low, reported)):
        raise ValueError('Nonfinite signed Ds operands')
    delta = high - low
    rebuilt = np.abs(delta).mean(1)
    if not np.allclose(rebuilt, reported, rtol=0, atol=1e-12):
        raise ValueError('Signed Ds operands do not reconstruct the official metric')
    return dict(schema='ABLR2_SIGNED_DS_v1', operand_shape=list(high.shape), Q_high=high.tolist(), Q_low=low.tolist(),
                delta=delta.tolist(), abs_delta=np.abs(delta).tolist(), signed_mean=float(delta.mean()),
                abs_mean=float(np.abs(delta).mean()), positive_fraction=float((delta > 0).mean()),
                reconstructed_d_s=float(rebuilt.mean()), reconstruction_max_abs_error=float(np.max(np.abs(rebuilt - reported))),
                reference='native_PAN_and_original_LMS', support='full512', masking=False)


def validate_signed_ds(fr):
    detail = fr.get('signed_ds', {})
    if len(fr.get('per_scene', [])) != 20:
        raise ValueError('Missing official per-scene native metrics')
    expected = signed_ds_details(detail.get('Q_high'), detail.get('Q_low'), [r['d_s'] for r in fr['per_scene']])
    if detail != expected:
        raise ValueError('Signed Ds payload differs from actual operands')
    if any(not np.isclose(fr[k], np.mean([r[k] for r in fr['per_scene']]), rtol=0, atol=1e-12) for k in FR_KEYS):
        raise ValueError('FR scene-mean aggregation changed')
    for row in fr['per_scene']:
        if not np.isclose(row['hqnr'], (1 - row['d_lambda']) * (1 - row['d_s']), rtol=0, atol=1e-12):
            raise ValueError('HQNR is not the native per-scene product')
    return detail


class FRMetrics:
    def __init__(self, dataset, spec=None, wald=None):
        from tools.metrics.eval_fr import load_dlpan, imresize_matlab, _blockproc_uqi
        self.spec = _spec(spec or dataset.spec)
        self.order = canonical_band_indices(self.spec)
        self.inverse_order = np.argsort(self.order)
        self.wald = wald or load_dlpan(os.environ.get('PANCRAFTER_DLPAN', str(Path(__file__).resolve().parents[2] / 'DLPan-Toolbox')))
        self.wald_sha256 = sha256(self.wald.__file__) if getattr(self.wald, '__file__', None) else None
        with h5py.File(dataset.raw_h5_path, 'r') as source:
            self.lms = np.asarray(source['lms'], dtype=np.float64).transpose(0, 2, 3, 1)
            self.pan = np.asarray(source['pan'], dtype=np.float64)[:, 0]
        if self.lms.shape != (20, 512, 512, self.spec.num_bands) or self.pan.shape != (20, 512, 512):
            raise ValueError('FR requires exactly 20 native full512 scenes and sensor bands')
        if not np.isfinite(self.lms).all() or not np.isfinite(self.pan).all():
            raise ValueError('Nonfinite native FR references')
        self.reference = []
        for ms, pan in zip(self.lms, self.pan):
            low = self.wald.interp23tap(imresize_matlab(pan, 1 / 4)[..., None], 4)[..., 0]
            self.reference.append([_blockproc_uqi(ms[..., b], low, 32) for b in range(self.spec.num_bands)])

    def __call__(self, sr, deadline=None):
        from tools.metrics.eval_fr import mtf_filter, _blockproc_uqi
        sr = np.asarray(sr, dtype=np.float64)
        if sr.shape != (20, self.spec.num_bands, 512, 512) or not np.isfinite(sr).all():
            raise ValueError('Invalid full-frame sensor prediction')
        rows, operands = [], []
        for i, fused in enumerate(sr.transpose(0, 2, 3, 1)):
            check_deadline(deadline)
            filtered = mtf_filter(fused[..., list(self.order)], self.spec.sensor.lower(), 4, self.wald)[..., self.inverse_order]
            dl = float(1 - q2n(self.lms[i], filtered, 32, 32)[0])
            high = [_blockproc_uqi(fused[..., b], self.pan[i], 32) for b in range(self.spec.num_bands)]
            operands.append(high)
            ds = float(np.abs(np.asarray(high) - self.reference[i]).mean())
            rows.append(dict(d_lambda=dl, d_s=ds, hqnr=float((1 - dl) * (1 - ds))))
        result = _summary(rows, FR_KEYS)
        result.update(sensor=self.spec.sensor, num_bands=self.spec.num_bands, max_dn=self.spec.max_dn,
                      band_order=list(self.spec.band_order), mtf_sensor=self.spec.sensor,
                      interp23tap_source_sha256=self.wald_sha256,
                      reference='native_PAN', support='full512', masking=False,
                      aggregation='mean_per_scene_HQNR', hqnr_variant='raw-original', official_complete=True,
                      signed_ds=signed_ds_details(operands, self.reference, [r['d_s'] for r in rows]))
        if self.spec.sensor == 'GF2':
            result.update(mtf_gnyq_canonical=[.3]*4,
                mtf_note='GF2 explicit existing DLPan otherwise GNyq=0.3 preset; not QB band-specific gains')
        return result


def fr_jqm(sr, ms, pan, spec, deadline=None):
    from tools.metrics.jqm import jqm, GNYQ_PAN
    spec = _spec(spec)
    order = canonical_band_indices(spec)
    sr, ms, pan = np.asarray(sr), np.asarray(ms), np.asarray(pan)
    if (sr.shape != (20, spec.num_bands, 512, 512) or ms.shape != (20, spec.num_bands, 128, 128)
            or pan.shape != (20, 1, 512, 512)):
        raise ValueError('JQM requires complete native FR20 sensor arrays')
    if not all(np.isfinite(x).all() for x in (sr, ms, pan)):
        raise FloatingPointError('Nonfinite JQM input')
    rows = []
    for i in range(20):
        check_deadline(deadline)
        result = jqm(sr[i, list(order)].transpose(1, 2, 0), ms[i, list(order)].transpose(1, 2, 0), pan[i, 0],
                     spec.sensor, ratio=4, R=spec.max_dn, lpf='mtf', window=None, v1=.5)
        rows.append(dict(jqm=float(result['JQM']), qlr=float(result['QLR']), qhr=float(result['QHR']),
                         w=np.asarray(result['w'])[np.argsort(order)].tolist(), w_source=result['w_source']))
    result = dict(_summary(rows, ('jqm', 'qlr', 'qhr')), variant=JQM_VARIANT, sensor=spec.sensor,
                max_dn=spec.max_dn, band_order=list(spec.band_order), pan_gnyq=.15 if spec.sensor=='GF2' else GNYQ_PAN[spec.sensor],
                reference='native_PAN_and_native_LRMS', support='full512', masking=False)
    if spec.sensor == 'GF2':
        result['gf2_pan_note']='Existing JQM default 0.15; surrogate, not measured GF2 spectral response'
    return result


@torch.no_grad()
def infer(model, dataset, device, deadline=None, batch_size=1):
    spec = _spec(dataset.spec)
    was_training = model.training
    model.eval()
    sr, deltas = [], []
    try:
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            check_deadline(deadline)
            _lms, ms, lp, pan = batch[:-1][-4:]
            out = model(pan.to(device), ms.to(device), lp.to(device))
            if out['y'].shape[1] != spec.num_bands or not torch.isfinite(out['y']).all() or not torch.isfinite(out['delta']).all():
                raise FloatingPointError('Invalid/nonfinite sensor inference')
            sr.append(((out['y'].float().clamp(-1, 1) + 1) * spec.inverse_scale).cpu().numpy())
            deltas.append(out['delta'].float().cpu().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(sr), np.concatenate(deltas)


@torch.no_grad()
def validation_ergas(model, dataset, device, deadline=None):
    from qg40.evaluation import validation_ergas as unchanged_validation
    _spec(dataset.spec)
    return unchanged_validation(model, dataset, device, deadline)


def native_gt(dataset):
    with h5py.File(dataset.raw_h5_path, 'r') as source:
        return np.asarray(source['gt'], dtype=np.float64)


def evaluate_model(model, datasets, device, engine=None, deadline=None, include_q=True, with_val=True, include_jqm=True):
    started = time.monotonic()
    spec = _spec(datasets['rr'].spec)
    if any(_spec(d.spec).sensor != spec.sensor for d in datasets.values()):
        raise ValueError('Mixed sensor evaluation splits')
    engine = engine or FRMetrics(datasets['fr'])
    fr_sr, fd = infer(model, datasets['fr'], device, deadline)
    fr = engine(fr_sr, deadline)
    if include_jqm:
        with h5py.File(datasets['fr'].raw_h5_path, 'r') as source:
            extra = fr_jqm(fr_sr, np.asarray(source['ms']), np.asarray(source['pan']), spec, deadline)
        fr.update(jqm=extra['jqm'], jqm_details=extra, jqm_status='measured', jqm_variant=JQM_VARIANT)
    else:
        fr.update(jqm=None, jqm_status='not_measured', jqm_reason='Explicitly disabled', jqm_variant=JQM_VARIANT)
    del fr_sr
    rr_sr, rd = infer(model, datasets['rr'], device, deadline)
    rr = rr_metrics(rr_sr, native_gt(datasets['rr']), spec, include_q)
    val = validation_ergas(model, datasets['val'], device, deadline) if with_val else None
    return dict(fr=fr, rr=rr, val_ergas=val, seconds=time.monotonic() - started,
                shift=dict(fr_mean=fd.mean(0).tolist(), fr_max_abs=float(np.abs(fd).max()),
                           rr_mean=rd.mean(0).tolist(), rr_max_abs=float(np.abs(rd).max())))
