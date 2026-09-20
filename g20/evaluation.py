"""Sensor-bound C4 RR/FR evaluation using unchanged, existing metric primitives."""
from pathlib import Path
import os
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from fh12.data import check_deadline
from tools.eval_dlpan import scc_dlpan, psnr_global, ssim_skimage
from tools.metrics.eval_rr import sam, ergas
from tools.metrics.q2n import q2n
from g20.data import canonical_band_indices, sha256_file
from g20.plan import sensor_spec

RR_KEYS = ('ergas', 'scc', 'psnr', 'sam', 'q4', 'ssim', 'rmse', 'cc')
FR_KEYS = ('hqnr', 'd_lambda', 'd_s')
JQM_VARIANT = 'SRF-substitute (NNLS-normalized), MTF41, phase2, global CMSC, v1=0.5; not SIPSA-equivalent'


def _spec(value):
    value = sensor_spec(value) if isinstance(value, str) else value
    if value.sensor != 'GF2' or value.num_bands != 4 or value.max_dn != 1023:
        raise ValueError('G20 evaluation is GF2/C4/DN1023 only')
    return value


SIGNED_DS_PROTOCOL = 'native_Ds_blockproc_UQI_Q32_LMS_interp23tap_matlab_down4; not NCC'


def signed_ds_details(q_high, q_low, official_per_scene):
    """Export the actual two operands of native Ds, without changing Ds itself."""
    high, low = np.asarray(q_high, dtype=np.float64), np.asarray(q_low, dtype=np.float64)
    reported = np.asarray(official_per_scene, dtype=np.float64)
    if high.shape != (20, 4) or low.shape != high.shape or reported.shape != (20,):
        raise ValueError('Signed Ds requires actual scene20 x band4 operands')
    if not all(np.isfinite(a).all() for a in (high, low, reported)):
        raise ValueError('Nonfinite signed Ds operand')
    delta = high - low
    absolute = np.abs(delta)
    rebuilt = absolute.mean(axis=1)
    error = float(np.max(np.abs(rebuilt - reported)))
    if not np.allclose(rebuilt, reported, rtol=0, atol=1e-12):
        raise ValueError('Signed operands do not reconstruct official native Ds')
    return dict(schema='G20_SIGNED_DS_v1', protocol=SIGNED_DS_PROTOCOL,
                operand_shape=[20, 4], Q_high=high.tolist(), Q_low=low.tolist(),
                delta=delta.tolist(), abs_delta=absolute.tolist(),
                signed_mean=float(delta.mean()), abs_mean=float(absolute.mean()),
                positive_fraction=float((delta > 0).mean()),
                per_scene=dict(signed_mean=delta.mean(1).tolist(), abs_mean=rebuilt.tolist(),
                               positive_fraction=(delta > 0).mean(1).tolist()),
                per_band=dict(signed_mean=delta.mean(0).tolist(), abs_mean=absolute.mean(0).tolist(),
                              positive_fraction=(delta > 0).mean(0).tolist()),
                reconstructed_d_s=float(rebuilt.mean()), reconstruction_max_abs_error=error,
                reconstruction_atol=1e-12, reconstruction_verified=True,
                reference='native_PAN_and_original_LMS', support='full512', masking=False,
                Q_high_definition='blockproc_uqi(fused_band,native_PAN,32)',
                Q_low_definition='blockproc_uqi(original_LMS_band,interp23tap(imresize_matlab(native_PAN,1/4),4),32)')


def validate_signed_ds(fr):
    detail = fr.get('signed_ds')
    if not isinstance(detail, dict) or len(fr.get('per_scene', [])) != 20:
        raise ValueError('Missing signed Ds actual operands/per-scene official metrics')
    expected = signed_ds_details(detail.get('Q_high'), detail.get('Q_low'),
                                 [r['d_s'] for r in fr['per_scene']])
    if detail != expected or not np.isclose(fr['d_s'], expected['reconstructed_d_s'], rtol=0, atol=1e-12):
        raise ValueError('Signed Ds payload or official reconstruction changed')
    for row in fr['per_scene']:
        if not all(np.isfinite(row[k]) for k in FR_KEYS) or not np.isclose(
                row['hqnr'], (1 - row['d_lambda']) * (1 - row['d_s']), rtol=0, atol=1e-12):
            raise ValueError('Per-scene raw HQNR does not use the native Ds operands')
    if any(not np.isclose(fr[k], np.mean([r[k] for r in fr['per_scene']]), rtol=0, atol=1e-12) for k in FR_KEYS):
        raise ValueError('FR mean-per-scene aggregation changed')
    return detail


def _summary(rows, keys):
    out = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    if not all(np.isfinite(v) for v in out.values()):
        raise FloatingPointError('Nonfinite official metric')
    return dict(out, per_scene=rows, n_scenes=len(rows),
                standard_deviation={k: float(np.std([r[k] for r in rows], ddof=1)) for k in keys})


def rr_metrics(sr, gt, spec, include_q=True):
    """Real four-band Q2n (Q4), RR20, support20:-21, Q32 and sensor DN range."""
    spec = _spec(spec)
    sr, gt = np.asarray(sr, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if sr.shape != (20, 4, 256, 256) or gt.shape != sr.shape:
        raise ValueError('Official G20 RR requires corresponding 20x4x256x256 arrays')
    if not np.isfinite(sr).all() or not np.isfinite(gt).all():
        raise FloatingPointError('Nonfinite RR input, including excluded borders')
    rows = []
    for i, (a, b) in enumerate(zip(sr.transpose(0, 2, 3, 1)[:, 20:-21, 20:-21],
                                  gt.transpose(0, 2, 3, 1)[:, 20:-21, 20:-21])):
        if a.std() == 0 or b.std() == 0:
            raise ValueError(f'Undefined flattened Pearson CC at RR scene {i}')
        row = dict(ergas=ergas(a, b), sam=sam(a, b), scc=scc_dlpan(a, b),
                   psnr=psnr_global(a, b, spec.max_dn), ssim=ssim_skimage(a, b, spec.max_dn),
                   rmse=float(np.sqrt(np.mean((a - b) ** 2))), cc=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
        if include_q:
            row['q4'] = float(q2n(b, a, 32, 32)[0])
        row['band_relative_mse'] = (np.mean((a - b) ** 2, axis=(0, 1)) / np.mean(b, axis=(0, 1)) ** 2).tolist()
        rows.append(row)
    out = _summary(rows, RR_KEYS if include_q else tuple(k for k in RR_KEYS if k != 'q4'))
    out.update(sensor=spec.sensor, num_bands=4, max_dn=spec.max_dn, crop='20:-21', q_block=32,
               official_complete=bool(include_q),
               protocol='DLPan_RR_dim21_Q4_Q32_SCC2Dzero_PSNRglobal_SSIMgauss11; scene-global DN RMSE/flattened Pearson CC')
    return out


class FRMetrics:
    """Full native PAN/LMS references; mean of per-scene raw-original HQNR."""
    def __init__(self, dataset, spec=None, wald=None):
        from tools.metrics.eval_fr import load_dlpan, imresize_matlab, _blockproc_uqi
        self.spec = _spec(spec or dataset.spec)
        self.order = canonical_band_indices(self.spec.band_order)
        self.inverse_order = np.argsort(self.order)
        self.wald = wald or load_dlpan(os.environ.get('PANCRAFTER_DLPAN', str(Path(__file__).resolve().parents[2] / 'DLPan-Toolbox')))
        self.wald_sha256 = sha256_file(self.wald.__file__) if getattr(self.wald, '__file__', None) else None
        with h5py.File(dataset.raw_h5_path, 'r') as src:
            self.lms = np.asarray(src['lms'], dtype=np.float64).transpose(0, 2, 3, 1)
            self.pan = np.asarray(src['pan'], dtype=np.float64)[:, 0]
        if self.lms.shape != (20, 512, 512, 4) or self.pan.shape != (20, 512, 512):
            raise ValueError('G20 FR requires 20 native full512 C4 scenes')
        if not np.isfinite(self.lms).all() or not np.isfinite(self.pan).all():
            raise ValueError('Nonfinite original full-frame FR references')
        self.reference = []
        for ms, pan in zip(self.lms, self.pan):
            low = self.wald.interp23tap(imresize_matlab(pan, 1 / 4)[..., None], 4)[..., 0]
            self.reference.append([_blockproc_uqi(ms[..., b], low, 32) for b in range(4)])

    def __call__(self, sr, deadline=None):
        from tools.metrics.eval_fr import mtf_filter, _blockproc_uqi
        sr = np.asarray(sr, dtype=np.float64)
        if sr.shape != (20, 4, 512, 512) or not np.isfinite(sr).all():
            raise ValueError('Invalid/nonfinite C4 full512 prediction')
        rows, high_operands = [], []
        for i, fused in enumerate(sr.transpose(0, 2, 3, 1)):
            check_deadline(deadline)
            # Canonical-order filtering restores original source-band order afterward.
            filtered = mtf_filter(fused[..., list(self.order)], self.spec.mtf_sensor.lower(), 4, self.wald)[..., self.inverse_order]
            dl = float(1 - q2n(self.lms[i], filtered, 32, 32)[0])
            q_high = [_blockproc_uqi(fused[..., b], self.pan[i], 32) for b in range(4)]
            high_operands.append(q_high)
            ds = float(np.mean(np.abs(np.asarray(q_high) - self.reference[i])))
            rows.append(dict(d_lambda=dl, d_s=ds, hqnr=float((1 - dl) * (1 - ds))))
        out = _summary(rows, FR_KEYS)
        out['signed_ds'] = signed_ds_details(high_operands, self.reference, [r['d_s'] for r in rows])
        out.update(sensor=self.spec.sensor, num_bands=4, max_dn=self.spec.max_dn,
                   band_order=list(self.spec.band_order), mtf_sensor=self.spec.mtf_sensor,
                   mtf_gnyq_canonical=[.34, .32, .30, .22] if self.spec.sensor == 'QB' else [.3] * 4,
                   mtf_note='GF2 explicit existing DLPan otherwise GNyq=0.3 preset' if self.spec.sensor == 'GF2' else 'QB band-specific preset',
                   interp23tap_source_sha256=getattr(self, 'wald_sha256', None),
                   eval_mode='A_ON', reference='native_PAN', support='full512', masking=False,
                   aggregation='mean_per_scene_HQNR', hqnr_variant='raw-original', official_complete=True)
        return out


def fr_jqm(sr, ms, pan, spec, deadline=None):
    """Existing SRF-substitute JQM, with explicit sensor, DN and original bands."""
    from tools.metrics.jqm import jqm
    spec = _spec(spec)
    order = canonical_band_indices(spec.band_order)
    sr, ms, pan = np.asarray(sr), np.asarray(ms), np.asarray(pan)
    if sr.shape != (20, 4, 512, 512) or ms.shape != (20, 4, 128, 128) or pan.shape != (20, 1, 512, 512):
        raise ValueError('JQM requires complete native C4 FR20')
    if not all(np.isfinite(a).all() for a in (sr, ms, pan)):
        raise FloatingPointError('Nonfinite JQM native inputs')
    rows = []
    for i in range(20):
        check_deadline(deadline)
        value = jqm(sr[i, list(order)].transpose(1, 2, 0), ms[i, list(order)].transpose(1, 2, 0),
                    pan[i, 0], spec.sensor, ratio=4, R=spec.max_dn, lpf='mtf', window=None, v1=.5)
        rows.append(dict(jqm=float(value['JQM']), qlr=float(value['QLR']), qhr=float(value['QHR']),
                         w=np.asarray(value['w'])[np.argsort(order)].tolist(), w_source=value['w_source']))
    out = _summary(rows, ('jqm', 'qlr', 'qhr'))
    out.update(variant=JQM_VARIANT, reference='native_PAN_and_native_LRMS', support='full512', masking=False,
               sensor=spec.sensor, max_dn=spec.max_dn, band_order=list(spec.band_order),
               estimation_data='same-scene native LRMS and MTF-filtered native PAN; NNLS normalized',
               pan_gnyq=.15, gf2_pan_note='Existing JQM default 0.15; surrogate, not measured GF2 spectral response' if spec.sensor == 'GF2' else '')
    return out


@torch.no_grad()
def infer(model, dataset, device, deadline=None, batch_size=1):
    _spec(dataset.spec)
    was_training = model.training
    model.eval()
    sr, deltas = [], []
    try:
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            check_deadline(deadline)
            data = batch[:-1]
            _lms, ms, lp, pan = data[-4:]
            out = model(pan.to(device), ms.to(device), lp.to(device))
            if out['y'].shape[1] != 4 or not torch.isfinite(out['y']).all() or not torch.isfinite(out['delta']).all():
                raise FloatingPointError('Invalid C4 inference')
            sr.append(((out['y'].float().clamp(-1, 1) + 1) * dataset.spec.inverse_scale).cpu().numpy())
            deltas.append(out['delta'].float().cpu().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(sr), np.concatenate(deltas)


@torch.no_grad()
def validation_ergas(model, dataset, device, deadline=None):
    if not dataset.has_gt or dataset.split != 'val' or dataset.augment:
        raise ValueError('Fixed unaugmented validation split required')
    was_training = model.training
    model.eval()
    vals = []
    try:
        for gt, _lms, ms, lp, pan, _meta in DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0):
            check_deadline(deadline)
            y = model(pan.to(device), ms.to(device), lp.to(device))['y']
            y = (y.float().clamp(-1, 1) + 1) * dataset.spec.inverse_scale
            target = (gt.to(device).float() + 1) * dataset.spec.inverse_scale
            per = 25 * torch.sqrt((((y - target) ** 2).mean((2, 3)) / target.mean((2, 3)).clamp_min(1e-12) ** 2).mean(1))
            vals.extend(per.cpu().tolist())
    finally:
        model.train(was_training)
    if len(vals) != dataset.base_count or not np.isfinite(vals).all():
        raise ValueError('Incomplete/nonfinite validation')
    return float(np.mean(vals))


def native_gt(dataset):
    with h5py.File(dataset.raw_h5_path, 'r') as src:
        return np.asarray(src['gt'], dtype=np.float64)


def evaluate_model(model, datasets, device, engine=None, deadline=None, include_q=True, with_val=True, include_jqm=True):
    start = time.monotonic()
    spec = _spec(datasets['rr'].spec)
    if any(d.spec.sensor != spec.sensor for d in datasets.values()):
        raise ValueError('Mixed sensor evaluation splits')
    engine = engine or FRMetrics(datasets['fr'])
    fr_sr, fd = infer(model, datasets['fr'], device, deadline)
    fr = engine(fr_sr, deadline)
    if include_jqm:
        with h5py.File(datasets['fr'].raw_h5_path, 'r') as src:
            extra = fr_jqm(fr_sr, np.asarray(src['ms']), np.asarray(src['pan']), spec, deadline)
        fr.update(jqm=extra['jqm'], jqm_details=extra, jqm_status='measured', jqm_variant=JQM_VARIANT)
    else:
        fr.update(jqm=None, jqm_status='not_measured', jqm_reason='Explicitly disabled for this evaluation', jqm_variant=JQM_VARIANT)
    del fr_sr
    rr_sr, rd = infer(model, datasets['rr'], device, deadline)
    rr = rr_metrics(rr_sr, native_gt(datasets['rr']), spec, include_q)
    val = validation_ergas(model, datasets['val'], device, deadline) if with_val else None
    return dict(fr=fr, rr=rr, val_ergas=val, seconds=time.monotonic() - start,
                shift=dict(fr_mean=fd.mean(0).tolist(), fr_max_abs=float(np.abs(fd).max()),
                           rr_mean=rd.mean(0).tolist(), rr_max_abs=float(np.abs(rd).max())))
