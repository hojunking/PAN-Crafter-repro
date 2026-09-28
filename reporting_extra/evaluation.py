"""Pinned-checkpoint RMSE/CC/JQM, without changing official selection or metrics.

RMSE and CC use the existing Sheet convention: each RR scene's whole 8-band
215 x 215 crop, DN RMSE and flattened Pearson CC, then a mean across 20 scenes.
JQM uses the existing explicitly labelled NNLS/SRF-substitute implementation on
all 20 full-resolution scenes, native unshifted PAN and original LRMS. It is an
additional metric, not a checkpoint selector or a claim of SIPSA equivalence.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import math
import re

import h5py
import numpy as np
import scipy
import skimage
import torch
import yaml

from fh12.common import ROOT, object_sha, read_json, sha256, utcnow
from fh12.data import build_dataset, write_immutable_json
from fh12.evaluation import infer, native_gt
from fh12.postrun import select_records
from fh20r1.common import load_checkpoint_model
from tools.metrics.jqm import jqm

SCHEMA = 'PAN_SUPPLEMENTAL_METRICS_v1'
SELECTIONS = {
    'raw_max': ('RAW_MAX', 'raw_max'),
    'target_selection': ('TARGET', 'target'),
    'exact50k': ('EXACT50K', 'exact50k'),
    'rr_val_selected': ('RR_VAL_SELECTED', 'rr_val_selected'),
    'e_min_diag': ('E_MIN_DIAG50', 'e_min_diag'),
}
INFERENCE_FILES = (
    'fh12/common.py', 'fh12/data.py', 'fh12/model.py', 'fh12/evaluation.py',
    'fh12/postrun.py', 'model/pancrafter_paper.py', 'model/pancrafter.py',
    'model/swin.py', 'pa/aligner.py', 'pa/warp.py', 'pa/losses.py',
    'pa/offset.py', 'tools/repair_lpan.py', 'tools/eval_dlpan.py',
    'tools/metrics/eval_rr.py', 'tools/metrics/eval_fr.py', 'tools/metrics/q2n.py',
)
JQM_VARIANT = 'SRF-substitute (NNLS-normalized), MTF41, phase2, global CMSC, v1=0.5; not SIPSA-equivalent'


def _finite(value, name):
    if not np.isfinite(np.asarray(value)).all():
        raise FloatingPointError(f'Nonfinite {name}')


def _summary(rows, keys):
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out['standard_deviation'] = {k: float(np.std([r[k] for r in rows], ddof=1)) for k in keys}
    out.update(n_scenes=len(rows), per_scene=rows)
    return out


def rr_metrics(sr, gt):
    """Same crop and flattened Pearson definition as gspread_upload._rr_extra."""
    sr, gt = np.asarray(sr), np.asarray(gt)
    if sr.shape != (20, 8, 256, 256) or sr.shape != gt.shape:
        raise ValueError(f'Expected corresponding 20x8x256x256 RR arrays: {sr.shape}, {gt.shape}')
    _finite(sr, 'RR prediction, including border'); _finite(gt, 'RR GT, including border')
    rows = []
    for i, (pred, truth) in enumerate(zip(sr, gt)):
        a = np.asarray(pred.transpose(1, 2, 0)[20:-21, 20:-21], dtype=np.float64).ravel()
        b = np.asarray(truth.transpose(1, 2, 0)[20:-21, 20:-21], dtype=np.float64).ravel()
        if np.std(a) == 0 or np.std(b) == 0:
            raise ValueError(f'Undefined Pearson CC in constant RR scene {i}; not replacing with zero')
        row = dict(scene_index=i, rmse=float(np.sqrt(np.mean((a - b) ** 2))),
                   cc=float(np.corrcoef(a, b)[0, 1]))
        _finite([row['rmse'], row['cc']], f'RR scene {i}')
        rows.append(row)
    out = _summary(rows, ('rmse', 'cc'))
    out.update(crop='20:-21', unit='DN (peak 2047)',
               protocol='scene-global all-band RMSE and flattened Pearson CC; mean20, std(ddof=1)')
    return out


def fr_metrics(sr, ms, pan):
    """JQM references are native LRMS and native PAN; no mask, warp or crop."""
    sr, ms, pan = np.asarray(sr), np.asarray(ms), np.asarray(pan)
    if (sr.shape != (20, 8, 512, 512) or ms.shape != (20, 8, 128, 128)
            or pan.shape != (20, 1, 512, 512)):
        raise ValueError(f'Invalid native FR geometry: {sr.shape}, {ms.shape}, {pan.shape}')
    for value, name in ((sr, 'FR predictions'), (ms, 'FR LRMS'), (pan, 'FR PAN')):
        _finite(value, name)
    rows = []
    for i in range(20):
        value = jqm(sr[i].transpose(1, 2, 0), ms[i].transpose(1, 2, 0), pan[i, 0],
                    'WV3', ratio=4, R=2047., lpf='mtf', window=None, v1=.5)
        row = dict(scene_index=i, jqm=float(value['JQM']), qlr=float(value['QLR']),
                   qhr=float(value['QHR']), w=value['w'], w_source=value['w_source'])
        _finite([row['jqm'], row['qlr'], row['qhr'], *row['w']], f'JQM scene {i}')
        if any(not -1e-12 <= row[k] <= 1 + 1e-12 for k in ('jqm', 'qlr', 'qhr')):
            raise ValueError('JQM terms are outside their declared range')
        rows.append(row)
    out = _summary(rows, ('jqm', 'qlr', 'qhr'))
    sources = sorted({r['w_source'] for r in rows})
    out.update(reference='native_PAN_and_native_LRMS', support='full512', masking=False,
               w_source=sources[0] if len(sources) == 1 else sources,
               variant=JQM_VARIANT, protocol='tools.metrics.jqm.jqm, WV3 ratio4 R2047 MTF41 phase2')
    return out


def evaluator_identity(root=ROOT):
    root = Path(root)
    paths = (*INFERENCE_FILES, 'fh20r1/common.py', 'reporting_extra/evaluation.py', 'tools/metrics/jqm.py')
    files = {path: sha256(root / path) for path in paths}
    return dict(schema=SCHEMA, files=files, content_sha256=object_sha(files),
                torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                skimage=skimage.__version__, precision='fp32', inference_batch_size=1,
                output_conversion='clamp[-1,1]; (y+1)*1023.5; no DN rounding',
                protocol='RR20_dim21_global_RMSE_CC; FR20_native_JQM_NNLS_MTF_v1')


def _validate_origin(origin, root, current):
    files = origin.get('files', {})
    if origin.get('content_sha256') != object_sha(files):
        raise ValueError('Corrupt original source identity')
    for name in INFERENCE_FILES:
        if files.get(name) != current['files'].get(name):
            raise ValueError(f'Pinned inference/selection implementation changed: {name}')
    for package in ('torch', 'numpy', 'scipy', 'skimage'):
        if origin.get(package) != current[package]:
            raise ValueError(f'Pinned numerical runtime changed: {package}')


def _run_dir(run, root):
    if not re.fullmatch(r'(FH12|FH20R1)_[A-Za-z0-9_]+', run):
        raise ValueError('Expected a single FH12/FH20R1 run ID, not a path')
    return Path(root) / 'work_dir' / run


def _context(run, root):
    """Read-only validation; recompute selectors but never rewrite their results."""
    root = Path(root); wd = _run_dir(run, root)
    cfg = yaml.safe_load((wd / 'meta/config.resolved.yaml').read_text())
    section = 'fh20r1' if 'fh20r1' in cfg else 'fh12'
    f = cfg[section]
    if f['run_id'] != run or f['server_id'] not in {f's{i}' for i in range(1, 6)}:
        raise ValueError('Wrong run/server identity')
    if (wd / 'meta/reuse_reference.json').exists():
        raise ValueError('Semantic-reuse link is not a newly trained run; evaluate its source run explicitly')
    status = read_json(wd / 'official/postrun_status.json')
    training = read_json(wd / 'meta/training_status.json')
    if (not status.get('official_complete') or not training.get('training_complete')
            or training.get('actual_updates') != 50000):
        raise ValueError('Supplemental evaluation requires completed official 50K results')
    data_path = root / f.get('dataset_manifest', f"work_dir/_fh12/{f['server_id']}/dataset_manifest.json")
    data = read_json(data_path)
    cfg_sha, data_sha = object_sha(cfg), object_sha(data)
    if data.get('sensor') != 'WV3' or data.get('max_pixel') != 2047.:
        raise ValueError('Supplemental protocol is WV3 2047DN only')
    for split in ('rr', 'fr'):
        item = data['splits'][split]
        if item['count'] != 20:
            raise ValueError(f'{split} is not the complete 20-scene dataset')
        for path_key, hash_key in (('dataroot', 'sha256'), ('lpan_path', 'lpan_sha256')):
            if sha256(root / item[path_key]) != item[hash_key]:
                raise ValueError(f'{split} native input/LP cache changed: {path_key}')
    grid_path = wd / 'official/raw_grid.json'
    grid = read_json(grid_path)
    if (not grid.get('complete') or grid.get('config_sha256') != cfg_sha
            or grid.get('data_sha256') != data_sha
            or grid.get('run_id') != run or grid.get('campaign_id') != f['campaign_id']):
        raise ValueError('Original official grid/config/data/campaign mismatch')
    current = evaluator_identity(root)
    _validate_origin(grid['source_identity'], root, current)
    selected = select_records(grid['records'])
    official, official_shas, candidates = {}, {}, {}
    for name, (label, selector) in SELECTIONS.items():
        path = wd / 'official' / f'{name}.json'
        doc = read_json(path); record = selected[selector]
        if (doc.get('run_id') != run or doc.get('campaign_id') != f['campaign_id']
                or doc.get('selection_id') != label or not doc.get('official_complete')
                or doc.get('config_sha256') != cfg_sha or doc.get('data_sha256') != data_sha
                or doc.get('source_identity') != grid['source_identity']):
            raise ValueError(f'Official {name} identity mismatch')
        if record is None:
            if (doc.get('selection', 'missing') is not None or doc.get('step') is not None
                    or doc.get('target_status') != 'no_eligible'):
                raise ValueError('No-eligible target has fabricated selected checkpoint')
        else:
            step = record['update']; ident = record['checkpoint_identity']
            if (doc.get('step') != step or doc.get('checkpoint_identity') != ident
                    or doc.get('checkpoint_sha256') != ident['model_sha256']
                    or any(doc.get(k) != record.get(k) for k in ('rr', 'fr', 'val_ergas'))
                    or doc.get('eval_mode') != 'A_ON' or doc.get('precision') != 'fp32'):
                raise ValueError(f'Official {name} differs from unchanged selected grid record')
            if (doc['rr'].get('n_scenes') != 20 or doc['rr'].get('crop') != '20:-21'
                    or doc['fr'].get('n_scenes') != 20 or doc['fr'].get('reference') != 'native_PAN'
                    or doc['fr'].get('support') != 'full512'):
                raise ValueError(f'Official {name} is not native full20 RR/FR protocol')
            if step not in candidates:
                folder = wd / 'candidates' / str(step)
                if (read_json(folder / 'identity.json') != ident
                        or sha256(folder / 'model.safetensors') != ident['model_sha256']
                        or ident.get('config_sha256') != cfg_sha or ident.get('data_sha256') != data_sha
                        or ident.get('source_identity') != grid['source_identity']):
                    raise ValueError(f'Checkpoint {step} bytes/provenance mismatch')
                candidates[step] = ident
        official[name] = doc; official_shas[name] = sha256(path)
    binding = dict(run_id=run, campaign_id=f['campaign_id'], server_id=f['server_id'],
                   config_sha256=cfg_sha, data_sha256=data_sha, grid_sha256=sha256(grid_path),
                   official_report_sha256=official_shas, evaluator_identity=current,
                   original_source_identity=grid['source_identity'])
    return dict(wd=wd, cfg=cfg, data=data, binding=binding, official=official, candidates=candidates)


def _seal(value):
    return dict(value, payload_sha256=object_sha(value))


def _check_seal(value):
    body = {k: v for k, v in value.items() if k != 'payload_sha256'}
    if value.get('payload_sha256') != object_sha(body):
        raise ValueError('Supplemental artifact payload checksum mismatch')


def _step_identity(ctx, step):
    b = ctx['binding']
    return dict(schema=SCHEMA, run_id=b['run_id'], step=step,
                checkpoint_sha256=ctx['candidates'][step]['model_sha256'],
                config_sha256=b['config_sha256'], data_sha256=b['data_sha256'],
                origin_source_sha256=b['original_source_identity']['content_sha256'],
                evaluator_identity=b['evaluator_identity'])


def _check_metrics(value):
    for group, keys in (('rr', ('rmse', 'cc')), ('fr', ('jqm', 'qlr', 'qhr'))):
        item = value[group]
        if item.get('n_scenes') != 20 or len(item.get('per_scene', [])) != 20:
            raise ValueError('Incomplete supplemental scene coverage')
        for key in keys:
            numbers = [r[key] for r in item['per_scene']]
            _finite(numbers, key)
            if (not math.isclose(item[key], float(np.mean(numbers)), rel_tol=1e-12, abs_tol=1e-12)
                    or not math.isclose(item['standard_deviation'][key], float(np.std(numbers, ddof=1)),
                                        rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(f'Corrupt supplemental mean/std for {key}')
    if value['rr'].get('crop') != '20:-21' or value['fr'].get('masking') is not False:
        raise ValueError('Supplemental evaluation support changed')


@contextmanager
def _precision(origin, device):
    dev = torch.device(device)
    if dev.type not in ('cpu', 'cuda'):
        raise ValueError('Only explicit CPU or CUDA inference is supported')
    if dev.type == 'cuda':
        if (origin.get('cuda') != torch.version.cuda
                or origin.get('cudnn') != torch.backends.cudnn.version()):
            raise ValueError('CUDA/cuDNN differs from original evaluator')
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(origin['tf32_matmul'])
        torch.backends.cudnn.allow_tf32 = bool(origin['tf32_cudnn'])
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def _evaluate_step(ctx, step, datasets, device):
    model, _ = load_checkpoint_model(ctx['cfg'], ctx['wd'] / 'candidates' / str(step), device,
                                     expected_source=ctx['binding']['original_source_identity'])
    try:
        with _precision(ctx['binding']['original_source_identity'], device):
            rr_sr, _ = infer(model, datasets['rr'], device, batch_size=1)
            rr = rr_metrics(rr_sr, native_gt(datasets['rr']))
            del rr_sr
            fr_sr, _ = infer(model, datasets['fr'], device, batch_size=1)
            with h5py.File(datasets['fr'].raw_h5_path, 'r') as f:
                fr = fr_metrics(fr_sr, np.asarray(f['ms']), np.asarray(f['pan']))
        return dict(rr=rr, fr=fr)
    finally:
        del model


def validate_report(run, root=ROOT):
    """Read-only integrity check, suitable immediately before a Sheet write."""
    ctx = _context(run, root); folder = ctx['wd'] / 'supplemental_metrics'
    report = read_json(folder / 'report.json'); _check_seal(report)
    if report.get('schema') != SCHEMA or report.get('binding') != ctx['binding']:
        raise ValueError('Supplemental report is stale or belongs to a different release/data/run')
    if (not report.get('complete') or report.get('official_selection_and_core9_unchanged') is not True
            or any(report.get(k) != ctx['binding'][k] for k in
                   ('run_id', 'campaign_id', 'server_id', 'config_sha256', 'data_sha256', 'evaluator_identity'))):
        raise ValueError('Supplemental report top-level provenance mismatch')
    if set(report.get('selections', {})) != set(SELECTIONS):
        raise ValueError('Supplemental report selection set incomplete')
    inference_devices = set()
    for name, doc in ctx['official'].items():
        row = report['selections'][name]
        if row.get('official_report_sha256') != ctx['binding']['official_report_sha256'][name]:
            raise ValueError('Supplemental selection was not derived from this official JSON')
        if row.get('selection_id') != doc['selection_id']:
            raise ValueError('Supplemental selector label mismatch')
        if 'step' not in doc:
            if (row.get('status') != 'no_eligible' or row.get('metrics') is not None
                    or row.get('step') is not None or row.get('checkpoint_sha256') is not None):
                raise ValueError('No-eligible target must retain empty metrics')
            continue
        step = doc['step']; cached = read_json(folder / f'step_{step}.json')
        _check_seal(cached); _check_metrics(cached)
        if cached.get('device') not in ('cpu', 'cuda') and not re.fullmatch(r'cuda:\d+', cached.get('device', '')):
            raise ValueError('Supplemental cache has an invalid inference device')
        inference_devices.add(cached['device'])
        if (cached.get('identity') != _step_identity(ctx, step)
                or row.get('step_cache_sha256') != sha256(folder / f'step_{step}.json')
                or row.get('step') != step or row.get('checkpoint_sha256') != doc['checkpoint_sha256']
                or row.get('status') != 'complete' or any(row.get(k) != cached[k] for k in ('rr', 'fr'))
                or row.get('metrics') != dict(rmse=cached['rr']['rmse'], cc=cached['rr']['cc'], jqm=cached['fr']['jqm'])):
            raise ValueError('Supplemental selection/checkpoint/cache mismatch')
    if report.get('inference_devices') != sorted(inference_devices):
        raise ValueError('Supplemental backend provenance mismatch')
    return report


def process(run, root=ROOT, device='cpu'):
    """Compute missing selected steps once; publish only additive immutable JSONs."""
    ctx = _context(run, root); folder = ctx['wd'] / 'supplemental_metrics'
    if (folder / 'report.json').exists():
        return validate_report(run, root)
    datasets = None; steps = {}
    for step in sorted(ctx['candidates']):
        path = folder / f'step_{step}.json'; identity = _step_identity(ctx, step)
        if path.exists():
            cached = read_json(path); _check_seal(cached); _check_metrics(cached)
            if cached.get('identity') != identity:
                raise ValueError(f'Stale supplemental step cache {step}; refusing overwrite')
        else:
            if datasets is None:
                datasets = {s: build_dataset(ctx['data'], s, root=root) for s in ('rr', 'fr')}
            metrics = _evaluate_step(ctx, step, datasets, device)
            _check_metrics(metrics)
            cached = _seal(dict(identity=identity, **metrics, device=str(torch.device(device)),
                                 precision='fp32', completed_at_utc=utcnow(),
                                 note='CPU/CUDA backend is explicit; no claim of bitwise GPU-output equivalence'))
            write_immutable_json(path, cached)
        steps[step] = cached
    if _context(run, root)['binding'] != ctx['binding']:
        raise ValueError('Original official data changed during supplemental evaluation')
    selections = {}
    for name, doc in ctx['official'].items():
        row = dict(selection_id=doc['selection_id'],
                   official_report_sha256=ctx['binding']['official_report_sha256'][name])
        if 'step' not in doc:
            row.update(status='no_eligible', step=None, checkpoint_sha256=None, metrics=None)
        else:
            step = doc['step']; result = steps[step]
            row.update(status='complete', step=step, checkpoint_sha256=doc['checkpoint_sha256'],
                       step_cache_sha256=sha256(folder / f'step_{step}.json'),
                       metrics=dict(rmse=result['rr']['rmse'], cc=result['rr']['cc'], jqm=result['fr']['jqm']),
                       rr=result['rr'], fr=result['fr'])
        selections[name] = row
    b = ctx['binding']
    report = _seal(dict(schema=SCHEMA, complete=True, run_id=run, campaign_id=b['campaign_id'],
                        server_id=b['server_id'], config_sha256=b['config_sha256'], data_sha256=b['data_sha256'],
                        evaluator_identity=b['evaluator_identity'], binding=b, selections=selections,
                        n_unique_checkpoints=len(steps), completed_at_utc=utcnow(),
                        inference_devices=sorted({value['device'] for value in steps.values()}),
                        jqm_variant=JQM_VARIANT, official_selection_and_core9_unchanged=True))
    write_immutable_json(folder / 'report.json', report)
    return validate_report(run, root)
