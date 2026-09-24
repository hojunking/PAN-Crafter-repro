"""Original FH12 native metrics, with an explicit MAIN-A population contract.

The candidate selector consumes *only* the mean of the 20 per-scene raw PAN
HQNRs.  Neither validation patches nor shifted/masked PAN enter this module.
"""
from contextlib import contextmanager
from pathlib import Path
import copy
import hashlib
import math
import random
import time

import numpy as np
import torch

from maina_hqnr.common import ROOT, atomic_json, object_sha, sha256

RR_KEYS = ('ergas', 'sam', 'psnr', 'scc', 'ssim', 'q8', 'rmse', 'cc')
FR_KEYS = ('hqnr', 'd_lambda', 'd_s')
JQM_VARIANT = 'SRF-substitute (NNLS-normalized), MTF41, phase2, global CMSC, v1=0.5; not SIPSA-equivalent'
PROTOCOL = dict(schema='MAINA_HQNR_NATIVE_FR20_RR20_v1', sensor='WV3', bands=8,
    max_dn=2047, ratio=4, fr_count=20, rr_count=20,
    prediction='original fh12.evaluation.infer: FP32 clamp[-1,1], (y+1)*1023.5; no rounding',
    fr_support='full512', hqnr_reference='RAW_ORIGINAL_PAN', masking=False,
    alignment=False, aggregation='mean_per_scene_HQNR',
    fr_kernel='original fh12.evaluation.FRMetrics',
    rr_kernel='original fh12.evaluation.rr_metrics', rr_crop='20:-21', q_block=32,
    rmse='mean of scene global all-band DN RMSE on 20:-21',
    cc='mean of scene flattened Pearson on 20:-21', jqm=JQM_VARIANT)


@contextmanager
def preserve_rng():
    """Even DataLoader construction and an interrupted evaluation leave RNG intact."""
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    try:
        yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])


def _check(stopcheck):
    if stopcheck is not None and stopcheck():
        raise InterruptedError('Safe MAIN-A evaluation pause; completed candidate ledger retained')


class _CheckedDataset:
    def __init__(self, dataset, stopcheck):
        self.dataset, self.stopcheck, self.has_gt = dataset, stopcheck, dataset.has_gt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        _check(self.stopcheck)
        return self.dataset[index]


def evaluator_identity(wald=None, root=ROOT):
    import scipy
    import skimage
    if wald is None:
        import os
        from tools.metrics.eval_fr import load_dlpan
        wald = load_dlpan(os.environ.get('PANCRAFTER_DLPAN', str(Path(root).parent/'DLPan-Toolbox')))
    paths = ('maina_hqnr/evaluation.py', 'fh12/evaluation.py', 'fh12/data.py',
             'tools/eval_dlpan.py', 'tools/metrics/eval_rr.py', 'tools/metrics/eval_fr.py',
             'tools/metrics/q2n.py', 'tools/metrics/jqm.py')
    files = {name: sha256(Path(root)/name) for name in paths}
    source = getattr(wald, '__file__', None)
    if source is None:
        raise ValueError('Production evaluation requires the actual pinned DLPan source')
    files['external/DLPan/wald_utilities.py'] = sha256(source)
    if files['external/DLPan/wald_utilities.py'] != 'c7d076395eec052dd4ed391e6f51e5fa79661a3fc283816a63ccdb100ff5bdbf':
        raise ValueError('The original DLPan MTF/interp23tap dependency changed')
    result = dict(files=files, numpy=np.__version__, torch=torch.__version__,
                  scipy=scipy.__version__, skimage=skimage.__version__,
                  runtime=dict(device_type='cuda', precision='fp32', cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(), matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark,
                    cudnn_deterministic=torch.backends.cudnn.deterministic, num_threads=torch.get_num_threads()))
    result['sha256'] = object_sha(result)
    return result


def population(dataset, split):
    """Bind every scene to its full, uncropped native inputs and LP cache bytes."""
    if split not in ('rr', 'fr') or len(dataset) != 20 or getattr(dataset, 'augment', False):
        raise ValueError('Only all 20 unaugmented native RR/FR scenes are accepted')
    size = 256 if split == 'rr' else 512
    arrays = dataset.arrays
    expected = dict(pan=(20, 1, size, size), lms=(20, 8, size, size),
                    ms=(20, 8, size//4, size//4), lpan=(20, 1, size//4, size//4))
    if split == 'rr':
        expected['gt'] = (20, 8, size, size)
    for key, shape in expected.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f'Invalid full native {split} {key} population')
    source_sha = sha256(dataset.raw_h5_path)
    rows = []
    for index in range(20):
        hashes = {}
        for key in sorted(expected):
            value = np.ascontiguousarray(arrays[key][index])
            digest = hashlib.sha256(value.tobytes()).hexdigest()
            hashes[key] = dict(sha256=digest, dtype=value.dtype.str, shape=list(value.shape))
        rows.append(dict(scene_index=index, scene_id=f'{source_sha}:{index}',
                         native_scene_sha256=object_sha(hashes), input_hashes=hashes))
    return dict(split=split, count=20, raw_h5_sha256=source_sha,
                lpan_sha256=sha256(dataset.lp_path), scenes=rows, scene_manifest_sha256=object_sha(rows))


def _attach(report, pop, checkpoint_sha256):
    result = copy.deepcopy(report)
    if len(result['per_scene']) != 20:
        raise ValueError('Metric kernel omitted native scenes')
    for value, scene in zip(result['per_scene'], pop['scenes']):
        value.update({key: scene[key] for key in ('scene_index', 'scene_id', 'native_scene_sha256')})
        value['checkpoint_sha256'] = checkpoint_sha256
        value['row_sha256'] = object_sha(value)
    result.update(n_scenes=20, scene_ids=[s['scene_id'] for s in pop['scenes']],
                  scene_manifest_sha256=pop['scene_manifest_sha256'], discarded_samples=0,
                  official_complete=True, checkpoint_sha256=checkpoint_sha256)
    return result


def _jqm(sr, dataset, stopcheck):
    from tools.metrics.jqm import jqm
    rows = []
    for index, image in enumerate(sr):
        _check(stopcheck)
        item = jqm(image.astype(np.float64).transpose(1, 2, 0),
                   np.asarray(dataset.arrays['ms'][index], dtype=np.float64).transpose(1, 2, 0),
                   np.asarray(dataset.arrays['pan'][index, 0], dtype=np.float64),
                   'WV3', ratio=4, R=2047, lpf='mtf', window=None, v1=.5)
        values = dict(jqm=float(item['JQM']), qlr=float(item['QLR']), qhr=float(item['QHR']),
                      jqm_weights=np.asarray(item['w']).tolist(), jqm_w_source=item['w_source'])
        if not np.isfinite([values[k] for k in ('jqm', 'qlr', 'qhr')]+list(values['jqm_weights'])).all():
            raise FloatingPointError('Nonfinite JQM')
        if any(not -1e-12 <= values[key] <= 1+1e-12 for key in ('jqm', 'qlr', 'qhr')):
            raise ValueError('JQM terms exceed their declared range')
        rows.append(values)
    return rows


def validate_report(report, split, checkpoint_sha256=None, require_jqm=False):
    keys = RR_KEYS if split == 'rr' else FR_KEYS + (('jqm',) if require_jqm else ())
    if (report.get('n_scenes') != 20 or len(report.get('per_scene', [])) != 20
            or len(report.get('scene_ids', [])) != 20 or len(set(report['scene_ids'])) != 20
            or report.get('discarded_samples') != 0 or report.get('official_complete') is not True):
        raise ValueError('Full native scene population missing')
    digest = checkpoint_sha256 or report.get('checkpoint_sha256')
    if not isinstance(digest, str) or len(digest) != 64 or report.get('checkpoint_sha256') != digest:
        raise ValueError('RR/FR checkpoint SHA mismatch')
    for index, row in enumerate(report['per_scene']):
        if (row.get('scene_index') != index or row.get('scene_id') != report['scene_ids'][index]
                or row.get('checkpoint_sha256') != digest or len(row.get('native_scene_sha256', '')) != 64
                or row.get('row_sha256') != object_sha({k: v for k, v in row.items() if k != 'row_sha256'})):
            raise ValueError('Scene/checkpoint/hash correspondence differs')
        if any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) for k in keys):
            raise ValueError('Nonfinite/missing official metric')
        if split == 'fr' and row['hqnr'] != (1-row['d_lambda'])*(1-row['d_s']):
            raise ValueError('HQNR is not the per-scene raw product')
    for key in keys:
        if report.get(key) != float(np.mean([row[key] for row in report['per_scene']])):
            raise ValueError('Official mean is not the full-precision scene mean')
    if split == 'fr':
        expected = dict(support='full512', reference='native_PAN', masking=False, alignment=False,
                        hqnr_variant='raw-original', aggregation='mean_per_scene_HQNR')
        if any(report.get(k) != v for k, v in expected.items()):
            raise ValueError('V64, masked, aligned or nonnative HQNR cannot select a checkpoint')
        if require_jqm and report.get('jqm_variant') != JQM_VARIANT:
            raise ValueError('JQM SRF-substitute variant must remain explicit')
    elif report.get('crop') != '20:-21' or report.get('q_block') != 32:
        raise ValueError('Original RR crop/Q8 protocol changed')
    return report


def validate_candidate(record):
    metadata = record['metadata']
    if (metadata.get('protocol') != PROTOCOL or metadata.get('protocol_sha256') != object_sha(PROTOCOL)
            or metadata.get('optimizer_updates_during_evaluation') != 0):
        raise ValueError('Candidate evaluation protocol differs')
    digest = record['checkpoint_sha256']
    if metadata.get('checkpoint_sha256') != digest:
        raise ValueError('Candidate A/U evaluation identity mismatch')
    validate_report(record['fr'], 'fr', digest)
    if record['fr']['scene_manifest_sha256'] != metadata['population']['scene_manifest_sha256']:
        raise ValueError('Full native scene hash differs')
    scenes = metadata['population']['scenes']
    if (len(scenes) != 20 or metadata['population'].get('count') != 20
            or object_sha(scenes) != metadata['population']['scene_manifest_sha256']):
        raise ValueError('Full native population manifest is invalid')
    for row, scene in zip(record['fr']['per_scene'], scenes):
        if any(row[k] != scene[k] for k in ('scene_index', 'scene_id', 'native_scene_sha256')):
            raise ValueError('Scene evidence differs from the fixed native population')
    return record


def evaluate_candidate(model, dataset, device, checkpoint_sha256, step, output_dir,
                       *, engine=None, pop=None, evaluator=None, best_hqnr=-math.inf,
                       data_sha256=None, stopcheck=None):
    """One resumable candidate boundary; JQM only for running winners/exact50K.

    Every final winner necessarily was a running winner.  Its JQM therefore
    comes from the very same native prediction batch, without 50 extra passes.
    """
    from fh12.evaluation import FRMetrics, infer
    started = time.monotonic()
    _check(stopcheck)
    with preserve_rng():
        engine = engine or FRMetrics(dataset)
        pop = pop or population(dataset, 'fr')
        evaluator = evaluator or evaluator_identity(engine.wald)
        sr, _ = infer(model, _CheckedDataset(dataset, stopcheck), device, batch_size=1)
        _check(stopcheck)
        fr = engine(sr)
        if fr['hqnr'] > best_hqnr or step == 50000:
            extras = _jqm(sr, dataset, stopcheck)
            for row, extra in zip(fr['per_scene'], extras):
                row.update(extra)
            fr.update(jqm=float(np.mean([x['jqm'] for x in extras])), jqm_variant=JQM_VARIANT)
        fr.update(masking=False, alignment=False, hqnr_variant='raw-original', aggregation='mean_per_scene_HQNR')
        fr = _attach(fr, pop, checkpoint_sha256)
    record = dict(update=step, checkpoint_sha256=checkpoint_sha256, fr=fr,
                  metadata=dict(protocol=PROTOCOL, protocol_sha256=object_sha(PROTOCOL),
                                checkpoint_sha256=checkpoint_sha256, data_sha256=data_sha256,
                                population=pop, evaluator=evaluator, optimizer_updates_during_evaluation=0),
                  seconds=time.monotonic()-started)
    validate_candidate(record)
    atomic_json(Path(output_dir)/'fr.json', record)
    return record


def evaluate_rr(model, dataset, device, checkpoint_sha256, *, pop=None, stopcheck=None):
    from fh12.evaluation import infer, native_gt, rr_metrics
    started = time.monotonic()
    with preserve_rng():
        pop = pop or population(dataset, 'rr')
        sr, _ = infer(model, _CheckedDataset(dataset, stopcheck), device, batch_size=1)
        _check(stopcheck)
        gt = native_gt(dataset)
        report = rr_metrics(sr, gt, include_q=True)
        for row, prediction, target in zip(report['per_scene'], sr, gt):
            # Exact historical reporting_extra.evaluation HWC flatten ordering:
            # CHW is algebraically equivalent but changes floating reductions.
            a = np.asarray(prediction.transpose(1, 2, 0)[20:-21, 20:-21], dtype=np.float64).ravel()
            b = np.asarray(target.transpose(1, 2, 0)[20:-21, 20:-21], dtype=np.float64).ravel()
            if a.std() == 0 or b.std() == 0:
                raise ValueError('RR CC undefined for a constant scene')
            row.update(rmse=float(np.sqrt(np.mean((a-b)**2))), cc=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
        for key in ('rmse', 'cc'):
            report[key] = float(np.mean([x[key] for x in report['per_scene']]))
            report['standard_deviation'][key] = float(np.std([x[key] for x in report['per_scene']], ddof=1))
        report = _attach(report, pop, checkpoint_sha256)
    validate_report(report, 'rr', checkpoint_sha256)
    return dict(rr=report, population=pop, seconds=time.monotonic()-started)
