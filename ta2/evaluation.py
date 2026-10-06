"""TA2 native, complete-20 evaluation and independently recoverable HQNR selection.

No training- or structural-support mask enters these metrics. Ground truth is
read by the RR metric only, never by the inference adapter. Raw exports precede
clipping; the metric conversion retains the existing DLPan reporting protocol.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

SENSORS = {'WV3': (8, 2047.), 'QB': (4, 2047.), 'GF2': (4, 1023.)}
JQM_VARIANT = 'SRF-substitute (NNLS-normalized), MTF41 phase2, global CMSC v1=0.5; not SIPSA-equivalent'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value, immutable=False):
    path = Path(path)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Conflicting immutable TA2 evidence: ' + str(path))
        return
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        if immutable:
            os.link(temp, path)
        else:
            os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def seal(value):
    return dict(value, payload_sha256=digest(value))


def check_seal(value):
    body = {k: v for k, v in value.items() if k != 'payload_sha256'}
    if value.get('payload_sha256') != digest(body):
        raise ValueError('TA2 evaluation payload checksum mismatch')
    return value


@contextmanager
def preserve_runtime(model):
    """Diagnostics/evaluation must not perturb a resumed training trajectory."""
    modes = [(module, module.training) for module in model.modules()]
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
    old_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        model.eval()
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_tf32


def assert_checkpoint_model(model, checkpoint_path):
    """An arbitrary supplied model cannot borrow the identity of another file."""
    if Path(checkpoint_path).suffix == '.safetensors':
        from safetensors.torch import load_file
        saved = load_file(str(checkpoint_path), device='cpu')
    elif Path(checkpoint_path).suffix == '.pt':
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)['model']
    else:
        raise ValueError('Only tensor-only safetensors or trusted TA2 .pt checkpoints are supported')
    live = model.state_dict()
    if set(saved) != set(live):
        raise ValueError('In-memory model keys differ from evaluated checkpoint')
    for key, tensor in live.items():
        if tensor.shape != saved[key].shape or tensor.dtype != saved[key].dtype or not torch.equal(tensor.detach().cpu(), saved[key]):
            raise ValueError('In-memory model differs from evaluated checkpoint: ' + key)


def hqnr_grid(total_updates):
    if total_updates not in (50000, 100000):
        raise ValueError('Only preregistered fresh50K/fresh100K budgets are valid')
    interval = 1010 if total_updates == 50000 else 2020
    return tuple(interval * i for i in range(1, 50)) + (total_updates,)


def _validate_metric_group(group, keys):
    rows = group.get('per_scene', [])
    if (group.get('n_scenes') != 20 or len(rows) != 20
            or [r.get('scene_index') for r in rows] != list(range(20))):
        raise ValueError('Every candidate requires all original 20 scenes in order')
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all() or not np.isfinite(group[key]):
            raise ValueError('Nonfinite selection metric: ' + key)
        if float(values.mean()) != group[key]:
            raise ValueError('Mean is not the full-precision scene-wise mean: ' + key)


def select_hqnr_max50(records, total_updates):
    """Strict fifty-point selector; ERGAS is deliberately not inspected."""
    by_step = {}
    for record in records:
        step = record['completed_step']
        if step in by_step:
            raise ValueError('Duplicate checkpoint candidate')
        if not record.get('complete') or not record.get('checkpoint_sha256'):
            raise ValueError('Incomplete candidate')
        if record.get('record_type', 'NATIVE') != 'NATIVE':
            raise ValueError('Oracle/counterfactual cannot enter official selection')
        fr = record['fr']
        if (fr.get('reference') != 'original_PAN_and_native_LMS' or fr.get('masking') is not False
                or fr.get('support') != 'full512'):
            raise ValueError('HQNR candidate is not original full-frame native FR20')
        _validate_metric_group(fr, ('hqnr',))
        by_step[step] = record
    if set(by_step) != set(hqnr_grid(total_updates)):
        raise ValueError('HQNR_SELECTION_PENDING: exact 50/50 grid coverage required')
    winner = max(by_step.values(), key=lambda r: (r['fr']['hqnr'], -r['completed_step']))
    return dict(status='HQNR_SELECTION_COMPLETE', selector='HQNR_MAX50',
                completed_step=winner['completed_step'], checkpoint_sha256=winner['checkpoint_sha256'],
                hqnr=winner['fr']['hqnr'], candidates_complete=50, candidates_expected=50,
                test_aware=True, selection_split='FR20', independent_test=False,
                tie_break='earliest_completed_step_at_equal_full_precision_HQNR',
                exact_final_step=total_updates, exact_final_sha256=by_step[total_updates]['checkpoint_sha256'])


class EvaluationLedger:
    """Evaluation recovery never requires rerunning an optimizer update."""
    def __init__(self, directory, identity, total_updates):
        self.directory = Path(directory)
        self.total_updates = int(total_updates)
        self.grid = hqnr_grid(self.total_updates)
        self.identity = dict(identity)
        atomic_json(self.directory / 'identity.json', dict(identity=self.identity, grid=list(self.grid)), immutable=True)

    def register(self, report):
        check_seal(report)
        step = report['completed_step']
        if step not in self.grid and not (self.total_updates == 100000 and step == 50000):
            raise ValueError('Unregistered evaluation step')
        if report.get('context') != self.identity or not report.get('complete'):
            raise ValueError('Wrong or incomplete evaluation identity')
        _validate_metric_group(report['fr'], ('hqnr', 'd_s', 'd_lambda'))
        _validate_metric_group(report['rr'], ('scc', 'ergas'))
        atomic_json(self.directory / f'step_{step:06d}.json', report, immutable=True)
        return self.status()

    add = register

    def records(self):
        records = []
        for path in sorted(self.directory.glob('step_*.json')):
            value = check_seal(json.loads(path.read_text()))
            if value['context'] != self.identity:
                raise ValueError('Mixed source/config/data ledger')
            records.append(value)
        return records

    def status(self):
        candidates = [r for r in self.records() if r['completed_step'] in self.grid]
        missing = sorted(set(self.grid) - {r['completed_step'] for r in candidates})
        if missing:
            return dict(status='HQNR_SELECTION_PENDING', candidates_complete=len(candidates),
                        candidates_expected=50, missing_steps=missing, training_restart_required=False)
        selected = select_hqnr_max50(candidates, self.total_updates)
        atomic_json(self.directory / 'selection.json', selected, immutable=True)
        return selected


def evaluator_identity(root=None):
    import scipy
    import skimage
    from tools.metrics.eval_fr import load_dlpan
    root = Path(root or Path(__file__).resolve().parents[1])
    wald = load_dlpan(os.environ.get('PANCRAFTER_DLPAN', str(root.parent / 'DLPan-Toolbox')))
    files = ('ta2/evaluation.py', 'tools/eval_dlpan.py', 'tools/metrics/eval_rr.py',
             'tools/metrics/eval_fr.py', 'tools/metrics/q2n.py', 'tools/metrics/jqm.py')
    hashes = {p: file_sha(root / p) for p in files}
    return dict(files=hashes, external_wald_sha256=file_sha(wald.__file__),
                torch=str(torch.__version__), numpy=np.__version__, scipy=scipy.__version__, skimage=skimage.__version__,
                precision='FP32; TF32 disabled', prediction_export='unclipped float32 DN; no rounding',
                metric_conversion='clip DN [0,maxDN], no rounding except internal Q2n',
                rr_protocol='all20 dim_cut21 [20:-21], Q32, SCC2Dzero, PSNRglobal, SSIMgauss11',
                fr_protocol='all20 full512 originalPAN/nativeLMS; scene-wise HQNR mean; no alignment mask',
                jqm_variant=JQM_VARIANT)


def _summary(rows, keys, **metadata):
    result = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    if not np.isfinite(list(result.values())).all():
        raise FloatingPointError('Nonfinite official quality metric')
    return dict(result, n_scenes=len(rows), per_scene=rows,
                standard_deviation={key: float(np.std([r[key] for r in rows], ddof=1)) for key in keys}, **metadata)


def rr_scene(prediction, truth, sensor):
    from tools.eval_dlpan import scc_dlpan, psnr_global, ssim_skimage
    from tools.metrics.eval_rr import sam, ergas
    from tools.metrics.q2n import q2n
    bands, peak = SENSORS[sensor]
    if prediction.shape != (bands, 256, 256) or truth.shape != prediction.shape:
        raise ValueError('RR must be native band-count x 256 x 256')
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise FloatingPointError('Nonfinite RR arrays, including discarded border')
    p = np.clip(prediction, 0, peak).transpose(1, 2, 0)[20:-21, 20:-21].astype(np.float64)
    t = np.asarray(truth).transpose(1, 2, 0)[20:-21, 20:-21].astype(np.float64)
    if p.std() == 0 or t.std() == 0:
        raise ValueError('Undefined constant-scene Pearson CC')
    return dict(ergas=ergas(p, t), sam=sam(p, t), scc=scc_dlpan(p, t),
                psnr=psnr_global(p, t, peak), ssim=ssim_skimage(p, t, peak),
                **{'q8' if bands == 8 else 'q4': float(q2n(t, p, 32, 32)[0])},
                rmse=float(np.sqrt(np.mean((p-t)**2))), cc=float(np.corrcoef(p.ravel(), t.ravel())[0, 1]))


def fr_scene(prediction, pan, lms, ms, sensor, wald):
    from tools.metrics.eval_fr import d_lambda_k, d_s
    from tools.metrics.jqm import jqm
    bands, peak = SENSORS[sensor]
    if (prediction.shape != (bands, 512, 512) or lms.shape != prediction.shape
            or pan.shape != (1, 512, 512) or ms.shape != (bands, 128, 128)):
        raise ValueError('FR requires native full512 PAN/LMS and128 LRMS')
    if any(not np.isfinite(a).all() for a in (prediction, pan, lms, ms)):
        raise FloatingPointError('Nonfinite native FR arrays')
    p = np.clip(prediction, 0, peak).transpose(1, 2, 0).astype(np.float64)
    l = lms.transpose(1, 2, 0).astype(np.float64)
    original_pan = pan[0].astype(np.float64)
    dl = d_lambda_k(p, l, sensor.lower(), 4, 32, wald)
    ds = d_s(p, l, original_pan, 4, 32, wald)
    jq = jqm(p, ms.transpose(1, 2, 0), original_pan, sensor, ratio=4, R=peak, lpf='mtf', window=None, v1=.5)
    return dict(hqnr=float((1-dl)*(1-ds)), d_s=float(ds), d_lambda=float(dl), jqm=float(jq['JQM']),
                jqm_weight_source=jq['w_source'], jqm_weights=jq['w'])


def _arrays(dataset):
    return dataset if isinstance(dataset, dict) else dataset.arrays


def _save_array(path, array):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.save(stream, np.asarray(array, dtype=np.float32), allow_pickle=False)
            stream.flush(); os.fsync(stream.fileno())
        if path.exists():
            if file_sha(temp) != file_sha(path):
                raise ValueError('Raw prediction changed on evaluation recovery')
        else:
            os.link(temp, path)
    finally:
        os.unlink(temp)
    return file_sha(path)


def evaluate_checkpoint(model, datasets, output_dir, *, sensor, checkpoint_path,
                        completed_step, context=None, device='cuda', predict=None,
                        export_raw=False):
    """Adapter: predict(model, pan, ms, lms, split, index) -> prediction/correction.

    It receives only the three normalized observations, never GT. For offline
    controls the runner closes over their PAN/LMS-only, split-specific shifts.
    Candidate metrics may omit pixel dumps; selected/final exports use a
    separate directory with export_raw=True (no destructive report upgrade).
    """
    sensor = sensor.upper()
    bands, peak = SENSORS[sensor]
    from tools.metrics.eval_fr import load_dlpan
    root = Path(__file__).resolve().parents[1]
    checkpoint_sha = file_sha(checkpoint_path)
    assert_checkpoint_model(model, checkpoint_path)
    context = dict(context or {})
    evaluator=evaluator_identity(root)
    pinned=context.get('runtime',{}).get('evaluator')
    if pinned is not None and pinned != evaluator:
        raise ValueError('Pinned evaluator/runtime source changed since campaign preflight')
    identity = dict(context=context, sensor=sensor, checkpoint_sha256=checkpoint_sha,
                    completed_step=int(completed_step), evaluator=evaluator, export_raw=bool(export_raw))
    output_dir = Path(output_dir)
    atomic_json(output_dir / 'identity.json', identity, immutable=True)
    report_path = output_dir / 'report.json'
    if report_path.exists():
        old = check_seal(json.loads(report_path.read_text()))
        if old.get('identity') != identity:
            raise ValueError('Cached evaluation uses different identities')
        for rel, expected in old.get('file_hashes', {}).items():
            if file_sha(output_dir / rel) != expected:
                raise ValueError('Evaluation artifact changed: ' + rel)
        return old
    wald = load_dlpan(os.environ.get('PANCRAFTER_DLPAN', str(root.parent / 'DLPan-Toolbox')))
    adapter = predict or (lambda m, p, ms, l, split, index: m(p, ms, l))
    started = time.monotonic(); rows = {}; files = {}
    with preserve_runtime(model):
        for split in ('fr', 'rr'):
            arrays = _arrays(datasets[split])
            if any(len(arrays[k]) != 20 for k in ('pan', 'ms', 'lms')):
                raise ValueError('Exactly20 complete native scenes required')
            rows[split] = []
            for index in range(20):
                scene_path = output_dir / split / f'scene_{index:02d}.json'
                if scene_path.exists():
                    row = check_seal(json.loads(scene_path.read_text()))
                    if row.get('evaluation_identity_sha256') != digest(identity):
                        raise ValueError('Scene receipt identity mismatch')
                    for rel, expected in row.get('file_hashes', {}).items():
                        if file_sha(output_dir / rel) != expected:
                            raise ValueError('Raw prediction export was modified')
                    rows[split].append(row)
                    files[str(scene_path.relative_to(output_dir))] = file_sha(scene_path)
                    files.update(row.get('file_hashes', {}))
                    continue
                inputs = [torch.from_numpy(np.array(arrays[k][index:index+1], dtype=np.float32, copy=True)).to(device)
                          .mul(2 / peak).sub(1) for k in ('pan', 'ms', 'lms')]
                with torch.no_grad():
                    out = adapter(model, *inputs, split, index)
                    pred, correction = out['prediction'], out['correction']
                    if (pred.shape != inputs[2].shape or correction.shape != (1, 2)
                            or not torch.isfinite(pred).all() or not torch.isfinite(correction).all()):
                        raise ValueError('Invalid/nonfinite inference output')
                    raw = ((pred.float()[0] + 1) * (peak / 2)).cpu().numpy()
                    c = correction[0].cpu().tolist()
                if split == 'rr':
                    metrics = rr_scene(raw, np.asarray(arrays['gt'][index]), sensor)
                else:
                    metrics = fr_scene(raw, *(np.asarray(arrays[k][index]) for k in ('pan', 'lms', 'ms')), sensor, wald)
                artifact_hashes = {}
                if export_raw:
                    rel = f'{split}/unclipped_DN/scene_{index:02d}.npy'
                    artifact_hashes[rel] = _save_array(output_dir / rel, raw)
                row = seal(dict(metrics, scene_index=index, correction_dy=float(c[0]), correction_dx=float(c[1]),
                                raw_prediction_sha256=hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
                                raw_min=float(raw.min()), raw_max=float(raw.max()),
                                file_hashes=artifact_hashes, evaluation_identity_sha256=digest(identity)))
                atomic_json(scene_path, row, immutable=True)
                files[str(scene_path.relative_to(output_dir))] = file_sha(scene_path)
                files.update(artifact_hashes); rows[split].append(row)
    if file_sha(checkpoint_path) != checkpoint_sha:
        raise ValueError('Checkpoint changed during evaluation')
    if evaluator_identity(root) != evaluator:
        raise ValueError('Evaluator source/runtime changed during evaluation')
    assert_checkpoint_model(model, checkpoint_path)
    rr_keys = ('ergas', 'scc', 'sam', 'psnr', 'ssim', 'q8' if bands == 8 else 'q4', 'rmse', 'cc')
    report = seal(dict(schema='TA2_NATIVE_EVALUATION_v1', record_type='NATIVE', complete=True,
                       completed_step=int(completed_step), checkpoint_sha256=checkpoint_sha,
                       context=context, identity=identity, file_hashes=files, elapsed_seconds=time.monotonic()-started,
                       rr=_summary(rows['rr'], rr_keys, crop='20:-21', q_block=32),
                       fr=_summary(rows['fr'], ('hqnr', 'd_s', 'd_lambda', 'jqm'), masking=False,
                                   reference='original_PAN_and_native_LMS', support='full512', jqm_variant=JQM_VARIANT),
                       rr_fr_same_checkpoint=True, gt_in_inference=False, raw_exported=bool(export_raw)))
    atomic_json(report_path, report, immutable=True)
    print(f"[TA2] step={completed_step} HQNR={report['fr']['hqnr']:.9f} SCC={report['rr']['scc']:.9f} ERGAS={report['rr']['ergas']:.9f}", flush=True)
    return report
