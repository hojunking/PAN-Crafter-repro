"""Synchronous in-memory official metrics and strictly qualified persistence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from pa.warp import warp_pan
from ta2.evaluation import (JQM_VARIANT, SENSORS, _save_array, atomic_json, digest,
                            evaluator_identity, file_sha, fr_scene, rr_scene)
from .plan import METHOD_CONTRACT, qualify_evaluation, validate_method_config
from .training import model_state_hash


def state_hash(stage):
    """Logical A/T tensor digest, not a file SHA or an optimizer/RNG digest."""
    return model_state_hash(stage)


def identity(trainer, config, sources, evaluator_revision):
    validate_method_config(config)
    return dict(**METHOD_CONTRACT, run_id=config['run_id'], attempt=config['attempt'], seed=config['seed'],
                completed_update=trainer.updates, model_state_sha256=state_hash(trainer.stage),
                inference_policy_digest=digest({'baseline': config['baseline_resolved'],
                    'method': METHOD_CONTRACT, 'aligner_head': trainer.stage.aligner.conditioning_manifest(),
                    'metrics': 'native-original-PAN-LMS-no-mask'}),
                evaluator_revision=evaluator_revision,
                prior_rule_digest=sources.mapping_sha256)


def official_evaluation(trainer, config, sources, *, device, evaluator_revision,
                        evaluator_pin, metric_functions=None, on_scene=None, deadline_unix=None):
    """No checkpoint/raw/image writes. Returned CPU arrays live only until gate."""
    if metric_functions is None:
        from tools.metrics.eval_fr import load_dlpan
        root = Path(__file__).resolve().parents[1]
        if evaluator_identity(root) != evaluator_pin:
            raise ValueError('Evaluator changed since admission')
        wald = load_dlpan(os.environ.get('PANCRAFTER_DLPAN', str(root.parent / 'DLPan-Toolbox')))
        metric_functions = {'rr': lambda p, a: rr_scene(p, a['gt'], sources.dataset),
                            'fr': lambda p, a: fr_scene(p, a['pan'], a['lms'], a['ms'], sources.dataset, wald)}
    started = time.monotonic()
    record = {'full_method_eligible': True, 'rr': {}, 'fr': {}, 'test_aware': True,
              'jqm_variant': JQM_VARIANT, 'eval_protocol_id': digest(evaluator_pin),
              'official_reference': 'original_PAN/native_LMS; FR full512 unmasked; RR dim_cut21 all20'}
    raw_outputs = {}
    with trainer.evaluation_context(), torch.no_grad():
        record['state_before'] = identity(trainer, config, sources, evaluator_revision)
        for split in ('rr', 'fr'):
            tick = time.monotonic(); rows = []
            raw_outputs[split] = []
            group = split + '_eval'
            for index, scene_id in enumerate(sources.expected_ids(group)):
                if deadline_unix is not None and time.time() >= deadline_unix:
                    raise TimeoutError('Campaign deadline reached; incomplete metrics cannot qualify')
                batch = sources.batch(group, [index], device=device)
                out = trainer.stage.native_prediction(batch)
                raw = ((out['y'][0].float()+1)*(sources.peak/2)).cpu().numpy()
                arrays = sources.arrays(group, index)
                metrics = metric_functions[split](raw, arrays)
                keys = ('ergas', 'scc', 'sam', 'psnr', 'ssim', 'q8' if sources.bands == 8 else 'q4', 'rmse', 'cc') if split == 'rr' else ('hqnr', 'd_s', 'd_lambda', 'jqm')
                rename = {'d_s': 'D_s', 'd_lambda': 'D_lambda'}
                values = {rename.get(k, k.upper()): float(metrics[k]) for k in keys}
                if not np.isfinite(list(values.values())).all():
                    raise FloatingPointError('Nonfinite official metrics; never replace by zero')
                correction = out['correction'][0].detach().cpu().tolist()
                rows.append({'scene_id': scene_id, 'metrics': values, 'correction_dy_dx': correction,
                    'raw_prediction_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                    'raw_min': float(raw.min()), 'raw_max': float(raw.max()),
                    'jqm_weights': np.asarray(metrics.get('jqm_weights', [])).tolist(),
                    'jqm_weight_source': metrics.get('jqm_weight_source')})
                if on_scene is not None:
                    on_scene({'branch':split,'identity':record['state_before'],**rows[-1]})
                raw_outputs[split].append(raw)
            record[split] = {'per_scene': rows,
                'metrics': {k: float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']},
                'std': {k: float(np.std([r['metrics'][k] for r in rows], ddof=1)) for k in rows[0]['metrics']},
                'elapsed_seconds': time.monotonic()-tick}
            record[split + '_state'] = identity(trainer, config, sources, evaluator_revision)
        record['state_after'] = identity(trainer, config, sources, evaluator_revision)
    if evaluator_pin and metric_functions and evaluator_revision != 'TEST':
        if evaluator_identity() != evaluator_pin:
            raise ValueError('Evaluator changed during evaluation')
    record['elapsed_seconds'] = time.monotonic()-started
    record['qualification'] = qualify_evaluation(record, server=config['server'],
        expected_rr_ids=sources.expected_ids('rr_eval'), expected_fr_ids=sources.expected_ids('fr_eval'))
    print(f"[Teacher48] step={trainer.updates} HQNR={record['fr']['metrics']['HQNR']:.6f} "
          f"SCC={record['rr']['metrics']['SCC']:.6f} ERGAS={record['rr']['metrics']['ERGAS']:.6f}", flush=True)
    return record, raw_outputs


def _atomic_checkpoint(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Qualified checkpoint path already exists; do not overwrite')
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.qualified-')
    try:
        with os.fdopen(fd, 'wb') as f:
            torch.save(payload, f); f.flush(); os.fsync(f.fileno())
        # Only our just-produced local serialization is loaded; never an
        # untrusted supplied pickle. Verify every A/T tensor before publishing.
        loaded = torch.load(temporary, map_location='cpu', weights_only=False)
        for role in ('aligner', 'teacher'):
            for key, tensor in payload[role].items():
                if not torch.equal(tensor.cpu(), loaded[role][key]):
                    raise ValueError('Qualified checkpoint round-trip failed')
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return file_sha(path)


def retain_qualified(root, trainer, config, sources, evaluation, raw_outputs, *, stream_states, provenance):
    """Caller must durably journal evaluation BEFORE invoking this writer."""
    validate_method_config(config)
    gate = qualify_evaluation(evaluation, server=config['server'],
        expected_rr_ids=sources.expected_ids('rr_eval'), expected_fr_ids=sources.expected_ids('fr_eval'))
    if not gate['joint_pass']:
        raise ValueError('STRICT_TARGET_ONLY: unqualified weights/images cannot be persisted')
    ident = evaluation['state_after']
    if state_hash(trainer.stage) != ident['model_state_sha256'] or trainer.updates != ident['completed_update']:
        raise ValueError('Live state moved after same-checkpoint evaluation')
    path = Path(root) / config['run_id'] / f"attempt_{config['attempt']}" / f"step_{ident['completed_update']}" / ident['model_state_sha256']
    path.mkdir(parents=True, exist_ok=True)
    payload = {'aligner': trainer.stage.aligner.state_dict(), 'teacher': trainer.stage.teacher.state_dict(),
               'training_state': trainer.snapshot_state(), 'data_stream_states': stream_states,
               'config': config, 'identity': ident, 'provenance': provenance}
    sha = _atomic_checkpoint(path/'teacher_aligner.pt', payload)
    # Save all provenance needed to re-bind the exact inference scene priors.
    atomic_json(path/'source_manifest.json', sources.manifest, immutable=True)
    atomic_json(path/'metrics.json', evaluation, immutable=True)
    manifest = {'file_sha256': sha, 'model_state_sha256': ident['model_state_sha256'],
                'identity': ident, 'qualification': gate, 'provenance': provenance,
                'config': config, 'status': 'QUAL_PENDING', 'raw_exporter': 'ta2.evaluation._save_array',
                'display': config['baseline_resolved']['qual_export']}
    atomic_json(path/'model_manifest.json', manifest)
    try:
        files = export_qualified(path/'qual', sources, evaluation, raw_outputs, config['baseline_resolved']['qual_export'])
        manifest.update(status='QUAL_COMPLETE', qualitative_files=files)
    except Exception as exc:
        manifest.update(status='QUAL_PENDING', export_error=f'{type(exc).__name__}: {exc}')
    atomic_json(path/'model_manifest.json', manifest)
    return {'artifact_path': str(path), 'checkpoint_file_sha256': sha, **manifest}


def export_qualified(path, sources, evaluation, raw_outputs, settings):
    """All-band unclipped DN + fixed-range display, same snapshots as metrics."""
    from PIL import Image
    path = Path(path); files = {}
    rgb = settings['rgb_indices']
    if settings['display_range_DN'] != [0, sources.peak] or settings['roi'] != 'FULL_SCENE':
        raise ValueError('This exporter requires explicitly pinned full-scene, fixed sensor-DN display')
    def png(name, array):
        target = path / name; target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.round(np.clip(array, 0, 1)*255).astype(np.uint8)).save(target)
        files[name] = file_sha(target)
    def rgb_view(a):
        return a[list(rgb)].transpose(1, 2, 0)/sources.peak
    def edges(a):
        a = a.mean(0) if a.ndim == 3 else a
        dy, dx = np.gradient(a/sources.peak)
        return np.clip(np.hypot(dy, dx)*settings['edge_gain'], 0, 1)
    for split in ('rr', 'fr'):
        rows = evaluation[split]['per_scene']
        if len(raw_outputs[split]) != len(rows):
            raise ValueError('Missing qualified per-scene raw predictions')
        for i, (raw, row) in enumerate(zip(raw_outputs[split], rows)):
            if hashlib.sha256(raw.tobytes()).hexdigest() != row['raw_prediction_sha256']:
                raise ValueError('Raw output differs from evaluated snapshot')
            arrays = sources.arrays(split+'_eval', i)
            prefix = f'{split}/scene_{i:02d}'
            name = prefix+'/prediction_unclipped_DN.npy'
            files[name] = _save_array(path/name, raw)
            png(prefix+'/prediction.png', rgb_view(raw))
            reference = arrays['gt'] if split == 'rr' else arrays['lms']
            png(prefix+('/GT.png' if split == 'rr' else '/native_LMS.png'), rgb_view(reference))
            png(prefix+'/MS.png', rgb_view(arrays['ms']))
            png(prefix+'/PAN.png', np.repeat((arrays['pan'][0]/sources.peak)[..., None], 3, axis=2))
            if split == 'rr':
                png(prefix+'/absolute_error.png', rgb_view(np.abs(raw-reference))*settings['error_gain'])
            pan = torch.from_numpy(arrays['pan'])[None]
            corrected = warp_pan(pan, pan.new_tensor([row['correction_dy_dx']]))[0].numpy()
            png(prefix+'/corrected_PAN.png', np.repeat((corrected[0]/sources.peak)[..., None], 3, axis=2))
            refedge = edges(reference[list(rgb)])
            for label, p in (('native', arrays['pan']), ('corrected', corrected)):
                overlay = np.stack((edges(p), refedge, np.zeros_like(refedge)), axis=2)
                png(prefix+f'/{label}_edge_overlay.png', overlay)
    atomic_json(path/'files.json', files, immutable=True)
    return files
