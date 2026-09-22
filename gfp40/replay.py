"""Read-only P40 interventions; no checkpoint is eligible as a main candidate.

Source models are never mutated. Crossed modules are ephemeral independent
copies, and native correction freezes exist only for individual forwards.
"""
from __future__ import annotations

import copy
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from g20.model import state_hash
from gfp40.common import (ROOT, atomic_json, camp, check_deadline, immutable_json,
    object_sha, read, read_config, read_json, resolved_path, source_identity)
from gfp40.diagnostics import native_quality, merge_quality, preserve_runtime, _stats

FIXED_SCENES = (0, 4, 9, 14, 19)
FIXED_CROPS = ((0, 0, 64), (64, 64, 64), (128, 128, 64))
PRIMARY_PAIRS = {'s3': ('A02', 'A03'), 's4': ('B03', 'B04'), 's5': ('C01', 'C02')}


def crossed_model(ctrl, mix, combination):
    if combination not in ('CC', 'CM', 'MC', 'MM'):
        raise ValueError('Registered A/U replay combinations only')
    sources = dict(C=ctrl, M=mix)
    model = copy.deepcopy(sources[combination[1]])
    model.aligner = copy.deepcopy(sources[combination[0]].aligner)
    model.requires_grad_(False).eval()
    if set(map(id, model.parameters())).intersection(
            set(map(id, ctrl.parameters())) | set(map(id, mix.parameters()))):
        raise ValueError('Diagnostic replay shares source parameters')
    return model


def factorial_effects(records):
    if set(records) != {'CC', 'CM', 'MC', 'MM'}:
        raise ValueError('All four interventions required')
    result = {}
    for split, keys in [('fr', ('hqnr', 'd_s', 'd_lambda')), ('rr', ('ergas', 'scc'))]:
        for metric in keys:
            cc, cm, mc, mm = [float(records[k][split][metric]) for k in ('CC', 'CM', 'MC', 'MM')]
            if not np.isfinite([cc, cm, mc, mm]).all():
                raise ValueError('Nonfinite factorial result')
            result[metric] = dict(U_at_CTRL_A=cm-cc, A_at_CTRL_U=mc-cc,
                interaction=mm-cm-mc+cc, total=mm-cc)
    return dict(effects=result, diagnostic_only=True,
        interpretation='module intervention responses, not unique causal attribution of training',
        sign_convention='algebraic metric differences; lower better for E/Ds/Dlambda')


def leave_one_scene_out(summary):
    """Interpretation only; the official full20 aggregation is never replaced."""
    rows = summary['per_scene']
    if len(rows) != 20:
        raise ValueError('All20 scenes required for interpretive leave-one-out')
    keys = [k for k in ('hqnr', 'd_s', 'd_lambda', 'ergas', 'scc') if k in rows[0]]
    result = {}
    for key in keys:
        x = np.asarray([row[key] for row in rows], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite per-scene interpretation')
        result[key] = dict(full20_mean=float(x.mean()),
            leave_one_out_means=((x.sum()-x)/19).tolist(),
            scene_std=float(x.std(ddof=1)))
    return dict(metrics=result, diagnostic_only=True, official_n_scenes=20,
                scene_exclusion_allowed=False)


@torch.no_grad()
def native_split_probe(model, dataset, device='cuda', deadline=None, parent_delta=None):
    if not dataset.has_gt or dataset.augment or dataset.split not in ('val', 'rr'):
        raise ValueError('Native validation or full RR20 split required')
    corrections, quality = [], []
    with preserve_runtime(model):
        model.eval()
        for gt, _lms, ms, lp, pan, _meta in DataLoader(dataset, batch_size=16 if dataset.split == 'val' else 1,
                                                    shuffle=False, num_workers=0):
            check_deadline(deadline)
            gt, ms, lp, pan = [v.to(device) for v in (gt, ms, lp, pan)]
            out = model(pan, ms, lp)
            quality.append(native_quality(out['y'], gt, out['ms_base'], reduction=False))
            corrections.append(out['delta'].detach().cpu())
    current = torch.cat(corrections)
    parent = current if parent_delta is None else torch.as_tensor(parent_delta, dtype=current.dtype)
    if parent.shape != current.shape:
        raise ValueError('Parent correction scene correspondence differs')
    return dict(corrections=current.tolist(), correction_norm=_stats(current.norm(dim=1)),
                drift_from_parent=_stats((current-parent).norm(dim=1)),
                quality=merge_quality(quality), correction_is_ground_truth=False,
                independent_estimator='ESTIMATOR_NOT_AVAILABLE')


@torch.no_grad()
def save_fixed_visuals(model, datasets, output_dir, device='cuda', deadline=None):
    """Fixed linear display transforms, native outputs; no selected pretty crops."""
    from PIL import Image
    from g20.data import canonical_band_indices
    output_dir = Path(output_dir)
    manifest = dict(scene_indices=list(FIXED_SCENES), crops=[list(c) for c in FIXED_CROPS],
        display='fixed normalized[-1,1] to DN[0,1023]; RGB/NIR clip display only',
        error_display='absolute normalized error/2, fixed full DN range',
        edge_display='mean absolute first differences/2, diagnostic only',
        selection='preregistered indices, never picked by score', FR_has_GT=False)
    immutable_json(output_dir / 'visual_protocol.json', manifest)
    with preserve_runtime(model):
        model.eval()
        for split in ('rr', 'fr'):
            dataset = datasets[split]
            b, g, r, nir = canonical_band_indices(dataset.spec.band_order)
            for scene in FIXED_SCENES:
                check_deadline(deadline)
                row = dataset[scene]
                _lms, ms, lp, pan = row[:-1][-4:]
                out = model(pan[None].to(device), ms[None].to(device), lp[None].to(device))
                y = out['y'][0].cpu().numpy()
                gt = row[0].numpy() if split == 'rr' else None
                folder = output_dir / split / f'scene_{scene:02d}'
                folder.mkdir(parents=True, exist_ok=True)
                for crop_id, (top, left, size) in enumerate(FIXED_CROPS):
                    crop = y[:, top:top+size, left:left+size]
                    rgb = np.clip((crop[[r,g,b]].transpose(1,2,0)+1)/2, 0, 1)
                    n = np.clip((crop[nir]+1)/2, 0, 1)
                    edge = np.zeros((size,size), dtype=np.float32)
                    edge[1:,1:] = (np.abs(np.diff(crop,axis=1))[:,:,1:]
                                      +np.abs(np.diff(crop,axis=2))[:,1:,:]).mean(0)/4
                    for label, array in [('rgb',rgb),('nir',n),('edge',np.clip(edge,0,1))]:
                        Image.fromarray(np.rint(array*255).astype(np.uint8)).save(folder/f'crop_{crop_id}_{label}.png')
                    payload = dict(output_normalized=crop, correction=out['delta'][0].cpu().numpy())
                    if gt is not None:
                        target = gt[:, top:top+size, left:left+size]
                        error = np.abs(crop-target).mean(0)/2
                        payload.update(gt_normalized=target, abs_error_normalized=np.abs(crop-target))
                        Image.fromarray(np.rint(np.clip(error,0,1)*255).astype(np.uint8)).save(folder/f'crop_{crop_id}_rr_error.png')
                    np.savez_compressed(folder/f'crop_{crop_id}_raw.npz', **payload)
    return manifest


def _replay_pair(ctrl_config_path, mix_config_path, root=ROOT, device='cuda', deadline=None):
    from g20.data import build_dataset
    from g20.evaluation import FRMetrics, evaluate_model
    from gfp40.assets import load_parent
    from gfp40.plan import validate_config
    from gfp40.postrun import load_candidate
    root = Path(root)
    configs = [read_config(p) for p in (ctrl_config_path, mix_config_path)]
    cases = [validate_config(c, require_bound=True) for c in configs]
    first, second = cases
    if (not all(c.is_ft and c.updates == 20000 for c in cases)
            or first.profile != 'CTRL' or second.profile != 'MIX'
            or any(config['gfp40']['family'] != 'G025' for config in configs)
            or (first.parent_id, first.stream_seed, first.server) != (second.parent_id, second.stream_seed, second.server)):
        raise ValueError('Replay requires the registered same-parent/stream G025 CTRL/MIX20K pair')
    directories = [resolved_path(c['work_dir'], root) for c in configs]
    for wd in directories:
        if not read(wd/'official/summary.json').get('complete'):
            raise ValueError('Official endpoints must be complete before diagnostic replay')
    initial = [read_json(wd/'init_manifest.json')['hashes'] for wd in directories]
    if initial[0] != initial[1]:
        raise ValueError('Cross-parent module swap is forbidden')
    release = source_identity(root)
    models_and_ids = [load_candidate(cfg, wd/'candidates/20000', device, expected_source=release)
                      for cfg, wd in zip(configs, directories)]
    ctrl, mix = [v[0] for v in models_and_ids]
    output = camp(root, first.server)/'replay'/f'{first.case_id}_{second.case_id}'
    context = dict(schema='GFP40_DIAGNOSTIC_REPLAY_v1', pair=[first.case_id, second.case_id],
        source_identity=release, checkpoint_identities=[v[1] for v in models_and_ids],
        parent_id=first.parent_id, source_init=initial[0], diagnostic_only=True,
        candidate_eligible=False, new_parent_eligible=False)
    immutable_json(output/'identity.json', context)
    if (output/'complete.json').exists():
        return read_json(output/'complete.json')
    data = read_json(resolved_path(configs[0]['gfp40']['dataset_manifest'], root))
    if object_sha(data) != models_and_ids[1][1]['data_sha256']:
        raise ValueError('Replay native datasets differ')
    datasets = {split: build_dataset(data, split, root=root) for split in ('val','rr','fr')}
    parent, _ = load_parent(resolved_path(configs[0]['gfp40']['parent_manifest'], root),
                            device, root=root, local_dataset_manifest=data)
    parents = {}
    started = time.monotonic()
    for split in ('val','rr'):
        path = output/f'parent_{split}.json'
        if not path.exists():
            immutable_json(path, native_split_probe(parent, datasets[split], device, deadline))
        parents[split] = read_json(path)
    source_hashes = [state_hash(m.state_dict()) for m in (ctrl,mix,parent)]
    engine, results = FRMetrics(datasets['fr']), {}
    for combination in ('CC','CM','MC','MM'):
        check_deadline(deadline)
        path = output/f'{combination}_metrics.json'
        model = crossed_model(ctrl,mix,combination).to(device)
        if not path.exists():
            value = evaluate_model(model,datasets,device,engine,deadline,include_q=True,with_val=True)
            value.update(diagnostic_only=True,candidate_eligible=False,
                A_source=combination[0],U_source=combination[1])
            immutable_json(path,value)
        results[combination] = read_json(path)
        if combination in ('CC','MM'):
            for split in ('val','rr'):
                detail_path = output/f'{combination}_{split}_quality.json'
                if not detail_path.exists():
                    immutable_json(detail_path,native_split_probe(model,datasets[split],device,deadline,
                        parents[split]['corrections']))
            save_fixed_visuals(model,datasets,output/'visuals'/combination,device,deadline)
        del model
    if source_hashes != [state_hash(m.state_dict()) for m in (ctrl,mix,parent)]:
        raise ValueError('Replay mutated immutable source modules')
    report = dict(context, complete=True, factorial=factorial_effects(results),
        leave_one_out={combo:{split:leave_one_scene_out(record[split]) for split in ('rr','fr')}
                      for combo,record in results.items()},
        independent_estimator='ESTIMATOR_NOT_AVAILABLE', seconds=time.monotonic()-started)
    immutable_json(output/'complete.json', report)
    return report


def replay_pair(ctrl_config_path, mix_config_path, root=ROOT, device='cuda', deadline=None):
    """Account for failed/partial replay attempts too; no hidden setup debt."""
    from gfp40.plan import validate_config
    first = validate_config(read_config(ctrl_config_path), require_bound=True)
    second = validate_config(read_config(mix_config_path), require_bound=True)
    output = camp(root, first.server)/'replay'/f'{first.case_id}_{second.case_id}'
    cost_path = output/'costs.json'
    costs = read(cost_path, dict(attempts=0, wall_seconds=0., complete=False))
    started = time.monotonic()
    try:
        result = _replay_pair(ctrl_config_path, mix_config_path, root, device, deadline)
        costs['complete'] = bool(result.get('complete'))
        return result
    finally:
        costs['attempts'] += 1
        costs['wall_seconds'] += time.monotonic()-started
        costs['diagnostic_only'] = True
        atomic_json(cost_path, costs)
