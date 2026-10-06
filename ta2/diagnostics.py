"""Measured TA2 P01--P08 diagnostics, kept outside official FR selection.

Callbacks make intervention semantics explicit: no GT or resized diagnostic
reference is silently introduced into the native inference function.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pa.warp import warp_pan, warp_support_mask
from ta2.evaluation import atomic_json, digest, seal

PROBE_SEED = 261006900


def response_grid(seed=PROBE_SEED):
    rows = []
    for probe, directions in (('AXIS16', ((1, 0), (-1, 0), (0, 1), (0, -1))),
                              ('DIAG16', ((1, 1), (1, -1), (-1, 1), (-1, -1)))):
        for radius in (.25, .5, 1., 2.):
            for direction in directions:
                v = np.asarray(direction) * (radius / np.linalg.norm(direction))
                rows.append(dict(probe=probe, radius=radius, epsilon=v.tolist(), ood=False))
    rng = np.random.Generator(np.random.PCG64(seed))
    radii = 2 * np.sqrt(rng.uniform(size=64)); angle = rng.uniform(0, 2 * np.pi, size=64)
    for r, a in zip(radii, angle):
        rows.append(dict(probe='DISK64_HELD', radius=float(r), epsilon=[float(r*np.sin(a)), float(r*np.cos(a))], ood=False))
    for radius in (4., 8.):
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rows.append(dict(probe='OOD_AXIS', radius=radius, epsilon=[radius*dy, radius*dx], ood=True))
    return [dict(row, offset_index=i) for i, row in enumerate(rows)]


def response_metrics(delta, epsilon):
    delta, epsilon = np.asarray(delta, dtype=np.float64), np.asarray(epsilon, dtype=np.float64)
    if delta.shape != epsilon.shape or delta.ndim != 2 or delta.shape[1] != 2 or not len(delta):
        raise ValueError('Response arrays must be equal nonempty Nx2 (dy,dx)')
    if not np.isfinite(delta).all() or not np.isfinite(epsilon).all():
        raise ValueError('Nonfinite response')
    numerator = float(-np.sum(delta * epsilon)); denominator = float(np.sum(epsilon**2))
    residual = delta + epsilon
    out = dict(gain=None if denominator == 0 else numerator/denominator,
               numerator=numerator, denominator=denominator,
               component_mae=float(np.abs(residual).mean()), epe=float(np.linalg.norm(residual, axis=1).mean()), n=len(delta))
    for j, axis in enumerate(('y', 'x')):
        num, den = float(-np.sum(delta[:, j]*epsilon[:, j])), float(np.sum(epsilon[:, j]**2))
        out.update({f'gain_{axis}': None if den == 0 else num/den,
                    f'numerator_{axis}': num, f'denominator_{axis}': den,
                    f'mae_{axis}': float(np.abs(residual[:, j]).mean())})
    return out


def _cluster_interval(rows, seed=PROBE_SEED, repeats=1000):
    clusters = sorted({r['scene_index'] for r in rows})
    if len(clusters) < 2:
        return dict(gain_ci95=None, reason='fewer_than_two_scene_clusters', n_clusters=len(clusters))
    totals = np.asarray([[sum(-np.dot(r['delta'], r['epsilon']) for r in rows if r['scene_index'] == c),
                          sum(np.dot(r['epsilon'], r['epsilon']) for r in rows if r['scene_index'] == c)] for c in clusters])
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = totals[rng.integers(len(totals), size=(repeats, len(totals)))].sum(1)
    valid = samples[:, 1] > 0
    values = samples[valid, 0]/samples[valid, 1]
    return dict(gain_ci95=np.quantile(values, [.025, .975]).tolist() if len(values) else None,
                n_clusters=len(clusters), bootstrap_repeats=repeats, cluster_unit='scene_index',
                geographic_independence='unverified', training_seed_std=False)


def summarize_responses(rows):
    """Never pool held-disk/OOD radii with in-range axis or diagonal probes."""
    output = []
    keys = sorted({(r['split'], r['probe'], r['mode'], r['measurement_domain'], r['support_scope']) for r in rows})
    for key in keys:
        group = [r for r in rows if (r['split'], r['probe'], r['mode'], r['measurement_domain'], r['support_scope']) == key]
        radii = sorted({r['radius'] for r in group}) if key[1] != 'DISK64_HELD' else []
        for radius in [None] + radii:
            subset = [r for r in group if radius is None or r['radius'] == radius]
            value = response_metrics([r['delta'] for r in subset], [r['epsilon'] for r in subset])
            output.append(dict(value, split=key[0], probe=key[1], mode=key[2], measurement_domain=key[3],
                               support_scope=key[4], radius=radius, ood=key[1] == 'OOD_AXIS',
                               **_cluster_interval(subset)))
    return output


@torch.no_grad()
def measure_responses(correction, samples, *, split, mode='REESTIMATE_CURRENT_INPUT',
                      grid=None, measurement_domain='NATIVE_PATCH', support_scope='normal'):
    """correction(pan,lms,index,mode) returns correction in current PAN pixels."""
    rows = []
    for index, p, l in samples:
        c0 = correction(p, l, index, mode).detach()
        if c0.shape != (1, 2):
            raise ValueError('Response probe operates on batch1 scenes')
        for probe in grid or response_grid():
            eps = p.new_tensor(probe['epsilon']).reshape(1, 2)
            pe = warp_pan(p, eps)
            ce = c0 if mode == 'HOLD_NATIVE_CACHE' else correction(pe, l, index, mode).detach()
            delta = (ce-c0)[0].cpu().tolist()
            rows.append(dict(probe, split=split, scene_index=int(index), mode=mode,
                             measurement_domain=measurement_domain, support_scope=support_scope,
                             native_c=c0[0].cpu().tolist(), shifted_c=ce[0].cpu().tolist(), delta=delta,
                             clean_support_fraction=float(warp_support_mask(*p.shape[-2:], eps).float().mean())))
    return dict(record_type='RESPONSE', rows=rows, summary=summarize_responses(rows), independent_training_seeds=0)


def correction_statistics(corrections):
    c = np.asarray(corrections, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2 or not len(c) or not np.isfinite(c).all():
        raise ValueError('Finite nonempty Nx2 correction required')
    norms = np.linalg.norm(c, axis=1)
    return dict(n=len(c), mean_vector=c.mean(0).tolist(), std_vector=c.std(0, ddof=1).tolist() if len(c)>1 else None,
                mean_norm=float(norms.mean()), norm_mean=float(np.linalg.norm(c.mean(0))),
                p95_norm=float(np.quantile(norms, .95)), median_vector=np.median(c, axis=0).tolist())


@torch.no_grad()
def native_geometry(pan, reference, correction, *, estimator, target='LMS', split='fr', max_dn=1.):
    """Residual is independently estimated on the actually warped PAN."""
    if target == 'GT' and split == 'fr':
        return dict(status='NOT_APPLICABLE', reason='FR_has_no_GT', physical_displacement_ground_truth=False)
    warped = warp_pan(pan, correction)
    before = pan[0].cpu().numpy(); after = warped[0].cpu().numpy(); ref = reference[0].cpu().numpy()
    rows = {}
    for descriptor in ('gradient', 'intensity'):
        kwargs = dict(descriptor=descriptor, target=target, max_dn=max_dn, split=split,
                      oracle=target == 'GT' and split != 'train')
        original = estimator(before, ref, **kwargs)
        residual = estimator(after, ref, **kwargs)
        rows[descriptor] = dict(before=original, after=residual,
                                before_epe=float(np.linalg.norm(original['shift'])),
                                after_epe=float(np.linalg.norm(residual['shift'])),
                                valid_pair=bool(original['valid'] and residual['valid']))
    return dict(status='MEASURED', target=target, split=split, correction=correction[0].cpu().tolist(),
                proxies=rows, residual_reestimated_after_actual_warp=True, physical_displacement_ground_truth=False,
                disagreement_before=float(np.linalg.norm(np.asarray(rows['gradient']['before']['shift']) - rows['intensity']['before']['shift'])),
                disagreement_after=float(np.linalg.norm(np.asarray(rows['gradient']['after']['shift']) - rows['intensity']['after']['shift'])))


def counterfactual_corrections(current, train_corrections):
    current, train = np.asarray(current), np.asarray(train_corrections)
    if current.ndim != 2 or train.ndim != 2 or current.shape[1] != 2 or train.shape[1] != 2 or len(current) < 2 or not len(train):
        raise ValueError('Counterfactuals require train-derived constants and at least two inference images')
    return dict(current=current.copy(), train_mean=np.broadcast_to(train.mean(0), current.shape).copy(),
                train_median=np.broadcast_to(np.median(train, axis=0), current.shape).copy(),
                shuffled=np.roll(current, -1, axis=0).copy(), zero=np.zeros_like(current))


@torch.no_grad()
def measure_counterfactuals(samples, current, train_corrections, *, predict_forced, quality):
    """quality(index,pred) may access RR GT; predict_forced(index,p,l,c) may not."""
    output = []
    for policy, corrections in counterfactual_corrections(current, train_corrections).items():
        rows = []
        for j, (index, p, l) in enumerate(samples):
            c = p.new_tensor(corrections[j:j+1])
            y = predict_forced(index, p, l, c)
            rows.append(dict(scene_index=index, correction=corrections[j].tolist(), **quality(index, y)))
        output.append(dict(record_type='COUNTERFACTUAL', policy=policy, rows=rows,
                           intervention='same_frozen_U; inference_only; not_retrained',
                           constant_source='train_split_only', shuffled_policy='fixed_cyclic_derangement',
                           excluded_from_native_selector=True, independent_training_seeds=0))
    return output


@torch.no_grad()
def ms_dependence(correction, samples, *, grid=None):
    rows = []
    samples = list(samples)
    if len(samples) < 2:
        raise ValueError('Other-scene LMS needs at least two scenes')
    grid = grid or [r for r in response_grid() if r['probe'] == 'AXIS16']
    for n, (index, p, l) in enumerate(samples):
        for probe in grid:
            eps = p.new_tensor(probe['epsilon'])[None]
            for mode, pp, ll, expected in (
                ('PAN_ONLY', warp_pan(p, eps), l, -eps),
                ('LMS_ONLY', p, warp_pan(l, eps), eps),
                ('JOINT', warp_pan(p, eps), warp_pan(l, eps), torch.zeros_like(eps))):
                for scope in ('normal', 'clean_support', 'border_only'):
                    # Clean support is cropped AFTER shifting; it never reads
                    # synthesized border pixels. Border-only is a declared
                    # occlusion intervention, not an official inference image.
                    def restrict(x):
                        if scope == 'normal':
                            return x
                        if scope == 'clean_support':
                            return x[..., 12:-12, 12:-12]
                        y = x.clone()
                        y[..., 12:-12, 12:-12] = x.mean((-2, -1), keepdim=True)
                        return y
                    p0, l0, p1, l1 = map(restrict, (p, l, pp, ll))
                    c0 = correction(p0, l0, index, 'REESTIMATE_CURRENT_INPUT')
                    delta = correction(p1, l1, index, 'REESTIMATE_CURRENT_INPUT')-c0
                    texture = float(torch.sqrt(torch.mean((l[..., 1:, :] - l[..., :-1, :])**2)))
                    rows.append(dict(scene_index=index, mode=mode, support_scope=scope, epsilon=probe['epsilon'],
                                     delta=delta[0].cpu().tolist(), expected=expected[0].cpu().tolist(),
                                     residual_epe=float(torch.linalg.vector_norm(delta-expected)),
                                     lms_diagnostic_only=True, target_vertical_difference_rms=texture,
                                     low_texture=texture < 1e-4,
                                     scope_operation='none' if scope=='normal' else 'crop12_after_warp' if scope=='clean_support' else 'interior12_mean_occlusion'))
        other = samples[(n+1) % len(samples)][2]
        if other.shape != l.shape:
            raise ValueError('Other-scene LMS must preserve geometry')
        c0 = correction(p, l, index, 'REESTIMATE_CURRENT_INPUT')
        delta = correction(p, other, index, 'REESTIMATE_CURRENT_INPUT')-c0
        rows.append(dict(scene_index=index, mode='OTHER_SCENE_LMS', donor_index=samples[(n+1)%len(samples)][0],
                         delta=delta[0].cpu().tolist(), expected=None, residual_epe=None,
                         lms_diagnostic_only=True))
    return dict(record_type='MS_DEPENDENCE', rows=rows, official_reconstruction_modified=False)


def nested_crops(pan, lms, sizes=(64, 128, 256, 512)):
    if pan.shape[-2:] != lms.shape[-2:]:
        raise ValueError('PAN/LMS must share HR coordinates')
    h, w = pan.shape[-2:]
    for size in sizes:
        if h < size or w < size:
            raise ValueError('Native FOV control cannot upscale missing source support')
        r, c = (h-size)//2, (w-size)//2
        yield size, pan[..., r:r+size, c:c+size], lms[..., r:r+size, c:c+size]


def same_fov(pan, lms, size):
    """Diagnostic antialiased bicubic mapping, never claimed as Wald data."""
    h, w = pan.shape[-2:]
    if h != w or size > h or size < 16:
        raise ValueError('Same-FOV control only accepts square downsampling')
    p = F.interpolate(pan, size=(size, size), mode='bicubic', align_corners=False, antialias=True)
    l = F.interpolate(lms, size=(size, size), mode='bicubic', align_corners=False, antialias=True)
    return p, l, dict(domain='SYNTHETIC_SAME_FOV', scale=size/h,
                     mapping='bicubic_antialias_align_corners_false; shared PAN/LMS geometry',
                     official_wald=False, recreated_lms_from_lr=False)


def _group(name):
    if 'fc2' in name:
        return 'fc2'
    if 'fc1' in name:
        return 'fc1'
    if 'stem' in name:
        return 'stem'
    return 'body'


def gradient_probe(losses, named_parameters, weights, *, before=None, after=None):
    """Raw gradients computed independently without altering parameter .grad.

    losses keys rec/eps/struct (+scale) are unweighted scalars. Additional
    parameter-update tensors are measured externally around the *actual*
    optimizer update, not fabricated from a norm times LR.
    """
    named = [(n, p) for n, p in named_parameters if p.requires_grad]
    if not named:
        return [dict(group='whole', status='NOT_APPLICABLE', reason='no_aligner_parameters')]
    params = [p for _, p in named]
    gradients = {}
    for key, loss in losses.items():
        gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True) if loss.requires_grad else (None,) * len(params)
        gradients[key] = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, gs)]
    return gradient_vector_summary(gradients, named, weights, before=before, after=after)


def gradient_vector_summary(gradients, named_parameters, weights, *, before=None, after=None):
    """Summarize exact effective-batch gradients after microbatch accumulation."""
    named = list(named_parameters)
    if not named:
        return [dict(group='whole', status='NOT_APPLICABLE', reason='no_aligner_parameters')]
    if any(len(gs) != len(named) for gs in gradients.values()):
        raise ValueError('Gradient vector/parameter group count mismatch')
    groups = dict(whole=list(range(len(named))))
    for i, (name, _) in enumerate(named):
        groups.setdefault(_group(name), []).append(i)
        if name.endswith('bias'):
            groups.setdefault('bias', []).append(i)
    rows = []
    for group, indices in groups.items():
        vectors = {key: torch.cat([gs[i].flatten() for i in indices]).double() for key, gs in gradients.items()}
        weighted = {key: vec * float(weights.get(key, 1.)) for key, vec in vectors.items()}
        raw_norm = {key: float(torch.linalg.vector_norm(vec)) for key, vec in vectors.items()}
        norm = {key: float(torch.linalg.vector_norm(vec)) for key, vec in weighted.items()}
        cosines = {}
        for a, b in (('rec', 'eps'), ('rec', 'struct'), ('eps', 'struct')):
            den = norm.get(a, 0) * norm.get(b, 0)
            cosines[a+'_'+b] = float(torch.dot(weighted[a], weighted[b]) / den) if den > 0 else None
        total = sum(weighted.values())
        update = None
        if before is not None or after is not None:
            if before is None or after is None:
                raise ValueError('Actual update requires both snapshots')
            diffs = [after[named[i][0]].detach().double().flatten()-before[named[i][0]].detach().double().flatten() for i in indices]
            update = float(torch.linalg.vector_norm(torch.cat(diffs)))
        rows.append(dict(record_type='GRAD', group=group, raw_norm=raw_norm, weighted_norm=norm,
                         ratio_to_rec={key: (val/norm['rec'] if norm.get('rec', 0)>0 else None) for key, val in norm.items()},
                         cosine=cosines, total_weighted_norm=float(torch.linalg.vector_norm(total)),
                         actual_parameter_update_norm=update,
                         update_na_reason=None if update is not None else 'no_actual_optimizer_snapshot',
                         zero_cosine_policy='N/A_when_either_norm_zero'))
    return rows


def _sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def cost_probe(register, infer, *, device='cuda', warmup=10, repeats=50, shape=None):
    """Registration is invoked afresh for every measured inference, no cache shortcut."""
    if warmup < 1 or repeats < 2:
        raise ValueError('Cost probe requires declared warmup and repeated timing')
    times = []
    for i in range(warmup+repeats):
        _sync(device); start = time.perf_counter()
        correction = register() if register is not None else None
        _sync(device); middle = time.perf_counter() if register is not None else start
        infer(correction)
        _sync(device); end = time.perf_counter()
        if i >= warmup:
            times.append(((middle-start)*1000, (end-middle)*1000, (end-start)*1000))
    a = np.asarray(times)
    return dict(record_type='COST', registration_ms=float(a[:, 0].mean()), warp_u_ms=float(a[:, 1].mean()),
                pipeline_ms=float(a[:, 2].mean()), pipeline_std_ms=float(a[:, 2].std(ddof=1)),
                warmup=warmup, repeats=repeats, batch_size=1, shape=shape, device=str(device),
                precision='FP32', cache_read_excluded=True,
                hardware=torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else 'CPU')


@torch.no_grad()
def known_shift_control(pan, epsilon, *, estimator):
    shifted = warp_pan(pan, epsilon)
    original = pan[0].cpu().numpy()
    # Three identical bands satisfy the fixed top3 estimator contract; the
    # synthetic P/P pair uses matched sigma .8/.8, not PAN/LMS blur matching.
    result = estimator(shifted[0].cpu().numpy(), np.repeat(original, 3, axis=0),
                       descriptor='intensity', target='GT', split='train', max_dn=1.)
    residual = np.asarray(result['shift']) + epsilon[0].cpu().numpy()
    return dict(record_type='ORACLE', scope='P_VS_SHIFTED_P_KNOWN_SHIFT', estimate=result,
                expected=(-epsilon[0]).cpu().tolist(), epe=float(np.linalg.norm(residual)),
                excluded_from_native_selector=True, independent_training_seeds=0)


def geometry_summary(rows):
    result = []
    for split, target in sorted({(r['split'], r['target']) for r in rows if r['status']=='MEASURED'}):
        subset = [r for r in rows if r.get('split')==split and r.get('target')==target and r['status']=='MEASURED']
        for descriptor in ('gradient', 'intensity'):
            valid = [r['proxies'][descriptor] for r in subset if r['proxies'][descriptor]['valid_pair']]
            all_proxies = [r['proxies'][descriptor] for r in subset]
            result.append(dict(split=split, target=target, descriptor=descriptor, n_total=len(subset), n_valid=len(valid),
                before_epe_mean=float(np.mean([r['before_epe'] for r in valid])) if valid else None,
                after_epe_mean=float(np.mean([r['after_epe'] for r in valid])) if valid else None,
                boundary_hit_count=sum(r[k].get('boundary_hit_count',0) for r in all_proxies for k in ('before','after')),
                low_texture_band_count=sum(r[k].get('band_failure_counts',{}).get('TARGET_LOW_TEXTURE',0) for r in all_proxies for k in ('before','after')),
                invalid_fraction=1-len(valid)/len(subset), physical_displacement_ground_truth=False))
    return result


@torch.no_grad()
def run_diagnostics(*, samples, correction, estimator, output_path=None,
                    checkpoint_sha256, context, offline=False, predict_forced=None,
                    quality=None, train_corrections=None, gradient_records=None,
                    cost_records=None, include_sizes=True):
    """Run deterministic common probes over caller's frozen per-split samples.

    samples[split] = [(index, PAN_[0,1], LMS_[0,1], optional_GT_[0,1], optional_Aref_[0,1]), ...].
    correction(split,p,reference,index,mode) receives the explicitly chosen A
    reference (B or L). The optional fifth field prevents substituting L for B
    during crops/resize. Geometry and U counterfactuals still receive native L.
    Optional counterfactual/gradient/cost callbacks are required for a COMPLETE
    report; omitted measurements remain explicit PENDING, never fake zeros.
    """
    response, geometry, dependence, size_rows, positive, counterfactual, oracle = [], [], [], [], [], [], []
    current = {}
    for split, entries in samples.items():
        basic = [(x[0], x[1], x[4] if len(x)>4 else x[2]) for x in entries]
        fn = lambda p, l, index, mode: correction(split, p, l, index, mode)
        measured = measure_responses(fn, basic, split=split)
        response.extend(measured['rows'])
        if offline:
            response.extend(measure_responses(fn, basic, split=split, mode='HOLD_NATIVE_CACHE')['rows'])
        dependence.append(dict(split=split, **ms_dependence(fn, basic)))
        current[split] = []
        for item in entries:
            index, p, l = item[:3]
            ref_a = item[4] if len(item)>4 else l
            c = fn(p, ref_a, index, 'REESTIMATE_CURRENT_INPUT').detach(); current[split].append(c[0].cpu().numpy())
            for target, ref in [('LMS', l)] + ([('GT', item[3])] if len(item)>3 and item[3] is not None and split!='fr' else []):
                geometry.append(dict(scene_index=index, **native_geometry(p, ref, c, estimator=estimator, target=target, split=split)))
            if split == 'fr' and include_sizes:
                for size, pp, ll in nested_crops(p, ref_a):
                    axis = [r for r in response_grid() if r['probe']=='AXIS16']
                    size_rows.extend(measure_responses(fn, [(index, pp, ll)], split=split, grid=axis,
                                                       measurement_domain=f'NATIVE_NESTED_{size}')['rows'])
                    pp, ll, mapping = same_fov(p, ref_a, size)
                    scaled = [dict(r, epsilon=(np.asarray(r['epsilon'])*mapping['scale']).tolist(),
                                   radius=r['radius']*mapping['scale']) for r in axis]
                    size_rows.extend(measure_responses(fn, [(index, pp, ll)], split=split, grid=scaled,
                                                       measurement_domain=f'SYNTHETIC_SAME_FOV_{size}')['rows'])
            if split == 'train':
                positive.append(dict(scene_index=index, **known_shift_control(p, p.new_tensor([[.5, -.25]]), estimator=estimator)))
            if split == 'rr' and len(item)>3 and item[3] is not None and predict_forced is not None and quality is not None:
                estimate = estimator(p[0].cpu().numpy(), item[3][0].cpu().numpy(), descriptor='gradient',
                                     target='GT', split='rr', oracle=True, max_dn=1.)
                override = p.new_tensor(estimate['shift'])[None]
                pred = predict_forced(split,index,p,l,override)
                oracle.append(dict(record_type='ORACLE', scene_index=index, split='rr',
                                   scope='ORACLE_RR_ONLY', estimate=estimate, **quality(split,index,pred),
                                   excluded_from_native_selector=True, independent_training_seeds=0))
    pending = []
    if predict_forced is not None and quality is not None and train_corrections is not None:
        for split in ('rr', 'fr'):
            basic = [(x[0], x[1], x[2]) for x in samples[split]]
            cf = measure_counterfactuals(basic, current[split], train_corrections,
                predict_forced=lambda i,p,l,c: predict_forced(split,i,p,l,c), quality=lambda i,y: quality(split,i,y))
            counterfactual.extend(dict(split=split, **r) for r in cf)
    else:
        pending.append('P03_COUNTERFACTUAL_CALLBACKS_REQUIRED')
    if not oracle:
        pending.append('P08_RR_GT_ORACLE_CALLBACK_REQUIRED')
    if not gradient_records:
        pending.append('P06_FIXED32_BATCH_GRADIENTS_REQUIRED')
    if not cost_records:
        pending.append('P07_UNCACHED_REGISTRATION_AND_INFERENCE_COST_REQUIRED')
    report = seal(dict(schema='TA2_DIAGNOSTICS_v1', status='COMPLETE' if not pending else 'PARTIAL',
                       pending=pending, context=context, checkpoint_sha256=checkpoint_sha256,
                       probe_seed=PROBE_SEED, probe_grid_sha256=digest(response_grid()),
                       sample_indices={s:[x[0] for x in a] for s,a in samples.items()},
                       response_rows=response, response_summary=summarize_responses(response),
                       geometry=geometry, geometry_summary=geometry_summary(geometry),
                       correction_summary={s:correction_statistics(c) for s,c in current.items()},
                       ms_dependence=dependence, size_rows=size_rows, size_summary=summarize_responses(size_rows),
                       counterfactual=counterfactual, gradient=gradient_records, cost=cost_records,
                       known_shift_positive_control=positive, rr_gt_oracle=oracle, independent_training_seeds=1,
                       official_metric_masking=False, physical_displacement_ground_truth=False))
    if output_path:
        atomic_json(Path(output_path), report, immutable=True)
    return report
