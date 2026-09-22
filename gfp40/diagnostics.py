"""Train-only fixed probes; never consume the optimization stream or RNG."""
from __future__ import annotations

from contextlib import contextmanager
import random
import numpy as np
import torch
from torch.nn import functional as F

from fh12.training import restore_rng
from gfp40.losses import student_losses


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


@contextmanager
def preserve_runtime(*models):
    saved = rng_state()
    modes = [(m, m.training) for model in models if model is not None for m in model.modules()]
    try:
        yield
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(saved)


def diagnostic_ids(base_count):
    from qg40.calibration import select_calibration_indices
    from gfp40.common import object_sha
    if base_count < 3072:
        raise ValueError('P40 requires the full train calibration population')
    calibration = select_calibration_indices(base_count)
    selected = np.random.default_rng(1234).choice(base_count, 128, replace=False).tolist()
    payload = dict(schema='GFP40_DIAGNOSTIC_IDS_v1', base_count=base_count, seed=1234,
        train128=selected, gradient24=selected[:24],
        calibration3072_sha256=object_sha(calibration.tolist()),
        train128_calibration_overlap=sorted(set(selected).intersection(calibration.tolist())),
        geometry='fixed-HV ROT0 train view, gamma1 for common gradient probes')
    return dict(payload, ids_sha256=object_sha(payload))


def _stats(value):
    x = torch.as_tensor(value).detach().double().flatten()
    if not x.numel() or not bool(torch.isfinite(x).all()):
        raise FloatingPointError('Empty/nonfinite diagnostic population')
    quantile = torch.quantile(x, torch.tensor([.5, .9, .95, .99], device=x.device, dtype=x.dtype))
    return dict(mean=float(x.mean()), minimum=float(x.min()), maximum=float(x.max()),
                median=float(quantile[0]), p90=float(quantile[1]), p95=float(quantile[2]), p99=float(quantile[3]),
                sum=float(x.sum()), rms=float(x.square().mean().sqrt()))


def border_summary(error, hard_map):
    if error.shape != hard_map.shape or error.ndim != 4:
        raise ValueError('Matching [B,1,H,W] error/hard maps required')
    result = {}
    for width in (1, 4):
        if min(error.shape[-2:]) <= 2 * width:
            raise ValueError('Diagnostic patch too small')
        mask = torch.ones_like(error, dtype=torch.bool)
        mask[..., width:-width, width:-width] = False
        result[str(width)] = dict(border_error=float(error[mask].mean()),
            interior_error=float(error[~mask].mean()), border_hard=float(hard_map[mask].mean()),
            interior_hard=float(hard_map[~mask].mean()),
            border_hard_mass=float(hard_map[mask].sum() / hard_map.sum().clamp_min(1e-30)),
            border_fraction=float(mask.float().mean()), loss_support_unchanged=True)
    return result


def gradient_probe(model, losses):
    """Actual weighted terms, not raw norm multiplied by a guessed coefficient."""
    params = tuple(p for p in model.backbone.parameters() if p.requires_grad)
    terms = dict(hard=losses['hard'], soft=losses['soft'], qedge=losses['edge_weighted'],
                 raw_soft=losses['raw_soft'], raw_edge=losses['edge'])
    vectors = {}
    for name, loss in terms.items():
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        vectors[name] = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                                   for p, g in zip(params, grads)]).double()
    norms = {name: float(g.norm()) for name, g in vectors.items()}
    hard_norm = norms['hard']
    cosine = lambda a, b: float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))
    return dict(norms=norms, soft_over_hard=norms['soft'] / max(hard_norm, 1e-30),
                qedge_over_hard=norms['qedge'] / max(hard_norm, 1e-30),
                hard_soft_cosine=cosine(vectors['hard'], vectors['soft']),
                hard_edge_cosine=cosine(vectors['hard'], vectors['qedge']),
                population=24, weighted_terms_exact=True,
                detached_advantage=True, detached_difficulty=True, detached_q=True)


def low_active_mass(records):
    by_step = {r.get('local_step'): r for r in records}
    required = (0, 5000, 20000)
    if not all(t in by_step and by_step[t].get('gradient') for t in required):
        return dict(status='PENDING', required_steps=list(required))
    low = all(by_step[t]['gradient']['soft_over_hard'] < 1e-3
              and by_step[t]['soft_active_fraction'] < 1e-4 for t in required)
    return dict(status='LOW_ACTIVE_MASS' if low else 'ACTIVE_MASS_OBSERVED',
                required_steps=list(required), coefficient_escalation_allowed=False)


def parameter_snapshot(model):
    return {name: [p.detach().clone() for p in module.parameters()]
            for name, module in [('U', model.backbone), ('A', model.aligner)]}


def parameter_update_norm(model, before):
    return {name: float(torch.stack([(p.detach() - old).double().square().sum()
                                    for p, old in zip(module.parameters(), before[name])]).sum().sqrt())
            for name, module in [('U', model.backbone), ('A', model.aligner)]}


def run_diagnostics(model, teacher, dataset, q, reference, profile, local_step, ids,
                    device='cuda', *, parent_delta=None, heavy=True, deadline=None):
    from fh12.common import check_deadline
    rows, corrections_s, corrections_t, gate_values, es_values, et_values = [], [], [], [], [], []
    hard_values, difficulty_values, advantage_values, weight_values = [], [], [], []
    losses_mass = {name: 0. for name in ('hard', 'soft', 'edge_weighted')}
    q = torch.as_tensor(q, device=device, dtype=torch.float32)
    gradient = None
    native_id = dataset.native_gamma_id
    quality = []
    with preserve_runtime(model, teacher):
        model.eval(); teacher.eval()
        for start in range(0, len(ids['train128']), 24):
            check_deadline(deadline)
            chosen = ids['train128'][start:start + 24]
            # Common native view retains the q-cache's fixed-HV geometry.
            batch = [dataset.get_view(i, rot=0, augment=True, gamma_id=native_id) for i in chosen]
            gt, _lms, ms, lp, pan, meta = [torch.stack([row[j] for row in batch]).to(device) for j in range(6)]
            with torch.set_grad_enabled(bool(heavy and start == 0)):
                out = model(pan, ms, lp)
                with torch.no_grad():
                    target = teacher(pan, ms, lp)
                values = q[meta[:, 0], meta[:, 1]]
                if q.ndim == 3:
                    values = values[:, native_id]
                weights = reference['q_ref'] / (reference['q_ref'] + values)
                loss = student_losses(out, target, gt, reference['tau_R'], weights,
                                      profile, completed_updates=local_step)
                if heavy and start == 0:
                    gradient = gradient_probe(model, loss)
            corrections_s.append(out['delta'].detach().cpu()); corrections_t.append(target['delta'].detach().cpu())
            quality.append(native_quality(out['y'].detach(), gt, out.get('ms_base'), reduction=False))
            for items, key in [(gate_values, 'soft_gate'), (es_values, 'e_student'),
                    (et_values, 'e_teacher'), (hard_values, 'hard_map'),
                    (difficulty_values, 'difficulty'), (advantage_values, 'advantage'),
                    (weight_values, 'q_weights')]:
                items.append(loss[key].detach().cpu())
            for key in losses_mass:
                losses_mass[key] += float(loss[key]) * len(chosen) / len(ids['train128'])
        cs, ct = torch.cat(corrections_s), torch.cat(corrections_t)
        cp = cs if parent_delta is None else torch.as_tensor(parent_delta, dtype=cs.dtype)
        if cp.shape != cs.shape:
            raise ValueError('Parent correction diagnostic IDs differ')
        advantage = torch.cat(advantage_values)
        result = dict(schema='GFP40_DIAGNOSTICS_v1', local_step=local_step,
            diagnostic_ids_sha256=ids['ids_sha256'], coefficients=loss['coefficients'],
            c_student=cs.tolist(), c_teacher=ct.tolist(), c_parent=cp.tolist(),
            c_delta_parent=_stats(cs - cp), c_delta_teacher=_stats(cs - ct),
            c_delta_parent_norm=_stats((cs - cp).norm(dim=1)),
            c_delta_teacher_norm=_stats((cs - ct).norm(dim=1)),
            c_student_components={axis: _stats(cs[:, i]) for i, axis in enumerate(('dy', 'dx'))},
            c_student_norm=_stats(cs.norm(dim=1)), c_teacher_norm=_stats(ct.norm(dim=1)),
            c_cosine_mean=float(F.cosine_similarity(cs, ct, dim=1).mean()),
            common_bias=(cs - ct).mean(0).tolist(), correction_is_ground_truth=False,
            e_student=_stats(torch.cat(es_values)), e_teacher=_stats(torch.cat(et_values)),
            difficulty=_stats(torch.cat(difficulty_values)), advantage=_stats(advantage),
            q_weights=_stats(torch.cat(weight_values)), soft_gate=_stats(torch.cat(gate_values)),
            soft_active_fraction=float((advantage > 0).float().mean()), gradient=gradient,
            loss_mass=losses_mass, border=border_summary(torch.cat(es_values), torch.cat(hard_values)),
            gradient_view='fixed-HV ROT0 gamma1; same 24 IDs for all profiles',
            independent_estimator='ESTIMATOR_NOT_AVAILABLE', loss_support_unchanged=True)
        result['native_quality'] = merge_quality(quality)
        # All three training views: report overshoot rather than clamping it.
        if profile in ('CTRL', 'MIX'):
            view_stats = {}
            for gamma in range(len(dataset.gammas)):
                check_deadline(deadline)
                pp, ll, hh, qq = [], [], [], []
                for i in ids['train128']:
                    row = dataset.get_view(i, rot=0, augment=True, gamma_id=gamma)
                    pan, lp = row[4], row[3]
                    low = F.interpolate(lp[None], scale_factor=4, mode='bicubic', align_corners=False)[0]
                    pp.append(pan); ll.append(low); hh.append(pan - low)
                    qq.append(q[i, 0, gamma].detach().cpu())
                view_stats[str(dataset.gammas[gamma])] = dict(P=_stats(torch.stack(pp)), L=_stats(torch.stack(ll)),
                    H=_stats(torch.stack(hh)), q=_stats(torch.stack(qq)), tau_R=reference['tau_R'],
                    pan_outside_normalized_range=float((torch.stack(pp).abs() > 1).float().mean()))
            result['gamma_views'] = view_stats
            if heavy:
                result['perturbation'] = perturbation_probe(model, dataset, ids['gradient24'],
                    device=device, deadline=deadline)
        return result


def native_quality(y, gt, base=None, reduction=True):
    """Raw normalized forward quality; no official clipping or support change."""
    from pa.losses import output_edge_loss_per_sample
    if y.shape != gt.shape or y.ndim != 4 or y.shape[1] != 4:
        raise ValueError('Native quality requires matching C4 tensors')
    absolute = (y - gt).abs()
    border = torch.ones_like(absolute, dtype=torch.bool)
    border[..., 1:-1, 1:-1] = False
    values = dict(pixel_mae=absolute.flatten(1).mean(1),
        edge_mae=output_edge_loss_per_sample(y, gt),
        border1_mae=absolute[border].reshape(len(y), -1).mean(1),
        interior_mae=absolute[..., 1:-1, 1:-1].flatten(1).mean(1),
        band_mae=absolute.flatten(2).mean(2),
        output_overshoot=(y.abs() > 1).float().flatten(1).mean(1),
        residual_energy=((y - base) if base is not None else torch.zeros_like(y)).square().flatten(1).mean(1))
    values = {k: v.detach().cpu().tolist() for k, v in values.items()}
    return merge_quality([values]) if reduction else values


def merge_quality(rows):
    keys = tuple(rows[0])
    arrays = {key: np.concatenate([np.asarray(row[key]) for row in rows]) for key in keys}
    return dict(n_samples=len(arrays['pixel_mae']),
                mean={key: value.mean(axis=0).tolist() for key, value in arrays.items()},
                per_sample={key: value.tolist() for key, value in arrays.items()},
                units='raw normalized forward, edge Scharr/32; no output clip',
                residual_definition='output - bicubic LRMS base', official_metric=False)


@torch.no_grad()
def perturbation_probe(model, dataset, ids, device='cuda', deadline=None):
    """Family-native c freeze affects diagnostic forwards only; parameters untouched."""
    from fh12.common import check_deadline
    reports = {}
    with preserve_runtime(model):
        model.eval()
        for gamma_id, gamma in enumerate(dataset.gammas):
            output_change, correction_change, fixed_change = [], [], []
            for index in ids:
                check_deadline(deadline)
                native = dataset.get_view(index, rot=0, augment=True, gamma_id=dataset.native_gamma_id)
                altered = dataset.get_view(index, rot=0, augment=True, gamma_id=gamma_id)
                ms, lp, pan = [native[i][None].to(device) for i in (2, 3, 4)]
                original = model(pan, ms, lp)
                ms1, lp1, pan1 = [altered[i][None].to(device) for i in (2, 3, 4)]
                current = model(pan1, ms1, lp1)
                frozen = model(pan1, ms1, lp1, delta_override=original['delta'].detach())
                output_change.append(float((current['y'] - original['y']).abs().mean()))
                correction_change.append(float((current['delta'] - original['delta']).norm()))
                fixed_change.append(float((frozen['y'] - original['y']).abs().mean()))
            reports[str(gamma)] = dict(output_mae=_stats(output_change),
                correction_shift=_stats(correction_change), fixed_native_c_output_mae=_stats(fixed_change))
    return dict(gamma_sweep=reports, source_ids=list(ids), native_c_freeze='diagnostic_forward_only',
                input_is_displacement_ground_truth=False)
