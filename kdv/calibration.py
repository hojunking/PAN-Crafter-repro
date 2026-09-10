"""train-only calibration (plan §6.4, §9.2, §9.3): tau_R · tau_V/eps_V · λ_V pilot. 고정 native train calibration set(증강 없음, 고정 무작위 부분집합).

  tau_R  = max(median_p e_T(p), eps_scale)
  tau_V  = max(median E_T^V, 1e-3·v_scale, 1e-12),  eps_V = max(1e-3·tau_V, 1e-12),  v_scale = median(|S_V(GT)| > 0)
  λ_V    = r_grad · median_batches( RMS(∂L_rec/∂Ŷ) / (RMS(∂L_V^H/∂Ŷ) + eps) )  — optimizer step 없이 pilot checkpoint 에서. 통계 gradient ≈ 0 이면 CALIBRATION_DEGENERATE.
결과는 work_dir/_kdv_calibration/<teacher_id>/ 에 json 캐시 (key: Teacher tensors sha · 통계 종류·창 · calibration set manifest)."""
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from kdv.losses_stat import statistic_map, positive_median, stat_transform
from pa.losses import output_edge_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def calibration_batches(args, n_patches=3072, seed=1234, batch_size=48):
    """train feeder(증강 off) 의 고정 무작위 부분집합. 반환 list[(gt, lms, ms, lpan, pan)] CPU 와 manifest."""
    from main import import_class
    Feeder = import_class(args.feeder)
    fa = dict(args.train_feeder_args); fa.update(hflip=False, vflip=False, rot=False, crop=False)
    ds = Feeder(**fa)
    g = torch.Generator(device='cpu'); g.manual_seed(int(seed))
    n = min(int(n_patches), len(ds))
    idx = torch.randperm(len(ds), generator=g)[:n].sort().values.tolist()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=int(batch_size), shuffle=False, num_workers=0, drop_last=False)
    batches = [tuple(t.clone() for t in b) for b in loader]
    man = dict(dataroot=fa['dataroot'], n_patches=n, dataset_len=len(ds), seed=int(seed), batch_size=int(batch_size), augmentation='none',
               index_sha256_16=hashlib.sha256(np.array(idx, dtype=np.int64).tobytes()).hexdigest()[:16], first_indices=idx[:8])
    return batches, man


@torch.no_grad()
def _pred(model, pan, ms, lpan, dev):
    ms, lpan, pan = (t.to(dev) for t in (ms, lpan, pan))
    return model(pan, ms, lpan)['y']


def ms_base_of(ms, dev):
    return F.interpolate(ms.to(dev).float(), scale_factor=4, mode='bicubic')


def stat_view(x, ms, dev, domain='final_hrms'):
    """통계 도메인 (W104 §5.3 REP-RESIDUAL): final_hrms 는 Ŷ 그대로, residual 은 Ŷ − M. 같은 M 을 세 tensor 에서 뺀다."""
    if domain == 'residual':
        return x.float() - ms_base_of(ms, dev)
    return x


def _quant(v):
    v = v.flatten().double()
    qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.99], dtype=torch.float64)
    q = torch.quantile(v, qs.to(v.device)).tolist() if v.numel() < 16_000_000 else torch.quantile(v[torch.randperm(v.numel(), device=v.device)[:16_000_000]], qs.to(v.device)).tolist()
    return dict(zip(['p10', 'p25', 'p50', 'p75', 'p90', 'p99'], q), mean=float(v.mean()), n=int(v.numel()))


@torch.no_grad()
def calibrate_rec(teacher, batches, dev, eps_scale=1e-6):
    es = []
    for gt, lms, ms, lpan, pan in batches:
        y_t = _pred(teacher, pan, ms, lpan, dev)
        es.append((y_t.float() - gt.to(dev).float()).abs().mean(dim=1).cpu())
    e = torch.cat(es, 0)
    q = _quant(e)
    return dict(tau_R=max(q['p50'], float(eps_scale)), eps_scale=float(eps_scale), e_T=q, intensity_convention='feeder [-1,1] (range 2 = [0,1]-scale ×2)', kind='reconstruction')


@torch.no_grad()
def calibrate_stat(teacher, batches, kind, window, dev, *, transform='none', transform_eps=1e-12, domain='final_hrms'):
    """τ_V·ε_V (plan §9.2). 표현(창·변환·도메인) 이 바뀌면 반드시 다시 잰다 — cache key 에 전부 들어간다."""
    if kind == 'edge':
        return dict(tau_V=1.0, eps_V=1e-6, degenerate=False, note='signed edge: criterion 미사용(H 만)', kind=kind)
    es, vs = [], []
    _S = lambda x: stat_transform(statistic_map(stat_view(x, ms, dev, domain).float(), kind, window), transform, transform_eps)
    for gt, lms, ms, lpan, pan in batches:
        y_t = _pred(teacher, pan, ms, lpan, dev); g = gt.to(dev)
        v_t = _S(y_t); v_g = _S(g)
        es.append((v_t - v_g).abs().mean(dim=1).cpu()); vs.append(v_g.cpu())
    e = torch.cat(es, 0); v = torch.cat(vs, 0)
    q = _quant(e); v_scale = positive_median(v)
    pos_frac = float((v > 0).float().mean())
    degenerate = v_scale is None
    e_scale = q['p50']
    tau_V = max(e_scale, 1e-3 * (v_scale or 0.0), 1e-12)
    return dict(tau_V=tau_V, eps_V=max(1e-3 * tau_V, 1e-12), e_scale=e_scale, v_scale=v_scale, E_T_V=q, gt_stat_positive_fraction=pos_frac,
                degenerate=degenerate, kind=kind, window=int(window), margin=1 + (window - 1) // 2, transform=transform, transform_eps=float(transform_eps), domain=domain)


def calibrate_lambda(pilot, batches, kind, window, r_grad, dev, *, windows=None, transform='none', transform_eps=1e-12, domain='final_hrms'):
    """pilot(Student 후보 checkpoint) 에서 optimizer step 없이 ∂L/∂Ŷ RMS 비를 잰다. 여러 창이면 창별 loss 의 평균(REP-MULTI357) 에 대해 잰다."""
    ws = [int(w) for w in (windows or [window])]
    g_rec, g_v, ratio = [], [], []
    for gt, lms, ms, lpan, pan in batches:
        pred = _pred(pilot, pan, ms, lpan, dev).float().detach().requires_grad_(True); g = gt.to(dev).float()
        l_rec = (pred - g).abs().mean()
        gr = torch.autograd.grad(l_rec, pred)[0]
        ps, gs = stat_view(pred, ms, dev, domain), stat_view(g, ms, dev, domain)
        l_v = (output_edge_loss(pred, g) if kind == 'edge' else
               sum((stat_transform(statistic_map(ps, kind, w), transform, transform_eps) - stat_transform(statistic_map(gs, kind, w), transform, transform_eps)).abs().mean() for w in ws) / len(ws))
        gv = torch.autograd.grad(l_v, pred)[0]
        a, b = float(gr.pow(2).mean().sqrt()), float(gv.pow(2).mean().sqrt())
        g_rec.append(a); g_v.append(b); ratio.append(a / (b + 1e-30))
    med_r, med_v = float(np.median(g_rec)), float(np.median(g_v))
    degenerate = (med_v <= 1e-9 * med_r) or not np.isfinite(med_v)
    lam = 0.0 if degenerate else float(r_grad) * float(np.median(ratio))
    return dict(lambda_V=lam, r_grad=float(r_grad), g_rec_rms_median=med_r, g_V_rms_median=med_v, ratio_median=float(np.median(ratio)),
                ratio_p10=float(np.percentile(ratio, 10)), ratio_p90=float(np.percentile(ratio, 90)), n_batches=len(batches), degenerate=bool(degenerate),
                status=('CALIBRATION_DEGENERATE' if degenerate else 'OK'), kind=kind, window=int(window), windows=ws, transform=transform, domain=domain)


def cached(path, key, compute):
    """json 캐시: 같은 key(dict) 면 재사용, 아니면 compute() 하고 저장. 반환 (result, from_cache)."""
    if os.path.exists(path):
        j = json.load(open(path))
        if j.get('key') == key:
            return j['result'], True
    r = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(dict(key=key, result=r, computed_at=time.strftime('%Y-%m-%dT%H:%M:%S')), open(path, 'w'), indent=1)
    return r, False


# ------------------------------------------------------------------ 이동량 covariance (계획 §11) — train-only, frozen Teacher
def calibrate_covariance(teacher, batches, spec, dev, geo_sigma=2.0, geo_margin=11):
    """출처별 Π_T 통계와 상수: k0 = median tr(Π_T)/2 (G1), rank/cap/valid 비율, geo 는 τ_abs·γ·s² 경계 + C01(FD↔autograd, h 일관성) 판정.
    반환 dict(status OK | GEO_JACOBIAN_MISMATCH | COV_DEGENERATE, k0, thresholds, stats ...)."""
    from kdv.alignment_kd import probe_set, eq_closure_moment, precision_from_moment, geo_curvature, geo_autograd_grad, precision_from_curvature, structure_tensor_mean, precision_stats
    src = spec['cov_source']
    if src == 'none':                                                    # G1K: 명시 k0, Teacher covariance 없음 (검토 지적: 이전에는 ValueError)
        return dict(status='OK', source='none', k0=float(spec['geom_k0']), note='explicit k0, no Teacher covariance calibration')
    A = teacher.aligner; mg = teacher.aligner_margin
    tr_half, valid_frac, extra = [], [], {}
    if src == 'struct':
        Js = []
        for gt, lms, ms, lpan, pan in batches:
            Js.append(structure_tensor_mean(gt.to(dev)).cpu())
        J = torch.cat(Js, 0); ev = torch.linalg.eigvalsh(J)
        k = float((J.diagonal(dim1=1, dim2=2).sum(1) / 2).median())
        return dict(status='OK', source=src, k0=k, k_struct=k, n=int(J.shape[0]), trace_half_quantiles=_quant(J.diagonal(dim1=1, dim2=2).sum(1) / 2), anisotropy_median=float(((ev[:, 1] + 1e-30) / (ev[:, 0] + 1e-30)).median()))
    if src == 'eq_closure':
        pr = spec['probes']; probes = probe_set(pr['K'], pr['radius_hr']); sm = spec['eq_sigma_min']
        closure, bias, caps, ranks = [], [], [], []
        for gt, lms, ms, lpan, pan in batches:
            pan, ms = pan.to(dev), ms.to(dev); mb = F.interpolate(ms, scale_factor=4, mode='bicubic')
            eq = eq_closure_moment(A, pan, mb, probes, mg, pr['guard_margin'])
            Pi, info = precision_from_moment(eq['Q'], sm)
            tr_half.append((Pi.diagonal(dim1=1, dim2=2).sum(1) / 2).cpu()); valid_frac.append(eq['valid'].double().cpu())
            closure.append(eq['R'].norm(dim=2).mean(1).cpu()); bias.append(eq['b'].norm(dim=1).cpu()); caps.append(info['cap_hit'].double().mean(1).cpu())
        th = torch.cat(tr_half); cl = torch.cat(closure)
        res = dict(status='OK', source=src, k0=float(th.median()), sigma_min=sm, probes=dict(K=pr['K'], radius_hr=pr['radius_hr'], guard_margin=pr['guard_margin'], set=probes.tolist()),
                   trace_half_quantiles=_quant(th), closure_norm_quantiles=_quant(cl), bias_norm_median=float(torch.cat(bias).median()), cap_hit_fraction=float(torch.cat(caps).mean()),
                   valid_fraction=float(torch.cat(valid_frac).mean()), probe_radius_mean_norm=float(probes.norm(dim=1).mean()),
                   note='closure ≈ probe 크기면 aligner 가 반응하지 않는 것(신뢰 낮음, Π 작음); 절대 정합 posterior 아님 (§11.4)')
        res['response_proxy_closure_over_probe'] = res['closure_norm_quantiles']['p50'] / max(res['probe_radius_mean_norm'], 1e-12)
        if not (th.median() > 0) or not torch.isfinite(th).all():
            res['status'] = 'COV_DEGENERATE'
        return res
    if src == 'geo_curvature':
        ge = spec['geo']; h = ge['h']
        Hs, s2s, fd_err, hcons = [], [], [], []
        for i, (gt, lms, ms, lpan, pan) in enumerate(batches):
            pan, ms, gt = pan.to(dev), ms.to(dev), gt.to(dev); mb = F.interpolate(ms, scale_factor=4, mode='bicubic')
            with torch.no_grad():
                mu_t = teacher.predict_delta(pan, mb)
            g = geo_curvature(pan, gt, mu_t, geo_sigma, geo_margin, h)
            Hs.append(g['H'].cpu()); s2s.append(g['s2'].cpu())
            if i < 2:                                                    # C01: FD Jᵀr ↔ autograd ∇½||r||², h/2·2h 일관성
                ga = geo_autograd_grad(pan, gt, mu_t, geo_sigma, geo_margin)
                fd_err.append(float((g['g_fd'] - ga).norm() / (ga.norm() + 1e-30)))
                g2 = geo_curvature(pan, gt, mu_t, geo_sigma, geo_margin, h / 2); g3 = geo_curvature(pan, gt, mu_t, geo_sigma, geo_margin, 2 * h)
                hcons.append(max(float((g2['H'] - g['H']).norm() / (g['H'].norm() + 1e-30)), float((g3['H'] - g['H']).norm() / (g['H'].norm() + 1e-30))))
        H = torch.cat(Hs); s2 = torch.cat(s2s); ev = torch.linalg.eigvalsh(H); lmax = ev[:, 1]
        tau_abs = ge['tau_abs'] if ge['tau_abs'] != 'calibrate' else float(1e-3 * lmax.median())
        gamma = ge['gamma'] if ge['gamma'] != 'calibrate' else float(1e-2 * lmax.median())
        s2b = ge['s2_bounds'] if ge['s2_bounds'] != 'calibrate' else [float(torch.quantile(s2, 0.01)), float(torch.quantile(s2, 0.99))]
        Pi, info = precision_from_curvature(H, s2, gamma, ge['sigma_min'], tau_abs, ge['tau_rel'], s2b[0], s2b[1])
        th = Pi.diagonal(dim1=1, dim2=2).sum(1) / 2
        res = dict(status='OK', source=src, k0=float(th.median()), h=h, gamma=float(gamma), sigma_min=ge['sigma_min'], tau_abs=float(tau_abs), tau_rel=ge['tau_rel'], s2_bounds=[float(x) for x in s2b],
                   lambda_max_quantiles=_quant(lmax), lambda_min_quantiles=_quant(ev[:, 0]), s2_quantiles=_quant(s2), trace_half_quantiles=_quant(th),
                   rank_deficient_fraction=float((info['rank'] < 2).double().mean()), cap_hit_fraction=float(info['cap_hit'].double().mean()),
                   fd_vs_autograd_rel_err=fd_err, h_consistency_rel=hcons, geometry=dict(sigma_hr=geo_sigma, margin_hr=geo_margin),
                   note='H 는 GT 구조 cost 의 곡률 — Teacher 가 상수를 내면 영상 구조의 날카로움을 잰다 (§11.3 마지막 문단)')
        if max(fd_err) > ge['max_fd_autograd_rel_err'] or max(hcons) > ge['max_h_consistency']:
            res['status'] = 'GEO_JACOBIAN_MISMATCH'
        elif not (th.median() > 0):
            res['status'] = 'COV_DEGENERATE'
        return res
    raise ValueError(src)


def teacher_precision_fn(teacher, spec, cov, geo_sigma=2.0, geo_margin=11):
    """학습 중 native step 마다 Π_T [B,2,2] 를 내는 함수 (no_grad). 반환 fn(pan, ms_base, gt, mu_t) -> (Pi, valid[B] bool, info)."""
    from kdv.alignment_kd import probe_set, eq_closure_moment, precision_from_moment, geo_curvature, precision_from_curvature, structure_tensor_mean
    src = spec['cov_source']
    if src == 'none':
        k = float(spec['geom_k0'])
        def fn(pan, ms_base, gt, mu_t):
            B = mu_t.shape[0]; Pi = k * torch.eye(2, dtype=torch.float64, device=mu_t.device).expand(B, 2, 2)
            return Pi, torch.ones(B, dtype=torch.bool, device=mu_t.device), dict(rank=torch.full((B,), 2, device=mu_t.device))
        return fn
    if src == 'struct':
        k = float(cov['k_struct'])
        def fn(pan, ms_base, gt, mu_t):
            J = structure_tensor_mean(gt)
            return J / k, torch.ones(J.shape[0], dtype=torch.bool, device=J.device), dict(rank=torch.full((J.shape[0],), 2, device=J.device))
        return fn
    if src == 'eq_closure':
        pr = spec['probes']; probes = probe_set(pr['K'], pr['radius_hr']); sm = spec['eq_sigma_min']
        def fn(pan, ms_base, gt, mu_t):
            eq = eq_closure_moment(teacher.aligner, pan, ms_base, probes, teacher.aligner_margin, pr['guard_margin'])
            Pi, info = precision_from_moment(eq['Q'], sm); info.update(closure_norm=eq['R'].norm(dim=2).mean(1), bias=eq['b'])
            return Pi, eq['valid'], info
        return fn
    if src == 'geo_curvature':
        ge = spec['geo']
        from pa.warp import support_margin_ok
        from pa.losses import geometry_support_margin
        guard = geometry_support_margin(geo_sigma, geo_margin)
        def fn(pan, ms_base, gt, mu_t):
            g = geo_curvature(pan, gt, mu_t, geo_sigma, geo_margin, ge['h'])
            Pi, info = precision_from_curvature(g['H'], g['s2'], cov['gamma'], cov['sigma_min'], cov['tau_abs'], cov['tau_rel'], cov['s2_bounds'][0], cov['s2_bounds'][1])
            H, W = pan.shape[-2:]; mu = mu_t.detach().float(); ok = torch.ones(mu.shape[0], dtype=torch.bool, device=mu.device)
            for i in range(2):
                for sgn in (1.0, -1.0):
                    e = torch.zeros_like(mu); e[:, i] = sgn * ge['h']; ok = ok & support_margin_ok(H, W, mu + e, guard)      # §11.2: 모든 ±h 에서 같은 고정 유효 support
            info.update(g_fd=g['g_fd'], r0_norm2=g['r0_norm2'], support_ok=ok)
            return Pi, ok, info
        return fn
    raise ValueError(src)


def calibrate_covhead(teacher, batches, spec, cov, dev, epochs=3, lr=1e-2, geo_sigma=2.0, geo_margin=11):
    """G5 Teacher covariance head: frozen Teacher aligner feature → Σ_head 를 출처 target Σ_T = Π_T⁻¹ 에 KL(N(0,Σ_head)‖N(0,Σ_T)) 로 맞춘 뒤 freeze (§11.6).
    반환 (state_dict, manifest)."""
    from kdv.alignment_kd import CovHead, gaussian_kl, sigma_from_precision
    head = CovHead().to(dev); opt = torch.optim.Adam(head.parameters(), lr=lr); fn = teacher_precision_fn(teacher, spec, cov, geo_sigma, geo_margin)
    hist = []; excluded = 0; seen = 0
    for ep in range(int(epochs)):
        tot, n = 0.0, 0
        for gt, lms, ms, lpan, pan in batches:
            pan, ms, gt = pan.to(dev), ms.to(dev), gt.to(dev); mb = F.interpolate(ms, scale_factor=4, mode='bicubic')
            with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
                mu_t, feat = teacher.aligner(teacher._view(pan.float()), teacher._view(mb.float()), return_features=True)
                Pi, valid, pinfo = fn(pan, mb, gt, mu_t)
                Sig_t, full = sigma_from_precision(Pi, pinfo)                # precision 0 방향(rank 부족) 은 target 에서 제외 (§11.3-3), inv(Pi+εI) 금지
                valid = valid & full; excluded += int((~valid).sum()); seen += int(valid.numel())
                Sig_t = torch.where(valid[:, None, None], Sig_t, torch.eye(2, dtype=torch.float64, device=dev).expand_as(Sig_t))
            Sig_h = head(feat)
            kl = gaussian_kl(mu_t, Sig_h, mu_t, Sig_t) * valid.double()
            loss = kl.sum() / max(1, int(valid.sum()))
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * pan.shape[0]; n += pan.shape[0]
        hist.append(tot / max(1, n))
    return {k: v.detach().cpu() for k, v in head.state_dict().items()}, dict(epochs=int(epochs), lr=lr, loss_per_epoch=hist, source=spec['cov_source'], n_patches=sum(b[0].shape[0] for b in batches),
                                                                            excluded_fraction=(excluded / max(1, seen)))


# ------------------------------------------------------------------ TRI-A/B/C (addendum §10)
@torch.no_grad()
def calibrate_component_tau(teacher, batches, dev, *, kind=None, window=5, transform='none', transform_eps=1e-12, domain='final_hrms'):
    """성분별 τ: rec(band c) τ_R,c = median_p e_T,c (kind None) · stat τ_V,j = max(median E_T,j, 1e-3 v_scale_j, 1e-12) (§4.6 BANDADV / §5.6 COMPADV)."""
    es, vs = [], []
    for gt, lms, ms, lpan, pan in batches:
        y_t = _pred(teacher, pan, ms, lpan, dev); g = gt.to(dev).float()
        if kind is None:
            es.append((y_t.float() - g).abs().flatten(2).cpu())
        else:
            _S = lambda x: stat_transform(statistic_map(stat_view(x, ms, dev, domain).float(), kind, window), transform, transform_eps)
            vt = _S(y_t); vg = _S(g)
            es.append((vt - vg).abs().flatten(2).cpu()); vs.append(vg.abs().flatten(2).cpu())
    e = torch.cat(es, 0).permute(1, 0, 2).flatten(1)                      # [D, N]
    med = e.double().median(dim=1).values
    if kind is None:
        tau = med.clamp_min(1e-6); eps = torch.full_like(tau, 1e-6)
    else:
        v = torch.cat(vs, 0).permute(1, 0, 2).flatten(1)
        vsc = torch.stack([positive_median(v[j]) or 0.0 for j in range(v.shape[0])]).double() if False else torch.tensor([positive_median(v[j]) or 0.0 for j in range(v.shape[0])], dtype=torch.float64)
        tau = torch.maximum(torch.maximum(med, 1e-3 * vsc), torch.full_like(med, 1e-12)); eps = torch.maximum(1e-3 * tau, torch.full_like(tau, 1e-12))
    return dict(tau=tau.tolist(), eps=eps.tolist(), median=med.tolist(), D=int(e.shape[0]), kind=kind, window=window, transform=transform, domain=domain)


@torch.no_grad()
def calibrate_sens(teacher, batches, dev, *, h=0.05, phi='identity', stat_kind=None, window=5, n_check=2):
    """§6.6 s_sens = median(양의 q_sens) · §6.9-1/2 h/2·h·2h J RMS 와 선형화 잔차 (fixed native train)."""
    from kdv.tri import teacher_fd_jacobian, sens_q, linearization_check, teacher_eval
    qs, checks = [], []
    for i, (gt, lms, ms, lpan, pan) in enumerate(batches):
        pan, ms, lpan = pan.to(dev), ms.to(dev), lpan.to(dev); mb = F.interpolate(ms, scale_factor=4, mode='bicubic')
        mu_t = teacher.predict_delta(pan, mb)
        J = teacher_fd_jacobian(teacher, pan, ms, lpan, mu_t, h=h, phi=phi, stat_kind=stat_kind, window=window)
        qs.append(sens_q(J).flatten().cpu())
        if i < n_check:
            ev = lambda d: teacher_eval(teacher, pan, ms, lpan, d, phi=phi, stat_kind=stat_kind, window=window)
            J2 = teacher_fd_jacobian(teacher, pan, ms, lpan, mu_t, h=h / 2, phi=phi, stat_kind=stat_kind, window=window)
            J3 = teacher_fd_jacobian(teacher, pan, ms, lpan, mu_t, h=2 * h, phi=phi, stat_kind=stat_kind, window=window)
            lin = linearization_check(ev, mu_t, J, torch.tensor([[0.1, -0.05]], device=dev).expand(mu_t.shape[0], 2))
            checks.append(dict(h=h, J_rms=float(J.pow(2).mean().sqrt()), J_rms_half=float(J2.pow(2).mean().sqrt()), J_rms_double=float(J3.pow(2).mean().sqrt()),
                               rel_half=float((J2 - J).norm() / (J.norm() + 1e-30)), rel_double=float((J3 - J).norm() / (J.norm() + 1e-30)), finite=bool(torch.isfinite(J).all()), **lin))
    q = torch.cat(qs); pos = q[torch.isfinite(q) & (q > 0)]
    s_sens = float(pos.median()) if pos.numel() else 0.0
    status = 'OK'
    if s_sens <= 0:
        status = 'SOURCE_UNRESPONSIVE'
    elif any((not c['finite']) or c['rel_half'] > 0.2 or c['rel_double'] > 0.5 for c in checks):
        status = 'BLOCKED_NUMERICS'
    return dict(status=status, s_sens=s_sens, q_quantiles=_quant(q), positive_fraction=float((q > 0).double().mean()), fd_checks=checks, h=h, phi=phi, stat_kind=stat_kind, window=window)


def calibrate_lambda_q(pilot, teacher, batches, dev, criterion, *, s_c):
    """§6.8 λ_Q: 공통 pilot 에서 QISO soft 의 ∂/∂Ŷ RMS 를 기준 R3 L1 soft 와 맞춘다 (FULL 에만 따로 유리하게 하지 않음)."""
    from kdv.tri import iso_quad
    g1, gq = [], []
    for gt, lms, ms, lpan, pan in batches:
        pred = _pred(pilot, pan, ms, lpan, dev).float().detach().requires_grad_(True); g = gt.to(dev).float(); y_t = _pred(teacher, pan, ms, lpan, dev).float()
        r = criterion(pred.detach(), y_t, g, return_maps=True); wk = r.maps['soft_weight']
        soft_l1 = (wk * (pred - y_t).abs().mean(1, keepdim=True)).mean()
        soft_q = (wk * iso_quad(pred - y_t, s_c=s_c)).mean()
        a = torch.autograd.grad(soft_l1, pred)[0]; b = torch.autograd.grad(soft_q, pred)[0]
        g1.append(float(a.pow(2).mean().sqrt())); gq.append(float(b.pow(2).mean().sqrt()))
    g1m, gqm = float(np.median(g1)), float(np.median(gq))
    degenerate = gqm <= 1e-12 * max(g1m, 1e-30)
    return dict(lambda_q=(0.0 if degenerate else g1m / gqm), g_l1_rms_median=g1m, g_q_rms_median=gqm, degenerate=bool(degenerate), status=('CALIBRATION_DEGENERATE' if degenerate else 'OK'), s_c=s_c)
