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

from kdv.losses_stat import statistic_map, positive_median
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
def calibrate_stat(teacher, batches, kind, window, dev):
    if kind == 'edge':
        return dict(tau_V=1.0, eps_V=1e-6, degenerate=False, note='signed edge: criterion 미사용(H 만)', kind=kind)
    es, vs = [], []
    for gt, lms, ms, lpan, pan in batches:
        y_t = _pred(teacher, pan, ms, lpan, dev); g = gt.to(dev)
        v_t = statistic_map(y_t.float(), kind, window); v_g = statistic_map(g.float(), kind, window)
        es.append((v_t - v_g).abs().mean(dim=1).cpu()); vs.append(v_g.cpu())
    e = torch.cat(es, 0); v = torch.cat(vs, 0)
    q = _quant(e); v_scale = positive_median(v)
    pos_frac = float((v > 0).float().mean())
    degenerate = v_scale is None
    e_scale = q['p50']
    tau_V = max(e_scale, 1e-3 * (v_scale or 0.0), 1e-12)
    return dict(tau_V=tau_V, eps_V=max(1e-3 * tau_V, 1e-12), e_scale=e_scale, v_scale=v_scale, E_T_V=q, gt_stat_positive_fraction=pos_frac,
                degenerate=degenerate, kind=kind, window=int(window), margin=1 + (window - 1) // 2)


def calibrate_lambda(pilot, batches, kind, window, r_grad, dev):
    """pilot(Student 후보 checkpoint) 에서 optimizer step 없이 ∂L/∂Ŷ RMS 비를 잰다."""
    g_rec, g_v, ratio = [], [], []
    for gt, lms, ms, lpan, pan in batches:
        pred = _pred(pilot, pan, ms, lpan, dev).float().detach().requires_grad_(True); g = gt.to(dev).float()
        l_rec = (pred - g).abs().mean()
        gr = torch.autograd.grad(l_rec, pred)[0]
        l_v = output_edge_loss(pred, g) if kind == 'edge' else (statistic_map(pred, kind, window) - statistic_map(g, kind, window)).abs().mean()
        gv = torch.autograd.grad(l_v, pred)[0]
        a, b = float(gr.pow(2).mean().sqrt()), float(gv.pow(2).mean().sqrt())
        g_rec.append(a); g_v.append(b); ratio.append(a / (b + 1e-30))
    med_r, med_v = float(np.median(g_rec)), float(np.median(g_v))
    degenerate = (med_v <= 1e-9 * med_r) or not np.isfinite(med_v)
    lam = 0.0 if degenerate else float(r_grad) * float(np.median(ratio))
    return dict(lambda_V=lam, r_grad=float(r_grad), g_rec_rms_median=med_r, g_V_rms_median=med_v, ratio_median=float(np.median(ratio)),
                ratio_p10=float(np.percentile(ratio, 10)), ratio_p90=float(np.percentile(ratio, 90)), n_batches=len(batches), degenerate=bool(degenerate),
                status=('CALIBRATION_DEGENERATE' if degenerate else 'OK'), kind=kind, window=int(window))


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
