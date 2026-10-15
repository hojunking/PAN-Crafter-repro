"""D50 — 같은 sample 의 loss gradient 진단과 A-only 국소 개입 (계획 §11). U-Net 은 업데이트하지 않는다."""
import copy, json, math, os
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from tools.eqrec4 import common as C
from tools.eqrec4.common import CAMP, DEV, QUADS

ALPHAS = (1e-5, 3e-5); OFF_EPS = [(p["ey"], p["ex"]) for p in C.probe_bank("A") if p["r"] == 1.0]     # L_off 진단용 고정 ε (r=1, 4 축) — training 의 무작위 원판이 아니라 결정적 대조


def _flat(gs, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gs, params)])


def grads_for(m, mg, pan, ms, lpan, gt):
    """g_r = ∇φ L_rec (native, A live), g_c = ∇φ L_off^SG (ε 4개 평균). 반환 (g_r, g_c, L_rec, L_off) — A 파라미터 순서."""
    A = list(m.aligner.parameters()); mb = F.interpolate(ms, scale_factor=4, mode="bicubic")
    o = m(pan, ms, lpan); lr = (o["y"] - gt).abs().mean(); gr = torch.autograd.grad(lr, A, retain_graph=True, allow_unused=True)
    c0 = o["delta"].detach(); lo = 0
    for ey, ex in OFF_EPS:
        e = torch.tensor([[ey, ex]], device=pan.device).expand(pan.shape[0], 2); ce = C.predict_c(m.aligner, C.warp_pan(pan, e).detach(), mb, mg); lo = lo + C.offset_loss(ce, c0, e, stop_reference=True) / len(OFF_EPS)
    gc = torch.autograd.grad(lo, A, allow_unused=True); return _flat(gr, A), _flat(gc, A), float(lr), float(lo), A


def head_slice(A):
    """fc2 (final head) 파라미터의 flat index 범위."""
    n = 0; out = []
    for p in A:
        out.append((n, n + p.numel())); n += p.numel()
    return out


@torch.no_grad()
def measure(L, m, pan, ms, lpan, gt):
    y = m(pan, ms, lpan); q = C.probe_responses(type("T", (), dict(m=m, mg=L.mg))(), pan.cpu(), ms.cpu())
    return dict(e_native=float((y["y"] - gt).abs().mean()), c0_dy=float(y["delta"][0, 0]), c0_dx=float(y["delta"][0, 1]), q_A=float(q["A"]["q"][0]), q_B=float(q["B"]["q"][0]), y=y["y"].detach())


def main(profile=False):
    d = C.load_json(os.path.join(CAMP, "detail_subsets.json")); ids = [i for q in QUADS for i in d["quadrants"][q]["grad"]]; quad = {i: q for q in QUADS for i in d["quadrants"][q]["grad"]}
    if profile:
        ids = ids[:4]
    gt, ms, lpan, pan = C.load_patches(ids)
    with C.Stage("D50" + ("-profile" if profile else ""), "gradient diagnostics · one-step A-only interventions"):
        rows = []
        for fam, seed, tag in (C.PRIMARY, ("L000", 2025, "best_raw")):
            if C.ckpt_dir(fam, seed, tag) is None:
                continue
            L = C.load_model(fam, seed, tag); m = copy.deepcopy(L.m).train(); m.backbone.requires_grad_(False); m.aligner.requires_grad_(True); A = list(m.aligner.parameters()); src = [p.detach().clone() for p in A]; hs = head_slice(A)
            names = [n for n, _ in m.aligner.named_parameters()]; head = [k for k, n in enumerate(names) if n.startswith("fc2")]; t0 = C.time.time()
            for i in range(len(ids)):
                p, q, l, g = (t[i:i + 1].to(DEV) for t in (pan, ms, lpan, gt)); gr, gc, lr, lo, _ = grads_for(m, L.mg, p, q, l, g)
                nr, nc = float(gr.norm()), float(gc.norm()); cos = (float((gr @ gc) / (nr * nc)) if nr > 1e-12 and nc > 1e-12 else None)
                hr = torch.cat([gr[a:b] for k, (a, b) in enumerate(hs) if k in head]); hc = torch.cat([gc[a:b] for k, (a, b) in enumerate(hs) if k in head]); cosh = (float((hr @ hc) / (hr.norm() * hc.norm())) if hr.norm() > 1e-12 and hc.norm() > 1e-12 else None)
                base = dict(model_key=L.key, sample_id=int(ids[i]), quadrant_primary=quad[ids[i]], L_rec=lr, L_off_sg=lo, g_r_norm=nr, g_c_norm=nc, cos_gr_gc=cos, R_g=(1e-4 * nc / (nr + 1e-12)), g_r_head_norm=float(hr.norm()), g_c_head_norm=float(hc.norm()), cos_head=cosh)
                m0 = measure(L, m, p, q, l, g); rows.append(dict(base, direction="J0", alpha=0.0, e_native_after=m0["e_native"], d_e=0.0, q_A_after=m0["q_A"], d_qA=0.0, q_B_after=m0["q_B"], d_qB=0.0, c0_after_dy=m0["c0_dy"], c0_after_dx=m0["c0_dx"], d_c0_norm=0.0, output_change=0.0))
                phin = math.sqrt(sum(float((p_ ** 2).sum()) for p_ in src)); dirs = dict(JR=gr, JC=gc, JRC=gr + 1e-4 * gc)
                for dn, gv in dirs.items():
                    for al in ALPHAS:
                        step = al * phin * gv / (gv.norm() + 1e-12); k = 0
                        with torch.no_grad():
                            for p_, s_ in zip(A, src):
                                n = p_.numel(); p_.copy_(s_ - step[k:k + n].view_as(p_)); k += n
                        mm = measure(L, m, p, q, l, g)
                        rows.append(dict(base, direction=dn, alpha=al, e_native_after=mm["e_native"], d_e=mm["e_native"] - m0["e_native"], q_A_after=mm["q_A"], d_qA=mm["q_A"] - m0["q_A"], q_B_after=mm["q_B"], d_qB=mm["q_B"] - m0["q_B"], c0_after_dy=mm["c0_dy"], c0_after_dx=mm["c0_dx"],
                                         d_c0_norm=float(math.hypot(mm["c0_dy"] - m0["c0_dy"], mm["c0_dx"] - m0["c0_dx"])), output_change=float((mm["y"] - m0["y"]).abs().mean())))
                        with torch.no_grad():
                            for p_, s_ in zip(A, src):
                                p_.copy_(s_)
                assert all(torch.equal(p_, s_) for p_, s_ in zip(A, src)), "A restore failed"
            print(f"  {L.key}: {C.time.time() - t0:.0f}s", flush=True); del L, m; torch.cuda.empty_cache()
        df = pd.DataFrame(rows); df.to_csv(os.path.join(CAMP, "gradient_interventions.csv"), index=False)
        summ = {}
        for mk, g in df.groupby("model_key"):
            j0 = g[g.direction == "J0"]; summ[mk] = dict(cos_gr_gc=dict(mean=float(j0.cos_gr_gc.mean()), median=float(j0.cos_gr_gc.median()), n_na=int(j0.cos_gr_gc.isna().sum())), R_g=dict(median=float(j0.R_g.median()), p90=float(j0.R_g.quantile(.9))),
                                                            by_quadrant={qd: dict(n=int(len(s)), cos_mean=float(s.cos_gr_gc.mean()), R_g_median=float(s.R_g.median())) for qd, s in j0.groupby("quadrant_primary")},
                                                            one_step={f"{dn}@{al}": {qd: dict(n=int(len(s)), d_e_mean=float(s.d_e.mean()), d_qA_mean=float(s.d_qA.mean()), d_qB_mean=float(s.d_qB.mean()), frac_e_down=float((s.d_e < 0).mean()), frac_qA_down=float((s.d_qA < 0).mean()), frac_qB_down=float((s.d_qB < 0).mean()))
                                                                                    for qd, s in g[(g.direction == dn) & (g.alpha == al)].groupby("quadrant_primary")} for dn in ("JR", "JC", "JRC") for al in ALPHAS},
                                                            note="normalized SGD one-step on A only (diagnostic); SG surrogate gradient ≠ derivative of the re-evaluated q — measured values reported")
        C.dump_json(os.path.join(CAMP, "gradient_summary.json"), summ); print(json.dumps(summ.get(C.mkey(*C.PRIMARY)), default=str)[:1500])
