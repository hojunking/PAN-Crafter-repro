"""D03 — 실제 gradient 충돌·민감도 진단 (계획 §7). gradient_conflicts.csv (pipeline × GRAD64 sample) · d03_summary.json (cell 별·checkpoint 별 집계, routing 검사).
같은 A 파라미터 φ 에서 g0 = ∇L0 · gH = ∇L_H · gE = ∇(λE L_E) · gK = ∇L_K · gC = ∇L_off (ε 는 r=1 4 축 고정 — 학습 sampler 의 무작위 ε 대신 결정적 대조; 기록) · gT = gH + gE.
T0 는 별도 clone 에서 A 만 requires_grad 로 연다(자산 불변). B0 도 같은 방식(A 가 frozen 이라 "받았을" gradient). B1 은 5050/25250/45450/50000 + best_hqnr. 다시 부르면 새 pipeline 분만 덧붙인다."""
import copy, os
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from tools.dcr12 import common as C
PHASE = os.environ.get("DCR12_PHASE", "")
OFF_EPS = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
ETA_VIRTUAL = 1e-5                                                         # A peak LR (SGD-like virtual step δφ = −η gC_raw; Adam 이 아님 — 1차 근사 검사용)


def open_A(P):
    m = copy.deepcopy(P.m).eval(); m.requires_grad_(False)
    for p in m.aligner.parameters():
        p.requires_grad_(True)
    return m


def terms(m, mg, pan, ms, lpan, gt, y_t, cal):
    o = m(pan, ms, lpan); y = o["y"]; c0 = o["delta"]; q = C.q12_terms(y, y_t, gt, cal); mb = o["ms_base"]; lo = 0
    for ey, ex in OFF_EPS:
        e = torch.tensor([[ey, ex]], device=pan.device).expand(pan.shape[0], 2)
        with torch.no_grad():
            pe = C.E.warp_pan(pan, e)
        ce = C.E.predict_c(m.aligner, pe, mb, mg); lo = lo + C.E.offset_loss(ce, c0, e, stop_reference=True) / len(OFF_EPS)
    return dict(L0=q["L0"], LH=q["LH"], LE=cal["lambda_E"] * q["LE"], LK=q["LK"], LC=lo, c0=c0, y=y)


def c_sensitivity(m, pan, ms, lpan, gt, c0, cal):
    """dL0/dc0, dLE/dc0 — correction 벡터에 대한 native task 민감도 (delta_override 를 leaf 로)."""
    from pa.losses import output_edge_loss
    cv = c0.detach().clone().requires_grad_(True); o = m(pan, ms, lpan, delta_override=cv); y = o["y"]; l0 = (y - gt).abs().mean(); le = cal["lambda_E"] * output_edge_loss(y.float(), gt.float())
    g0 = torch.autograd.grad(l0, cv, retain_graph=True)[0][0]; ge = torch.autograd.grad(le, cv)[0][0]; return g0.detach().cpu(), ge.detach().cpu()


def per_sample(P, m, A, pan, ms, lpan, gt, y_t, cal, floor):
    t = terms(m, P.mg, pan, ms, lpan, gt, y_t, cal); g = {}
    for k in ("L0", "LH", "LE", "LK", "LC"):
        g[k] = C.flat(torch.autograd.grad(t[k], A, retain_graph=True, allow_unused=True), A)
    gT = g["LH"] + g["LE"]; n = {k: float(v.norm()) for k, v in g.items()}; n["T"] = float(gT.norm()); n["C_w"] = 1e-4 * n["LC"]
    zero = {k: bool(n[k] < 10 * floor) for k in n}
    row = dict(**{f"norm_{k}": v for k, v in n.items()}, cos_g0_gC=(C.cos(g["L0"], g["LC"]) if not (zero["L0"] or zero["LC"]) else None), cos_gT_gC=(C.cos(gT, g["LC"]) if not (zero["T"] or zero["LC"]) else None),
               cos_gT_gK=(C.cos(gT, g["LK"]) if not (zero["T"] or zero["LK"]) else None), cos_gH_gE=(C.cos(g["LH"], g["LE"]) if not (zero["LH"] or zero["LE"]) else None), ratio_wC_over_T=(n["C_w"] / (n["T"] + 1e-30)),
               dot_gT_gC=float(gT @ g["LC"]), na_zero_grad=",".join(k for k, z in zero.items() if z) or "", L0=float(t["L0"]), LH=float(t["LH"]), LE_w=float(t["LE"]), LK=float(t["LK"]), LC=float(t["LC"]), c0_dy=float(t["c0"][0, 0]), c0_dx=float(t["c0"][0, 1]))
    s0, se = c_sensitivity(m, pan, ms, lpan, gt, t["c0"], cal); row.update(dL0_dc_y=float(s0[0]), dL0_dc_x=float(s0[1]), dLE_dc_y=float(se[0]), dLE_dc_x=float(se[1]))
    # virtual step δφ = −η gC_raw: 1차 예측 ΔL_task ≈ −η gT·gC vs 실제; closure 는 (a) 이 step 에서 고정한 sg(c0) 기준 (b) step 뒤 다시 계산한 c0' 기준 둘 다
    with torch.no_grad():
        backup = [p.detach().clone() for p in A]; off = 0
        for p in A:
            p.sub_(ETA_VIRTUAL * g["LC"][off:off + p.numel()].view_as(p)); off += p.numel()
        o2 = m(pan, ms, lpan); y2 = o2["y"]; c2 = o2["delta"]; q2 = C.q12_terms(y2, y_t, gt, cal); LT2 = float(q2["LH"] + cal["lambda_E"] * q2["LE"]); mb = o2["ms_base"]; clo_fixed = 0; clo_new = 0
        for ey, ex in OFF_EPS:
            e = torch.tensor([[ey, ex]], device=pan.device); ce = C.E.predict_c(m.aligner, C.E.warp_pan(pan, e), mb, P.mg)
            clo_fixed += float((ce + e - t["c0"].detach()).abs().mean()) / len(OFF_EPS); clo_new += float((ce + e - c2).abs().mean()) / len(OFF_EPS)
        for p, b in zip(A, backup):
            p.copy_(b)
    row.update(virtual_eta=ETA_VIRTUAL, dL_task_pred=-ETA_VIRTUAL * row["dot_gT_gC"], dL_task_actual=LT2 - float(t["LH"] + t["LE"]), closure_before=float(t["LC"]), closure_after_sg_fixed=clo_fixed, closure_after_recomputed=clo_new,
               virtual_target_policy="sg(c0) fixed at this virtual step; recomputed closure reported separately")
    return row


def routing_check(P, cal, ids):
    """B1 checkpoint 의 실제 trainer 경로: backward + _apply_routing 뒤 A .grad 와 ∇φ(L_H + λE L_E + 1e-4 L_off) 의 차이 (홀수 update, 2 sample)."""
    gt, ms, lpan, pan = (t.to(C.DEV) for t in C.load_patches(ids[:2])); S = copy.deepcopy(P.m).train().requires_grad_(True); T = C.load_pipe("T").m
    tr = C.trainer_stub("JK0", P.seed or C.SEEDS[0], S, T, cal); tot, info = tr._step(gt, ms, lpan, pan, 1); ap = list(tr.M.aligner.parameters())
    want = C.flat(torch.autograd.grad(info["_rec_hard_t"] + info["_edge_w_t"] + float(info["lam_off"]) * info["loss_off"], ap, retain_graph=True, allow_unused=True), ap)
    tot.backward(retain_graph=True); tr._apply_routing(info, tr.M); got = torch.cat([p.grad.flatten() for p in ap]); e = float((got - want).abs().max()); ref = float(want.abs().max())
    return dict(pipeline=P.key, max_abs_err=e, ref_max_abs=ref, rel_err=e / (ref + 1e-30), ok=bool(e <= 1e-4 * (1 + ref)), policy="qD=1,qK=0,qE=1", update=1)


def main(profile=False):
    with C.Stage("D03" + (f"-{PHASE}" if PHASE else ""), "aligner gradient decomposition on GRAD64"):
        panels = C.make_panels(); ids = panels.get("grad_ids") or panels["ids"]["DISC"][:C.GRAD_N]; cells = panels.get("grad_cells", {}); ids = ids[:8] if profile else ids
        cal = C.calibration(); T = C.load_pipe("T"); gt, ms, lpan, pan = (t.to(C.DEV) for t in C.load_patches(ids))
        with torch.no_grad():
            y_t_all = T.m(pan, ms, lpan)["y"]
        gp = os.path.join(C.CAMP, "gradient_conflicts.csv"); have = set(pd.read_csv(gp).pipeline.unique()) if os.path.exists(gp) else set()
        pipes = [T]
        for seed in C.SEEDS:
            for ck, tags in (("B0", ("best_hqnr",)), ("B1", tuple(f"cand:{s}" for s in C.D03_STEPS) + ("best_hqnr",))):
                for tag in tags:
                    if C.run_complete(ck, seed) and C.ckpt_path(ck, seed, tag):          # 완료 run 만 (candidate 는 불변이지만 run 단위로 일관되게)
                        pipes.append(C.load_pipe("S", ck, seed, tag))
        rows, rchecks = [], []
        for P in pipes:
            if P.key in have:
                continue
            m = open_A(P); A = list(m.aligner.parameters())
            t1 = terms(m, P.mg, pan[:1], ms[:1], lpan[:1], gt[:1], y_t_all[:1], cal); ga = C.flat(torch.autograd.grad(t1["L0"], A, retain_graph=False, allow_unused=True), A)
            t2 = terms(m, P.mg, pan[:1], ms[:1], lpan[:1], gt[:1], y_t_all[:1], cal); gb = C.flat(torch.autograd.grad(t2["L0"], A, allow_unused=True), A); floor = float((ga - gb).norm())
            for i, sid in enumerate(ids):
                r = per_sample(P, m, A, pan[i:i + 1], ms[i:i + 1], lpan[i:i + 1], gt[i:i + 1], y_t_all[i:i + 1], cal, floor)
                rows.append(dict(pipeline=P.key, role=P.role, case=P.case, seed=P.seed, tag=P.tag, step=P.step, sample_id=int(sid), group=C.group_of(sid), cell_T0=cells.get(str(sid)), aggregation="per_sample (batch 1)", grad_floor=floor, **r))
            if P.role == "S" and P.case == "JK0":
                rchecks.append(routing_check(P, cal, ids))
            print(f"[d03] {P.key} floor {floor:.2e}", flush=True)
        if rows:
            C.append_rows(gp, rows, ["pipeline", "sample_id"])
        df = pd.read_csv(gp); summ = {}
        for pipe, g in df.groupby("pipeline"):
            d = dict(n=int(len(g)), step=int(g.step.iloc[0]) if pd.notna(g.step.iloc[0]) else None, cos_gT_gC_mean=float(g.cos_gT_gC.mean()), cos_gT_gC_neg_frac=float((g.cos_gT_gC < 0).mean()), cos_g0_gC_mean=float(g.cos_g0_gC.mean()), cos_gT_gK_mean=float(g.cos_gT_gK.mean()),
                     ratio_wC_over_T_median=float(g.ratio_wC_over_T.median()), norm_T_median=float(g.norm_T.median()), norm_LC_median=float(g.norm_LC.median()), na_zero_count=int((g.na_zero_grad.fillna("") != "").sum()),
                     dL_task_actual_vs_pred_corr=C.pearson(g.dL_task_actual, g.dL_task_pred), dL_task_actual_neg_frac=float((g.dL_task_actual < 0).mean()),
                     by_cell={c: dict(n=int(len(h)), cos_gT_gC_mean=float(h.cos_gT_gC.mean()), neg_frac=float((h.cos_gT_gC < 0).mean()), cos_gT_gK_mean=float(h.cos_gT_gK.mean())) for c, h in g.groupby("cell_T0")})
            summ[pipe] = d
        rp = os.path.join(C.CAMP, "d03_routing_checks.json"); old = C.load_json(rp, []) or []; old = [x for x in old if x["pipeline"] not in {r["pipeline"] for r in rchecks}] + rchecks; C.dump_json(rp, old)
        summ["routing_checks"] = old; summ["off_eps"] = OFF_EPS; summ["note"] = "gC 는 고정 ε(r=1 4 축) 의 SG offset loss; 학습의 실제 offset gradient 는 홀수 update 에서 1e-4·gC(무작위 ε)"
        C.dump_json(os.path.join(C.CAMP, "d03_summary.json"), summ)
        for k, v in summ.items():
            if isinstance(v, dict) and "cos_gT_gC_mean" in v:
                print(f"[d03] {k}: cos(gT,gC) {v['cos_gT_gC_mean']:+.3f} (neg {v['cos_gT_gC_neg_frac']:.2f}) · cos(gT,gK) {v['cos_gT_gK_mean']:+.3f} · ‖1e-4 gC‖/‖gT‖ {v['ratio_wC_over_T_median']:.3g}", flush=True)
