"""D20 — 정합이 진짜 잘 되는가: (A) native pair 의 알려진 상대 이동 PAN-only/MS-only/common, (B) 동일 격자 합성 대조(절대 오차), (C) cross-modal 단서 (계획 §8)."""
import json, math, os
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from tools.eqrec4 import common as C
from tools.eqrec4.common import CAMP, DEV, QUADS
from tools.po10_diag import warp_generic

SYN_DELTAS = [(0.0, 0.0)] + [(s * r, 0.0) for r in (0.5, 1.0, 2.0) for s in (1, -1)] + [(0.0, s * r) for r in (0.5, 1.0, 2.0) for s in (1, -1)] + [(sy * r, sx * r) for r in (0.5, 1.0) for sy in (1, -1) for sx in (1, -1)]
SYN_WEIGHTS = dict(uniform_all=[1 / 8] * 8, even_index=[0.25 if b % 2 == 0 else 0.0 for b in range(8)], odd_index=[0.25 if b % 2 == 1 else 0.0 for b in range(8)])


def detail_ids():
    d = C.load_json(os.path.join(CAMP, "detail_subsets.json")); ids, quad = [], {}
    for q, v in d["quadrants"].items():
        for i in v["detail"]:
            ids.append(int(i)); quad[int(i)] = q
    return ids, quad


@torch.no_grad()
def modality_response(L, pan, ms, mode, bank="A", chunk=128):
    """§8 D20-A: pan_only (기대 −ε) / ms_only (+ε) / common (0). aligner 의 진단 입력만 바꾼다. 반환 sample 별 dict."""
    sgn = dict(pan_only=-1.0, ms_only=1.0, common=0.0)[mode]; probes = C.probe_bank(bank); N = pan.shape[0]; resid = torch.zeros(N, len(probes), 2); resp = torch.zeros(N, len(probes), 2)
    for s in range(0, N, chunk):
        p = pan[s:s + chunk].to(DEV); mb = F.interpolate(ms[s:s + chunk].to(DEV), scale_factor=4, mode="bicubic"); c0 = C.predict_c(L.m.aligner, p, mb, L.mg)
        for k, pr in enumerate(probes):
            e = torch.tensor([[pr["ey"], pr["ex"]]], device=DEV).expand(p.shape[0], 2)
            pe = C.warp_pan(p, e) if mode != "ms_only" else p; me = warp_generic(mb, e) if mode != "pan_only" else mb
            ce = C.predict_c(L.m.aligner, pe, me, L.mg); q = (ce - c0).cpu(); resp[s:s + chunk, k] = q; resid[s:s + chunk, k] = q - sgn * torch.tensor([pr["ey"], pr["ex"]])
    out = []
    for i in range(N):
        ft = C.fit_response(resp[i].numpy(), probes); r = resid[i]
        out.append(dict(mode=mode, expected_B=("-I" if mode == "pan_only" else "+I" if mode == "ms_only" else "0"), epe=float(r.norm(dim=1).mean()), q_component=float(r.abs().mean()),
                        **{f"epe_r{rad}": float(r[[k for k, p_ in enumerate(probes) if p_["r"] == rad]].norm(dim=1).mean()) for rad in C.RADII}, **ft))
    return out


@torch.no_grad()
def synthetic_control(L, fd):
    """§8 D20-B: RR GT 를 anchor Z 로 (원 격자·추가 phase 없음), P_syn = Σ w_c Z_c, M_syn = Z. 절대 오차 ||A(W(P_syn,δ),M_syn)+δ||₂ 와 baseline 을 뺀 relative q."""
    rows = []
    for i in range(len(fd["rr"])):
        gt = fd["rr"][i][0].unsqueeze(0).to(DEV).float(); k = torch.exp(-0.5 * (torch.arange(-3, 4, device=DEV).float() / 1.0) ** 2); k = (k / k.sum()).view(1, 1, 1, 7)
        gb = F.conv2d(F.pad(gt.view(-1, 1, *gt.shape[-2:]), (3, 3, 3, 3), mode="reflect"), k); gb = F.conv2d(gb, k.transpose(2, 3)).view_as(gt)
        for zname, Z in (("gt", gt), ("gt_blur_sigma1", gb)):
            for wname, w in SYN_WEIGHTS.items():
                P = (Z * torch.tensor(w, device=DEV).view(1, 8, 1, 1)).sum(1, keepdim=True); cb = C.predict_c(L.m.aligner, P, Z, L.mg)[0].cpu()
                for dy, dx in SYN_DELTAS:
                    e = torch.tensor([[dy, dx]], device=DEV); cd = C.predict_c(L.m.aligner, C.warp_pan(P, e) if (dy or dx) else P, Z, L.mg)[0].cpu()
                    rows.append(dict(model_key=L.key, scene=i, anchor=zname, weights=wname, delta_dy=dy, delta_dx=dx, r=round(math.hypot(dy, dx), 3), c_base_dy=float(cb[0]), c_base_dx=float(cb[1]), c_delta_dy=float(cd[0]), c_delta_dx=float(cd[1]),
                                     abs_synthetic_error=float(torch.hypot(cd[0] + dy, cd[1] + dx)), relative_q_l2=float(torch.hypot(cd[0] - cb[0] + dy, cd[1] - cb[1] + dx)), scalar_scope="probe"))
    return rows


@torch.no_grad()
def cue_checks(L, pan, ms):
    """§8 D20-C: 정상 / MS swap / band 상수 MS / PAN affine / bilinear kernel / reflection padding — q_A·B fit·c0 변화."""
    variants = dict(normal=dict(), ms_swap=dict(ms_mode="swap"), ms_const_band=dict(ms_mode="const"), pan_affine=dict(pan_mode="affine"), kernel_bilinear=dict(kernel="bilinear"), padding_reflection=dict(padding="reflection"))
    base = None; rows = []
    for name, kw in variants.items():
        pr = C.probe_responses(L, pan, ms, banks=("A",), **kw)
        if base is None:
            base = pr
        for i in range(pan.shape[0]):
            ft = C.fit_response(pr["A"]["resp"][i].numpy(), pr["A"]["probes"])
            rows.append(dict(model_key=L.key, variant=name, idx=i, q_A=float(pr["A"]["q"][i]), epe_A=float(pr["A"]["epe"][i]), c0_dy=float(pr["c0"][i, 0]), c0_dx=float(pr["c0"][i, 1]),
                             c0_change_vs_normal=float((pr["c0"][i] - base["c0"][i]).norm()), q_change_vs_normal=float(pr["A"]["q"][i] - base["A"]["q"][i]), **ft))
    return rows


def main(profile=False):
    fd = C.feeders(); ids, quad = detail_ids(); gt, ms, lpan, pan = C.load_patches(ids)
    with C.Stage("D20" + ("-profile" if profile else ""), "relative response · synthetic absolute · cross-modal cues"):
        # A: PAN-only summary from D10 (전수) + MS-only/common on detail subsets, RR20, FR20 (L1E4 3 seed best/last; L000 best; N2 last)
        nm = C.read_csv(os.path.join(CAMP, "native_sample_metrics.csv")); summ = []
        for mk, g in nm[nm.A_B_yy.notna()].groupby("model_key") if "A_B_yy" in nm else []:
            for sc, s in g.groupby("scale"):
                summ.append(dict(model_key=mk, scale=sc, mode="pan_only", n=int(len(s)), B_yy_mean=float(s.A_B_yy.mean()), B_xx_mean=float(s.A_B_xx.mean()), B_yx_mean=float(s.A_B_yx.mean()), B_xy_mean=float(s.A_B_xy.mean()), b_y_mean=float(s.A_b_y.mean()), b_x_mean=float(s.A_b_x.mean()),
                                 fit_rmse_mean=float(s.A_fit_rmse.mean()), sv_max_mean=float(s.A_sv_max.mean()), sv_min_mean=float(s.A_sv_min.mean()), epe_A_mean=float(s.epe_A.mean()), q_A_mean=float(s.q_A.mean()),
                                 q_over_qconst_A=float(s.q_A.mean() / C.load_json(os.path.join(CAMP, "probe_bank_A.json"))["q_const_component"]), **{f"q_A_r{r}_mean": float(s[f"q_A_r{r}"].mean()) for r in C.RADII}, source="D10 records (all samples)"))
        rows = []; models = [("L1E4", s, t) for s in (1234, 7777, 2025) for t in ("best_raw", "last")] + [("L000", s, "best_raw") for s in (1234, 7777, 2025)] + [("N2", 2025, "last")]
        rr_pan = torch.cat([fd["rr"][i][4].unsqueeze(0) for i in range(len(fd["rr"]))]); rr_ms = torch.cat([fd["rr"][i][2].unsqueeze(0) for i in range(len(fd["rr"]))])
        fr_pan = torch.cat([fd["fr"][i][3].unsqueeze(0) for i in range(len(fd["fr"]))]); fr_ms = torch.cat([fd["fr"][i][1].unsqueeze(0) for i in range(len(fd["fr"]))])
        syn = []; cues = []
        for fam, seed, tag in models:
            if C.ckpt_dir(fam, seed, tag) is None:
                continue
            L = C.load_model(fam, seed, tag); t0 = C.time.time()
            for mode in ("pan_only", "ms_only", "common"):
                for sc, (p, q, sid) in (("native64", (pan, ms, ids)), ("rr256", (rr_pan, rr_ms, list(range(20)))), ("fr512", (fr_pan, fr_ms, list(range(20))))):
                    if profile and sc != "native64":
                        continue
                    for i, r in enumerate(modality_response(L, p[:(16 if profile else None)], q[:(16 if profile else None)], mode)):
                        rows.append(dict(model_key=L.key, scale=sc, sample_id=int(sid[i]), quadrant_primary=(quad.get(int(sid[i])) if sc == "native64" else None), **r))
            if not profile:
                syn += synthetic_control(L, fd)
            if fam == "L1E4" and tag == "best_raw":
                cues += [dict(r, sample_id=int(ids[r["idx"]]), quadrant_primary=quad.get(int(ids[r["idx"]]))) for r in cue_checks(L, pan[:(16 if profile else None)], ms[:(16 if profile else None)])]
            print(f"  {L.key}: {C.time.time() - t0:.0f}s", flush=True); del L; torch.cuda.empty_cache()
        df = pd.DataFrame(rows); df.to_csv(os.path.join(CAMP, "geometry_modality.csv"), index=False)
        for (mk, sc, mode), s in df.groupby(["model_key", "scale", "mode"]):
            summ.append(dict(model_key=mk, scale=sc, mode=mode, n=int(len(s)), B_yy_mean=float(s.B_yy.mean()), B_xx_mean=float(s.B_xx.mean()), B_yx_mean=float(s.B_yx.mean()), B_xy_mean=float(s.B_xy.mean()), b_y_mean=float(s.b_y.mean()), b_x_mean=float(s.b_x.mean()),
                             fit_rmse_mean=float(s.fit_rmse.mean()), sv_max_mean=float(s.sv_max.mean()), sv_min_mean=float(s.sv_min.mean()), epe_mean=float(s.epe.mean()), epe_p90=float(s.epe.quantile(.9)), q_component_mean=float(s.q_component.mean()),
                             **{f"epe_r{r}_mean": float(s[f"epe_r{r}"].mean()) for r in C.RADII}, expected_B=s.expected_B.iloc[0], source="detail subsets (native64) / RR20 / FR20"))
        C.write_csv(os.path.join(CAMP, "response_matrices.csv"), summ)
        if syn:
            sdf = pd.DataFrame(syn); sdf.to_csv(os.path.join(CAMP, "geometry_controls.csv"), index=False)
            ss = sdf[sdf.r > 0].groupby(["model_key", "anchor", "weights"]).agg(abs_err_mean=("abs_synthetic_error", "mean"), abs_err_p90=("abs_synthetic_error", lambda x: float(np.percentile(x, 90))), rel_q_mean=("relative_q_l2", "mean"),
                                                                                   base_norm_mean=("c_base_dy", lambda x: float(np.hypot(x, sdf.loc[x.index, "c_base_dx"]).mean()))).reset_index()
            ss.to_csv(os.path.join(CAMP, "geometry_controls_summary.csv"), index=False)
        if cues:
            cdf = pd.DataFrame(cues); cdf.to_csv(os.path.join(CAMP, "geometry_cues.csv"), index=False)
            cdf.groupby(["model_key", "variant"]).agg(q_A_mean=("q_A", "mean"), epe_A_mean=("epe_A", "mean"), B_yy_mean=("B_yy", "mean"), B_xx_mean=("B_xx", "mean"), c0_change_mean=("c0_change_vs_normal", "mean"), q_change_mean=("q_change_vs_normal", "mean")).reset_index().to_csv(os.path.join(CAMP, "geometry_cues_summary.csv"), index=False)
        pk = C.mkey(*C.PRIMARY); print(json.dumps([s for s in summ if s["model_key"] == pk and s["scale"] == "native64"], default=str)[:1500])
