"""D40 — 출력 stress(H2)·PAN 민감도·spectral/band·edge profile·blur 대조 (계획 §10)."""
import json, math, os
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from tools.eqrec4 import common as C
from tools.eqrec4.common import CAMP, DEV, QUADS
from tools.eqrec4.d20 import detail_ids

STRESS_PROBES = [p for p in C.probe_bank("B") if p["r"] in (0.5, 1.0, 2.0)] + [dict(p, probe_id=p["probe_id"] + "_rep") for p in C.probe_bank("A") if p["r"] == 1.0]   # 주: bank B r∈{.5,1,2} 4 대각; 재현: bank A r=1 4 축
PATHS = ("response", "no_response", "known_inverse")


@torch.no_grad()
def stress_patches(L, ids, quad, gt, ms, lpan, pan, chunk=64):
    """§10 D40-A patch64/RR: y_ε = M + F([W(W(P,ε), c_path), M]) — response(cε) / no_response(c0) / known_inverse(c0−ε). native 와 같은 ROI; d_e = L1(yε)−L1(y0), s_Y = mean|yε−y0| (ROI)."""
    scale = "native64" if pan.shape[-1] == 64 else "rr256"; mg_roi = C.ROI_MARGIN[scale]; H = pan.shape[-1]; rows = []
    y0, c0 = C.native_forward(L, pan, ms, lpan, chunk=chunk); l0 = C.l1_roi(y0, gt, scale); sy, sx = C.roi_slice(scale)
    for pr in STRESS_PROBES:
        for s in range(0, pan.shape[0], chunk):
            p, q, l, g = (t[s:s + chunk].to(DEV) for t in (pan, ms, lpan, gt)); e = torch.tensor([[pr["ey"], pr["ex"]]], device=DEV).expand(p.shape[0], 2); pe = C.warp_pan(p, e)
            mb = F.interpolate(q, scale_factor=4, mode="bicubic"); ce = C.predict_c(L.m.aligner, pe, mb, L.mg); cb = c0[s:s + chunk].to(DEV)
            for path, cc in (("response", ce), ("no_response", cb), ("known_inverse", cb - e)):
                y = L.m(pe, q, l, delta_override=cc)["y"]; l1 = C.l1_roi(y, g, scale); sY = (y - y0[s:s + chunk].to(DEV)).abs()[..., sy, sx].mean(dim=(1, 2, 3)); eb = C.l1_roi(y, g, scale, per_band=True)
                for i in range(p.shape[0]):
                    j = s + i; rows.append(dict(model_key=L.key, scale=scale, sample_id=int(ids[j]), quadrant_primary=(quad.get(int(ids[j])) if scale == "native64" else None), probe_id=pr["probe_id"], probe_bank=pr["bank"], r=pr["r"], epsilon_dy_hr=pr["ey"], epsilon_dx_hr=pr["ex"],
                                                path=path, c_path_dy=float(cc[i, 0]), c_path_dx=float(cc[i, 1]), c0_dy=float(cb[i, 0]), c0_dx=float(cb[i, 1]), ce_dy=float(ce[i, 0]), ce_dx=float(ce[i, 1]), l1_native_roi=float(l0[j]), l1_stress_roi=float(l1[i]),
                                                d_e=float(l1[i] - l0[j]), s_Y=float(sY[i]), **{f"d_e_band{b}": float(eb[i, b]) for b in range(8)}, valid_support=C.two_stage_ok(H, H, (pr["ey"], pr["ex"]), cc[i], mg_roi), scalar_scope="probe"))
    return rows


@torch.no_grad()
def stress_fr(L, fd):
    """§10 D40-A FR20: 참조는 원 P·원 MS 로 고정(raw_valid, margin 96; aligned 는 W(P,c0) 고정 진단) — d_Q = HQNR0 − HQNRε."""
    from pa.evalviews import scene_views, STRESS_MARGIN
    from tools.eval_fr_paperset import sensor_of, load_dlpan
    sensor = sensor_of(fd["cfg"]); wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")); mp = fd["mp"]; rows = []
    for i in range(len(fd["fr"])):
        lms, ms, lpan, pan = (t.unsqueeze(0).to(DEV) for t in fd["fr"][i]); mb = F.interpolate(ms, scale_factor=4, mode="bicubic"); c0 = C.predict_c(L.m.aligner, pan, mb, L.mg); o0 = L.m(pan, ms, lpan, delta_override=c0)
        p = fd["pan_raw"][i]; pa_fixed = C.warp_pan(torch.from_numpy(p)[None, None], c0[0].double().cpu()[None])[0, 0].numpy(); d0 = c0[0].double().cpu().numpy()
        def views(y):
            sr = ((y[0].clip(-1, 1).float().cpu().numpy() + 1) / 2 * mp).astype(np.float64).transpose(1, 2, 0); v, ok, _ = scene_views(sr, fd["lms_raw"][i].transpose(1, 2, 0), p, pa_fixed, sensor, wald, d0, 4, mp, margin=STRESS_MARGIN); return v, ok
        v0, ok0 = views(o0["y"]); sy, sx = C.roi_slice("fr512")
        for pr in STRESS_PROBES:
            e = torch.tensor([[pr["ey"], pr["ex"]]], device=DEV); pe = C.warp_pan(pan, e); ce = C.predict_c(L.m.aligner, pe, mb, L.mg)
            for path, cc in (("response", ce), ("no_response", c0), ("known_inverse", c0 - e)):
                y = L.m(pe, ms, lpan, delta_override=cc)["y"]; v, ok = views(y)
                rows.append(dict(model_key=L.key, scale="fr512", sample_id=i, probe_id=pr["probe_id"], probe_bank=pr["bank"], r=pr["r"], epsilon_dy_hr=pr["ey"], epsilon_dx_hr=pr["ex"], path=path, c_path_dy=float(cc[0, 0]), c_path_dx=float(cc[0, 1]), c0_dy=float(c0[0, 0]), c0_dx=float(c0[0, 1]),
                                 hqnr_native_v96=float(v0["raw_valid"]["hqnr"]), hqnr_stress_v96=float(v["raw_valid"]["hqnr"]), d_Q=float(v0["raw_valid"]["hqnr"] - v["raw_valid"]["hqnr"]), d_fscc=float(v0["raw_valid"]["fscc"] - v["raw_valid"]["fscc"]),
                                 aligned_fixed_hqnr=float(v["aligned_valid"]["hqnr"]), s_Y=float((y - o0["y"]).abs()[..., sy, sx].mean()), valid_support=bool(ok and ok0 and C.two_stage_ok(512, 512, (pr["ey"], pr["ex"]), cc[0], 96)), scalar_scope="probe"))
    return rows


@torch.no_grad()
def pan_sensitivity(L, ids, quad, gt, ms, lpan, pan):
    """§10 D40-B: c0 고정 → PAN 을 blur σ1 / 다른 장면 PAN / 평균 상수로 바꿔 U-Net 만 검사; 그 뒤 A 재추정 열."""
    y0, c0 = C.native_forward(L, pan, ms, lpan); l0 = C.l1_roi(y0, gt, "native64"); k = torch.exp(-0.5 * (torch.arange(-3, 4).float()) ** 2); k = (k / k.sum()).view(1, 1, 1, 7)
    pb = F.conv2d(F.pad(pan, (3, 3, 3, 3), mode="reflect"), k); pb = F.conv2d(pb, k.transpose(2, 3)); variants = dict(blur_sigma1=pb, other_scene=torch.roll(pan, 1, 0), mean_constant=pan.mean(dim=(2, 3), keepdim=True).expand_as(pan).contiguous()); rows = []; sy, sx = C.roi_slice("native64")
    for name, pv in variants.items():
        yf, _ = C.native_forward(L, pv, ms, lpan, delta_override=c0); ya, ca = C.native_forward(L, pv, ms, lpan); lf = C.l1_roi(yf, gt, "native64"); la = C.l1_roi(ya, gt, "native64")
        for i in range(len(ids)):
            rows.append(dict(model_key=L.key, sample_id=int(ids[i]), quadrant_primary=quad.get(int(ids[i])), variant=name, l1_native=float(l0[i]), l1_c0_fixed=float(lf[i]), d_e_c0_fixed=float(lf[i] - l0[i]), s_Y_c0_fixed=float((yf[i] - y0[i]).abs()[..., sy, sx].mean()),
                             l1_A_reestimated=float(la[i]), d_e_A_reestimated=float(la[i] - l0[i]), c_reest_minus_c0_norm=float((ca[i] - c0[i]).norm()), note="OOD intervention; diagnostic of direction/magnitude, not proof of 'PAN unused'"))
    return rows


@torch.no_grad()
def band_alignment(L, fd):
    """§10 D40-D native RR: PAN 과 각 GT band 의 구조 정합 추정 (GT-informed) + band 별 L1 — 물리적 misalignment 와 spectral appearance 를 구분하지 못한 상태로 기록."""
    from tools.palsv18_validate import _est, blur_hr
    from tools.metrics.jqm import _pan_kernel
    from align.estimator import GATES
    kp = _pan_kernel("WV3", 4); G = dict(GATES, search_int=4, max_magnitude=4.0); mp = fd["mp"]; rows = []
    for i in range(len(fd["rr"])):
        gt, lms, ms, lpan, pan = (t.unsqueeze(0).to(DEV) for t in fd["rr"][i]); o = L.m(pan, ms, lpan); d = o["delta"][0].double().cpu(); p = (pan[0, 0].double().cpu().numpy() + 1) / 2 * mp; pt = C.warp_pan(torch.from_numpy(p)[None, None], d[None])[0, 0].numpy()
        gdn = (gt[0].double().cpu().numpy() + 1) / 2 * mp; l1b = (o["y"] - gt).abs()[0].mean(dim=(1, 2)).cpu().numpy()
        for b in range(8):
            a = _est(blur_hr(p, kp), gdn[b], G); c = _est(blur_hr(pt, kp), gdn[b], G)
            rows.append(dict(model_key=L.key, scene=i, band_index=b, band_l1=float(l1b[b]), before_dy=a["dy"], before_dx=a["dx"], before_mag=a["mag"], before_accepted=a["accepted"], after_dy=c["dy"], after_dx=c["dx"], after_mag=c["mag"], after_accepted=c["accepted"],
                             note="band names follow source metadata; even/odd index groups are not called visible/NIR"))
    return rows


def h2_stats(df, nm):
    out = {}
    for mk, g in df.groupby("model_key"):
        out[mk] = {}
        for sc, s in g.groupby("scale"):
            key = "d_e" if sc != "fr512" else "d_Q"; q = nm[(nm.model_key == mk) & (nm.scale == sc)].set_index("sample_id"); res = {}
            for path, sp in s.groupby("path"):
                per = {}
                for r, sr in sp[sp.valid_support].groupby("r"):
                    agg = sr.groupby("sample_id")[key].mean(); qa = q.q_A.reindex(agg.index); e_ = (q.e_native_full if sc != "fr512" else q.raw_original_hqnr).reindex(agg.index)
                    groups = q.source_group_id.reindex(agg.index).values if sc == "native64" else np.array([f"{sc}{i}" for i in agg.index])
                    per[str(r)] = dict(n=int(len(agg)), mean=float(agg.mean()), median=float(agg.median()), positive_loss_mean=float(agg[agg > 0].mean()) if (agg > 0).any() else None, p90=float(agg.quantile(.9)),
                                       worst_direction=(sr.groupby("probe_id")[key].mean().idxmax() if len(sr) else None), spearman_qA_d=C.spearman_boot(qa.values, agg.values, groups, seed=7),
                                       spearman_qA_d_by_native_tertile={t: C.spearman(qa.values[m_], agg.values[m_]) for t, m_ in zip(("low", "mid", "high"), [C.texture_tertile(e_.values) == t for t in ("low", "mid", "high")])} if len(agg) >= 24 else None)
                res[path] = per
            out[mk][sc] = dict(by_path=res, invalid_support_fraction=float(1 - s.valid_support.mean()), metric=key, sign=("d_e>0 = stress worse" if key == "d_e" else "d_Q>0 = stress worse"))
    return out


def main(profile=False):
    fd = C.feeders(); ids, quad = detail_ids(); gt, ms, lpan, pan = C.load_patches(ids); nm = C.read_csv(os.path.join(CAMP, "native_sample_metrics.csv"))
    from tools.palsv18_validate import edge_profile, energy_fr
    with C.Stage("D40" + ("-profile" if profile else ""), "output stress · PAN sensitivity · band/spectral · edge profiles"):
        models = [("L1E4", s, t) for s in (1234, 7777, 2025) for t in ("best_raw", "last")] + [("L000", s, "best_raw") for s in (1234, 7777, 2025)]
        rr = [torch.cat([fd["rr"][i][k].unsqueeze(0) for i in range(20)]) for k in range(5)]; st, ps, ba, ep, en = [], [], [], [], {}
        for fam, seed, tag in models + [("P0", s, "best_raw") for s in (1234, 7777, 2025)]:
            if C.ckpt_dir(fam, seed, tag) is None:
                continue
            L = C.load_model(fam, seed, tag); t0 = C.time.time()
            if L.has_aligner:
                st += stress_patches(L, ids if not profile else ids[:16], quad, *(t[:16] if profile else t for t in (gt, ms, lpan, pan)))
                if not profile:
                    st += stress_patches(L, list(range(20)), {}, rr[0], rr[2], rr[3], rr[4], chunk=4); st += stress_fr(L, fd); ba += band_alignment(L, fd)
                if (fam, seed, tag) == C.PRIMARY or (fam == "L1E4" and tag == "best_raw" and not profile):
                    ps += pan_sensitivity(L, ids, quad, gt, ms, lpan, pan)
            if not profile:
                rows, summ = edge_profile(L.m, fd["cfg"], fd["mp"], DEV, L.has_aligner); ep += [dict(model_key=L.key, **r) for r in rows]; en[L.key] = dict(edge_profile=summ, **(energy_fr(L.m, fd["fr"], fd["pan_raw"], DEV) if L.has_aligner else {}))
            print(f"  {L.key}: {C.time.time() - t0:.0f}s", flush=True); del L; torch.cuda.empty_cache()
        df = pd.DataFrame(st); df.to_csv(os.path.join(CAMP, "output_stress.csv"), index=False)
        if ps:
            pd.DataFrame(ps).to_csv(os.path.join(CAMP, "pan_sensitivity.csv"), index=False)
        if ba:
            pd.DataFrame(ba).to_csv(os.path.join(CAMP, "band_alignment.csv"), index=False)
        if ep:
            pd.DataFrame(ep).to_csv(os.path.join(CAMP, "band_edge_profiles.csv"), index=False); C.dump_json(os.path.join(CAMP, "edge_energy_summary.json"), en)
        h2 = h2_stats(df, nm); C.dump_json(os.path.join(CAMP, "h2_stats.json"), h2); pk = C.mkey(*C.PRIMARY); print(json.dumps(h2.get(pk, {}).get("native64"), default=str)[:1500])
