"""D30 — native 보정의 실효성·절대 기준·복원 최적점: (A) correction 치환, (B) 독립 native proxy, (C) bias 개입, (D) correction landscape (계획 §9)."""
import json, math, os
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tools.eqrec4 import common as C
from tools.eqrec4.common import CAMP, DEV, QUADS
from tools.eqrec4.d20 import detail_ids
from tools.palsv18_validate import _est, native_proxy, calibration_patches, blur_hr, up_bicubic
from tools.metrics.jqm import _pan_kernel
from align.estimator import GATES

GRID = [(-2.0 + 0.5 * i, -2.0 + 0.5 * j) for i in range(9) for j in range(9)]


def _const_vec(mk):
    nm = C.read_csv(os.path.join(CAMP, "native_sample_metrics.csv")); a = nm[(nm.model_key == mk) & (nm.split_role == "A") & (nm.scale == "native64")]
    return [float(a.c0_dy.median()), float(a.c0_dx.median())]


@torch.no_grad()
def patch_interventions(L, ids, quad, gt, ms, lpan, pan, const_vec, e_by_id):
    """§9 D30-A on patch64 detail subsets: I-L / I-Z / I-W / I-C / I-S(10 within-quadrant derangements + matched-e) → common ROI L1."""
    _, c0 = C.native_forward(L, pan, ms, lpan); N = len(ids); modes = {"I-L": c0, "I-Z": torch.zeros_like(c0), "I-W": -c0, "I-C": torch.tensor(const_vec).expand(N, 2).clone()}
    byq = {q: [i for i in range(N) if quad[ids[i]] == q] for q in QUADS}
    for k in range(1, 11):
        perm = torch.arange(N)
        for q, members in byq.items():
            if len(members) > 1:
                for a, b in zip(members, members[k % len(members):] + members[:k % len(members)]):
                    perm[a] = b
        modes[f"I-S{k}"] = c0[perm]
    perm = torch.arange(N)                                                  # matched-e shuffle: 같은 집단 안에서 e 순위가 비슷한 블록(8) 내 순환
    for q, members in byq.items():
        mem = sorted(members, key=lambda i: e_by_id[ids[i]])
        for s in range(0, len(mem), 8):
            blk = mem[s:s + 8]
            if len(blk) > 1:
                for a, b in zip(blk, blk[1:] + blk[:1]):
                    perm[a] = b
    modes["I-S-matched"] = c0[perm]; rows = []; ref = None
    for name, cvec in modes.items():
        y, _ = C.native_forward(L, pan, ms, lpan, delta_override=cvec); l1 = C.l1_roi(y, gt, "native64"); eb = C.l1_roi(y, gt, "native64", per_band=True); ee = C.edge_l1_roi(y, gt, "native64")
        if name == "I-L":
            ref = l1.clone()
        for i in range(N):
            rows.append(dict(model_key=L.key, scale="native64", sample_id=int(ids[i]), quadrant_primary=quad[ids[i]], mode=name, applied_dy=float(cvec[i, 0]), applied_dx=float(cvec[i, 1]), c0_dy=float(c0[i, 0]), c0_dx=float(c0[i, 1]),
                             applied_minus_learned_norm=float((cvec[i] - c0[i]).norm()), l1_roi=float(l1[i]), edge_l1_roi=float(ee[i]), **{f"l1_band{b}": float(eb[i, b]) for b in range(8)},
                             invalid_support=bool(not C.two_stage_ok(64, 64, (0, 0), cvec[i], 16)), scalar_scope="sample"))
    return rows


@torch.no_grad()
def scene_interventions(L, fd, const_vec, cgeo_rr, cgeo_fr, blur):
    """RR20 (GT L1·reduced metrics) 와 FR20 (raw views) 의 learned/zero/wrong/const/shuffle×10/c_geo/blur 치환."""
    from pa.evalviews import scene_views, VIEWS
    from tools.eval_fr_paperset import sensor_of, load_dlpan
    from utils import reduced_metrics
    sensor = sensor_of(fd["cfg"]); wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")); mp = fd["mp"]; rows = []
    for scale, ds in (("rr256", fd["rr"]), ("fr512", fd["fr"])):
        n = len(ds); learned = []
        for i in range(n):
            t = [x.unsqueeze(0).to(DEV) for x in ds[i]]; pan, ms, lpan = t[-1], t[-3], t[-2]; learned.append(L.m(pan, ms, lpan)["delta"][0].float().cpu())
        learned = torch.stack(learned); modes = {"I-L": learned, "I-Z": torch.zeros_like(learned), "I-W": -learned, "I-C": torch.tensor(const_vec).expand(n, 2).clone()}
        for k in range(1, 11):
            modes[f"I-S{k}"] = learned[[(i + k) % n for i in range(n)]]
        cg = cgeo_rr if scale == "rr256" else cgeo_fr
        if cg is not None:
            modes["I-G"] = torch.tensor([[v[0], v[1]] if v is not None else [float("nan"), float("nan")] for v in cg])
        if blur.get("blur_status") == "matched" and blur.get("sigma_star", 0) > 0:
            modes["I-BLUR"] = torch.zeros_like(learned)
        for name, cvec in modes.items():
            for i in range(n):
                t = [x.unsqueeze(0).to(DEV) for x in ds[i]]; pan, ms, lpan = t[-1], t[-3], t[-2]; c = cvec[i]
                if name == "I-G" and not torch.isfinite(c).all():
                    rows.append(dict(model_key=L.key, scale=scale, sample_id=i, mode=name, status="estimator_not_accepted")); continue
                pin = pan
                if name == "I-BLUR":
                    pin = torch.from_numpy(gaussian_filter(pan[0, 0].float().cpu().numpy(), blur["sigma_star"], mode="nearest"))[None, None].to(DEV)
                o = L.m(pin, ms, lpan, delta_override=c[None].to(DEV).float()); r = dict(model_key=L.key, scale=scale, sample_id=i, mode=name, status="ok", applied_dy=float(c[0]), applied_dx=float(c[1]), c0_dy=float(learned[i, 0]), c0_dx=float(learned[i, 1]),
                                                                                     applied_minus_learned_norm=float((c - learned[i]).norm()), scalar_scope="scene")
                if scale == "rr256":
                    gt = t[0]; y = o["y"]; r.update(l1_full=float((y - gt).abs().mean()), l1_roi=float(C.l1_roi(y, gt, "rr256")[0]), edge_l1_roi=float(C.edge_l1_roi(y, gt, "rr256")[0]), **{f"l1_band{b}": float(v) for b, v in enumerate(C.l1_roi(y, gt, "rr256", per_band=True)[0])},
                             **{f"rr_{k}": float(v) for k, v in reduced_metrics(x_true=gt, x_pred=y, max_pixel=mp).items()}, invalid_support=bool(not C.two_stage_ok(256, 256, (0, 0), c, 32)))
                else:
                    sr = ((o["y"][0].clip(-1, 1).float().cpu().numpy() + 1) / 2 * mp).astype(np.float64).transpose(1, 2, 0); p = fd["pan_raw"][i]; d = c.double().numpy()
                    pa_eval = C.warp_pan(torch.from_numpy(p)[None, None], torch.from_numpy(d)[None])[0, 0].numpy(); v, ok, _ = scene_views(sr, fd["lms_raw"][i].transpose(1, 2, 0), p, pa_eval, sensor, wald, d, 4, mp)
                    r.update(eligible_valid_views=bool(ok), **{f"{k}_{q}": float(v[k][q]) for k in VIEWS for q in ("hqnr", "d_lambda", "d_s", "fscc")})
                rows.append(r)
    return rows


@torch.no_grad()
def rr_proxy(L, fd):
    """§9 D30-B RR: 같은 해상도·phase 의 GT↔PAN 구조 (GT-informed diagnostic). canonical = −audit (PAN 을 MS 쪽으로 옮길 보정)."""
    sensor = fd["cfg"]["test_reduced_feeder_args"]["dataroot"]; kp = _pan_kernel("WV3", 4); G = dict(GATES, search_int=4, max_magnitude=4.0); rows = []; mp = fd["mp"]
    for i in range(len(fd["rr"])):
        gt, lms, ms, lpan, pan = (t.unsqueeze(0).to(DEV) for t in fd["rr"][i]); d = L.m(pan, ms, lpan)["delta"][0].double().cpu(); p = ((pan[0, 0].double().cpu().numpy() + 1) / 2 * mp); g = ((gt[0].double().cpu().numpy() + 1) / 2 * mp).mean(0)
        pt = C.warp_pan(torch.from_numpy(p)[None, None], d[None])[0, 0].numpy(); a = _est(blur_hr(p, kp), g, G); b = _est(blur_hr(pt, kp), g, G)
        rows.append(dict(model_key=L.key, scale="rr256", scene=i, delta_dy=float(d[0]), delta_dx=float(d[1]), audit_before_dy=a["dy"], audit_before_dx=a["dx"], canon_before_dy=-a["dy"], canon_before_dx=-a["dx"], before_mag=a["mag"], before_zncc=a["zncc"], before_accepted=a["accepted"],
                         audit_after_dy=b["dy"], audit_after_dx=b["dx"], after_mag=b["mag"], after_zncc=b["zncc"], after_accepted=b["accepted"], improved=bool(b["mag"] < a["mag"]), both_accepted=bool(a["accepted"] and b["accepted"]), reference="GT band-mean (GT-informed diagnostic)"))
    return rows


@torch.no_grad()
def bias_intervention(L, fd, ids, gt, ms, lpan, pan):
    """§9 D30-C: A_v = A + v — q 는 동일해야 하고(음성 대조), native e / HQNR 은 달라질 수 있다."""
    vs = [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5), (1.0, 0), (-1.0, 0), (0, 1.0), (0, -1.0)]; rows = []
    from pa.evalviews import scene_views
    from tools.eval_fr_paperset import sensor_of, load_dlpan
    sensor = sensor_of(fd["cfg"]); wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")); mp = fd["mp"]
    pr0 = C.probe_responses(L, pan, ms, banks=("A",)); y0, c0 = C.native_forward(L, pan, ms, lpan); e0 = C.l1_roi(y0, gt, "native64")
    for v in [(0.0, 0.0)] + vs:
        vt = torch.tensor(v, dtype=torch.float32); y, _ = C.native_forward(L, pan, ms, lpan, delta_override=c0 + vt); e = C.l1_roi(y, gt, "native64")
        qv = (pr0["A"]["resid"]).abs().mean(dim=(1, 2))        # (ĉε+v) + ε − (ĉ0+v) = r : 대수적으로 같다 — 수치 확인은 별도 forward 로
        p = pan[:8].to(DEV); mb = F.interpolate(ms[:8].to(DEV), scale_factor=4, mode="bicubic"); cbase = C.predict_c(L.m.aligner, p, mb, L.mg) + vt.to(DEV); qnum = []
        for k, prb in enumerate(pr0["A"]["probes"]):
            e_ = torch.tensor([[prb["ey"], prb["ex"]]], device=DEV).expand(8, 2); ce = C.predict_c(L.m.aligner, C.warp_pan(p, e_), mb, L.mg) + vt.to(DEV); qnum.append((ce + e_ - cbase).abs().mean(1))
        qnum = torch.stack(qnum, 1).mean(1).cpu()
        for i in range(len(ids)):
            rows.append(dict(model_key=L.key, scale="native64", sample_id=int(ids[i]), v_dy=v[0], v_dx=v[1], q_A_algebraic=float(qv[i]), q_A_numeric_check=(float(qnum[i]) if i < 8 else None), q_A_numeric_abs_diff=(float(abs(qnum[i] - qv[i])) if i < 8 else None),
                             e_roi=float(e[i]), e_roi_minus_base=float(e[i] - e0[i]), scalar_scope="sample"))
        for i in range(len(fd["fr"])):
            lms, msf, lpf, pf = (t.unsqueeze(0).to(DEV) for t in fd["fr"][i]); mbf = F.interpolate(msf, scale_factor=4, mode="bicubic"); cf = C.predict_c(L.m.aligner, pf, mbf, L.mg)[0].cpu() + vt
            o = L.m(pf, msf, lpf, delta_override=cf[None].to(DEV)); sr = ((o["y"][0].clip(-1, 1).float().cpu().numpy() + 1) / 2 * mp).astype(np.float64).transpose(1, 2, 0); pp = fd["pan_raw"][i]; d = cf.double().numpy()
            vv, ok, _ = scene_views(sr, fd["lms_raw"][i].transpose(1, 2, 0), pp, C.warp_pan(torch.from_numpy(pp)[None, None], torch.from_numpy(d)[None])[0, 0].numpy(), sensor, wald, d, 4, mp)
            rows.append(dict(model_key=L.key, scale="fr512", sample_id=i, v_dy=v[0], v_dx=v[1], raw_original_hqnr=float(vv["raw_original"]["hqnr"]), raw_original_fscc=float(vv["raw_original"]["fscc"]), scalar_scope="scene"))
    return rows


@torch.no_grad()
def landscape(L, ids, quad, gt, ms, lpan, pan):
    """§9 D30-D: 고정 U-Net 에서 c∈{−2..2}² (81) + c0 + c_geo — ROI L1·band·edge/non-edge L1·구조 objective(ZNCC of Scharr magnitude, W(P,c) vs up(MS) mean)."""
    kp = _pan_kernel("WV3", 4); G = dict(GATES, search_int=2, max_magnitude=2.0); rows = []; mp = C.MAX_PIXEL; sy, sx = C.roi_slice("native64")
    _, c0 = C.native_forward(L, pan, ms, lpan)
    for i in range(len(ids)):
        p = pan[i:i + 1].to(DEV); q = ms[i:i + 1].to(DEV); l = lpan[i:i + 1].to(DEV); g = gt[i:i + 1].to(DEV); mb = F.interpolate(q, scale_factor=4, mode="bicubic"); mref = mb.mean(1, keepdim=True)
        pdn = (p[0, 0].double().cpu().numpy() + 1) / 2 * mp; est = _est(blur_hr(pdn, kp), up_bicubic(((q[0].float().cpu().numpy() + 1) / 2 * mp).transpose(1, 2, 0)).mean(2), G)
        cgeo = (-est["dy"], -est["dx"]) if est["accepted"] else None; cands = [("grid", c) for c in GRID] + [("c0", (float(c0[i, 0]), float(c0[i, 1])))] + ([("c_geo", cgeo)] if cgeo else [])
        em = C.edge_mask_gt(g)[0]; hx, hy = C.scharr_t(mref); mm = (hx ** 2 + hy ** 2).sqrt()[..., sy, sx].flatten()
        for kind, (dy, dx) in cands:
            c = torch.tensor([[dy, dx]], device=DEV); o = L.m(p, q, l, delta_override=c); y = o["y"]; d = (y - g).abs()[0]
            l1 = float(d[:, sy, sx].mean()); eb = d[:, sy, sx].mean(dim=(1, 2)); e_edge = float(d[em][None].mean()) if em.any() else None; e_flat = float(d[~em].mean()) if (~em).any() else None
            gx, gy = C.scharr_t(o["pan_aligned"]); mw = (gx ** 2 + gy ** 2).sqrt()[..., sy, sx].flatten(); a_ = mw - mw.mean(); b_ = mm - mm.mean(); zn = float((a_ * b_).sum() / ((a_.norm() * b_.norm()) + 1e-12))
            rows.append(dict(model_key=L.key, sample_id=int(ids[i]), quadrant_primary=quad[ids[i]], kind=kind, c_dy=dy, c_dx=dx, l1_roi=l1, edge_l1=e_edge, flat_l1=e_flat, **{f"l1_band{b}": float(eb[b]) for b in range(8)}, struct_zncc=zn,
                             c_geo_accepted=bool(cgeo is not None), c_geo_dy=(cgeo[0] if cgeo else None), c_geo_dx=(cgeo[1] if cgeo else None), invalid_support=bool(not C.two_stage_ok(64, 64, (0, 0), (dy, dx), 16))))
    return rows


def landscape_summary(df):
    out = []
    for (mk, sid), g in df.groupby(["model_key", "sample_id"]):
        gr = g[g.kind == "grid"]; rec = gr.loc[gr.l1_roi.idxmin()]; st = gr.loc[gr.struct_zncc.idxmax()]; c0 = g[g.kind == "c0"].iloc[0]; cg = g[g.kind == "c_geo"]
        out.append(dict(model_key=mk, sample_id=int(sid), quadrant_primary=g.quadrant_primary.iloc[0], c_rec_grid_dy=float(rec.c_dy), c_rec_grid_dx=float(rec.c_dx), l1_at_c_rec=float(rec.l1_roi), search_boundary_rec=bool(abs(rec.c_dy) == 2 or abs(rec.c_dx) == 2),
                        c_struct_grid_dy=float(st.c_dy), c_struct_grid_dx=float(st.c_dx), l1_at_c_struct=float(st.l1_roi), search_boundary_struct=bool(abs(st.c_dy) == 2 or abs(st.c_dx) == 2), c0_dy=float(c0.c_dy), c0_dx=float(c0.c_dx), l1_at_c0=float(c0.l1_roi),
                        l1_at_c_geo=(float(cg.l1_roi.iloc[0]) if len(cg) else None), dist_c0_to_c_rec=float(math.hypot(c0.c_dy - rec.c_dy, c0.c_dx - rec.c_dx)), dist_struct_to_rec=float(math.hypot(st.c_dy - rec.c_dy, st.c_dx - rec.c_dx)),
                        gain_c_rec_vs_c0=float(c0.l1_roi - rec.l1_roi), gain_c_geo_vs_c0=(float(c0.l1_roi - cg.l1_roi.iloc[0]) if len(cg) else None), oracle_note="c_rec_grid* is a GT-using fixed-U-Net task oracle, not a registration GT or an inference method"))
    return out


def main(profile=False):
    fd = C.feeders(); ids, quad = detail_ids(); gt, ms, lpan, pan = C.load_patches(ids); nm = C.read_csv(os.path.join(CAMP, "native_sample_metrics.csv"))
    with C.Stage("D30" + ("-profile" if profile else ""), "correction interventions · native proxy · bias · landscape"):
        models = [("L1E4", s, t) for s in (1234, 7777, 2025) for t in ("best_raw", "last")] + [("L000", s, "best_raw") for s in (1234, 7777, 2025)] + [("N2", 2025, "last"), ("L3E4", 2025, "best_raw")]
        inter, prox, bias, land = [], [], [], []; blur_status = {}
        for fam, seed, tag in models:
            if C.ckpt_dir(fam, seed, tag) is None:
                continue
            L = C.load_model(fam, seed, tag); t0 = C.time.time(); e_by = dict(zip(nm[(nm.model_key == L.key) & (nm.scale == "native64")].sample_id, nm[(nm.model_key == L.key) & (nm.scale == "native64")].e_native_full)); cv = _const_vec(L.key)
            blur = calibration_patches(L.m, L.cfg, DEV) if not profile else dict(blur_status="not_run", sigma_star=None); blur_status[L.key] = blur
            if fam in ("L1E4",):
                inter += patch_interventions(L, ids, quad, gt, ms, lpan, pan, cv, e_by)
            if not profile:
                prows, psum = native_proxy(L.m, fd["fr"], fd["pan_raw"], "wv3", DEV); rr_rows = rr_proxy(L, fd)
                prox += [dict(model_key=L.key, scale="fr512", **r) for r in prows] + rr_rows; C.dump_json(os.path.join(CAMP, f"native_proxy_summary_{L.key}.json"), dict(fr512=psum))
                cg_fr = [((r["canon_before_dy"], r["canon_before_dx"]) if r["before_accepted"] else None) for r in prows]; cg_rr = [((r["canon_before_dy"], r["canon_before_dx"]) if r["before_accepted"] else None) for r in rr_rows]
                if fam in ("L1E4", "L000"):
                    inter += scene_interventions(L, fd, cv, cg_rr, cg_fr, blur)
            if (fam, seed, tag) == C.PRIMARY and not profile:
                bias += bias_intervention(L, fd, ids, gt, ms, lpan, pan)
            if fam == "L1E4" and tag == "best_raw" and (seed == 2025 or not profile):
                lids = [i for q in QUADS for i in C.load_json(os.path.join(CAMP, "detail_subsets.json"))["quadrants"][q]["landscape"]]; sel = [ids.index(i) for i in lids]
                land += landscape(L, lids, quad, gt[sel], ms[sel], lpan[sel], pan[sel])
            print(f"  {L.key}: {C.time.time() - t0:.0f}s", flush=True); del L; torch.cuda.empty_cache()
        C.dump_json(os.path.join(CAMP, "blur_control_status.json"), blur_status)
        df = pd.DataFrame(inter); df.to_csv(os.path.join(CAMP, "correction_interventions.csv"), index=False)
        if prox:
            pd.DataFrame(prox).to_csv(os.path.join(CAMP, "native_geometry_proxy.csv"), index=False)
        if bias:
            pd.DataFrame(bias).to_csv(os.path.join(CAMP, "bias_intervention.csv"), index=False)
        if land:
            ld = pd.DataFrame(land); ld.to_csv(os.path.join(CAMP, "correction_landscape.csv"), index=False); C.write_csv(os.path.join(CAMP, "correction_landscape_summary.csv"), landscape_summary(ld))
        # H3a/H3b: paired learned−zero gain per scale, block bootstrap by source group; Spearman(q_A, g)
        h3 = {}
        for mk, g in df[df.status.fillna("ok") == "ok"].groupby("model_key") if "status" in df else df.groupby("model_key"):
            h3[mk] = {}
            for sc, s in g.groupby("scale"):
                key = "l1_roi" if sc != "fr512" else "raw_original_hqnr"; L_ = s[s["mode"] == "I-L"].set_index("sample_id")[key]; Z = s[s["mode"] == "I-Z"].set_index("sample_id")[key]; W = s[s["mode"] == "I-W"].set_index("sample_id")[key]; Cc = s[s["mode"] == "I-C"].set_index("sample_id")[key]
                gain = (Z - L_) if sc != "fr512" else (L_ - Z); gw = (W - L_) if sc != "fr512" else (L_ - W); gc = (Cc - L_) if sc != "fr512" else (L_ - Cc)
                groups = nm[(nm.model_key == mk) & (nm.scale == "native64")].set_index("sample_id").source_group_id.reindex(gain.index).fillna("scene").values if sc == "native64" else np.array([f"{sc}{i}" for i in gain.index])
                q = nm[(nm.model_key == mk) & (nm.scale == sc)].set_index("sample_id").q_A.reindex(gain.index)
                sh = [s[s["mode"] == f"I-S{k}"].set_index("sample_id")[key] for k in range(1, 11)]; gs = pd.concat([(x - L_) if sc != "fr512" else (L_ - x) for x in sh], axis=1).mean(1)
                h3[mk][sc] = dict(n=int(len(gain)), gain_learned_vs_zero=C.block_bootstrap(gain.values, groups, seed=3), gain_learned_vs_wrong=C.block_bootstrap(gw.values, groups, seed=4), gain_learned_vs_constant=C.block_bootstrap(gc.values, groups, seed=5),
                                  gain_learned_vs_shuffle_mean10=C.block_bootstrap(gs.values, groups, seed=6), positive_fraction_vs_zero=float((gain > 0).mean()), H3b_spearman_qA_gain=C.spearman(q.values, gain.values),
                                  H3b_spearman_qA_gain_conditioned_on_zero_error_tertile={t: C.spearman(q.values[m_], gain.values[m_]) for t, m_ in zip(("low", "mid", "high"), [C.texture_tertile(Z.values) == t for t in ("low", "mid", "high")])} if sc != "fr512" else None,
                                  metric=key, sign="gain>0 = learned better")
        C.dump_json(os.path.join(CAMP, "h3_stats.json"), h3); pk = C.mkey(*C.PRIMARY); print(json.dumps(h3.get(pk), default=str)[:2000])
