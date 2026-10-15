"""D30B — 감사(research_log/PAN_EQREC4_Implementation_Experiment_Audit_2026-09-15.md) 보충 단계:
Q06 L1E4 6 checkpoint 의 native64 **전수** I-L/I-Z/I-W/I-C 와 **자기 checkpoint 4분면** 집계 · Q07 분할 A 에서의 blur calibration(energy·overshoot·고주파 비) · Q09 D20 MS-only/common 의 bank B, native64 상세 subset 의 I-G(estimator, secondary 포함),
RR bias 개입(A+v) 전후 위치 proxy, FR/RR native proxy 의 전체 estimator 필드."""
import json, math, os
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tools.eqrec4 import common as C
from tools.eqrec4.common import CAMP, DEV, QUADS
from tools.eqrec4.d20 import detail_ids, modality_response
from tools.palsv18_validate import blur_hr, up_bicubic, energy
from tools.metrics.jqm import _pan_kernel
from align.estimator import GATES

SIGMAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0); BIAS_V = [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5), (1.0, 0), (-1.0, 0), (0, 1.0), (0, -1.0)]


@torch.no_grad()
def core_interventions(L, man, gt, ms, lpan, pan, const_vec):
    own = C.own_quadrants(L.key); prim = C.own_quadrants(C.mkey(*C.PRIMARY)); _, c0 = C.native_forward(L, pan, ms, lpan); N = len(man)
    modes = {"I-L": c0, "I-Z": torch.zeros_like(c0), "I-W": -c0, "I-C": torch.tensor(const_vec).expand(N, 2).clone()}; per = {}
    for name, cvec in modes.items():
        y, _ = C.native_forward(L, pan, ms, lpan, delta_override=cvec); per[name] = (C.l1_roi(y, gt, "native64"), (y - gt).abs().mean(dim=(1, 2, 3)), C.edge_l1_roi(y, gt, "native64"))
    rows = []
    for i in range(N):
        sid = int(man.sample_id.iloc[i]); r = dict(model_key=L.key, sample_id=sid, split_role=man.split_role.iloc[i], source_group_id=man.source_group_id.iloc[i], quadrant_own=own.get(sid), quadrant_primary=prim.get(sid), c0_dy=float(c0[i, 0]), c0_dx=float(c0[i, 1]),
                                                 invalid_support=bool(not C.two_stage_ok(64, 64, (0, 0), c0[i], 16)))
        for name in modes:
            r[f"{name}_l1_roi"] = float(per[name][0][i]); r[f"{name}_l1_full"] = float(per[name][1][i]); r[f"{name}_edge_roi"] = float(per[name][2][i])
        r["gain_learned_vs_zero_roi"] = r["I-Z_l1_roi"] - r["I-L_l1_roi"]; r["gain_learned_vs_wrong_roi"] = r["I-W_l1_roi"] - r["I-L_l1_roi"]; r["gain_learned_vs_const_roi"] = r["I-C_l1_roi"] - r["I-L_l1_roi"]; rows.append(r)
    return rows


def h3_core(df, nm):
    out = {}
    for mk, g in df.groupby("model_key"):
        q = nm[(nm.model_key == mk) & (nm.scale == "native64")].set_index("sample_id"); res = {}
        for role, s in (("ALL", g), ("D", g[g.split_role == "D"])):
            qa = q.q_A.reindex(s.sample_id).values
            res[role] = dict(n=int(len(s)), gain_learned_vs_zero=C.block_bootstrap(s.gain_learned_vs_zero_roi.values, s.source_group_id.values, seed=31), gain_learned_vs_wrong=C.block_bootstrap(s.gain_learned_vs_wrong_roi.values, s.source_group_id.values, seed=32),
                             gain_learned_vs_const=C.block_bootstrap(s.gain_learned_vs_const_roi.values, s.source_group_id.values, seed=33), positive_fraction=float((s.gain_learned_vs_zero_roi > 0).mean()),
                             H3b_spearman_qA_gain=C.spearman_boot(qa, s.gain_learned_vs_zero_roi.values, s.source_group_id.values, seed=34),
                             by_quadrant_own={qd: dict(n=int(len(x)), n_groups=int(x.source_group_id.nunique()), gain_mean=float(x.gain_learned_vs_zero_roi.mean()), gain_median=float(x.gain_learned_vs_zero_roi.median()), positive_fraction=float((x.gain_learned_vs_zero_roi > 0).mean()),
                                                       spearman_qA_gain=C.spearman(q.q_A.reindex(x.sample_id).values, x.gain_learned_vs_zero_roi.values)) for qd, x in s.groupby("quadrant_own")},
                             by_quadrant_primary={qd: dict(n=int(len(x)), gain_mean=float(x.gain_learned_vs_zero_roi.mean())) for qd, x in s.groupby("quadrant_primary")},
                             own_vs_primary_label_agreement=float((s.quadrant_own == s.quadrant_primary).mean()))
        out[mk] = res
    return out


@torch.no_grad()
def blur_calibration_A(L, pan_A, ms_A):
    """감사 Q07: 계획 §10-E 대로 분할 A 에서 zero-phase symmetric blur 를 맞춘다 — Scharr energy 비(2%) + overshoot(0) + 고주파 power 비(5%) 를 모두 만족해야 matched."""
    mp = C.MAX_PIXEL; c0s, rw, ov, hf_w, rb, hf_b = [], [], [], [], {s: [] for s in SIGMAS}, {s: [] for s in SIGMAS}
    def hf_ratio(a):
        Fm = np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean()))) ** 2; h, w = a.shape; yy, xx = np.mgrid[:h, :w]; r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2); return float(Fm[r > 0.5].sum() / (Fm.sum() + 1e-12))
    for s in range(0, pan_A.shape[0], 128):
        p = pan_A[s:s + 128].to(DEV); mb = F.interpolate(ms_A[s:s + 128].to(DEV), scale_factor=4, mode="bicubic"); c0 = C.predict_c(L.m.aligner, p, mb, L.mg).double().cpu()
        for i in range(p.shape[0]):
            p0 = (p[i, 0].double().cpu().numpy() + 1) / 2 * mp; pw = C.warp_pan(torch.from_numpy(p0)[None, None], c0[i][None])[0, 0].numpy(); e0 = energy(p0, 4); c0s.append(c0[i].numpy())
            rw.append(energy(pw, 4) / max(e0, 1e-12)); ov.append(float(((pw < p0.min() - 1e-6) | (pw > p0.max() + 1e-6)).mean())); h0 = hf_ratio(p0); hf_w.append(hf_ratio(pw) / max(h0, 1e-12))
            for sg in SIGMAS:
                pb = gaussian_filter(p0, sg, mode="nearest") if sg > 0 else p0; rb[sg].append(energy(pb, 4) / max(e0, 1e-12)); hf_b[sg].append(hf_ratio(pb) / max(h0, 1e-12))
    target = float(np.mean(rw)); ratios = {s: float(np.mean(v)) for s, v in rb.items()}; hfr = {s: float(np.mean(v)) for s, v in hf_b.items()}; best = min(SIGMAS, key=lambda s: abs(ratios[s] - target)); hf_target = float(np.mean(hf_w))
    checks = dict(energy_within_2pct=bool(abs(ratios[best] - target) <= 0.02), warp_not_sharpening=bool(target <= 1.0), overshoot_zero=bool(np.mean(ov) <= 1e-3), highfreq_within_5pct=bool(abs(hfr[best] - hf_target) <= 0.05 * max(hf_target, 1e-12)), sigma_positive=bool(best > 0))
    return dict(model_key=L.key, n=int(pan_A.shape[0]), split="A", c0_median_dy_dx=np.median(np.array(c0s), 0).tolist(), warp_energy_ratio_target=target, warp_overshoot_fraction_mean=float(np.mean(ov)), warp_highfreq_ratio_target=hf_target,
                blur_energy_ratio_by_sigma=ratios, blur_highfreq_ratio_by_sigma=hfr, sigma_star=float(best), checks=checks, blur_status=("matched" if all(checks.values()) else "unmatched_blur_control"),
                note="A-split calibration (plan §10-E); D30's blur rows used the PALSV18 train-256 calibration and are exploratory")


@torch.no_grad()
def native64_ig(L, ids, quad, gt, ms, lpan, pan):
    """감사 Q09: 상세 subset 의 I-G — estimator(primary+secondary) 로 canonical c_geo, accepted 만 적용 (미수용은 0 으로 채우지 않는다)."""
    kp = _pan_kernel("WV3", 4); G = dict(GATES, search_int=2, max_magnitude=2.0); mp = C.MAX_PIXEL; own = C.own_quadrants(L.key); _, c0 = C.native_forward(L, pan, ms, lpan); rows = []; cg = torch.full_like(c0, float("nan")); ests = []
    for i in range(len(ids)):
        p0 = (pan[i, 0].double().numpy() + 1) / 2 * mp; up = up_bicubic(((ms[i].double().numpy() + 1) / 2 * mp).transpose(1, 2, 0)).mean(2); e = C.est_full(blur_hr(p0, kp), up, G); ests.append(e)
        if e["accepted"]:
            cg[i] = torch.tensor([-e["dy"], -e["dx"]])
    yl, _ = C.native_forward(L, pan, ms, lpan, delta_override=c0); ll = C.l1_roi(yl, gt, "native64"); ok = torch.isfinite(cg).all(1); yg, _ = C.native_forward(L, pan, ms, lpan, delta_override=torch.where(ok[:, None], cg, c0)); lg = C.l1_roi(yg, gt, "native64")
    for i in range(len(ids)):
        e = ests[i]; rows.append(dict(model_key=L.key, sample_id=int(ids[i]), quadrant_primary=quad.get(int(ids[i])), quadrant_own=own.get(int(ids[i])), c0_dy=float(c0[i, 0]), c0_dx=float(c0[i, 1]), **{f"est_{k}": v for k, v in e.items()},
                         c_geo_dy=(float(cg[i, 0]) if ok[i] else None), c_geo_dx=(float(cg[i, 1]) if ok[i] else None), l1_learned=float(ll[i]), l1_geo=(float(lg[i]) if ok[i] else None), gain_geo_vs_learned=((float(ll[i] - lg[i])) if ok[i] else None), status=("ok" if ok[i] else "estimator_not_accepted")))
    return rows


@torch.no_grad()
def proxies_full(L, fd):
    """FR PAN↔up(MS) 와 RR PAN↔GT-mean 의 native before/after proxy — estimator 전체 필드 + identity/known-shift 검사 (감사 Q09)."""
    kp = _pan_kernel("WV3", 4); G = dict(GATES, search_int=4, max_magnitude=4.0); mp = fd["mp"]; rows = []
    p0 = fd["pan_raw"][0]; ident = C.est_full(blur_hr(p0, kp), blur_hr(p0, kp), G); known = C.est_full(blur_hr(C.warp_pan(torch.from_numpy(p0)[None, None], torch.tensor([[1.0, 0.0]], dtype=torch.float64))[0, 0].numpy(), kp), blur_hr(p0, kp), G)
    sign_ok = bool(abs(-known["dy"] + 1.0) < 0.25 and abs(known["dx"]) < 0.25)
    for scale, ds in (("fr512", fd["fr"]), ("rr256", fd["rr"])):
        for i in range(len(ds)):
            t = [x.unsqueeze(0).to(DEV) for x in ds[i]]; pan, ms, lpan = t[-1], t[-3], t[-2]; d = L.m(pan, ms, lpan)["delta"][0].double().cpu()
            p = fd["pan_raw"][i] if scale == "fr512" else (pan[0, 0].double().cpu().numpy() + 1) / 2 * mp; pt = C.warp_pan(torch.from_numpy(p)[None, None], d[None])[0, 0].numpy()
            ref = up_bicubic(((ms[0].float().cpu().numpy() + 1) / 2 * mp).transpose(1, 2, 0)).mean(2) if scale == "fr512" else ((t[0][0].double().cpu().numpy() + 1) / 2 * mp).mean(0)
            a = C.est_full(blur_hr(p, kp), ref, G); b = C.est_full(blur_hr(pt, kp), ref, G)
            rows.append(dict(model_key=L.key, scale=scale, scene=i, reference=("up(MS) mean" if scale == "fr512" else "GT band-mean (GT-informed)"), delta_dy=float(d[0]), delta_dx=float(d[1]), **{f"before_{k}": v for k, v in a.items()}, canon_before_dy=-a["dy"], canon_before_dx=-a["dx"],
                             **{f"after_{k}": v for k, v in b.items()}, improved=bool(b["mag"] < a["mag"]), both_accepted=bool(a["accepted"] and b["accepted"]), high_confidence=bool(a["accepted"] and b["accepted"] and (a["primary_secondary_diff"] or 9) <= 0.25 and (b["primary_secondary_diff"] or 9) <= 0.25),
                             identity_check_mag=ident["mag"], known_shift_sign_ok=sign_ok))
    return rows


@torch.no_grad()
def rr_bias_proxy(L, fd):
    kp = _pan_kernel("WV3", 4); G = dict(GATES, search_int=4, max_magnitude=4.0); mp = fd["mp"]; rows = []
    for i in range(len(fd["rr"])):
        gt, lms, ms, lpan, pan = (t.unsqueeze(0).to(DEV) for t in fd["rr"][i]); d = L.m(pan, ms, lpan)["delta"][0].double().cpu(); p = (pan[0, 0].double().cpu().numpy() + 1) / 2 * mp; g = ((gt[0].double().cpu().numpy() + 1) / 2 * mp).mean(0)
        for v in [(0.0, 0.0)] + BIAS_V:
            pt = C.warp_pan(torch.from_numpy(p)[None, None], (d + torch.tensor(v, dtype=torch.float64))[None])[0, 0].numpy(); b = C.est_full(blur_hr(pt, kp), g, G)
            rows.append(dict(model_key=L.key, scene=i, v_dy=v[0], v_dx=v[1], applied_dy=float(d[0] + v[0]), applied_dx=float(d[1] + v[1]), **{f"after_{k}": vv for k, vv in b.items()}))
    return rows


def main(profile=False):
    fd = C.feeders(); man = C.make_manifest(); ids, quad = detail_ids(); nm = C.read_csv(os.path.join(CAMP, "native_sample_metrics.csv"))
    with C.Stage("D30B" + ("-profile" if profile else ""), "audit supplement: core interventions (own quadrant) · A-split blur calibration · bank B modality · I-G · proxies"):
        gtA, msA, lpA, pnA = C.load_patches(man[man.split_role == "A"].sample_id.tolist()); gtD, msD, lpD, pnD = C.load_patches(ids); gt, ms, lpan, pan = C.load_patches(man.sample_id.tolist()) if not profile else (gtA, msA, lpA, pnA)
        rr_pan = torch.cat([fd["rr"][i][4].unsqueeze(0) for i in range(20)]); rr_ms = torch.cat([fd["rr"][i][2].unsqueeze(0) for i in range(20)]); fr_pan = torch.cat([fd["fr"][i][3].unsqueeze(0) for i in range(20)]); fr_ms = torch.cat([fd["fr"][i][1].unsqueeze(0) for i in range(20)])
        core, blur, modB, ig, prox, bias = [], {}, [], [], [], []
        for fam, seed, tag in [("L1E4", s, t) for s in (1234, 7777, 2025) for t in ("best_raw", "last")] + [("L000", s, "best_raw") for s in (1234, 7777, 2025)]:
            if C.ckpt_dir(fam, seed, tag) is None:
                continue
            L = C.load_model(fam, seed, tag); t0 = C.time.time(); a = nm[(nm.model_key == L.key) & (nm.split_role == "A") & (nm.scale == "native64")]; cv = [float(a.c0_dy.median()), float(a.c0_dx.median())]
            if fam == "L1E4":
                core += core_interventions(L, man if not profile else man[man.split_role == "A"], gt, ms, lpan, pan, cv); blur[L.key] = blur_calibration_A(L, pnA, msA)
                if tag == "best_raw" or (fam, seed, tag) == C.PRIMARY:
                    for mode in ("ms_only", "common", "pan_only"):
                        for sc, (p, q, sid) in (("native64", (pnD, msD, ids)), ("rr256", (rr_pan, rr_ms, list(range(20)))), ("fr512", (fr_pan, fr_ms, list(range(20))))):
                            modB += [dict(model_key=L.key, scale=sc, sample_id=int(sid[i]), probe_bank="B", **r) for i, r in enumerate(modality_response(L, p, q, mode, bank="B"))]
                    ig += native64_ig(L, ids, quad, gtD, msD, lpD, pnD)
            if not profile:
                prox += proxies_full(L, fd)
            if (fam, seed, tag) == C.PRIMARY and not profile:
                bias += rr_bias_proxy(L, fd)
            print(f"  {L.key}: {C.time.time() - t0:.0f}s", flush=True); del L; torch.cuda.empty_cache()
        df = pd.DataFrame(core); df.to_csv(os.path.join(CAMP, "correction_interventions_core.csv"), index=False); C.dump_json(os.path.join(CAMP, "h3_core_stats.json"), h3_core(df, nm)); C.dump_json(os.path.join(CAMP, "blur_control_status_A.json"), blur)
        if modB:
            mb = pd.DataFrame(modB); mb.to_csv(os.path.join(CAMP, "geometry_modality_bankB.csv"), index=False)
            summ = [dict(model_key=mk, scale=sc, mode=mode, probe_bank="B", n=int(len(s)), B_yy_mean=float(s.B_yy.mean()), B_xx_mean=float(s.B_xx.mean()), B_yx_mean=float(s.B_yx.mean()), B_xy_mean=float(s.B_xy.mean()), b_y_mean=float(s.b_y.mean()), b_x_mean=float(s.b_x.mean()), fit_rmse_mean=float(s.fit_rmse.mean()),
                         epe_mean=float(s.epe.mean()), epe_p90=float(s.epe.quantile(.9)), q_component_mean=float(s.q_component.mean()), **{f"epe_r{r}_mean": float(s[f"epe_r{r}"].mean()) for r in C.RADII}, expected_B=s.expected_B.iloc[0], source="bank B (D30B)") for (mk, sc, mode), s in mb.groupby(["model_key", "scale", "mode"])]
            C.write_csv(os.path.join(CAMP, "response_matrices_bankB.csv"), summ)
        if ig:
            pd.DataFrame(ig).to_csv(os.path.join(CAMP, "native64_ig.csv"), index=False)
        if prox:
            pd.DataFrame(prox).to_csv(os.path.join(CAMP, "native_geometry_proxy_full.csv"), index=False)
        if bias:
            pd.DataFrame(bias).to_csv(os.path.join(CAMP, "bias_position_proxy_rr.csv"), index=False)
        pk = C.mkey(*C.PRIMARY); h = C.load_json(os.path.join(CAMP, "h3_core_stats.json")).get(pk, {}); print(json.dumps(dict(all=h.get("ALL", {}).get("gain_learned_vs_zero"), by_own=h.get("ALL", {}).get("by_quadrant_own"), blur=blur.get(pk, {}).get("blur_status")), default=str)[:1200])
