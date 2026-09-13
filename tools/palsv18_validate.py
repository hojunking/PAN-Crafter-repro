#!/usr/bin/env python
"""PALSV18 정합 검증 V2–V4 (계획 research_log/PAN_L1E4_Refinement_AlignmentValidation_S1_18GPUh_2026-09-13.md §8–§10) — 한 run·한 checkpoint 의 사후 진단 (학습 그래프·RNG 불변, no_grad).

    python tools/palsv18_validate.py --run <run> --ckpt best_hqnr|last [--parts v2,v3,v4] [--stress-dirs 4|8]

V1(signed 2D 반응, palsv18 probe) 은 tools/po10_diag.py --probe-set palsv18 이 맡는다. 여기서는
  V2  모달리티 대응: pan_only / ms_only / common (기대 c_e−c_0 ≈ −e / +e / 0), 장면별 B_i fit·잔차 EPE, 반경별            → palsv18/<ckpt>_modality.json (+ rows csv)
  V3  native 위치 proxy(독립 추정기 Scharr-ZNCC + census, PAN_b←up(MS) before/after, 장면별 벡터·신뢰도) · RR edge_profile_v1 (GT 고정 edge mask 의 오차, 50% crossing·10–90% 폭)
      · 보간/선명도: warp 전후 Scharr energy 비, zero-shift identity, coordinate ramp, energy-match blur 대조(train 256 patch calibration)                          → palsv18/<ckpt>_{native_proxy.csv, edge_profile.csv, energy.json}
  V4  correction 치환(learned / zero / wrong_sign / scene_shuffle / constant_calibration / blur_energy_match) 의 FR view · native-reference stress (r {0.5,1,2} × 방향) → palsv18/<ckpt>_{interventions.json, stress.json}
요약 results/palsv18_<ckpt>.json. 판정에 쓰지 않는다 — 주 판정은 best_raw raw_original HQNR → fSCC. 여기 값은 proxy 근거이며 센서 GT 정합 오차가 아니다."""
import argparse, csv, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F, h5py, yaml
from scipy.ndimage import map_coordinates, gaussian_filter, correlate
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tools.po10_diag import load_run, datasets, warp_generic, fixed_probes, write_csv, stress_hqnr_native   # noqa: E402
from pa.offset import predict_c                                                                             # noqa: E402
from pa.warp import warp_pan                                                                                # noqa: E402
from pa.evalviews import scene_views, VIEWS                                                                 # noqa: E402
from tools.pa_diag import refs, dn                                                                          # noqa: E402
from tools.eval_fr_paperset import sensor_of, load_dlpan                                                    # noqa: E402
from tools.align_after_training_diag import blur_hr, up_bicubic                                             # noqa: E402
from align.estimator import estimate_shift, GATES                                                           # noqa: E402
from tools.metrics.jqm import _pan_kernel                                                                   # noqa: E402
from main import import_class                                                                               # noqa: E402

TOOL_VERSION = "2026-09-13.2"          # 리뷰 반영: canonical proxy 방향, 적격 장면만 평균, shortcut 대조 best/last, stress 8 방향 + c0−ε 기준선 + RR GT stress
CAMP = os.path.join(ROOT, "work_dir", "_palsv18_campaign"); RR_STRESS_MARGIN = 32
KX = np.array([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]) / 32.0; KY = KX.T.copy()
SIGMAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0); BLUR_TOL = 0.02; EDGE_TOP = 0.30; CELL = 16; PROF_T = np.linspace(-4, 4, 33)


def scharr(a):
    return correlate(a, KX, mode="nearest"), correlate(a, KY, mode="nearest")


def energy(a, crop=8):
    gx, gy = scharr(a); return float((gx[crop:-crop, crop:-crop] ** 2 + gy[crop:-crop, crop:-crop] ** 2).mean())


def fit_scene(es, qs):
    E = np.array([[e[0], e[1], 1.0] for e in es]); Q = np.array(qs); c, *_ = np.linalg.lstsq(E, Q, rcond=None); B = c[:2].T; res = Q - E @ c
    return dict(B=B.tolist(), a=c[2].tolist(), sv=np.linalg.svd(B, compute_uv=False).tolist(), fit_rmse=float(np.sqrt((res ** 2).mean())))


# ------------------------------------------------------------------ V2
@torch.no_grad()
def modality(m, ds, R, mg, dev):
    probes = [p for p in fixed_probes(R, "palsv18")[0] if p != (0.0, 0.0)]; expect = dict(pan_only=-1.0, ms_only=+1.0, common=0.0)
    out, rows = {}, []
    for name, samples in ds.items():
        per = {k: [] for k in expect}
        for i, (pan, ms) in enumerate(samples):
            pan, ms = pan.to(dev), ms.to(dev); mb = F.interpolate(ms, scale_factor=4, mode="bicubic"); c0 = predict_c(m.aligner, pan, mb, mg)[0].cpu().numpy()
            for mode, sgn in expect.items():
                es, qs, rs = [], [], []
                for ey, ex in probes:
                    e = torch.tensor([[ey, ex]], device=dev)
                    pe = warp_generic(pan, e) if mode != "ms_only" else pan; me = warp_generic(mb, e) if mode != "pan_only" else mb     # MS-only: upsampled M 의 8 band 에 같은 HR shift
                    ce = predict_c(m.aligner, pe, me, mg)[0].cpu().numpy(); q = ce - c0; r_ = q - sgn * np.array([ey, ex])
                    es.append((ey, ex)); qs.append(q.tolist()); rs.append(r_)
                    rows.append(dict(scale=name, sample=i, mode=mode, ey=ey, ex=ex, c0_dy=float(c0[0]), c0_dx=float(c0[1]), q_dy=float(q[0]), q_dx=float(q[1]), resid_norm=float(np.linalg.norm(r_))))
                rs = np.array(rs); ft = fit_scene(es, qs)
                per[mode].append(dict(sample=i, epe=float(np.linalg.norm(rs, axis=1).mean()), mae_component=float(np.abs(rs).mean()), **ft,
                                      by_radius={f"{round(math.hypot(a, b), 2)}": float(np.mean([np.linalg.norm(r) for (a2, b2), r in zip(es, rs) if math.isclose(math.hypot(a2, b2), math.hypot(a, b))])) for a, b in es}))
        out[name] = {}
        for mode, lst in per.items():
            ep = np.array([x["epe"] for x in lst]); Bs = np.array([x["B"] for x in lst])
            out[name][mode] = dict(expected_B=("-I" if mode == "pan_only" else "+I" if mode == "ms_only" else "0"), n_scenes=len(lst), epe_mean=float(ep.mean()), epe_p50=float(np.median(ep)), epe_p90=float(np.percentile(ep, 90)),
                                   mae_component_mean=float(np.mean([x["mae_component"] for x in lst])), B_mean=Bs.mean(0).tolist(), B_diag_mean=[float(Bs[:, 0, 0].mean()), float(Bs[:, 1, 1].mean())], B_cross_mean=[float(Bs[:, 0, 1].mean()), float(Bs[:, 1, 0].mean())],
                                   sv_mean=np.mean([x["sv"] for x in lst], 0).tolist(), fit_rmse_mean=float(np.mean([x["fit_rmse"] for x in lst])),
                                   epe_by_radius={k: float(np.mean([x["by_radius"][k] for x in lst])) for k in sorted(lst[0]["by_radius"], key=float)}, per_scene=lst)
    out["probe_manifest"] = dict(radii_hr=[0.25, 0.5, 1.0, 2.0], directions=8, n_nonzero=len(probes), no_response_reference_epe=float(np.mean([math.hypot(a, b) for a, b in probes])), view_margin=mg,
                                 note="probe on the full input first, then the same aligner view (margin 4); common-shift zero is not a hard pass line (finite crop/padding)")
    return out, rows


# ------------------------------------------------------------------ V3.1 native proxy
def _est(ref, mov, G):
    r = estimate_shift(ref.astype(np.float32), mov.astype(np.float32), G)
    return dict(dy=r["dy_lr_raw"], dx=r["dx_lr_raw"], mag=r["magnitude_raw"], zncc=r.get("peak_zncc"), margin=r.get("peak_margin"), accepted=bool(r.get("accepted")), boundary=bool(r.get("boundary_hit")))


@torch.no_grad()
def native_proxy(m, ds, pan_raw, sensor, dev):
    G = dict(GATES, search_int=4, max_magnitude=4.0); kp = _pan_kernel(sensor.upper(), 4); rows = []
    p0 = pan_raw[0]; ident = _est(blur_hr(p0, kp), blur_hr(p0, kp), G)
    known = _est(blur_hr(p0, kp), blur_hr(warp_pan(torch.from_numpy(p0)[None, None], torch.tensor([[1.0, 0.0]], dtype=torch.float64))[0, 0].numpy(), kp), G)
    for i in range(len(ds)):
        lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i]); d = m(pan, ms, lpan)["delta"][0].double().cpu()
        p = pan_raw[i]; pt = warp_pan(torch.from_numpy(p)[None, None], d[None])[0, 0].numpy(); up = up_bicubic(((ms[0].float().cpu().numpy() + 1) / 2 * 2047.0).transpose(1, 2, 0)).mean(2)
        a = _est(blur_hr(p, kp), up, G); b = _est(blur_hr(pt, kp), up, G)
        cb = np.array([-a["dy"], -a["dx"]]); dv = d.numpy(); cosang = float(np.dot(cb, dv) / (np.linalg.norm(cb) * np.linalg.norm(dv) + 1e-12)) if np.linalg.norm(cb) > 1e-9 and np.linalg.norm(dv) > 1e-9 else None
        rows.append(dict(scene=i, delta_dy=float(d[0]), delta_dx=float(d[1]), delta_norm=float(d.norm()),
                         audit_before_dy=a["dy"], audit_before_dx=a["dx"], canon_before_dy=float(-a["dy"]), canon_before_dx=float(-a["dx"]), before_mag=a["mag"], before_zncc=a["zncc"], before_accepted=a["accepted"], before_boundary=a["boundary"],
                         audit_after_dy=b["dy"], audit_after_dx=b["dx"], canon_after_dy=float(-b["dy"]), canon_after_dx=float(-b["dx"]), after_mag=b["mag"], after_zncc=b["zncc"], after_accepted=b["accepted"], after_boundary=b["boundary"],
                         cos_canon_before_vs_model_c=cosang, improved=bool(b["mag"] < a["mag"]), both_accepted=bool(a["accepted"] and b["accepted"])))
    acc = [r for r in rows if r["both_accepted"]]
    summ = dict(n=len(rows), n_both_accepted=len(acc), before_mag_median=float(np.median([r["before_mag"] for r in rows])), after_mag_median=float(np.median([r["after_mag"] for r in rows])),
                before_mag_median_accepted=(float(np.median([r["before_mag"] for r in acc])) if acc else None), after_mag_median_accepted=(float(np.median([r["after_mag"] for r in acc])) if acc else None),
                n_improved=int(sum(r["improved"] for r in rows)), n_improved_accepted=int(sum(r["improved"] for r in acc)),
                estimator="align/estimator.estimate_shift (Scharr -> median/MAD -> top-30% edge -> ZNCC -> quadratic subpixel; secondary census5 gate inside)", secondary_independent_estimator="not_available (census is a gate of the same estimator, not an independent phase-correlation)",
                cos_canon_before_vs_model_c_mean=(float(np.mean([r["cos_canon_before_vs_model_c"] for r in rows if r["cos_canon_before_vs_model_c"] is not None])) if any(r["cos_canon_before_vs_model_c"] is not None for r in rows) else None),
                identity_check=ident, known_shift_check=dict(applied_pan_shift_dy_dx=[1.0, 0.0], audit_measured=known, canonical_pan_to_ms_of_audit=[-known["dy"], -known["dx"]],
                                                            note="audit = shift of moving(MS) w.r.t. reference(PAN_b); P = W(M,(+1,0)) gives audit ≈ (+1,0) while the PAN correction needed is (−1,0) → canonical = −audit (plan §9.1)"),
                note="proxy evidence for native position; not a sensor ground truth. 0.557 / 1.79 px are never substituted as per-scene GT")
    return rows, summ


# ------------------------------------------------------------------ V3.2 edge_profile_v1 (RR, GT 고정 edge)
def _profile(img, y, x, ny, nx):
    yy = y + PROF_T * ny; xx = x + PROF_T * nx
    if yy.min() < 0 or xx.min() < 0 or yy.max() > img.shape[0] - 1 or xx.max() > img.shape[1] - 1:
        return None
    return map_coordinates(img, [yy, xx], order=1, mode="nearest")


def _crossings(prof, level, tmax=2.0):
    s = np.sign(prof - level); idx = [k for k in range(len(prof) - 1) if s[k] != s[k + 1] and s[k] != 0 and abs(PROF_T[k]) <= tmax]
    return [float(PROF_T[k] + (level - prof[k]) / (prof[k + 1] - prof[k]) * (PROF_T[k + 1] - PROF_T[k])) for k in idx]


@torch.no_grad()
def edge_profile(m, cfg, mp, dev, is_pa):
    Feeder = import_class(cfg["feeder"]); ds = Feeder(**cfg["test_reduced_feeder_args"]); rows = []
    for i in range(len(ds)):
        gt, lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i]); o = m(pan, ms, lpan) if is_pa else m(pan, ms, lpan, aligner_enabled=False)
        G_ = dn(gt[0], mp); Y_ = dn(o["y"][0].clamp(-1, 1), mp)
        for b in range(G_.shape[0]):
            g, y = G_[b], Y_[b]; gx, gy = scharr(g); mag = np.hypot(gx, gy); yx, yy = scharr(y)
            thr = np.percentile(mag, 100 * (1 - EDGE_TOP)); mask = mag >= thr
            edge_mae = float(np.abs(y - g)[mask].mean()); flat_mae = float(np.abs(y - g)[~mask].mean()); edge_gmae = float((np.abs(yx - gx) + np.abs(yy - gy))[mask].mean() * 0.5)
            cmin = 0.1 * (np.percentile(g, 99) - np.percentile(g, 1)); H, W = g.shape; n_c = n_simple = n_meas = n_miss = n_multi = 0; offs, wg, wo = [], [], []
            cand = mag >= np.percentile(mag, 70)
            for cy in range(0, H - CELL + 1, CELL):
                for cx in range(0, W - CELL + 1, CELL):
                    sub = mag[cy:cy + CELL, cx:cx + CELL] * cand[cy:cy + CELL, cx:cx + CELL]
                    if sub.max() <= 0:
                        continue
                    k = int(np.argmax(sub)); py, px = cy + k // CELL, cx + k % CELL; ny, nx = gy[py, px] / max(mag[py, px], 1e-9), gx[py, px] / max(mag[py, px], 1e-9)
                    pg = _profile(g, py, px, ny, nx); po = _profile(y, py, px, ny, nx)
                    if pg is None or po is None:
                        continue
                    n_c += 1; lo, hi = pg[PROF_T <= -2].mean(), pg[PROF_T >= 2].mean()
                    if abs(hi - lo) < cmin:
                        continue
                    mid = (lo + hi) / 2; cg = _crossings(pg, mid)
                    if len(cg) != 1:
                        continue
                    n_simple += 1; lvl10, lvl90 = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo); c10, c90 = _crossings(pg, lvl10, 4.0), _crossings(pg, lvl90, 4.0)
                    if not (c10 and c90):
                        continue
                    w_gt = abs(min(c90, key=lambda t: abs(t - cg[0])) - min(c10, key=lambda t: abs(t - cg[0]))); co = _crossings(po, mid)
                    if len(co) == 0:
                        n_miss += 1; continue
                    if len(co) > 1:
                        n_multi += 1; continue
                    o10, o90 = _crossings(po, lvl10, 4.0), _crossings(po, lvl90, 4.0)
                    if not (o10 and o90):
                        n_miss += 1; continue
                    n_meas += 1; offs.append(co[0] - cg[0]); wg.append(w_gt); wo.append(abs(min(o90, key=lambda t: abs(t - co[0])) - min(o10, key=lambda t: abs(t - co[0]))))
            rows.append(dict(scene=i, band=b, n_candidates=n_c, n_simple=n_simple, n_measured=n_meas, n_missing=n_miss, n_multiple=n_multi, crossing_offset_mean=(float(np.mean(offs)) if offs else None),
                             crossing_offset_mean_abs=(float(np.mean(np.abs(offs))) if offs else None), width_gt_mean=(float(np.mean(wg)) if wg else None), width_out_mean=(float(np.mean(wo)) if wo else None),
                             width_ratio=(float(np.mean(wo) / np.mean(wg)) if wg and wo else None), edge_mae_dn=edge_mae, flat_mae_dn=flat_mae, edge_grad_mae_dn=edge_gmae, contrast_min_dn=float(cmin)))
    def mean_of(k):
        v = [r[k] for r in rows if r.get(k) is not None]; return float(np.mean(v)) if v else None
    summ = dict(spec="edge_profile_v1: GT Scharr top-30% mask (errors); candidates top-30% gradient, strongest per 16x16 cell, profile +-4 HR px @0.25 along GT gradient normal; simple edge = one 50% crossing within |t|<=2 and plateau contrast >= 0.1*(p99-p1)",
                n_rows=len(rows), n_candidates=int(sum(r["n_candidates"] for r in rows)), n_simple=int(sum(r["n_simple"] for r in rows)), n_measured=int(sum(r["n_measured"] for r in rows)), n_missing=int(sum(r["n_missing"] for r in rows)), n_multiple=int(sum(r["n_multiple"] for r in rows)),
                crossing_offset_mean=mean_of("crossing_offset_mean"), crossing_offset_mean_abs=mean_of("crossing_offset_mean_abs"), width_gt_mean=mean_of("width_gt_mean"), width_out_mean=mean_of("width_out_mean"), width_ratio_mean=mean_of("width_ratio"),
                edge_mae_dn=mean_of("edge_mae_dn"), flat_mae_dn=mean_of("flat_mae_dn"), edge_grad_mae_dn=mean_of("edge_grad_mae_dn"), note="diagnostic only (not a loss/selector); GT not warped; RR test 20 scenes x 8 bands")
    return rows, summ


# ------------------------------------------------------------------ V3.3 energy / blur calibration
@torch.no_grad()
def calibration_patches(m, cfg, dev, n=256, seed=20260913):
    """train 256 patch (원 좌표, augmentation 없음, 고정 index) 에서 c0·warp energy 비·blur energy 비 — 점수(FR/RR) 를 보지 않는 calibration."""
    with h5py.File(cfg["train_feeder_args"]["dataroot"]) as f:
        N = f["pan"].shape[0]; idx = np.sort(np.random.RandomState(seed).choice(N, n, replace=False))
        pan = np.asarray(f["pan"][idx], dtype=np.float32); ms = np.asarray(f["ms"][idx], dtype=np.float32)
    mp = float(cfg["max_pixel"]); c0s, rw, rb = [], [], {s: [] for s in SIGMAS}
    for i in range(n):
        p = torch.from_numpy(pan[i:i + 1]) * 2 / mp - 1; q = torch.from_numpy(ms[i:i + 1]) * 2 / mp - 1
        c0 = m.predict_delta(p.to(dev), F.interpolate(q.to(dev), scale_factor=4, mode="bicubic"))[0].double().cpu(); c0s.append(c0.numpy())
        p0 = pan[i, 0].astype(np.float64); pw = warp_pan(torch.from_numpy(p0)[None, None], c0[None])[0, 0].numpy(); e0 = energy(p0, 4); rw.append(energy(pw, 4) / max(e0, 1e-12))
        for s in SIGMAS:
            rb[s].append((energy(gaussian_filter(p0, s, mode="nearest"), 4) if s > 0 else e0) / max(e0, 1e-12))
    c0s = np.array(c0s); target = float(np.mean(rw)); ratios = {s: float(np.mean(v)) for s, v in rb.items()}
    best = min(SIGMAS, key=lambda s: abs(ratios[s] - target)); status = ("unmatched_blur_control" if (target > 1.0 or abs(ratios[best] - target) > BLUR_TOL) else "matched")
    return dict(n=n, index_seed=seed, c0_median_dy_dx=np.median(c0s, 0).tolist(), c0_mean_dy_dx=c0s.mean(0).tolist(), c0_norm_median=float(np.median(np.linalg.norm(c0s, axis=1))),
                warp_energy_ratio_target=target, blur_energy_ratio_by_sigma=ratios, sigma_star=float(best), blur_status=status, tolerance=BLUR_TOL,
                note="constant_calibration vector = median c0 over train patches in original orientation (no augmentation); blur sigma fitted on calibration energy ratio only (no FR/RR scores)")


@torch.no_grad()
def energy_fr(m, ds, pan_raw, dev):
    rows = []
    for i in range(len(ds)):
        lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i]); d = m(pan, ms, lpan)["delta"][0].double().cpu(); p = pan_raw[i]
        pw = warp_pan(torch.from_numpy(p)[None, None], d[None])[0, 0].numpy(); e0 = energy(p); rows.append(dict(scene=i, energy_ratio=energy(pw) / max(e0, 1e-12), overshoot_frac=float(((pw < p.min()) | (pw > p.max())).mean())))
    p = pan_raw[0]; ident = float(np.abs(warp_pan(torch.from_numpy(p)[None, None], torch.zeros(1, 2, dtype=torch.float64))[0, 0].numpy() - p).max())
    ramp = np.tile(np.arange(p.shape[1], dtype=np.float64), (p.shape[0], 1)); rs = warp_pan(torch.from_numpy(ramp)[None, None], torch.tensor([[0.0, 1.0]], dtype=torch.float64))[0, 0].numpy()
    return dict(per_scene=rows, energy_ratio_mean=float(np.mean([r["energy_ratio"] for r in rows])), zero_shift_identity_max_abs=ident, coordinate_ramp_shift_dx1=float((rs - ramp)[8:-8, 8:-8].mean()), ramp_note="W(P,c)[y,x]=P[y+cy,x+cx] -> ramp value increases by +1 for dx=+1")


# ------------------------------------------------------------------ V4 interventions
@torch.no_grad()
def interventions(m, ds, lms_raw, pan_raw, sensor, wald, mp, dev, const_vec, sigma_star, blur_status):
    learned = []
    for i in range(len(ds)):
        lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i]); learned.append(m(pan, ms, lpan)["delta"][0].float().cpu().numpy())
    n = len(ds); modes = dict(learned=lambda i: learned[i], zero=lambda i: np.zeros(2, np.float32), wrong_sign=lambda i: -learned[i], scene_shuffle=lambda i: learned[(i + 1) % n], constant_calibration=lambda i: np.array(const_vec, np.float32))
    out = {}
    blur_ok = sigma_star is not None and sigma_star > 0 and blur_status == "matched"          # σ*=0 / unmatched 면 blur 대조는 zero 와 같아 의미가 없다 — 상태만 남긴다 (§9.3)
    for mode, fn in list(modes.items()) + ([("blur_energy_match", None)] if blur_ok else []):
        acc = {v: dict(hqnr=[], fscc=[], d_s=[], d_lambda=[]) for v in VIEWS}; ok_n = 0; per = []; oks = []
        for i in range(n):
            lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i])
            if mode == "blur_energy_match":
                pb = torch.from_numpy(gaussian_filter(pan[0, 0].float().cpu().numpy(), sigma_star, mode="nearest"))[None, None].to(dev); o = m(pb, ms, lpan, aligner_enabled=False); d = np.zeros(2, np.float32)
            else:
                d = fn(i); o = m(pan, ms, lpan, delta_override=torch.from_numpy(d)[None].to(dev)) if mode != "zero" else m(pan, ms, lpan, aligner_enabled=False)
            sr = dn(o["y"][0], mp).transpose(1, 2, 0); p = pan_raw[i]; pa_eval = warp_pan(torch.from_numpy(p)[None, None], torch.from_numpy(d.astype(np.float64))[None])[0, 0].numpy()
            v, ok, _ = scene_views(sr, lms_raw[i].transpose(1, 2, 0), p, pa_eval, sensor, wald, d, 4, mp); ok_n += int(ok); per.append(v["raw_original"]["hqnr"]); oks.append(bool(ok))
            for k in VIEWS:
                for q in acc[k]:
                    acc[k][q].append(v[k][q])
        def vmean(k, q):                                                    # raw_original 은 전체 프레임(항상 유효); V64 view 는 적격 장면만, 0 이면 None (조용한 제외 금지 → 수를 남긴다)
            vals = acc[k][q] if k == "raw_original" else [x for x, o_ in zip(acc[k][q], oks) if o_]
            return float(np.mean(vals)) if vals else None
        out[mode] = dict(views={k: {q: vmean(k, q) for q in acc[k]} for k in VIEWS}, per_scene_raw_hqnr=per, per_scene_eligible=oks, n_eligible=ok_n, n_scenes=n, applied=("sigma %.2f, correction 0 (%s)" % (sigma_star, blur_status) if mode == "blur_energy_match" else mode))
    # zero 의 수치 동치: aligner_enabled=False 와 delta_override=0 (같은 sampler 경로) 의 출력 차이
    lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[0]); z1 = m(pan, ms, lpan, aligner_enabled=False)["y"]; z2 = m(pan, ms, lpan, delta_override=torch.zeros(1, 2, device=dev))["y"]
    out["zero_equivalence_max_abs"] = float((z1 - z2).abs().max()); out["constant_vector_dy_dx"] = list(map(float, const_vec)); out["scene_shuffle"]["derangement"] = "c_{(i+1) mod 20} (fixed)"
    if not blur_ok:
        out["blur_energy_match"] = dict(status=(blur_status or "not_run"), sigma_star=sigma_star, note="bicubic warp did not reduce Scharr energy on calibration patches (ratio >= 1) or no sigma matched within tolerance — blur control not applicable (unmatched_blur_control)")
    out["note"] = "same weights, correction substituted; raw views only (aligned view uses the substituted correction as reference). Not a substitute for a P0/constant-trained control (§10.1)"
    return out


@torch.no_grad()
def fr_stress_baseline(m, cfg, R, mg, dev, refm, probes_hr):
    """계획 §10.2 보간 기준선: 입력 P_ε 에 진단 correction c0−ε (c0 = native 예측) 를 적용한 출력의 native-reference 점수 (raw = 원 P, aligned = W(P, c_D), V96). learned c_ε 는 stress_hqnr_native 가 준다."""
    from pa.evalviews import STRESS_MARGIN
    from tools.po10_diag import native_stress_eligible
    sensor = sensor_of(cfg); wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")); Feeder = import_class(cfg["feeder"]); ds = Feeder(**cfg["test_full_feeder_args"]); mp = float(ds.max_pixel); lms_raw, pan_raw = refs(cfg)
    out = {}
    for ey, ex in probes_hr:
        vals, el = [], []
        for i in range(len(ds)):
            lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i]); mb = F.interpolate(ms, scale_factor=4, mode="bicubic")
            c0 = predict_c(m.aligner, pan, mb, mg)[0].double().cpu(); c_d = predict_c(refm.aligner, pan, mb, refm.aligner_margin)[0].double().cpu()
            e = torch.tensor([[ey, ex]], device=dev); pe = warp_pan(pan, e).to(pan.dtype) if (ey or ex) else pan; cb = (c0 - torch.tensor([ey, ex], dtype=torch.float64))
            o = m(pe, ms, lpan, delta_override=cb.float()[None].to(dev)); sr = dn(o["y"][0], mp).transpose(1, 2, 0); p = pan_raw[i]; pa_fixed = warp_pan(torch.from_numpy(p)[None, None], c_d[None])[0, 0].numpy()
            v, ok, _ = scene_views(sr, lms_raw[i].transpose(1, 2, 0), p, pa_fixed, sensor, wald, c_d.numpy(), 4, mp, margin=STRESS_MARGIN); sup = native_stress_eligible(p.shape[0], p.shape[1], [ey, ex], cb.numpy(), c_d.numpy())
            vals.append((v["raw_valid"]["hqnr"], v["aligned_valid"]["hqnr"])); el.append(bool(ok and sup))
        out[f"({ey:+.1f},{ex:+.1f})"] = dict(n=len(vals), n_eligible=int(sum(el)), eligible_all=bool(all(el)), raw_native_hqnr=(float(np.mean([a for a, _ in vals])) if all(el) else None), aligned_fixed_hqnr=(float(np.mean([b for _, b in vals])) if all(el) else None),
                                            raw_native_hqnr_eligible_subset=(float(np.mean([a for (a, _), o_ in zip(vals, el) if o_])) if any(el) else None))
    return dict(by_eps=out, note="diagnostic correction c0−ε (exactly cancels the synthetic shift; two raster samplings remain) — coordinate baseline, not an inference method")


@torch.no_grad()
def rr_stress(m, cfg, mg, dev, probes_hr, is_pa):
    """계획 §10.2 RR stress: P_ε 입력, GT Y 는 고정, 공통 ROI(margin 32, 두 warp support) 에서 오차. learned c_ε 와 기준선 c0−ε 둘 다."""
    from utils import reduced_metrics
    Feeder = import_class(cfg["feeder"]); ds = Feeder(**cfg["test_reduced_feeder_args"]); mp = float(ds.max_pixel); out = {}
    for ey, ex in probes_hr:
        agg = {"learned": [], "baseline_c0_minus_eps": []}
        for i in range(len(ds)):
            gt, lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i]); H, W = gt.shape[-2:]; y0, y1, x0, x1 = RR_STRESS_MARGIN, H - RR_STRESS_MARGIN, RR_STRESS_MARGIN, W - RR_STRESS_MARGIN
            e = torch.tensor([[ey, ex]], device=dev); pe = warp_pan(pan, e).to(pan.dtype) if (ey or ex) else pan
            if is_pa:
                mb = F.interpolate(ms, scale_factor=4, mode="bicubic"); c0 = predict_c(m.aligner, pan, mb, mg)[0].double().cpu(); cb = (c0 - torch.tensor([ey, ex], dtype=torch.float64)).float()[None].to(dev)
                variants = {"learned": m(pe, ms, lpan)["y"], "baseline_c0_minus_eps": m(pe, ms, lpan, delta_override=cb)["y"]}
            else:
                variants = {"learned": m(pe, ms, lpan, aligner_enabled=False)["y"], "baseline_c0_minus_eps": m(pe, ms, lpan, delta_override=(-e).float())["y"]}   # P0: correction 0 이 learned; 기준선은 −ε
            for k, y in variants.items():
                agg[k].append(reduced_metrics(x_true=gt[..., y0:y1, x0:x1], x_pred=y[..., y0:y1, x0:x1], max_pixel=mp))
        out[f"({ey:+.1f},{ex:+.1f})"] = {k: {q: float(np.mean([r[q] for r in v])) for q in v[0]} for k, v in agg.items()}
    return dict(by_eps=out, roi=f"[{RR_STRESS_MARGIN}:H-{RR_STRESS_MARGIN}, {RR_STRESS_MARGIN}:W-{RR_STRESS_MARGIN}] (block 32 aligned; two warp supports)", note="GT not moved; RR test 20 scenes; metrics = utils.reduced_metrics (training-log definitions)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--ckpt", default="best_hqnr"); ap.add_argument("--parts", default="v2,v3,v4"); ap.add_argument("--stress-dirs", type=int, default=8, choices=(4, 8))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--ref-run", default="PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT"); ap.add_argument("--ref-ckpt", default="last")
    a = ap.parse_args(); dev = torch.device(a.device); parts = set(a.parts.split(",")); t0 = time.time()
    wd, cfg, m, R, mg = load_run(a.run, a.ckpt, dev); od = os.path.join(wd, "palsv18"); os.makedirs(od, exist_ok=True); is_pa = m.aligner is not None
    meta = json.load(open(os.path.join(wd, f"{a.ckpt}_meta.json"))) if os.path.exists(os.path.join(wd, f"{a.ckpt}_meta.json")) else {}
    from kdv.teacher_assets import sha256_file
    prev = json.load(open(os.path.join(wd, "results", f"palsv18_{a.ckpt}.json"))) if os.path.exists(os.path.join(wd, "results", f"palsv18_{a.ckpt}.json")) else {}
    sha = sha256_file(os.path.join(wd, a.ckpt, "model.safetensors"))
    summ = {k: v for k, v in prev.items() if prev.get("ckpt_sha256") == sha and prev.get("tool_version") == TOOL_VERSION}                     # 같은 checkpoint·같은 도구면 이전 parts 누적
    summ.update(run=a.run, ckpt=a.ckpt, step=meta.get("step"), ckpt_sha256=sha, aligner=is_pa, radius_hr=R, view_margin=mg, parts=sorted(set(summ.get("parts", [])) | parts), tool_version=TOOL_VERSION)
    os.makedirs(CAMP, exist_ok=True)
    if not os.path.exists(os.path.join(CAMP, "probe_manifest.json")):
        pm = fixed_probes(R, "palsv18")[0]
        json.dump(dict(v1_v2_response=dict(radii_hr=[0.25, 0.5, 1.0, 2.0], directions=8, angles_deg=[45 * k for k in range(8)], n_nonzero=len(pm) - 1, zero_for_identity_only=True, probes_dy_dx=pm, scales=dict(native64="valid split 64 fixed patches (linspace ids)", rr256="RR test 20", fr512="FR mat20 20"),
                                           no_response_reference_epe=float(np.mean([math.hypot(x, y) for x, y in pm[1:]])), modes=["pan_only", "ms_only", "common"]),
                       v4_stress=dict(radii_hr=[0.5, 1.0, 2.0], directions=8, fr_reference="native original P / frozen N2 donor correction, V96 (STRESS_MARGIN 96), two-stage support eligibility", rr_roi=f"margin {RR_STRESS_MARGIN}", baseline="c0−ε diagnostic correction"),
                       generated=time.strftime("%Y-%m-%dT%H:%M:%S"), tool_version=TOOL_VERSION), open(os.path.join(CAMP, "probe_manifest.json"), "w"), indent=1)
    if not os.path.exists(os.path.join(CAMP, "estimator_contract.json")):
        from align.estimator import GATES as _G
        json.dump(dict(estimator="align/estimator.estimate_shift", pipeline="Scharr magnitude -> robust (median/MAD) normalization -> top-30% edge mask of reference -> integer ZNCC search ±4 HR px -> quadratic subpixel; census5 Hamming secondary as gate",
                       reference="PAN blurred with sensor PAN MTF (tools.metrics.jqm._pan_kernel)", moving="bicubic-upsampled MS band mean (HR grid)", gates=dict(_G, search_int=4, max_magnitude=4.0), unit="HR px", sampling_phase="HR grid, no crop",
                       sign="audit vector = shift of moving(MS) relative to reference(PAN); canonical PAN->MS correction = -audit (checked with a known +1 px PAN shift)", secondary_independent="not_available (phase-correlation not implemented; census is a gate of the same estimator)",
                       confidence_fields=["peak_zncc", "peak_margin", "accepted", "boundary_hit"], note="proxy for native position — not sensor ground truth; per-scene status kept, low-confidence scenes never dropped", tool_version=TOOL_VERSION),
                  open(os.path.join(CAMP, "estimator_contract.json"), "w"), indent=1)
    sensor = sensor_of(cfg); Feeder = import_class(cfg["feeder"]); dsf = Feeder(**cfg["test_full_feeder_args"]); mp = float(dsf.max_pixel); lms_raw, pan_raw = refs(cfg)
    if "v2" in parts and is_pa:
        mo, rows = modality(m, datasets(cfg), R, mg, dev); json.dump(mo, open(os.path.join(od, f"{a.ckpt}_modality.json"), "w"), indent=1); write_csv(os.path.join(od, f"{a.ckpt}_modality_rows.csv"), rows)
        summ["v2"] = {sc: {md: dict(B_diag=v["B_diag_mean"], B_cross=v["B_cross_mean"], epe=v["epe_mean"], epe_p90=v["epe_p90"]) for md, v in mo[sc].items()} for sc in ("native64", "rr256", "fr512")}
        from tools.po10_diag import interpolation_controls
        ic = interpolation_controls(m, datasets(cfg)["native64"], R, mg, dev); json.dump(ic, open(os.path.join(od, f"{a.ckpt}_shortcut_controls.json"), "w"), indent=1)     # §8.3 shortcut 대조를 best/last 둘 다 (리뷰 P2-4)
        summ["v2_shortcut"] = dict(ms_swap_B_diag=ic["ms_swap"]["B_diag"], ms_swap_closure=ic["ms_swap"]["closure_mean"], ms_const_B_diag=ic["ms_const"]["B_diag"], ms_const_closure=ic["ms_const"]["closure_mean"], kernel_bilinear_B_diag=ic["kernel_bilinear"]["B_diag"], padding_max_abs_diff=ic["padding_border_vs_reflection_max_abs_diff"])
        f = mo["fr512"]; print(f"  V2 fr512: pan_only B diag {f['pan_only']['B_diag_mean']} EPE {f['pan_only']['epe_mean']:.3f} | ms_only B diag {f['ms_only']['B_diag_mean']} EPE {f['ms_only']['epe_mean']:.3f} | common B diag {f['common']['B_diag_mean']} EPE {f['common']['epe_mean']:.3f} (no-response ref {mo['probe_manifest']['no_response_reference_epe']:.4f})")
    if "v3" in parts:
        if is_pa:
            rows, ns = native_proxy(m, dsf, pan_raw, sensor, dev); write_csv(os.path.join(od, f"{a.ckpt}_native_proxy.csv"), rows); summ["v3_native_proxy"] = ns
            print(f"  V3.1 native proxy |δ| median before {ns['before_mag_median']:.3f} → after {ns['after_mag_median']:.3f} HR px (accepted {ns['n_both_accepted']}/20, improved {ns['n_improved']}/20, cos(canonical, model c) {ns['cos_canon_before_vs_model_c_mean']}) | identity {ns['identity_check']['mag']:.3f} known PAN +1 → audit ({ns['known_shift_check']['audit_measured']['dy']:+.2f},{ns['known_shift_check']['audit_measured']['dx']:+.2f}) canonical {ns['known_shift_check']['canonical_pan_to_ms_of_audit']}")
            en = energy_fr(m, dsf, pan_raw, dev); cal = calibration_patches(m, cfg, dev); json.dump(dict(fr=en, calibration=cal), open(os.path.join(od, f"{a.ckpt}_energy.json"), "w"), indent=1)
            summ["v3_energy"] = dict(fr_energy_ratio_mean=en["energy_ratio_mean"], zero_shift_identity_max_abs=en["zero_shift_identity_max_abs"], ramp_dx1=en["coordinate_ramp_shift_dx1"], calib_target_ratio=cal["warp_energy_ratio_target"], sigma_star=cal["sigma_star"], blur_status=cal["blur_status"], const_vec=cal["c0_median_dy_dx"])
            print(f"  V3.3 warp energy ratio FR {en['energy_ratio_mean']:.4f} (calib {cal['warp_energy_ratio_target']:.4f}) → blur σ* {cal['sigma_star']} ({cal['blur_status']}) | identity {en['zero_shift_identity_max_abs']:.1e} ramp {en['coordinate_ramp_shift_dx1']:+.3f} | const c0 median {cal['c0_median_dy_dx']}")
        rows, es = edge_profile(m, cfg, mp, dev, is_pa); write_csv(os.path.join(od, f"{a.ckpt}_edge_profile.csv"), rows); summ["v3_edge"] = {k: v for k, v in es.items() if k not in ("spec", "note")}
        print(f"  V3.2 edge_profile_v1: measured {es['n_measured']}/{es['n_simple']} simple (missing {es['n_missing']}, multiple {es['n_multiple']}) | crossing offset {es['crossing_offset_mean']:+.3f} (|·| {es['crossing_offset_mean_abs']:.3f}) px | width GT {es['width_gt_mean']:.2f} → out {es['width_out_mean']:.2f} | edge MAE {es['edge_mae_dn']:.2f} flat {es['flat_mae_dn']:.2f} gradMAE {es['edge_grad_mae_dn']:.2f} DN")
    if "v4" in parts:
        wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
        if is_pa:
            cal = summ.get("v3_energy") or {}; cv = cal.get("const_vec") or calibration_patches(m, cfg, dev)["c0_median_dy_dx"]
            iv = interventions(m, dsf, lms_raw, pan_raw, sensor, wald, mp, dev, cv, cal.get("sigma_star"), cal.get("blur_status")); json.dump(iv, open(os.path.join(od, f"{a.ckpt}_interventions.json"), "w"), indent=1)
            summ["v4_interventions"] = {k: dict(raw_hqnr=v["views"]["raw_original"]["hqnr"], raw_fscc=v["views"]["raw_original"]["fscc"], raw_v64_hqnr=v["views"]["raw_valid"]["hqnr"], n_eligible=v["n_eligible"], n_scenes=v["n_scenes"]) for k, v in iv.items() if isinstance(v, dict) and "views" in v}
            summ["v4_blur_control"] = (iv.get("blur_energy_match") or {}).get("status", "run") if "views" not in (iv.get("blur_energy_match") or {}) else "matched"
            print("  V4 interventions raw HQNR: " + " | ".join(f"{k} {v['raw_hqnr']:.4f} (fSCC {v['raw_fscc']:.4f})" for k, v in summ["v4_interventions"].items()) + f" | zero-equivalence {iv['zero_equivalence_max_abs']:.1e}")
        import math as _m
        dirs = [(round(_m.sin(k * _m.pi / (a.stress_dirs / 2)), 12), round(_m.cos(k * _m.pi / (a.stress_dirs / 2)), 12)) for k in range(a.stress_dirs)]
        probes = tuple([(0.0, 0.0)] + [(r / R * dy, r / R * dx) for r in (0.5, 1.0, 2.0) for dy, dx in dirs])      # stress_hqnr_native 는 R 배수로 받는다 → HR px {0.5,1,2}
        refm = load_run(a.ref_run, a.ref_ckpt, dev)[2]; st = stress_hqnr_native(m, cfg, R, mg, dev, wd, refm, probes=probes, tag=f"palsv18_{a.ckpt}"); st["probe_manifest"] = dict(radii_hr=[0.5, 1.0, 2.0], directions=a.stress_dirs, reference="native original P / frozen N2 donor correction, V96")
        probes_hr = [(r * dy, r * dx) for r in (0.5, 1.0, 2.0) for dy, dx in dirs]
        if is_pa:
            st["baseline_c0_minus_eps"] = fr_stress_baseline(m, cfg, R, mg, dev, refm, probes_hr)
        st["rr_stress"] = rr_stress(m, cfg, mg, dev, [(0.0, 0.0)] + probes_hr, is_pa)
        json.dump(st, open(os.path.join(od, f"{a.ckpt}_stress.json"), "w"), indent=1); by = st["by_eps"]
        summ["v4_stress"] = {k: dict(raw=v.get("raw_native_hqnr"), fixed=v.get("aligned_fixed_hqnr"), eligible=v.get("eligible_all"), baseline_raw=((st.get("baseline_c0_minus_eps") or {}).get("by_eps", {}).get(k) or {}).get("raw_native_hqnr")) for k, v in by.items()}
        summ["v4_rr_stress"] = {k: dict(learned_ergas=v["learned"].get("ergas"), baseline_ergas=v["baseline_c0_minus_eps"].get("ergas"), learned_sam=v["learned"].get("sam")) for k, v in st["rr_stress"]["by_eps"].items()}
        rr0 = st["rr_stress"]["by_eps"].get("(+0.0,+0.0)", {}).get("learned", {}); print(f"  V4 RR stress (ROI margin {RR_STRESS_MARGIN}): ε=0 ERGAS {rr0.get('ergas')} | mean over nonzero learned {np.mean([v['learned']['ergas'] for k, v in st['rr_stress']['by_eps'].items() if k != '(+0.0,+0.0)']):.4f} baseline {np.mean([v['baseline_c0_minus_eps']['ergas'] for k, v in st['rr_stress']['by_eps'].items() if k != '(+0.0,+0.0)']):.4f}")
        e0 = by.get("(+0.0,+0.0)", {}); print(f"  V4 stress native-reference (V96, {len(probes) - 1} probes): ε=0 raw {e0.get('raw_native_hqnr')} fixed {e0.get('aligned_fixed_hqnr')} | mean over nonzero raw {np.nanmean([v['raw_native_hqnr'] for k, v in by.items() if k != '(+0.0,+0.0)' and v.get('raw_native_hqnr') is not None]):.4f}")
    summ["seconds"] = round(time.time() - t0, 1); os.makedirs(os.path.join(wd, "results"), exist_ok=True); json.dump(summ, open(os.path.join(wd, "results", f"palsv18_{a.ckpt}.json"), "w"), indent=1)
    print(f"  -> {os.path.relpath(os.path.join(wd, 'results', f'palsv18_{a.ckpt}.json'), ROOT)} ({summ['seconds']} s)")


if __name__ == "__main__":
    main()
