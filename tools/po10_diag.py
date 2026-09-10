#!/usr/bin/env python
"""PO10 사후 진단 (명세 §10.3–10.4·§13.1): 추가 변위 반응 · 보간/padding/MS 대조. 학습이 끝난 run 에 한 번 (tools/_upload.sh 가 PO10_ run 에 자동 실행).

    PANCRAFTER_DLPAN=... python tools/po10_diag.py --run PO10_N2_OFFSG_W96_D124_WV3_S2025 [--ckpt best_hqnr]
    python tools/po10_diag.py --run PA_A1_REC_W96_D124_9CH_S2025          # 과거 A1 (전체 view) 도 같은 진단 (배경 기준)

산출 work_dir/<run>/{offset_response_native64.csv, offset_response_rr256.csv, offset_response_fr512.csv, interpolation_controls.json} + results/po10_diag.json
  §10.3 반응: q(ε) = ĉε − ĉ0, 2D 선형 fit q ≈ Bε + b (목표 B≈−I, b≈0). 절대 probe 0·±0.5·±1·±1.5·±2 HR px 축 방향(|e|≤R 만 in-range, 나머지 stress) + 원판 무작위. |ε| 0.5 px 구간별 closure.
        closure error ||ĉε − ĉ0 + ε||₂ 의 mean/P50/P90, 고정 R 로 나눈 상대값. native ĉ0 mean/median/IQR.
        64² 는 학습·calibration 과 분리된 valid_wv3.h5 patch(고정 sample id), 256² 는 RR test, 512² 는 FR 논문 세트.
  §10.4 대조: (1) outer padding 변경(border→reflection)에 대한 ĉε 불변 (2) MS 교체/상수 MS 에서의 closure (3) zero/learned/wrong-sign 은 pa_diag
        (4) W(W(P,ε), ĉ0−ε) vs W(P, ĉ0) 차이 — **두 단계 sampling support 의 교집합 안에서만**(두 번 보간 바닥) (5) corruption kernel 을 bilinear 로 바꿨을 때 반응 유지 여부.
  §6.3 stress HQNR: FR 20장에 ε∈{0, ±R 축} 을 넣은 P_ε 로 추론 → 별도 stress ROI(margin 96, 두 단계 적격성 |ε|+|ĉ| ≤ MAX_ELIGIBLE_TWO_STAGE = 40)에서 raw_valid·aligned_valid → stress_hqnr_fr512.csv.
"""
import argparse, csv, json, os, sys
import numpy as np, torch, yaml, h5py
import torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from main import import_class                                           # noqa: E402
from pa.aligner import PANGlobalAligner                                 # noqa: E402
from pa.model import PAModel                                            # noqa: E402
from pa.warp import warp_pan                                            # noqa: E402
from pa.offset import aligner_margin, predict_c, sample_offsets         # noqa: E402
from pa.warp import warp_support_mask, two_stage_support_mask           # noqa: E402
from pa.evalviews import scene_views, STRESS_MARGIN, MAX_ELIGIBLE_TWO_STAGE, MAX_ELIGIBLE_SHIFT, stress_roi_manifest, VIEWS   # noqa: E402
from pa.warp import support_margin_ok   # noqa: E402
from tools.metrics.eval_fr import load_dlpan                            # noqa: E402
import tools.eval_fr_paperset as efp                                    # noqa: E402
import h5py as _h5                                                      # noqa: E402


def load_run(run, ckpt, dev):
    wd = os.path.join(ROOT, "work_dir", run); cfg = yaml.safe_load(open(os.path.join(wd, "meta", "config.yaml")))
    from safetensors.torch import load_file
    tr = cfg.get("trainer"); assert tr in ("pa", "po", "kdv"), "aligner 가 있는 run 만"
    if tr == "kdv":                                                          # KDV/NF16: skeleton 은 run config 대로 (donor view margin, A-ID 면 aligner 없음)
        from kdv.teacher_assets import skeleton_from_cfg
        m, info = skeleton_from_cfg(cfg, import_class(cfg["model"])); m.load_state_dict(load_file(os.path.join(wd, ckpt, "model.safetensors")), strict=True)
        k = cfg.get("kdv") or {}; R = float((k.get("corruption") or {}).get("radius_hr", 0) or 2.0); mg = info["margin"]
        return wd, cfg, m.to(dev).eval(), R, mg
    R = float((cfg.get("po") or {}).get("radius_hr", 1.0)); mg = aligner_margin(R) if tr == "po" else 0
    m = PAModel(import_class(cfg["model"])(**cfg["model_args"]), PANGlobalAligner(int(cfg["num_bands"])), aligner_margin=mg)
    m.load_state_dict(load_file(os.path.join(wd, ckpt, "model.safetensors")), strict=True)
    return wd, cfg, m.to(dev).eval(), R, mg


def warp_generic(p, d, mode="bicubic", padding="border"):
    B, _, h, w = p.shape; ys = torch.arange(h, device=p.device, dtype=torch.float32); xs = torch.arange(w, device=p.device, dtype=torch.float32); yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    gx = 2.0 * (xx[None] + d[:, 1].view(B, 1, 1) + 0.5) / w - 1.0; gy = 2.0 * (yy[None] + d[:, 0].view(B, 1, 1) + 0.5) / h - 1.0
    return F.grid_sample(p.float(), torch.stack((gx, gy), -1), mode=mode, padding_mode=padding, align_corners=False)


def datasets(cfg):
    """(name, list of (pan, ms) float tensors [1,·,H,W] normalized) for 64 / 256 / 512."""
    out = {}
    with h5py.File(cfg["val_feeder_args"]["dataroot"]) as f:                  # 학습·calibration 과 분리된 valid split, 고정 sample id
        n = f["pan"].shape[0]; idx = np.linspace(0, n - 1, 64).astype(int)
        pan = torch.from_numpy(np.asarray(f["pan"][idx], dtype=np.float32)); ms = torch.from_numpy(np.asarray(f["ms"][idx], dtype=np.float32))
    mp = float(cfg["max_pixel"]); out["native64"] = [(2 * pan[i:i + 1] / mp - 1, 2 * ms[i:i + 1] / mp - 1) for i in range(len(idx))]
    for name, key in (("rr256", "test_reduced_feeder_args"), ("fr512", "test_full_feeder_args")):
        with h5py.File(cfg[key]["dataroot"]) as f:
            pan = torch.from_numpy(np.asarray(f["pan"], dtype=np.float32)); ms = torch.from_numpy(np.asarray(f["ms"], dtype=np.float32))
        out[name] = [(2 * pan[i:i + 1] / mp - 1, 2 * ms[i:i + 1] / mp - 1) for i in range(pan.shape[0])]
    return out


@torch.no_grad()
def response(m, samples, R, mg, dev, seed=12345, kernel="bicubic", ms_mode="own"):
    """probe: 0, ±R/2, ±R 축 방향(8) + 원판 무작위 8 + ±2R 축 stress(4). 반환 rows(list of dict), fit dict."""
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    # 절대 probe (HR px) — R100/R200 을 같은 변위 구간에서 비교 (변경 명세 §6.3). |e| ≤ R 은 in-range 'probe', 그 밖은 'stress'
    AX = (0.5, 1.0, 1.5, 2.0)
    probes = [(0.0, 0.0)] + [(sg * f, 0.0) for sg in (1, -1) for f in AX if f <= R + 1e-9] + [(0.0, sg * f) for sg in (1, -1) for f in AX if f <= R + 1e-9]
    stress = [(sg * f, 0.0) for sg in (1, -1) for f in AX if f > R + 1e-9] + [(0.0, sg * f) for sg in (1, -1) for f in AX if f > R + 1e-9]
    rows = []
    for i, (pan, ms) in enumerate(samples):
        pan, ms = pan.to(dev), ms.to(dev)
        ms_use = ms
        if ms_mode == "swap":
            ms_use = samples[(i + 1) % len(samples)][1].to(dev)
        elif ms_mode == "const":
            ms_use = torch.zeros_like(ms) + ms.mean()
        mb = F.interpolate(ms_use, scale_factor=4, mode="bicubic")
        c0 = predict_c(m.aligner, pan, mb, mg)[0].cpu().numpy()
        rnd = sample_offsets(8, R, g).numpy().tolist()
        for kind, eps_list in (("probe", probes), ("random", rnd), ("stress", stress)):
            for ey, ex in eps_list:
                pe = warp_generic(pan, torch.tensor([[ey, ex]], device=dev), mode=kernel)
                ce = predict_c(m.aligner, pe, mb, mg)[0].cpu().numpy()
                rows.append(dict(sample=i, kind=kind, ey=ey, ex=ex, c0_dy=float(c0[0]), c0_dx=float(c0[1]), ce_dy=float(ce[0]), ce_dx=float(ce[1]),
                                 q_dy=float(ce[0] - c0[0]), q_dx=float(ce[1] - c0[1]), closure=float(np.hypot(ce[0] - c0[0] + ey, ce[1] - c0[1] + ex))))
    return rows, fit(rows, R)


def fit(rows, R):
    """q ≈ B ε + b (least squares, in-range: probe+random)."""
    r = [x for x in rows if x["kind"] != "stress"]
    E = np.array([[x["ey"], x["ex"], 1.0] for x in r]); Q = np.array([[x["q_dy"], x["q_dx"]] for x in r])
    coef, *_ = np.linalg.lstsq(E, Q, rcond=None)                              # [3,2]: rows (ey, ex, 1), cols (q_dy, q_dx)
    B = coef[:2].T; b = coef[2]
    cl = np.array([x["closure"] for x in r]); cs = np.array([x["closure"] for x in rows if x["kind"] == "stress"])
    c0 = np.array([[x["c0_dy"], x["c0_dx"]] for x in rows if x["kind"] == "probe" and x["ey"] == 0 and x["ex"] == 0])
    en = np.array([np.hypot(x["ey"], x["ex"]) for x in rows]); ca = np.array([x["closure"] for x in rows])
    bins = {}
    for lo in (0.0, 0.5, 1.0, 1.5):
        sel = (en > lo) & (en <= lo + 0.5) if lo > 0 else (en >= 0) & (en <= 0.5)
        bins[f"{lo:.1f}-{lo + 0.5:.1f}"] = dict(n=int(sel.sum()), closure_mean=(float(ca[sel].mean()) if sel.any() else None), in_range=bool(lo + 0.5 <= R + 1e-9))
    return dict(B=B.tolist(), b=b.tolist(), B_diag=[float(B[0, 0]), float(B[1, 1])], B_cross=[float(B[0, 1]), float(B[1, 0])], ideal_B_diag=-1.0,
                closure_mean=float(cl.mean()), closure_p50=float(np.median(cl)), closure_p90=float(np.percentile(cl, 90)), closure_rel_R=float(cl.mean() / R),
                stress_closure_mean=(float(cs.mean()) if len(cs) else None), stress_probes_abs_hr=[f for f in (0.5, 1.0, 1.5, 2.0) if f > R + 1e-9], closure_by_eps_bin=bins, n_in_range=int(len(r)),
                native_c0_mean=c0.mean(0).tolist(), native_c0_median=np.median(c0, 0).tolist(), native_c0_iqr=[np.subtract(*np.percentile(c0[:, k], [75, 25])) for k in range(2)])


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


@torch.no_grad()
def interpolation_controls(m, samples, R, mg, dev):
    out = {}
    pan, ms = samples[0]; pan, ms = pan.to(dev), ms.to(dev); mb = F.interpolate(ms, scale_factor=4, mode="bicubic")
    e = torch.tensor([[0.7, -0.6]], device=dev)
    # (1) padding 변경 — crop 안 입력이 같아야 ĉε 동일
    cb = predict_c(m.aligner, warp_generic(pan, e, padding="border"), mb, mg); cr = predict_c(m.aligner, warp_generic(pan, e, padding="reflection"), mb, mg)
    out["padding_border_vs_reflection_max_abs_diff"] = float((cb - cr).abs().max())
    # (2) MS 교체 / 상수 MS 에서의 반응 (shortcut 의심 진단)
    for mode in ("swap", "const"):
        rows, ft = response(m, samples[:16], R, mg, dev, ms_mode=mode); out[f"ms_{mode}"] = dict(B_diag=ft["B_diag"], closure_mean=ft["closure_mean"])
    # (4) 두 번 보간 바닥: W(W(P,ε), ĉ0−ε) vs W(P, ĉ0) — 두 경로의 sampling support 교집합 안에서만 (검토 지적: 고정 [8:-8] 은 경계 복제값을 섞는다)
    c0 = predict_c(m.aligner, pan, mb, mg)
    a = warp_pan(warp_pan(pan, e), c0 - e); b = warp_pan(pan, c0)
    H, W = pan.shape[-2:]
    common = two_stage_support_mask(H, W, e, c0 - e) & warp_support_mask(H, W, c0)
    diff = (a - b).abs()[common]
    out["double_interp_floor_common_support_mean_abs"] = float(diff.mean()) if diff.numel() else None
    out["double_interp_floor_common_support_max_abs"] = float(diff.max()) if diff.numel() else None
    out["double_interp_floor_common_support_frac"] = float(common.float().mean()); out["double_interp_floor_c0"] = c0[0].tolist(); out["double_interp_floor_eps"] = e[0].tolist()
    # padding 을 바꿔도 공통 support 안 값은 같아야 한다 (경계 복제가 섞이지 않았다는 확인)
    a_r = warp_generic(warp_generic(pan, e, padding="reflection"), c0 - e, padding="reflection")
    out["double_interp_padding_independence_inside_support_max_abs"] = float((a - a_r).abs()[common].max()) if diff.numel() else None
    # (5) corruption kernel 을 bilinear 로 — 반응이 남는가
    rows, ft = response(m, samples[:16], R, mg, dev, kernel="bilinear"); out["kernel_bilinear"] = dict(B_diag=ft["B_diag"], closure_mean=ft["closure_mean"])
    return out


@torch.no_grad()
def stress_hqnr(m, cfg, R, mg, dev, wd, probes=((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))):
    """PO10 §6.3·§10.3: FR 논문 세트에 ε 를 넣은 P_ε 로 추론한 SR 을 **별도 stress ROI**(margin 96)에서 평가. raw_valid(원 P) · aligned_valid(P̃ε = W(W(P_raw,ε),ĉε) float64).
    적격성은 두 단계: |ε|∞ + |ĉε|∞ ≤ MAX_ELIGIBLE_TWO_STAGE. 최종 diagnostic 만 — 선택에 쓰지 않는다."""
    sensor = efp.sensor_of(cfg); wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
    Feeder = import_class(cfg["feeder"]); ds = Feeder(**cfg["test_full_feeder_args"]); mp = float(ds.max_pixel)
    with _h5.File(cfg["test_full_feeder_args"]["dataroot"]) as f:
        lms_raw = np.asarray(f["lms"], dtype=np.float64); pan_raw = np.asarray(f["pan"], dtype=np.float64)[:, 0]
    H, W = pan_raw.shape[-2:]; man = stress_roi_manifest(H, W); json.dump(man, open(os.path.join(wd, "stress_roi_manifest.json"), "w"), indent=1)
    rows = []
    for ey, ex in probes:
        e = torch.tensor([[ey * R, ex * R]], device=dev)
        for i in range(len(ds)):
            lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i])
            pe = warp_pan(pan, e).to(pan.dtype) if (ey or ex) else pan
            o = m(pe, ms, lpan); d = o["delta"][0].double().cpu()
            sr = ((o["y"][0].clip(-1, 1).float().cpu().numpy() + 1) / 2 * mp).astype(np.float64).transpose(1, 2, 0)
            p = pan_raw[i]; pe_raw = warp_pan(torch.from_numpy(p)[None, None], e.double().cpu())[0, 0].numpy() if (ey or ex) else p
            pa_eval = warp_pan(torch.from_numpy(pe_raw)[None, None], d[None])[0, 0].numpy()
            v, _, _ = scene_views(sr, lms_raw[i].transpose(1, 2, 0), p, pa_eval, sensor, wald, d.numpy(), 4, mp, margin=STRESS_MARGIN)
            two = float(max(abs(ey * R), abs(ex * R)) + float(d.abs().max()))
            rows.append(dict(ey=ey * R, ex=ex * R, scene=i, c_dy=float(d[0]), c_dx=float(d[1]), two_stage_abs_shift=two, eligible=bool(two <= MAX_ELIGIBLE_TWO_STAGE and np.isfinite([v[k]["hqnr"] for k in VIEWS]).all()),
                             raw_valid_hqnr=v["raw_valid"]["hqnr"], raw_valid_fscc=v["raw_valid"]["fscc"], aligned_valid_hqnr=v["aligned_valid"]["hqnr"], aligned_valid_fscc=v["aligned_valid"]["fscc"],
                             d_lambda=v["raw_valid"]["d_lambda"], raw_valid_d_s=v["raw_valid"]["d_s"], aligned_valid_d_s=v["aligned_valid"]["d_s"], roi_hash=man["roi_hash"]))
    write_csv(os.path.join(wd, "stress_hqnr_fr512.csv"), rows)
    summ = {}
    for ey, ex in probes:
        r = [x for x in rows if x["ey"] == ey * R and x["ex"] == ex * R]; el = [x for x in r if x["eligible"]]
        summ[f"({ey * R:+.1f},{ex * R:+.1f})"] = dict(n_eligible=len(el), n=len(r), eligible_all=bool(len(el) == len(r) and len(r) > 0), raw_valid_hqnr=(float(np.mean([x["raw_valid_hqnr"] for x in el])) if el else None),
                                                    aligned_valid_hqnr=(float(np.mean([x["aligned_valid_hqnr"] for x in el])) if el else None),
                                                    raw_valid_fscc=(float(np.mean([x["raw_valid_fscc"] for x in el])) if el else None), aligned_valid_fscc=(float(np.mean([x["aligned_valid_fscc"] for x in el])) if el else None),
                                                    c_median=[float(np.median([x["c_dy"] for x in r])), float(np.median([x["c_dx"] for x in r]))])
    return dict(roi_margin=STRESS_MARGIN, max_eligible_two_stage=MAX_ELIGIBLE_TWO_STAGE, by_eps=summ)


@torch.no_grad()
def native_stress_eligible(H, W, eps, c, c_d, margin=STRESS_MARGIN):
    """NF16 §8.4 적격: (1) 모델 경로의 두 단계 sampling W(W(P,ε),ĉε) 가 ROI(margin) 안에서 관측만 읽고, (2) 고정 참조 W(P,c_D) 의 support 도 ROI 안에서 유효. 둘 다 아니면 부적격 (검토 지적 3)."""
    e = torch.as_tensor(eps, dtype=torch.float32).view(1, 2); cc = torch.as_tensor(c, dtype=torch.float32).view(1, 2); cd = torch.as_tensor(c_d, dtype=torch.float32).view(1, 2)
    m2 = two_stage_support_mask(H, W, e, cc)[0, 0, margin:H - margin, margin:W - margin]
    return bool(m2.all()) and bool(support_margin_ok(H, W, cd, margin).all())


def stress_hqnr_native(m, cfg, R, mg, dev, wd, ref_model, probes=((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))):
    """NF16 §8.4: 입력만 P_ε 로 바꾸고 **참조는 native 로 고정** — raw = 원 P, aligned = W(P_raw, c_D) (c_D = 고정 donor 의 native 예측; ref_model None 이면 자기 native 예측). ROI margin 96.
    ε 마다 참조를 따라 옮겨 손실이 줄어든 것처럼 만들지 않는다. 출력 좌표(M-frame) 도 그대로다."""
    sensor = efp.sensor_of(cfg); wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
    Feeder = import_class(cfg["feeder"]); ds = Feeder(**cfg["test_full_feeder_args"]); mp = float(ds.max_pixel)
    with _h5.File(cfg["test_full_feeder_args"]["dataroot"]) as f:
        lms_raw = np.asarray(f["lms"], dtype=np.float64); pan_raw = np.asarray(f["pan"], dtype=np.float64)[:, 0]
    rows = []; refm = ref_model if ref_model is not None else m
    for ey, ex in probes:
        e = torch.tensor([[ey * R, ex * R]], device=dev)
        for i in range(len(ds)):
            lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i])
            mb = F.interpolate(ms, scale_factor=4, mode="bicubic")
            with torch.no_grad():
                c_d = (predict_c(refm.aligner, pan, mb, refm.aligner_margin)[0].double().cpu() if refm.aligner is not None else torch.zeros(2, dtype=torch.float64))   # native 참조 보정 (입력 ε 와 무관)
            pe = warp_pan(pan, e).to(pan.dtype) if (ey or ex) else pan
            with torch.no_grad():
                o = m(pe, ms, lpan); d = o["delta"][0].double().cpu()
                sr = ((o["y"][0].clip(-1, 1).float().cpu().numpy() + 1) / 2 * mp).astype(np.float64).transpose(1, 2, 0)
            p = pan_raw[i]; pa_fixed = warp_pan(torch.from_numpy(p)[None, None], c_d[None])[0, 0].numpy()
            v, ok, _ = scene_views(sr, lms_raw[i].transpose(1, 2, 0), p, pa_fixed, sensor, wald, c_d.numpy(), 4, mp, margin=STRESS_MARGIN)
            H_, W_ = p.shape; sup = native_stress_eligible(H_, W_, [ey * R, ex * R], d.numpy(), c_d.numpy())
            rows.append(dict(ey=ey * R, ex=ex * R, scene=i, c_dy=float(d[0]), c_dx=float(d[1]), ref_dy=float(c_d[0]), ref_dx=float(c_d[1]), eligible=bool(ok and sup and np.isfinite([v[k]["hqnr"] for k in VIEWS]).all()), support_two_stage_ok=bool(sup),
                             raw_native_hqnr=v["raw_valid"]["hqnr"], raw_native_fscc=v["raw_valid"]["fscc"], aligned_fixed_hqnr=v["aligned_valid"]["hqnr"], aligned_fixed_fscc=v["aligned_valid"]["fscc"],
                             d_lambda=v["raw_valid"]["d_lambda"], raw_native_d_s=v["raw_valid"]["d_s"], aligned_fixed_d_s=v["aligned_valid"]["d_s"]))
    write_csv(os.path.join(wd, "stress_hqnr_native_fr512.csv"), rows)
    return dict(roi_margin=STRESS_MARGIN, reference="native P / W(P, c_D) fixed", ref_model=("donor" if ref_model is not None else "self_native"),
                rule="model-input two-stage support(ε→ĉε) inside V96 AND fixed-reference W(P,c_D) support; scene eligible only if both hold and all view metrics finite",
                max_eligible_two_stage=MAX_ELIGIBLE_TWO_STAGE, max_eligible_reference_shift=MAX_ELIGIBLE_SHIFT, aggregation="all-eligible-only (부분 평균은 *_eligible_subset)",
                by_eps=aggregate_native(rows, probes, R))


def aggregate_native(rows, probes, R):
    """집계: 장면을 빼고 평균하지 않는다 — 전부 적격일 때만 값, 아니면 None + eligible_all False (명세 §6.2-7). 부분 평균은 *_eligible_subset 으로만 남긴다."""
    summ = {}
    for ey, ex in probes:
        r = [x for x in rows if x["ey"] == ey * R and x["ex"] == ex * R]; el = [x for x in r if x["eligible"]]; all_ok = len(el) == len(r) and len(r) > 0
        summ[f"({ey * R:+.1f},{ex * R:+.1f})"] = dict(n_eligible=len(el), n=len(r), eligible_all=bool(all_ok), raw_native_hqnr=(float(np.mean([x["raw_native_hqnr"] for x in r])) if all_ok else None),
                                                    aligned_fixed_hqnr=(float(np.mean([x["aligned_fixed_hqnr"] for x in r])) if all_ok else None),
                                                    raw_native_hqnr_eligible_subset=(float(np.mean([x["raw_native_hqnr"] for x in el])) if el and not all_ok else None),
                                                    aligned_fixed_hqnr_eligible_subset=(float(np.mean([x["aligned_fixed_hqnr"] for x in el])) if el and not all_ok else None),
                                                    c_median=[float(np.median([x["c_dy"] for x in r])), float(np.median([x["c_dx"] for x in r]))])
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--ckpt", default="best_hqnr"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--native-reference", action="store_true", help="NF16 §8.4: 참조를 native P / 고정 donor 보정으로 고정한 stress 도 계산")
    ap.add_argument("--ref-run", default=None); ap.add_argument("--ref-ckpt", default="last")
    ap.add_argument("--out", default="po10_diag", help="results/<out>.json (기본 po10_diag; last 진단은 po10_diag_last 권장)")
    a = ap.parse_args(); dev = torch.device(a.device)
    wd, cfg, m, R, mg = load_run(a.run, a.ckpt, dev)
    if m.aligner is None:                                                    # A-ID(P0): 반응 진단 없음, native stress 만 (참조 = donor 필요)
        out = dict(run=a.run, ckpt=a.ckpt, radius_hr=R, view_margin=mg, response={}, note="aligner 없음 (A-ID)")
        if a.native_reference and a.ref_run:
            _, _, refm, _, _ = load_run(a.ref_run, a.ref_ckpt, dev); out["stress_hqnr_native_fr512"] = stress_hqnr_native(m, cfg, R, mg, dev, wd, refm)
        os.makedirs(os.path.join(wd, "results"), exist_ok=True); json.dump(out, open(os.path.join(wd, "results", f"{a.out}.json"), "w"), indent=1); print("  aligner 없음 — native stress 만 기록"); return
    ds = datasets(cfg); out = dict(run=a.run, ckpt=a.ckpt, radius_hr=R, view_margin=mg, response={})
    print(f"[{a.run}] §10.3 추가 변위 반응 (R={R}, view margin {mg}, ckpt {a.ckpt})")
    for name, samples in ds.items():
        rows, ft = response(m, samples, R, mg, dev); write_csv(os.path.join(wd, f"offset_response_{name}.csv"), rows); out["response"][name] = ft
        print(f"  {name:9s} B diag ({ft['B_diag'][0]:+.3f}, {ft['B_diag'][1]:+.3f}) cross ({ft['B_cross'][0]:+.3f}, {ft['B_cross'][1]:+.3f}) b ({ft['b'][0]:+.3f},{ft['b'][1]:+.3f}) "
              f"| closure mean {ft['closure_mean']:.3f} p50 {ft['closure_p50']:.3f} p90 {ft['closure_p90']:.3f} (rel R {ft['closure_rel_R']:.2f}) | stress(>R) {ft['stress_closure_mean']} "
              f"| native ĉ0 median ({ft['native_c0_median'][0]:+.3f},{ft['native_c0_median'][1]:+.3f})")
    out["interpolation_controls"] = ic = interpolation_controls(m, ds["native64"], R, mg, dev)
    json.dump(ic, open(os.path.join(wd, "interpolation_controls.json"), "w"), indent=1)
    print(f"  §10.4 padding border→reflection |Δĉ| {ic['padding_border_vs_reflection_max_abs_diff']:.2e} | MS swap B diag {ic['ms_swap']['B_diag']} | MS const {ic['ms_const']['B_diag']} "
          f"| double-interp floor (common support {ic['double_interp_floor_common_support_frac']:.2f}) mean {ic['double_interp_floor_common_support_mean_abs']:.2e}, padding-indep inside {ic['double_interp_padding_independence_inside_support_max_abs']:.1e} | bilinear kernel B diag {ic['kernel_bilinear']['B_diag']}")
    out["stress_hqnr_fr512"] = st = stress_hqnr(m, cfg, R, mg, dev, wd)
    print("  §6.3/§10.3 stress HQNR (FR, ROI margin 96, 두 단계 적격): " + " | ".join(f"ε{k}: raw_valid {v['raw_valid_hqnr']:.4f} aligned_valid {v['aligned_valid_hqnr']:.4f} ({v['n_eligible']}/{v['n']})" for k, v in st["by_eps"].items() if v["raw_valid_hqnr"] is not None))
    if a.native_reference:
        refm = load_run(a.ref_run, a.ref_ckpt, dev)[2] if a.ref_run else None
        out["stress_hqnr_native_fr512"] = sn = stress_hqnr_native(m, cfg, R, mg, dev, wd, refm)
        print("  NF16 §8.4 native-reference stress (raw = 원 P, aligned = W(P, c_D) 고정, V96): " + " | ".join((f"ε{k}: raw {v['raw_native_hqnr']:.4f} fixed {v['aligned_fixed_hqnr']:.4f}" if v["eligible_all"] else f"ε{k}: INELIGIBLE({v['n_eligible']}/{v['n']})") for k, v in sn["by_eps"].items()))
    os.makedirs(os.path.join(wd, "results"), exist_ok=True); json.dump(out, open(os.path.join(wd, "results", f"{a.out}.json"), "w"), indent=1)
    print(f"  -> {os.path.relpath(os.path.join(wd, 'results', a.out + '.json'), ROOT)}")


if __name__ == "__main__":
    main()
