#!/usr/bin/env python
"""PO10 사후 진단 (명세 §10.3–10.4·§13.1): 추가 변위 반응 · 보간/padding/MS 대조. 학습이 끝난 run 에 한 번 (tools/_upload.sh 가 PO10_ run 에 자동 실행).

    PANCRAFTER_DLPAN=... python tools/po10_diag.py --run PO10_N2_OFFSG_W96_D124_WV3_S2025 [--ckpt best_hqnr]
    python tools/po10_diag.py --run PA_A1_REC_W96_D124_9CH_S2025          # 과거 A1 (전체 view) 도 같은 진단 (배경 기준)

산출 work_dir/<run>/{offset_response_native64.csv, offset_response_rr256.csv, offset_response_fr512.csv, interpolation_controls.json} + results/po10_diag.json
  §10.3 반응: q(ε) = ĉε − ĉ0, 2D 선형 fit q ≈ Bε + b (목표 B≈−I, b≈0). probe 0·±R/2·±R 축 방향 + 원판 무작위 + ±2R 외삽 stress (별도 표시).
        closure error ||ĉε − ĉ0 + ε||₂ 의 mean/P50/P90, 고정 R 로 나눈 상대값. native ĉ0 mean/median/IQR.
        64² 는 학습·calibration 과 분리된 valid_wv3.h5 patch(고정 sample id), 256² 는 RR test, 512² 는 FR 논문 세트.
  §10.4 대조: (1) outer padding 변경(border→reflection)에 대한 ĉε 불변 (2) MS 교체/상수 MS 에서의 closure (3) zero/learned/wrong-sign 은 pa_diag
        (4) W(W(P,ε), ĉ0−ε) vs W(P, ĉ0) 내부 차이(두 번 보간 바닥) (5) corruption kernel 을 bilinear 로 바꿨을 때 반응 유지 여부.
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


def load_run(run, ckpt, dev):
    wd = os.path.join(ROOT, "work_dir", run); cfg = yaml.safe_load(open(os.path.join(wd, "meta", "config.yaml")))
    from safetensors.torch import load_file
    tr = cfg.get("trainer"); assert tr in ("pa", "po"), "aligner 가 있는 run 만"
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
    probes = [(0.0, 0.0)] + [(s * R * f, 0.0) for s in (1, -1) for f in (0.5, 1.0)] + [(0.0, s * R * f) for s in (1, -1) for f in (0.5, 1.0)]
    stress = [(2 * R, 0.0), (-2 * R, 0.0), (0.0, 2 * R), (0.0, -2 * R)]
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
        for kind, eps_list in (("probe", probes), ("random", rnd), ("stress2R", stress)):
            for ey, ex in eps_list:
                pe = warp_generic(pan, torch.tensor([[ey, ex]], device=dev), mode=kernel)
                ce = predict_c(m.aligner, pe, mb, mg)[0].cpu().numpy()
                rows.append(dict(sample=i, kind=kind, ey=ey, ex=ex, c0_dy=float(c0[0]), c0_dx=float(c0[1]), ce_dy=float(ce[0]), ce_dx=float(ce[1]),
                                 q_dy=float(ce[0] - c0[0]), q_dx=float(ce[1] - c0[1]), closure=float(np.hypot(ce[0] - c0[0] + ey, ce[1] - c0[1] + ex))))
    return rows, fit(rows, R)


def fit(rows, R):
    """q ≈ B ε + b (least squares, in-range: probe+random)."""
    r = [x for x in rows if x["kind"] != "stress2R"]
    E = np.array([[x["ey"], x["ex"], 1.0] for x in r]); Q = np.array([[x["q_dy"], x["q_dx"]] for x in r])
    coef, *_ = np.linalg.lstsq(E, Q, rcond=None)                              # [3,2]: rows (ey, ex, 1), cols (q_dy, q_dx)
    B = coef[:2].T; b = coef[2]
    cl = np.array([x["closure"] for x in r]); cs = np.array([x["closure"] for x in rows if x["kind"] == "stress2R"])
    c0 = np.array([[x["c0_dy"], x["c0_dx"]] for x in rows if x["kind"] == "probe" and x["ey"] == 0 and x["ex"] == 0])
    return dict(B=B.tolist(), b=b.tolist(), B_diag=[float(B[0, 0]), float(B[1, 1])], B_cross=[float(B[0, 1]), float(B[1, 0])], ideal_B_diag=-1.0,
                closure_mean=float(cl.mean()), closure_p50=float(np.median(cl)), closure_p90=float(np.percentile(cl, 90)), closure_rel_R=float(cl.mean() / R),
                stress2R_closure_mean=(float(cs.mean()) if len(cs) else None), n_in_range=int(len(r)),
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
    # (4) 두 번 보간 바닥: W(W(P,ε), ĉ0−ε) vs W(P, ĉ0) 내부 차이
    c0 = predict_c(m.aligner, pan, mb, mg)
    a = warp_pan(warp_pan(pan, e), c0 - e); b = warp_pan(pan, c0)
    out["double_interp_floor_interior_mean_abs"] = float((a - b)[..., 8:-8, 8:-8].abs().mean()); out["double_interp_floor_interior_max_abs"] = float((a - b)[..., 8:-8, 8:-8].abs().max())
    # (5) corruption kernel 을 bilinear 로 — 반응이 남는가
    rows, ft = response(m, samples[:16], R, mg, dev, kernel="bilinear"); out["kernel_bilinear"] = dict(B_diag=ft["B_diag"], closure_mean=ft["closure_mean"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--ckpt", default="best_hqnr"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = torch.device(a.device)
    wd, cfg, m, R, mg = load_run(a.run, a.ckpt, dev)
    ds = datasets(cfg); out = dict(run=a.run, ckpt=a.ckpt, radius_hr=R, view_margin=mg, response={})
    print(f"[{a.run}] §10.3 추가 변위 반응 (R={R}, view margin {mg}, ckpt {a.ckpt})")
    for name, samples in ds.items():
        rows, ft = response(m, samples, R, mg, dev); write_csv(os.path.join(wd, f"offset_response_{name}.csv"), rows); out["response"][name] = ft
        print(f"  {name:9s} B diag ({ft['B_diag'][0]:+.3f}, {ft['B_diag'][1]:+.3f}) cross ({ft['B_cross'][0]:+.3f}, {ft['B_cross'][1]:+.3f}) b ({ft['b'][0]:+.3f},{ft['b'][1]:+.3f}) "
              f"| closure mean {ft['closure_mean']:.3f} p50 {ft['closure_p50']:.3f} p90 {ft['closure_p90']:.3f} (rel R {ft['closure_rel_R']:.2f}) | ±2R stress {ft['stress2R_closure_mean']:.3f} "
              f"| native ĉ0 median ({ft['native_c0_median'][0]:+.3f},{ft['native_c0_median'][1]:+.3f})")
    out["interpolation_controls"] = ic = interpolation_controls(m, ds["native64"], R, mg, dev)
    json.dump(ic, open(os.path.join(wd, "interpolation_controls.json"), "w"), indent=1)
    print(f"  §10.4 padding border→reflection |Δĉ| {ic['padding_border_vs_reflection_max_abs_diff']:.2e} | MS swap B diag {ic['ms_swap']['B_diag']} | MS const {ic['ms_const']['B_diag']} "
          f"| double-interp floor mean {ic['double_interp_floor_interior_mean_abs']:.2e} | bilinear kernel B diag {ic['kernel_bilinear']['B_diag']}")
    os.makedirs(os.path.join(wd, "results"), exist_ok=True); json.dump(out, open(os.path.join(wd, "results", "po10_diag.json"), "w"), indent=1)
    print(f"  -> {os.path.relpath(os.path.join(wd, 'results', 'po10_diag.json'), ROOT)}")


if __name__ == "__main__":
    main()
