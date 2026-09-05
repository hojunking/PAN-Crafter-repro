#!/usr/bin/env python
"""Local alignment 추론 진단 (계획 §12) — 학습 없이, first conv 의 PAN 기여에만 밀집 field 를 적용해 L0~L4 를 비교한다.

  python tools/local_align_diag.py --run work_dir/<winner> [--ckpt best_hqnr] [--global audit|g1|none]

Field: TV-L1 은 이 환경(opencv-python 본체)에 없어 **DIS optical flow** 로 대체. GT 미사용 — PAN_HF 구조맵과
bicubic ↑MS 구조맵(밴드 평균 Scharr 크기, 표본별 정규화) 사이의 flow. 전역 보정 뒤 잔차 field 만 쓴다:
  global: g1 모델이면 Δ̂ (모델 자체), 아니면 audit Δ(−4Δ_LR HR px, outputs/global_shift_cache/wv3_fr.csv) — 진단용 비교 target.
Gate(§12.4): edge energy 상위 30% · forward-backward 일치(≤0.5px) · |flow| ≤ 2 HR px, 나머지 0.
비교: L0 no-align / L1 global only / L2 global+gated local / L3 global+ungated local / L4 wrong-sign gated local.
판정(§12.6): HQNR drop ≤ 0.0005 · fSCC gain ≥ 0.005 · 12-19 중 6장 이상 동일 방향 · L2 > L3 · L4 비개선.
"""
import argparse, csv, os, sys
import cv2, numpy as np, torch, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tools.sr_infer_diag import load                                   # noqa: E402
from sr.forward import build_inputs, x11                               # noqa: E402
from sr.pan_align import split_first_conv, edge_weight, apply_local_field   # noqa: E402
from align.resample import warp_hr                                     # noqa: E402
from align.shiftnet import scharr_mag_t, robust_norm_t                 # noqa: E402
from feeders.feeder import PanFeeder                                   # noqa: E402
from utils import tensor2img, SCC_full_numpy                           # noqa: E402
from tools.metrics.eval_fr import load_dlpan, d_lambda_k, d_s          # noqa: E402


def struct_map(x):                    # [1,C,H,W] -> [H,W] float32 (0..1 근사)
    g = robust_norm_t(scharr_mag_t(x.mean(1, keepdim=True)))[0, 0]
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    return g.float().cpu().numpy().astype(np.float32)


def dis_flow(prev, nxt):
    """prev(x) ≈ nxt(x + flow(x)). cv2 flow 는 (dx, dy) 순."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    p8, n8 = (np.clip(prev * 255, 0, 255)).astype(np.uint8), (np.clip(nxt * 255, 0, 255)).astype(np.uint8)
    return dis.calc(p8, n8, None)


def local_field(m_struct, p_struct, top_frac=0.30, fb_tol=0.5, max_mag=2.0):
    """M 구조맵 → P 구조맵 flow (dy,dx) 와 gate 맵."""
    f = dis_flow(m_struct, p_struct); b = dis_flow(p_struct, m_struct)
    H, W = m_struct.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bx = cv2.remap(b[..., 0], xx + f[..., 0], yy + f[..., 1], cv2.INTER_LINEAR)
    by = cv2.remap(b[..., 1], xx + f[..., 0], yy + f[..., 1], cv2.INTER_LINEAR)
    fb = np.sqrt((f[..., 0] + bx) ** 2 + (f[..., 1] + by) ** 2)
    mag = np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2)
    edge = m_struct >= np.quantile(m_struct, 1 - top_frac)
    gate = (edge & (fb <= fb_tol) & (mag <= max_mag)).astype(np.float32)
    flow_dydx = np.stack([f[..., 1], f[..., 0]], 0)
    return flow_dydx, gate, dict(gate_ratio=float(gate.mean()), edge_ratio=float(edge.mean()), fb_ok=float((fb <= fb_tol).mean()),
                                 mag_p50=float(np.median(mag[edge])) if edge.any() else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--ckpt", default="best_hqnr")
    ap.add_argument("--global", dest="glob", default="auto", choices=["auto", "audit", "g1", "none"])
    ap.add_argument("--indices", default="12-19")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, cfg, v = load(a.run, a.ckpt); m = m.to(dev); bb, corr = m.backbone, m.correlator
    glob = a.glob if a.glob != "auto" else ("g1" if v == "g1" else "audit")
    wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
    lo, hi = (int(x) for x in a.indices.split("-")); maxp = float(cfg.get("max_pixel", 2047.0))
    fr = PanFeeder(**cfg["test_full_feeder_args"])
    import pandas as pd
    audit = torch.tensor(pd.read_csv(os.path.join(ROOT, "outputs/global_shift_cache/wv3_fr.csv")).sort_values("sample_id")[["dy_lr_raw", "dx_lr_raw"]].values, dtype=torch.float32)
    dn = lambda t: np.round((((t + 1) / 2).clamp(0, 1).float().cpu().numpy().astype(np.float64)) * maxp)
    per = {k: [] for k in ("L0", "L1", "L2", "L3", "L4")}; gstats = []
    with torch.no_grad():
        for i in range(lo, hi + 1):
            lms, ms, lpan, pan = (t[None].to(dev) for t in fr[i])
            ms_base, lpan_u, pan_hf = build_inputs(pan, lpan, ms)
            xc = x11(pan, lpan_u, pan_hf, ms_base)
            f_p, f_m = split_first_conv(bb.input, xc)
            # global
            if glob == "g1" and corr is not None:
                info = corr(f_p, f_m, edge_weight(pan_hf)); f_pg = corr.apply(f_p, info["delta"], info["gate"], 1.0)
            elif glob == "audit":
                f_pg = warp_hr(f_p, (-4.0 * audit[i:i + 1]).to(dev))
            else:
                f_pg = f_p
            # local field on structure maps (global-corrected PAN feature 의 구조 vs M 구조)
            m_s = struct_map(ms_base); p_s = struct_map(f_pg[:, :8] if False else torch.stack([pan_hf[0, 0]], 0)[None])
            flow, gate, st = local_field(m_s, p_s); gstats.append(st)
            fl = torch.tensor(flow, device=dev)[None]; gt_ = torch.tensor(gate, device=dev)[None, None]
            variants = {"L0": f_p, "L1": f_pg, "L2": apply_local_field(f_pg, fl, gt_),
                        "L3": apply_local_field(f_pg, fl, torch.ones_like(gt_)), "L4": apply_local_field(f_pg, -fl, gt_)}
            sw = torch.ones(1, device=dev)
            for k, fp in variants.items():
                y = ms_base + bb(None, None, None, sw, f_in=f_m + fp)
                sr_np, lm, pn = dn(y[0]).transpose(1, 2, 0), dn(lms[0]).transpose(1, 2, 0), dn(pan[0, 0])
                dl = d_lambda_k(sr_np, lm, "wv3", 4, 32, wald); ds = d_s(sr_np, lm, pn, 4, 32, wald)
                per[k].append(((1 - dl) * (1 - ds), SCC_full_numpy(tensor2img(pan, maxp), tensor2img(y, maxp)), dl, ds))
    print(f"global={glob}  gate 비율 {np.mean([s['gate_ratio'] for s in gstats]):.3f}  edge 내 |flow| p50 {np.mean([s['mag_p50'] for s in gstats]):.2f} px")
    rows = []
    for k, L in per.items():
        A = np.array(L); rows.append(dict(case=k, hqnr=A[:, 0].mean(), fscc=A[:, 1].mean(), d_lambda=A[:, 2].mean(), d_s=A[:, 3].mean()))
        print(f"  {k}: HQNR {A[:,0].mean():.4f}  fSCC {A[:,1].mean():.5f}  D_l {A[:,2].mean():.4f} D_s {A[:,3].mean():.4f}   장면별 HQNR " + " ".join(f"{x:.3f}" for x in A[:, 0]))
    L1, L2, L3, L4 = (np.array(per[k]) for k in ("L1", "L2", "L3", "L4"))
    same_dir = int(((L2[:, 1] - L1[:, 1]) > 0).sum())
    ok = dict(hqnr_drop=(L1[:, 0].mean() - L2[:, 0].mean()) <= 0.0005, fscc_gain=(L2[:, 1].mean() - L1[:, 1].mean()) >= 0.005,
              scenes=same_dir >= 6, gated_gt_ungated=L2[:, 0].mean() > L3[:, 0].mean(), wrong_sign_no_gain=L4[:, 1].mean() <= L1[:, 1].mean() + 1e-6)
    print("§12.6 gate:", {k: bool(x) for k, x in ok.items()}, "->", "PASS(다음 캠페인 진입)" if all(ok.values()) else "FAIL(local alignment 종료)")
    out = os.path.join(a.run, "results", "local_diag.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("->", out)


if __name__ == "__main__":
    main()
