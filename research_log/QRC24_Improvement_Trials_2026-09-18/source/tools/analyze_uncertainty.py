#!/usr/bin/env python
"""Teacher uncertainty percentile 감사 — s2 계획 Appendix A.

check_calibration.py(게이트용 최소 검사)의 확장판. 재학습 없이 checkpoint 로:
  - 장면별 percentile 기준 Top-10/20/30%·Bottom-10% error lift (A.3)
  - disjoint 10분위 bin MAE
  - risk-coverage 곡선 (θ 상위 k% 제외 시 잔여 MAE)
  - θ vs GT 국소분산 상관 (중복성 진단, A.4-5)
  - edge / smooth 영역별 Spearman (A.4-6)
결과: <workdir>/uncertainty_audit.json + 요약 stdout.

  python tools/analyze_uncertainty.py work_dir/T1_c6_unc
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_kd import load_teacher  # noqa: E402
from main import import_class  # noqa: E402
from kd.ops import LocalVarianceMap, AbsoluteGradient  # noqa: E402


def main():
    wd = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "work_dir", "T1_c6_unc")
    cfg_path = os.path.join(wd, "meta", "config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(ROOT, "config", os.path.basename(wd.rstrip("/")) + ".yaml")
    cfg = yaml.safe_load(open(cfg_path))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, has_unc = load_teacher(cfg_path, os.path.join(wd, "best_hqnr"), dev, torch.float32)
    assert has_unc, "uncertainty head 없는 checkpoint"

    Feeder = import_class(cfg["feeder"])
    ds = Feeder(**cfg["val_feeder_args"])
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    var_op = LocalVarianceMap().to(dev)
    grad_op = AbsoluteGradient().to(dev)

    # 장면(sample)별로 percentile 을 계산해 평균한다 (A.3 per-image-quantile)
    lifts = {k: [] for k in ("top10", "top20", "top30", "bot10")}
    decile_mae = np.zeros(10)
    decile_cnt = 0
    risk_cov = {k: [] for k in (0, 10, 20, 30)}          # 상위 k% 제외 시 잔여 MAE
    th_all, er_all, var_all, edge_mask_all = [], [], [], []
    n_img = 0
    with torch.no_grad():
        for i, (gt, lms, ms, lpan, pan) in enumerate(loader):
            if i >= 32:
                break
            gt, ms, lpan, pan = (x.to(dev, dtype=torch.float32) for x in (gt, ms, lpan, pan))
            out = model(pan, lpan, ms, torch.ones(pan.shape[0], device=dev))
            recon = out + F.interpolate(ms, scale_factor=4, mode="bicubic")
            err = (recon - gt).abs().mean(dim=1)                     # (B,H,W)
            th = model.theta().squeeze(1)                            # (B,H,W)
            gvar = var_op(gt).squeeze(1)
            edge = (grad_op(pan).squeeze(1) >
                    grad_op(pan).squeeze(1).flatten(1).median(dim=1).values[:, None, None])
            for b in range(err.shape[0]):
                e, t = err[b].flatten(), th[b].flatten()
                q = torch.quantile(t, torch.tensor([0.1, 0.7, 0.8, 0.9], device=dev))
                mean_all = e.mean()
                lifts["top10"].append((e[t >= q[3]].mean() / mean_all).item())
                lifts["top20"].append((e[t >= q[2]].mean() / mean_all).item())
                lifts["top30"].append((e[t >= q[1]].mean() / mean_all).item())
                lifts["bot10"].append((e[t <= q[0]].mean() / mean_all).item())
                dq = torch.quantile(t, torch.linspace(0.1, 0.9, 9, device=dev))
                bins = torch.bucketize(t, dq)
                for d in range(10):
                    m = bins == d
                    if m.any():
                        decile_mae[d] += e[m].mean().item()
                decile_cnt += 1
                for k in risk_cov:
                    keep = t <= torch.quantile(t, 1 - k / 100) if k else torch.ones_like(t, dtype=torch.bool)
                    risk_cov[k].append(e[keep].mean().item())
                n_img += 1
            sl = (slice(None), slice(None, None, 2), slice(None, None, 2))
            th_all.append(th[sl].flatten().cpu().numpy())
            er_all.append(err[sl].flatten().cpu().numpy())
            var_all.append(gvar[sl].flatten().cpu().numpy())
            edge_mask_all.append(edge[sl].flatten().cpu().numpy())

    th_np = np.concatenate(th_all); er_np = np.concatenate(er_all)
    var_np = np.concatenate(var_all); em = np.concatenate(edge_mask_all).astype(bool)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(th_np), min(200_000, len(th_np)), replace=False)
    rho_all = spearmanr(th_np[sel], er_np[sel]).statistic
    rho_var = spearmanr(th_np[sel], var_np[sel]).statistic
    e_sel, s_sel = sel[em[sel]], sel[~em[sel]]
    rho_edge = spearmanr(th_np[e_sel], er_np[e_sel]).statistic
    rho_smooth = spearmanr(th_np[s_sel], er_np[s_sel]).statistic

    L = {k: float(np.mean(v)) for k, v in lifts.items()}
    rc = {f"top{k}%_제외시_잔여MAE": float(np.mean(v)) for k, v in risk_cov.items()}
    dec = (decile_mae / decile_cnt).tolist()
    mono = all(dec[i + 1] >= dec[i] - 1e-4 for i in range(9))
    ok = rho_all > 0 and L["top10"] > 1.5 and mono
    out = {"n_images": n_img, "spearman_all": float(rho_all),
           "spearman_edge_region": float(rho_edge), "spearman_smooth_region": float(rho_smooth),
           "spearman_theta_vs_gt_variance": float(rho_var),
           "error_lift": L, "decile_mae": dec, "decile_monotonic": mono,
           "risk_coverage": rc, "pass": bool(ok)}
    json.dump(out, open(os.path.join(wd, "uncertainty_audit.json"), "w"), indent=1)
    print(f"[audit] {os.path.basename(wd)} ({n_img}장, 장면별 percentile)")
    print(f"  error lift: Top10 {L['top10']:.2f}x  Top20 {L['top20']:.2f}x  "
          f"Top30 {L['top30']:.2f}x  Bottom10 {L['bot10']:.2f}x")
    print(f"  Spearman: 전체 {rho_all:.3f}  edge 영역 {rho_edge:.3f}  "
          f"smooth 영역 {rho_smooth:.3f}  |  θ~GT분산 {rho_var:.3f}")
    print(f"  risk-coverage: " + "  ".join(f"{k}: {v:.4f}" for k, v in rc.items()))
    print(f"  10분위 단조: {mono}  ->  {'유효' if ok else '재검토'}")


if __name__ == "__main__":
    main()
