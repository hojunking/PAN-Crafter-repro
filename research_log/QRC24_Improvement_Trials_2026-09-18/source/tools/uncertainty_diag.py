#!/usr/bin/env python
"""Teacher uncertainty 심화 진단 — check_calibration.py 의 확장판.

check_calibration.py 는 게이트 통과/불통과만 본다(Spearman + 5분위 단조).
이 도구는 uncertainty 를 "쓸 수 있는가" 를 판정하는 데 필요한 것을 전부 낸다:

  1. Spearman(theta, |e|) / Pearson(순위 아닌 값)
  2. 조건부 오차 순서   E[e|Top10] > E[e|Top20] > E[e|Top30] > E[e] > E[e|Bottom10]
     (Top = theta 가 큰 쪽 = 모델이 모른다고 한 쪽)
  3. 10분위 error curve
  4. risk-coverage curve (+ oracle / random 기준선, AURC 와 정규화 excess)
  5. theta vs GT 국소분산(kd.ops.LocalVarianceMap)의 상관

  python tools/uncertainty_diag.py T1_c6_unc [--batches 32]

산출: work_dir/<run>/uncertainty_diag.json
      results_log/ex_log_notion/assets/<run>_unc.png
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_kd import load_teacher      # noqa: E402
from main import import_class          # noqa: E402
from kd.ops import LocalVarianceMap    # noqa: E402

ASSETS = os.path.join(ROOT, "results_log", "ex_log_notion", "assets")


def collect(run, n_batches):
    """(theta, |err|, gt_local_var) 픽셀 벡터 3개를 val set 에서 모은다."""
    wd = os.path.join(ROOT, "work_dir", run)
    cfg_path = os.path.join(wd, "meta", "config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(ROOT, "config", run + ".yaml")
    cfg = yaml.safe_load(open(cfg_path))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, has_unc = load_teacher(cfg_path, os.path.join(wd, "best_hqnr"), dev, torch.float32)
    assert has_unc, f"{run}: uncertainty head 가 없다 — T1/T2 계열만 대상"

    Feeder = import_class(cfg["feeder"])
    ds = Feeder(**cfg["val_feeder_args"])
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    varmap = LocalVarianceMap(kernel_size=5).to(dev)

    th_all, er_all, gv_all = [], [], []
    with torch.no_grad():
        for i, (gt, lms, ms, lpan, pan) in enumerate(loader):
            if i >= n_batches:
                break
            gt, ms, lpan, pan = (x.to(dev, dtype=torch.float32) for x in (gt, ms, lpan, pan))
            ones = torch.ones(pan.shape[0], device=dev)
            out = model(pan, lpan, ms, ones)
            recon = out + F.interpolate(ms, scale_factor=4, mode="bicubic")
            err = (recon - gt).abs().mean(dim=1, keepdim=True)
            th = model.theta()
            gv = varmap(gt)
            # check_calibration.py 와 같은 2배 서브샘플 — 인접 픽셀 상관을 줄인다
            th_all.append(th[..., ::2, ::2].flatten().cpu().numpy())
            er_all.append(err[..., ::2, ::2].flatten().cpu().numpy())
            gv_all.append(gv[..., ::2, ::2].flatten().cpu().numpy())
    ckpt = os.path.join(wd, "best_hqnr", "model.safetensors")
    st = os.stat(ckpt)
    sig = f"{st.st_size}-{st.st_mtime_ns}"
    return (np.concatenate(th_all), np.concatenate(er_all),
            np.concatenate(gv_all), wd, sig)


def conditional_errors(th, er):
    """theta 상위 p% 픽셀의 평균 오차. Top = theta 큰 쪽 = '모델이 모르겠다'."""
    order = np.argsort(-th)                       # theta 내림차순
    n = len(th)
    out = {}
    for p in (10, 20, 30):
        out[f"top{p}"] = float(er[order[: int(n * p / 100)]].mean())
    out["all"] = float(er.mean())
    out["bottom10"] = float(er[order[-int(n * 0.10):]].mean())
    seq = [out["top10"], out["top20"], out["top30"], out["all"], out["bottom10"]]
    out["ordering_holds"] = bool(all(seq[i] > seq[i + 1] for i in range(4)))
    out["top10_over_bottom10"] = float(out["top10"] / out["bottom10"])
    return out


def risk_coverage(th, er, grid):
    """confidence(=theta 작은 순) 로 채택했을 때의 coverage-risk 곡선."""
    n = len(th)
    by_theta = er[np.argsort(th)]                 # 확신하는 것부터
    by_oracle = np.sort(er)                       # 실제 오차가 작은 것부터 (상한)
    cs_t, cs_o = np.cumsum(by_theta), np.cumsum(by_oracle)
    idx = np.clip((grid * n).astype(int) - 1, 0, n - 1)
    risk_t = cs_t[idx] / (idx + 1)
    risk_o = cs_o[idx] / (idx + 1)
    risk_r = np.full_like(risk_t, er.mean())      # 무작위 채택 = 상수
    aurc = lambda r: float(np.trapz(r, grid) / (grid[-1] - grid[0]))
    a_t, a_o, a_r = aurc(risk_t), aurc(risk_o), aurc(risk_r)
    return {"coverage": grid.tolist(),
            "risk_theta": risk_t.tolist(), "risk_oracle": risk_o.tolist(),
            "risk_random": risk_r.tolist(),
            "aurc_theta": a_t, "aurc_oracle": a_o, "aurc_random": a_r,
            # 0 = oracle 과 동일, 1 = 무작위와 동일. 낮을수록 좋다
            "excess_ratio": float((a_t - a_o) / (a_r - a_o))}


def plot(run, th, er, gv, D, fname):
    grid = np.array(D["risk_coverage"]["coverage"])
    dec = D["decile_mae"]
    ce = D["conditional_error"]
    fig, ax = plt.subplots(1, 4, figsize=(19.5, 4.2))

    # (1) 조건부 오차 순서  E[e|Top10] > Top20 > Top30 > E[e] > Bottom10
    lab = ["E[e|\nTop10]", "E[e|\nTop20]", "E[e|\nTop30]", "E[e]", "E[e|\nBot10]"]
    val = [ce["top10"], ce["top20"], ce["top30"], ce["all"], ce["bottom10"]]
    col = ["#08306b", "#2171b5", "#4292c6", "#969696", "#a1d99b"]
    b0 = ax[0].bar(range(5), val, color=col)
    for r, v in zip(b0, val):
        ax[0].text(r.get_x() + r.get_width() / 2, v, f"{v:.4f}",
                   ha="center", va="bottom", fontsize=8)
    ax[0].set_xticks(range(5)); ax[0].set_xticklabels(lab, fontsize=8)
    ax[0].set_ylabel("actual |error|  (MAE)")
    ax[0].set_ylim(0, max(val) * 1.2); ax[0].grid(axis="y", alpha=.3)
    ok = "HOLDS" if ce["ordering_holds"] else "VIOLATED"
    ax[0].set_title(f"(1) conditional error ordering: {ok}\n"
                    f"Top10 / Bottom10 = {ce['top10_over_bottom10']:.1f}x", fontsize=10)

    # (2) 10분위 error curve
    ax[1].bar(range(1, 11), dec, color=plt.cm.Blues(np.linspace(.35, .95, 10)))
    ax[1].axhline(ce["all"], color="k", ls="--", lw=1, label=f"E[e] = {ce['all']:.4f}")
    ax[1].set_xlabel("theta decile  (D1 = most confident, D10 = least)")
    ax[1].set_ylabel("actual |error|  (MAE)")
    ax[1].set_xticks(range(1, 11))
    ax[1].set_title(f"(2) decile error curve\nD10/D1 = {dec[-1]/dec[0]:.1f}x", fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=.3)

    # (3) risk-coverage
    rc = D["risk_coverage"]
    ax[2].plot(grid, rc["risk_theta"], color="#1f77b4", lw=2, label="select by theta")
    ax[2].plot(grid, rc["risk_oracle"], color="#2ca02c", lw=1.4, ls="--", label="oracle (true error)")
    ax[2].plot(grid, rc["risk_random"], color="#999999", lw=1.4, ls=":", label="random")
    ax[2].set_xlabel("coverage  (fraction of pixels kept, most confident first)")
    ax[2].set_ylabel("risk = mean |error| on kept pixels")
    ax[2].set_title(f"(3) risk-coverage\nAURC {rc['aurc_theta']:.4f} "
                    f"(oracle {rc['aurc_oracle']:.4f} / random {rc['aurc_random']:.4f})   "
                    f"excess {rc['excess_ratio']:.2f}", fontsize=10)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    # (4) theta vs GT local variance
    q = np.quantile(gv, np.linspace(0, 1, 21))
    q = np.unique(q)
    bi = np.clip(np.digitize(gv, q[1:-1]), 0, len(q) - 2)
    xs = [gv[bi == k].mean() for k in range(len(q) - 1) if (bi == k).sum() > 50]
    ys = [th[bi == k].mean() for k in range(len(q) - 1) if (bi == k).sum() > 50]
    ax[3].hexbin(gv, th, gridsize=45, bins="log", cmap="Blues", mincnt=1)
    ax[3].plot(xs, ys, color="#d62728", lw=2, marker="o", ms=3, label="binned mean theta")
    ax[3].set_xlabel("GT local variance  (5x5, robust-normalized [0,1])")
    ax[3].set_ylabel("theta")
    gvc = D["gt_variance_corr"]
    ax[3].set_title(f"(4) theta vs GT local variance\nSpearman {gvc['spearman']:.3f}  "
                    f"Pearson {gvc['pearson']:.3f}   (GTvar vs |e| = "
                    f"{gvc['spearman_gtvar_vs_err']:.3f})", fontsize=10)
    ax[3].legend(fontsize=8)

    fig.suptitle(f"{run}  -  uncertainty diagnostics  (n = {D['n_pixels']:,} px, "
                 f"Spearman(theta,|e|) = {D['spearman']:.3f})", fontsize=11)
    fig.tight_layout()
    os.makedirs(ASSETS, exist_ok=True)
    fig.savefig(os.path.join(ASSETS, fname), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--batches", type=int, default=32)
    a = ap.parse_args()

    th, er, gv, wd, sig = collect(a.run, a.batches)
    n = len(th)

    dq = np.quantile(th, np.linspace(0.1, 0.9, 9))
    dbin = np.digitize(th, dq)
    decile = [float(er[dbin == k].mean()) for k in range(10)]

    grid = np.linspace(0.05, 1.0, 96)
    D = {
        "run": a.run, "n_pixels": int(n), "ckpt_signature": sig,
        "spearman": float(spearmanr(th, er).correlation),
        "pearson": float(pearsonr(th, er)[0]),
        "decile_mae": decile,
        "decile_monotonic": bool(all(decile[i + 1] >= decile[i] - 1e-4 for i in range(9))),
        "conditional_error": conditional_errors(th, er),
        "risk_coverage": risk_coverage(th, er, grid),
        "gt_variance_corr": {
            "spearman": float(spearmanr(th, gv).correlation),
            "pearson": float(pearsonr(th, gv)[0]),
            # 참고: GT 분산이 오차를 직접 얼마나 설명하는가 (theta 의 경쟁 가설)
            "spearman_gtvar_vs_err": float(spearmanr(gv, er).correlation),
        },
        "theta_stats": {"mean": float(th.mean()), "std": float(th.std()),
                        "p05": float(np.quantile(th, .05)), "p95": float(np.quantile(th, .95))},
    }
    json.dump(D, open(os.path.join(wd, "uncertainty_diag.json"), "w"), indent=1)
    plot(a.run, th, er, gv, D, f"{a.run}_unc.png")

    ce = D["conditional_error"]
    print(f"[unc] {a.run}  n={n:,}")
    print(f"  Spearman(theta,|e|) = {D['spearman']:.4f}   Pearson = {D['pearson']:.4f}")
    print(f"  E[e|Top10] {ce['top10']:.5f} > E[e|Top20] {ce['top20']:.5f} > "
          f"E[e|Top30] {ce['top30']:.5f} > E[e] {ce['all']:.5f} > "
          f"E[e|Bot10] {ce['bottom10']:.5f}   순서성립 {ce['ordering_holds']}"
          f"  (Top10/Bot10 = {ce['top10_over_bottom10']:.1f}x)")
    print(f"  decile MAE {['%.4f' % d for d in decile]}  단조 {D['decile_monotonic']}")
    rc = D["risk_coverage"]
    print(f"  AURC theta {rc['aurc_theta']:.5f} / oracle {rc['aurc_oracle']:.5f} / "
          f"random {rc['aurc_random']:.5f}   excess {rc['excess_ratio']:.3f}")
    g = D["gt_variance_corr"]
    print(f"  theta vs GTvar: Spearman {g['spearman']:.4f}  "
          f"(참고 GTvar vs |e| = {g['spearman_gtvar_vs_err']:.4f})")


if __name__ == "__main__":
    main()
