#!/usr/bin/env python
"""CANConv 배포 가중치의 출력(컨테이너 tools/infer_h5.py 산출 sr h5)을 우리 평가기(지표 v2)로 재서
센서별 참조 run `work_dir/_ref_cannet_<sensor>/` 을 만들고 논문의 CANConv 행과 대조한다 — 평가기·데이터 anchor.

    python tools/make_cannet_reference.py --sensor qb --rr ../CANConv/data/datasets/qb/sr_cannet_rr.h5 --fr ../CANConv/data/datasets/qb/sr_cannet_mat20.h5
    python tools/make_cannet_reference.py --sensor wv2 --rr .../sr_cannet_rr_from_wv3.h5 --fr .../sr_cannet_mat20_from_wv3.h5 --weights cannet_wv3.pth

산출: results/reduced_best_val.mat (sr, NCHW DN), results/fr_mat20.json (provenance 포함). 시트는 EXTERNAL 로 올린다.
"""
import os, sys, json, argparse, datetime, hashlib
import numpy as np, h5py
from scipy.io import savemat, loadmat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.metrics.eval_fr import load_dlpan, d_lambda_k, d_s        # noqa: E402
from tools.eval_fr_paperset import EVAL_VERSION, h5_for, sha256_of   # noqa: E402
import importlib.util                                                  # noqa: E402
# 저장소의 gspread/ 는 pip 패키지 gspread 에 가려지므로(정규 패키지 우선) 경로로 읽는다
_spec = importlib.util.spec_from_file_location("_gu", os.path.join(ROOT, "gspread", "gspread_upload.py")); gu = importlib.util.module_from_spec(_spec); sys.modules["_gu"] = gu; _spec.loader.exec_module(gu)

# 두 논문(PAN-Crafter Table 1-3 · U-Know Table 2-3, 보충자료 ±) 의 CANConv 행 — 동일한 숫자
PAPER = {
    "wv3": dict(psnr=37.441, ssim=0.973, sam=2.927, ergas=2.163, scc=0.985, q2n=0.918, d_lambda=0.020, d_s=0.030, hqnr=0.951),
    "qb":  dict(psnr=37.795, ssim=0.960, sam=4.554, ergas=3.740, scc=0.982, q2n=0.935, d_lambda=0.039, d_s=0.070, hqnr=0.893),
    "gf2": dict(psnr=43.166, ssim=0.982, sam=0.722, ergas=0.653, scc=0.991, q2n=0.983, d_lambda=0.019, d_s=0.063, hqnr=0.919),
    "wv2": dict(psnr=29.005, ssim=0.837, sam=5.481, ergas=4.328, scc=0.918, q2n=0.841, d_lambda=0.068, d_s=0.060, hqnr=0.876),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensor", required=True, choices=list(PAPER)); ap.add_argument("--rr", required=True); ap.add_argument("--fr", default=None)
    ap.add_argument("--weights", default=None, help="기본 cannet_<sensor>.pth")
    a = ap.parse_args()
    s = a.sensor; wd = os.path.join(ROOT, "work_dir", f"_ref_cannet_{s}"); os.makedirs(os.path.join(wd, "results"), exist_ok=True)
    weights = a.weights or f"cannet_{s}.pth"
    with h5py.File(a.rr) as f:
        sr = np.asarray(f["sr"], dtype=np.float64)                      # (N,C,H,W) uint16 -> float
    mat = os.path.join(wd, "results", "reduced_best_val.mat"); savemat(mat, dict(sr=sr))
    rr = gu._rr(mat, s)
    print(f"[{s}] RR (CANConv {weights}, 20장, 지표 v2)  vs 논문 CANConv 행")
    for k in ("psnr", "ssim", "sam", "ergas", "scc", "q2n"):
        print(f"   {k:6s} {rr[k]:9.4f}   paper {PAPER[s][k]:8.3f}   ({100*(rr[k]-PAPER[s][k])/PAPER[s][k]:+.2f}%)")
    if a.fr:
        wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")); h5 = h5_for(s)
        with h5py.File(h5) as f:
            lms = np.asarray(f["lms"], dtype=np.float64).transpose(0, 2, 3, 1); pan = np.asarray(f["pan"], dtype=np.float64)[:, 0]
        with h5py.File(a.fr) as f:
            fsr = np.asarray(f["sr"], dtype=np.float64).transpose(0, 2, 3, 1)
        with h5py.File(h5) as f:
            ms_all = np.asarray(f["ms"], dtype=np.float64).transpose(0, 2, 3, 1)
        dl = np.array([d_lambda_k(fsr[i], lms[i], s, 4, 32, wald) for i in range(len(fsr))])
        dsv = np.array([d_s(fsr[i], lms[i], pan[i], 4, 32, wald) for i in range(len(fsr))]); h = (1 - dl) * (1 - dsv)
        print(f"   D_l    {dl.mean():9.4f}±{dl.std(ddof=1):.4f}   paper {PAPER[s]['d_lambda']:8.3f}")
        print(f"   D_s    {dsv.mean():9.4f}±{dsv.std(ddof=1):.4f}   paper {PAPER[s]['d_s']:8.3f}")
        print(f"   HQNR   {h.mean():9.4f}±{h.std(ddof=1):.4f}   paper {PAPER[s]['hqnr']:8.3f}   ({100*(h.mean()-PAPER[s]['hqnr'])/PAPER[s]['hqnr']:+.2f}%)")
        from tools.eval_fr_paperset import jqm_fields
        jq = jqm_fields([fsr[i] for i in range(len(fsr))], ms_all, pan, s, 1023.0 if s == "gf2" else 2047.0)
        print(f"   JQM    {jq['jqm']:9.4f}±{jq['jqm_sd']:.4f}   (논문 미보고; QLR {jq['qlr']:.4f} QHR {jq['qhr']:.4f})")
        j = dict(hqnr=float(h.mean()), hqnr_sd=float(h.std(ddof=1)), d_lambda=float(dl.mean()), d_lambda_sd=float(dl.std(ddof=1)), d_s=float(dsv.mean()), d_s_sd=float(dsv.std(ddof=1)),
                 per_scene_hqnr=[round(float(x), 6) for x in h], n=int(len(fsr)), checkpoint=f"../CANConv/weights/{weights} (released)", sensor=s,
                 forward="CANConv tools/infer_h5.py (container), sr uint16 반올림", input_h5=h5, input_sha256=sha256_of(h5), lpan_sha256="", config_sha256="",
                 eval_version=EVAL_VERSION, std="ddof=1 (MATLAB std)", evaluated_at=datetime.datetime.now().isoformat(timespec="seconds"), **jq)
        json.dump(j, open(os.path.join(wd, "results", "fr_mat20.json"), "w"), indent=1)
    print(f"   -> {os.path.relpath(wd, ROOT)}")


if __name__ == "__main__":
    main()
