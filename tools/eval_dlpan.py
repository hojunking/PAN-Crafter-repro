"""DLPan-Toolbox Demo_Reduced_Resolution.m 프로토콜로 reduced-resolution 지표를 낸다.

MATLAB 이 이 머신에 없어서, 옆 저장소 `../CANConv/tools/eval_rr.py` 의 파이썬 포팅을 재사용한다.
그 구현은 CANNet 논문 Table 1 과 SAM/ERGAS/Q2n 이 0.7% 이내로 일치함이 확인돼 있고,
같은 평가기를 쓰므로 CANConv 결과와 직접 비교도 된다.

  python tools/eval_dlpan.py work_dir/wv3_baseline/results/reduced_best_reduced.mat --preset wv3

프로토콜: flag_cut_bounds=1, dim_cut=21, thvalues=0(클리핑 없음), Qblocks_size=32
"""
import os, sys, argparse
import numpy as np, h5py
from scipy.io import loadmat
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.metrics.eval_rr import evaluate  # noqa: E402


def scc_dlpan(F, G):
    """DLPan-Toolbox Quality_Indices/SCC.m 정의.

    Sobel 기울기 크기의 **전역 코사인 유사도** 다. 평균을 빼지 않고 밴드도 나누지 않는다.
    ../CANConv/tools/eval_rr.py 의 scc() 는 3x3 Laplacian + 밴드별 Pearson 이라 다른 값을 낸다
    (WV3 에서 0.878 vs 0.990). 논문 Table 과 맞는 것은 이쪽이다.
    """
    lf = np.sqrt(sobel(F[1:-1, 1:-1, :], 0) ** 2 + sobel(F[1:-1, 1:-1, :], 1) ** 2)
    lg = np.sqrt(sobel(G[1:-1, 1:-1, :], 0) ** 2 + sobel(G[1:-1, 1:-1, :], 1) ** 2)
    return float((lf * lg).sum() / np.sqrt((lf ** 2).sum()) / np.sqrt((lg ** 2).sum()))


def psnr_global(F, G, peak):
    """전 밴드 통합 MSE 기준 PSNR. 두 논문 수치와 일관된 쪽이다.

    ../CANConv/tools/eval_rr.py 의 psnr() 은 밴드별 PSNR 평균이라 약 1.5 dB 높게 나온다
    (WV3 에서 39.07 vs 37.51). 밴드별 평균을 쓰면 다른 모든 지표가 논문보다 나쁜데
    PSNR 만 좋아지는 모순이 생기므로, 통합 MSE 쪽을 기본으로 쓴다.
    다만 **PSNR 은 DLPan 표준 프로토콜에 없어** 어느 쪽도 확정할 수 없다.
    """
    return float(10 * np.log10(peak ** 2 / ((F - G) ** 2).mean()))


def ssim_skimage(F, G, peak):
    """SSIM. **DLPan 표준 프로토콜(indexes_evaluation.m)에 없는 지표다.**

    두 논문 모두 SSIM 을 보고하지만 구현을 명시하지 않아 정확한 재현이 불가능하다.
    여기서는 skimage 구현을 쓰며, 논문 값과의 차이는 구현 차이일 수 있다.
    """
    return float(structural_similarity(G / peak, F / peak, data_range=1.0, channel_axis=-1))

SCALE = {"wv3": 2047.0, "qb": 2047.0, "wv2": 2047.0, "gf2": 1023.0}
GT_H5 = {s: f"data/PanCollection/{s.upper()}/reduced_examples_h5/test_{s}_multiExm1.h5"
         for s in ("wv3", "qb", "gf2", "wv2")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mat", nargs="+", help="reduced_*.mat (PAN-Crafter 출력)")
    ap.add_argument("--preset", required=True, choices=list(SCALE))
    ap.add_argument("--dim-cut", type=int, default=21)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--baseline", action="store_true", help="lms(보간 입력) 기준선도 계산")
    a = ap.parse_args()

    scale = SCALE[a.preset]
    with h5py.File(GT_H5[a.preset]) as f:                       # 정답은 원본 h5 에서 직접
        gt = np.asarray(f["gt"], dtype=np.float64).transpose(0, 2, 3, 1)
        lms = np.asarray(f["lms"], dtype=np.float64).transpose(0, 2, 3, 1)

    cut = a.dim_cut
    sl = slice(cut - 1, -cut) if cut > 0 else slice(None)
    gt_c, lms_c = gt[:, sl, sl, :], lms[:, sl, sl, :]

    print(f"preset={a.preset}  N={len(gt)}  평가 shape={gt_c.shape[1:]}  scale={scale:.0f}")
    print(f"dim_cut={cut}  Q2n block={a.block_size}  (thvalues=0)\n")
    hdr = (f"{'':34s} {'PSNR↑*':>17} {'SSIM↑*':>17} {'SAM↓':>17} "
           f"{'ERGAS↓':>17} {'SCC↑':>17} {'Q2n↑':>17}")
    print(hdr); print("-" * len(hdr))

    rows = []
    if a.baseline:
        rows.append(("lms (보간 입력, 기준선)", lms_c))
    for p in a.mat:
        sr = loadmat(p)["sr"].astype(np.float64)                # (N,C,H,W)
        if sr.shape[1] in (4, 8):
            sr = sr.transpose(0, 2, 3, 1)
        rows.append((os.path.relpath(p).replace("work_dir/", "").replace("/results", ""),
                     sr[:, sl, sl, :]))

    for name, data in rows:
        if data.shape != gt_c.shape:
            print(f"{name:34s} shape 불일치 {data.shape} != {gt_c.shape}"); continue
        m = evaluate(data, gt_c, scale, a.block_size)
        sc = [scc_dlpan(data[i], gt_c[i]) for i in range(len(gt_c))]
        pg = [psnr_global(data[i], gt_c[i], scale) for i in range(len(gt_c))]
        ss = [ssim_skimage(data[i], gt_c[i], scale) for i in range(len(gt_c))]
        m["SCC"] = (float(np.mean(sc)), float(np.std(sc)))      # DLPan 정의로 교체
        m["PSNRb"] = m["PSNR"]                                   # 밴드별 평균 (참고)
        m["PSNR"] = (float(np.mean(pg)), float(np.std(pg)))      # 통합 MSE (논문과 일관)
        m["SSIM"] = (float(np.mean(ss)), float(np.std(ss)))
        print(f"{name:34s} " + " ".join(
            f"{m[k][0]:>10.4f}±{m[k][1]:<5.3f}"
            for k in ("PSNR", "SSIM", "SAM", "ERGAS", "SCC", "Q2n")))
        print(f"{'':34s} {'(밴드별 PSNR 평균 = ':>17}{m['PSNRb'][0]:.4f})")
    print("\n* SAM/ERGAS/Q2n: ../CANConv/tools/eval_rr.py 재사용 (MATLAB 포팅, 검증본)")
    print("* SCC: DLPan SCC.m 정의로 이 파일에서 재계산 (CANConv 포팅은 다른 정의다)")
    print("* PSNR/SSIM: **DLPan 표준 프로토콜에 없는 지표**. 논문이 구현을 명시하지 않아 정확 재현 불가")
    print("  PSNR 은 전 밴드 통합 MSE 기준. 밴드별 평균은 약 1.5 dB 높다")


if __name__ == "__main__":
    main()
