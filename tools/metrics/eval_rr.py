"""RR(reduced-resolution) 참조 지표: PSNR / SAM / ERGAS / SCC / Q2n(Q8).

    python tools/eval_rr.py runs/eval_cannet_wv3_pretrained/reduced --preset wv3
    python tools/eval_rr.py <results_dir> --preset wv3 --baseline   # lms 입력 기준선도 함께

DLPan-Toolbox `Demo_Reduced_Resolution.m`의 프로토콜을 따른다.

    flag_cut_bounds = 1, dim_cut = 21   경계 21픽셀 제거 후 평가
    thvalues        = 0                 I_F를 클리핑하지 않음
    Qblocks_size    = 32                Q2n 블록 크기

검증 (WV3 RR, 사전학습 weights/cannet_wv3.pth, 20장):

    지표     본 구현            논문 Table 1
    SAM      2.9229 ± 0.5529    2.930 ± 0.593
    ERGAS    2.1716 ± 0.5074    2.158 ± 0.515
    Q8       0.9188 ± 0.0809    0.920 ± 0.084

PSNR과 SCC는 논문이 보고하지 않아 대조 기준이 없다. 나머지 셋은 0.7% 이내로 일치한다.
학습 중 모니터링과 체크포인트 선택에 쓰기 충분하다. 논문 비교표를 만들 때는
어느 구현으로 계산했는지 표에 명시할 것.
"""

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import scipy.io as sio
from scipy.ndimage import convolve

# 같은 디렉터리의 q2n 을 쓴다. 패키지로 import 되든 스크립트로 직접 실행되든 동작한다.
try:
    from .q2n import q2n
except ImportError:  # python tools/metrics/eval_rr.py 처럼 직접 실행한 경우
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from q2n import q2n  # noqa: E402

# SCC용 고역통과 필터 (DLPan의 ScoreCC와 동일한 3x3 Laplacian)
HIGHPASS = np.array([[-1.0, -1.0, -1.0],
                     [-1.0, 8.0, -1.0],
                     [-1.0, -1.0, -1.0]])


def psnr(sr: np.ndarray, gt: np.ndarray, peak: float) -> float:
    """밴드별 PSNR의 평균. sr/gt: (H, W, C)"""
    mse = ((sr - gt) ** 2).mean(axis=(0, 1))
    return float(np.mean(10 * np.log10(peak**2 / np.maximum(mse, 1e-12))))


def sam(sr: np.ndarray, gt: np.ndarray) -> float:
    """Spectral Angle Mapper (degree). 0에 가까울수록 분광 보존이 좋다."""
    num = (sr * gt).sum(axis=2)
    den = np.linalg.norm(sr, axis=2) * np.linalg.norm(gt, axis=2)
    valid = den > 1e-8
    cos = np.clip(num[valid] / den[valid], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def ergas(sr: np.ndarray, gt: np.ndarray, ratio: int = 4) -> float:
    """Erreur Relative Globale Adimensionnelle de Synthese."""
    rmse_b = np.sqrt(((sr - gt) ** 2).mean(axis=(0, 1)))
    mu_b = gt.mean(axis=(0, 1))
    return float(100.0 / ratio * np.sqrt(np.mean((rmse_b / np.maximum(mu_b, 1e-8)) ** 2)))


def scc(sr: np.ndarray, gt: np.ndarray) -> float:
    """고역통과 성분의 밴드별 상관계수 평균. 공간 디테일 보존을 본다."""
    vals = []
    for b in range(sr.shape[2]):
        a = convolve(sr[:, :, b], HIGHPASS, mode="reflect")
        c = convolve(gt[:, :, b], HIGHPASS, mode="reflect")
        a = a - a.mean()
        c = c - c.mean()
        den = np.sqrt((a**2).sum() * (c**2).sum())
        vals.append(float((a * c).sum() / den) if den > 1e-12 else 0.0)
    return float(np.mean(vals))


def evaluate(sr_all: np.ndarray, gt_all: np.ndarray, peak: float, block: int = 32) -> dict:
    """sr_all/gt_all: (N, H, W, C)"""
    rows = []
    for i in range(sr_all.shape[0]):
        sr, gt = sr_all[i], gt_all[i]
        rows.append({
            "PSNR": psnr(sr, gt, peak),
            "SAM": sam(sr, gt),
            "ERGAS": ergas(sr, gt),
            "SCC": scc(sr, gt),
            "Q2n": q2n(gt, sr, block, block)[0],
        })
    return {k: (float(np.mean([r[k] for r in rows])),
                float(np.std([r[k] for r in rows]))) for k in rows[0]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="output_mulExm_*.mat이 들어 있는 디렉터리")
    parser.add_argument("--preset", required=True, help="wv3/qb/gf2/wv2")
    parser.add_argument("--baseline", action="store_true",
                        help="lms(보간 입력) 기준선도 함께 계산")
    parser.add_argument("--dim-cut", type=int, default=21,
                        help="경계 제거 폭. Demo_Reduced_Resolution.m 기본값 21. 0이면 제거 안 함")
    parser.add_argument("--block-size", type=int, default=32, help="Q2n 블록 크기")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, "presets.json")) as fp:
        cfg = json.load(fp)[args.preset]
    scale = float(cfg["dataset_scale"])

    results_dir = args.results_dir
    if os.path.basename(results_dir) != "results" and os.path.isdir(os.path.join(results_dir, "results")):
        results_dir = os.path.join(results_dir, "results")

    files = sorted(glob.glob(os.path.join(results_dir, "output_mulExm_*.mat")),
                   key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    if not files:
        print(f"결과 .mat을 찾을 수 없다: {results_dir}")
        return 1

    sr_all = np.stack([sio.loadmat(f)["sr"].astype(np.float64) for f in files])  # (N,H,W,C)

    with h5py.File(cfg["test_reduced_data"], "r") as f:
        if "gt" not in f:
            print("RR 테스트 파일에 gt가 없다. FR 결과에는 이 스크립트를 쓸 수 없다.")
            return 1
        gt_all = np.asarray(f["gt"][:len(files)], dtype=np.float64)      # (N,C,H,W)
        lms_all = np.asarray(f["lms"][:len(files)], dtype=np.float64)
    gt_all = gt_all.transpose(0, 2, 3, 1)
    lms_all = lms_all.transpose(0, 2, 3, 1)

    if sr_all.shape != gt_all.shape:
        print(f"shape 불일치: sr={sr_all.shape} gt={gt_all.shape}")
        return 1

    # Demo_Reduced_Resolution.m: flag_cut_bounds=1, dim_cut=21 -> I(dim_cut:end-dim_cut)
    cut = args.dim_cut
    if cut > 0:
        sl = slice(cut - 1, -cut)
        sr_all = sr_all[:, sl, sl, :]
        gt_all = gt_all[:, sl, sl, :]
        lms_all = lms_all[:, sl, sl, :]

    print(f"preset={args.preset}  N={len(files)}  shape={sr_all.shape[1:]}  scale={scale:.0f}")
    print(f"dim_cut={cut}  Q2n block={args.block_size}  (th_values=0: 클리핑 없음)")
    print(f"결과: {results_dir}")
    print(f"정답: {cfg['test_reduced_data']}")
    print()
    header = f"{'':16s} {'PSNR↑':>16} {'SAM↓':>16} {'ERGAS↓':>16} {'SCC↑':>16} {'Q2n↑':>16}"
    print(header)
    print("-" * len(header))

    targets = [("model", sr_all)]
    if args.baseline:
        targets.append(("lms (보간 입력)", lms_all))

    for name, data in targets:
        m = evaluate(data, gt_all, scale, args.block_size)
        print(f"{name:16s} " + " ".join(
            f"{m[k][0]:>10.4f}±{m[k][1]:<5.3f}" for k in ("PSNR", "SAM", "ERGAS", "SCC", "Q2n")))

    print()
    print("* DLPan Demo_Reduced_Resolution.m 프로토콜. SAM/ERGAS/Q2n은 논문 수치와 0.7% 이내 일치 확인.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
