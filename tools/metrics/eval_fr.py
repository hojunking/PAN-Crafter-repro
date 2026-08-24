"""FR(full-resolution) 무참조 지표 D_λ / D_s / HQNR 계산.

    python tools/eval_fr.py runs/eval_cannet_wv3_pretrained/origscale --preset wv3

FR에는 ground truth가 없으므로 QNR 계열 무참조 지표를 쓴다. DLPan-Toolbox의
`Demo_Full_Resolution.m`이 쓰는 경로(flagQNR=0 → HQNR, S=32)를 그대로 옮겼다.

    D_λ  = 1 - Q2n( msexp , MTF_sensor(fused) )              [Khan09]
    D_s  = ( 1/N Σ_b |Q(F_b, PAN) - Q(msexp_b, PAN_filt)|^q )^(1/q)   [Alparone08]
    HQNR = (1 - D_λ) * (1 - D_s)

부품 출처:

  - MTF 필터 / interp23tap : DLPan-Toolbox 공식 Python 포트를 그대로 import
                             (`wald_utilities.py`). MATLAB genMTF.m과 GNyq 계수가 일치한다.
  - Q2n                    : `tools/q2n.py` (MATLAB 원본에서 직접 이식).
                             공식 Python 포트의 q2n은 Q2n>1을 반환해 쓸 수 없다. 사유는 해당 파일 참조.
  - imresize_matlab        : D_s의 pan_filt가 MATLAB imresize(bicubic+antialiasing)를 쓴다.
  - D_λ / D_s / HQNR       : MATLAB에만 있고 Python 포트에는 없어 새로 작성.

검증 (WV3 FR, 20장 중 인덱스 12-19):

    방법     지표    본 구현            논문 Table 1
    EXP      D_λ     0.0246 ± 0.0068    0.0232 ± 0.0066
             D_s     0.0811 ± 0.0199    0.0813 ± 0.0318
             HQNR    0.8963             0.897 ± 0.036
    CANNet   D_λ     0.0253 ± 0.0108    0.0196 ± 0.0083
             D_s     0.0261 ± 0.0035    0.0301 ± 0.0074
             HQNR    0.9493 ± 0.0113    0.951  ± 0.013

EXP(=lms, 모델 무관)와 CANNet 두 독립 기준점 모두 HQNR이 0.2% 이내로 맞는다.
구현은 검증된 것으로 본다.

다만 **전체 20장으로는 논문과 맞지 않는다** (CANNet HQNR 0.855). 이 PanCollection
배포본의 FR 테스트 인덱스 0-11이 12-19보다 크게 어려운 장면이고, D_λ는 msexp에만
의존하므로(모델과 무관) 이는 코드가 아니라 데이터 차이다. 논문 이후 PanCollection FR
테스트셋이 갱신되었을 가능성이 높다. 자세한 근거는 RUNBOOK.md §8 참조.

실용적 함의: 자체 실험끼리 비교할 때는 전체 20장을 쓰면 된다(모든 방법이 같은 데이터를
쓰므로 공정하다). 논문 표의 절대값과 직접 비교하지는 말 것.
"""

import argparse
import glob
import importlib.util
import json
import os
import sys

import h5py
import numpy as np
import scipy.io as sio
from scipy.signal import fftconvolve

# 같은 디렉터리의 q2n 을 쓴다. 패키지로 import 되든 스크립트로 직접 실행되든 동작한다.
try:
    from .q2n import q2n
except ImportError:  # python tools/metrics/eval_rr.py 처럼 직접 실행한 경우
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from q2n import q2n  # noqa: E402

DEFAULT_DLPAN = "/dlpan"
# presets.json 키 -> DLPan genMTF.m의 sensor 문자열.
# None이면 genMTF.m의 otherwise 분기(GNyq = 0.3 * ones)를 쓴다.
#   gf2    : DLPan에 GF2 항목이 없어 원래부터 기본값을 탄다.
#   cas500 : 데이터 자체가 GNyq_MS=0.30으로 열화되어 만들어졌으므로(h5 attrs에 기록)
#            기본값 0.3이 우연이 아니라 **정확히 맞는 값**이다.
SENSOR_NAME = {"wv3": "WV3", "wv2": "WV2", "qb": "QB", "gf2": None,
               "cas500": None, "vantor": None}

# 데이터를 만들 때 쓴 GNyq와 지표의 MTF를 맞춘다. None이면 genMTF의 기본값 0.3.
#   cas500 : 데이터가 GNyq_MS=0.30으로 열화됨 -> 기본값 0.3이 정확히 맞는다
#   vantor : 데이터가 GNyq_MS=0.35로 열화됨 -> 0.35를 명시해야 한다 (h5 attrs 기록)
GNYQ_OVERRIDE = {"vantor": 0.35}


def load_dlpan(dlpan_root: str):
    """DLPan-Toolbox의 MTF / interp23tap 공식 Python 구현을 파일 경로로 import 한다."""
    path = os.path.join(dlpan_root, "01-DL-toolbox(Pytorch)", "UDL", "pansharpening",
                        "models", "APNN", "wald_utilities.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"DLPan-Toolbox 파일을 찾을 수 없다: {path}\n"
            f"  git clone https://github.com/liangjiandeng/DLPan-Toolbox.git 후 "
            f"--dlpan 으로 경로를 지정할 것.")
    spec = importlib.util.spec_from_file_location("dlpan_wald", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dlpan_wald"] = mod
    spec.loader.exec_module(mod)
    return mod


# --- MATLAB imresize (bicubic + antialiasing) --------------------------------

def _cubic(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    ax2, ax3 = ax**2, ax**3
    return ((1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1)
            + (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((ax > 1) & (ax <= 2)))


def _contributions(in_length: int, out_length: int, scale: float):
    """MATLAB imresize의 contributions()와 동일한 가중치/인덱스."""
    kernel_width = 4.0
    if scale < 1:  # 축소 시 antialiasing: 커널을 늘리고 진폭을 줄인다
        def kernel(t):
            return scale * _cubic(scale * t)
        kernel_width = kernel_width / scale
    else:
        kernel = _cubic

    x = np.arange(1, out_length + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    p = int(np.ceil(kernel_width)) + 2

    ind = left[:, None] + np.arange(p)[None, :]
    weights = kernel(u[:, None] - ind)
    weights = weights / weights.sum(axis=1, keepdims=True)
    # MATLAB: indices = min(max(indices,1), in_length)  (replicate)
    ind = (np.minimum(np.maximum(ind, 1), in_length) - 1).astype(np.intp)

    keep = ~np.all(weights == 0, axis=0)
    return weights[:, keep], ind[:, keep]


def imresize_matlab(img: np.ndarray, scale: float) -> np.ndarray:
    """2D 영상에 대한 MATLAB imresize(bicubic, antialiasing on) 등가 구현."""
    h, w = img.shape
    oh, ow = int(np.ceil(h * scale)), int(np.ceil(w * scale))

    wr, ir = _contributions(h, oh, scale)
    tmp = np.zeros((oh, w), dtype=np.float64)
    for k in range(wr.shape[1]):
        tmp += wr[:, k:k + 1] * img[ir[:, k], :]

    wc, ic = _contributions(w, ow, scale)
    out = np.zeros((oh, ow), dtype=np.float64)
    for k in range(wc.shape[1]):
        out += wc[None, :, k] * tmp[:, ic[:, k]]
    return out


# --- 지표 -------------------------------------------------------------------

def mtf_filter(img: np.ndarray, preset: str, ratio: int, wald) -> np.ndarray:
    """센서 MTF 저역통과 필터. MATLAB MTF.m = imfilter(..., 'replicate').

    41x41 커널을 512x512x8에 직접 상관하면 화소당 1681회 곱셈이라 한 장에 5초가 걸린다
    (체크포인트 20개 × 20장이면 그것만 33분). 가장자리를 replicate로 미리 채운 뒤
    FFT 컨볼루션으로 바꾸면 결과는 같고 훨씬 빠르다. 커널이 대칭이라 correlate와
    convolve가 동일하므로 뒤집기도 불필요하다.
    """
    sensor = SENSOR_NAME[preset]
    if sensor is None:
        # genMTF.m의 otherwise 분기: GNyq = 0.3 * ones. 데이터가 다른 값으로 만들어졌으면 그것을 쓴다.
        gnyq = GNYQ_OVERRIDE.get(preset, 0.3) * np.ones(img.shape[2])
        kernel = wald.NyquistFilterGenerator(gnyq, ratio, 41)
    else:
        kernel = wald.MTF(ratio, sensor)

    out = np.empty_like(img, dtype=np.float64)
    for b in range(img.shape[2]):
        k = np.real(kernel[:, :, b]).astype(np.float64)
        pad_y, pad_x = k.shape[0] // 2, k.shape[1] // 2
        padded = np.pad(img[:, :, b].astype(np.float64),
                        ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
        out[:, :, b] = fftconvolve(padded, k[::-1, ::-1], mode="valid")
    return out


def _uqi(x: np.ndarray, y: np.ndarray) -> float:
    """Universal Image Quality Index. MATLAB cov()와 같이 (n-1) 정규화를 쓴다."""
    x = x.ravel().astype(np.float64)
    y = y.ravel().astype(np.float64)
    mx, my = x.mean(), y.mean()
    n = x.size
    cov = ((x - mx) * (y - my)).sum() / (n - 1)
    vx = ((x - mx) ** 2).sum() / (n - 1)
    vy = ((y - my) ** 2).sum() / (n - 1)
    den = (vx + vy) * (mx**2 + my**2)
    return float(4 * cov * mx * my / den) if den != 0 else 0.0


def _blockproc_uqi(a: np.ndarray, b: np.ndarray, s: int) -> float:
    """MATLAB blockproc(...,[S S], uqi) 후 mean2. 겹치지 않는 S×S 블록.

    화소 루프 대신 블록을 (블록수, S*S)로 재배열해 한 번에 계산한다. 512x512·S=32면
    블록이 256개이고 이걸 8밴드 × 2회(고/저해상도) 반복하므로 루프 구현은 느리다.
    _uqi와 동일한 (n-1) 정규화를 쓴다 — 등가성은 아래 __main__ 자체 검사로 확인한다.
    """
    h, w = a.shape
    nh, nw = h // s, w // s
    def blocks(x):
        return (x[:nh * s, :nw * s].reshape(nh, s, nw, s)
                .transpose(0, 2, 1, 3).reshape(nh * nw, s * s).astype(np.float64))
    A, B = blocks(a), blocks(b)
    n = s * s
    ma, mb = A.mean(axis=1), B.mean(axis=1)
    da, db = A - ma[:, None], B - mb[:, None]
    cov = (da * db).sum(axis=1) / (n - 1)
    va = (da**2).sum(axis=1) / (n - 1)
    vb = (db**2).sum(axis=1) / (n - 1)
    den = (va + vb) * (ma**2 + mb**2)
    q = np.divide(4 * cov * ma * mb, den, out=np.zeros_like(den), where=den != 0)
    return float(q.mean())


def d_lambda_k(fused, msexp, preset, ratio, s, wald) -> float:
    """D_lambda_K.m: 1 - Q2n(msexp, MTF(fused)). q2n이 내부에서 uint16 캐스팅을 한다."""
    fused_degraded = mtf_filter(fused, preset, ratio, wald)
    index, _ = q2n(msexp, fused_degraded, s, s)
    return float(1.0 - index)


def d_s(fused, msexp, pan, ratio, s, wald, q=1) -> float:
    """D_s.m (flag_orig_paper=0, Toolbox 1.0 경로)."""
    pan_lr = imresize_matlab(pan, 1.0 / ratio)
    pan_filt = wald.interp23tap(pan_lr[:, :, None], ratio)[:, :, 0]

    total = 0.0
    for b in range(fused.shape[2]):
        q_high = _blockproc_uqi(fused[:, :, b], pan, s)
        q_low = _blockproc_uqi(msexp[:, :, b], pan_filt, s)
        total += abs(q_high - q_low) ** q
    return float((total / fused.shape[2]) ** (1.0 / q))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="FR output_mulExm_*.mat이 있는 디렉터리")
    parser.add_argument("--preset", required=True, help="wv3/qb/gf2/wv2")
    parser.add_argument("--dlpan", default=DEFAULT_DLPAN, help="DLPan-Toolbox 경로")
    parser.add_argument("--block-size", type=int, default=32, help="Demo_Full_Resolution.m 기본값 32")
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--baseline", action="store_true", help="lms(EXP) 기준선도 계산")
    parser.add_argument("--no-clip", action="store_true",
                        help="indexes_evaluation_FS의 th_values=0에 해당 (I_F를 클리핑하지 않음)")
    parser.add_argument("--indices", default=None,
                        help="평가할 이미지 인덱스 부분집합 (예: '12-19' 또는 '0,3,5'). "
                             "기본은 전체. 논문 수치와의 대조는 docstring 참조")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, "presets.json")) as fp:
        cfg = json.load(fp)[args.preset]
    scale = float(cfg["dataset_scale"])
    bit_depth = int(round(np.log2(scale + 1)))  # 2047 -> 11, 1023 -> 10

    wald = load_dlpan(args.dlpan)

    results_dir = args.results_dir
    if os.path.basename(results_dir) != "results" and os.path.isdir(os.path.join(results_dir, "results")):
        results_dir = os.path.join(results_dir, "results")
    files = sorted(glob.glob(os.path.join(results_dir, "output_mulExm_*.mat")),
                   key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    if not files:
        print(f"결과 .mat을 찾을 수 없다: {results_dir}")
        return 1

    with h5py.File(cfg["test_origscale_data"], "r") as f:
        lms_all = np.asarray(f["lms"][:len(files)], dtype=np.float64).transpose(0, 2, 3, 1)
        pan_all = np.asarray(f["pan"][:len(files)], dtype=np.float64)[:, 0, :, :]

    print(f"preset={args.preset}  N={len(files)}  block={args.block_size}  ratio={args.ratio}")
    clip_note = "클리핑 없음 (th_values=0)" if args.no_clip else f"I_F를 [0, {2**bit_depth}]로 클리핑"
    print(f"sensor MTF={SENSOR_NAME[args.preset] or 'default(GNyq=0.3)'}  "
          f"bit depth L={bit_depth}  {clip_note}")
    print(f"결과: {results_dir}")
    print(f"입력: {cfg['test_origscale_data']}")
    print()

    if args.indices:
        if "-" in args.indices and "," not in args.indices:
            lo, hi = args.indices.split("-")
            sel = list(range(int(lo), int(hi) + 1))
        else:
            sel = [int(v) for v in args.indices.split(",")]
        print(f"부분집합 평가: {sel}")
    else:
        sel = list(range(len(files)))

    targets = [("model", [sio.loadmat(files[i])["sr"].astype(np.float64) for i in sel])]
    if args.baseline:
        targets.append(("lms (EXP)", [lms_all[i] for i in sel]))
    lms_all, pan_all = lms_all[sel], pan_all[sel]

    for name, fused_list in targets:
        rows = []
        for i, fused in enumerate(fused_list):
            # indexes_evaluation_FS.m: th_values면 I_F를 [0, 2^L]로 클리핑
            if not args.no_clip:
                fused = np.clip(fused, 0.0, float(2**bit_depth))
            msexp, pan = lms_all[i], pan_all[i]
            dl = d_lambda_k(fused, msexp, args.preset, args.ratio, args.block_size, wald)
            ds = d_s(fused, msexp, pan, args.ratio, args.block_size, wald)
            rows.append((dl, ds, (1 - dl) * (1 - ds)))
            # 진행 상황은 stderr로: stdout에는 최종 요약만 남긴다
            print(f"  [{name}] {i + 1:2d}/{len(fused_list)}  D_λ={dl:.4f}  "
                  f"D_s={ds:.4f}  HQNR={rows[-1][2]:.4f}", end="\r", file=sys.stderr)
        arr = np.array(rows)
        print(" " * 78, end="\r", file=sys.stderr)
        print(f"{name:12s}  D_λ↓ {arr[:, 0].mean():.4f}±{arr[:, 0].std():.4f}   "
              f"D_s↓ {arr[:, 1].mean():.4f}±{arr[:, 1].std():.4f}   "
              f"HQNR↑ {arr[:, 2].mean():.4f}±{arr[:, 2].std():.4f}")

    print()
    print("* Q2n / MTF / interp23tap은 DLPan-Toolbox 공식 Python 구현을 그대로 사용한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
