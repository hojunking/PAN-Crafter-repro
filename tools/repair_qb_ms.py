#!/usr/bin/env python
"""PanCollection QB 학습·검증셋의 `ms` 를 Wald 프로토콜대로 다시 만든다 (KNOWN_ISSUES F-3).

배포 `train_qb.h5`/`valid_qb.h5` 는 패치의 67~70% 에서 `ms` 가 `gt` 에 대해 LR 1픽셀(HR 4픽셀)만큼
어긋나 있다 — MTF↓gt 를 (2,2) 대신 (1,2) 또는 (2,1) 위상으로 데시메이션한 것과 정확히 같다(그 위상으로
맞추면 MAD 0.00 DN). WV3·GF2 학습셋과 QB 테스트셋은 전부 (2,2)로 정확하다. `ms` 를 입력으로 쓰는 모델
(PAN-Crafter 계열: up(MS) 입력·bicubic(ms) 잔차 base)은 학습 표본의 2/3 가 4px 어긋난 분광 입력을 보게 된다.

    python tools/repair_qb_ms.py            # QB/train_qb_msfix.h5, QB/valid_qb_msfix.h5 (+ *_msfix_pan.h5 링크)

레시피 (WV3/GF2 배포본을 MAD 0.00 으로 재현하는 것과 동일): ms = genMTF(QB)·replicate 필터(gt)[2::4, 2::4],
lms = interp23tap(ms). gt·pan 은 그대로. 원본은 건드리지 않는다.
"""
import os, sys, argparse, json, datetime
import numpy as np, h5py
from scipy.signal import fftconvolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.metrics.eval_fr import genmtf_matlab, GNYQ_TABLE, load_dlpan   # noqa: E402


def mtf_down(gt_chw, kernel):
    """gt (C,H,W) -> ms (C,H/4,W/4): imfilter replicate + [2::4, 2::4]."""
    out = []
    for b in range(gt_chw.shape[0]):
        k = kernel[:, :, b]; p = k.shape[0] // 2
        low = fftconvolve(np.pad(gt_chw[b], p, mode="edge"), k[::-1, ::-1], mode="valid")
        out.append(low[2::4, 2::4])
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensor", default="qb"); ap.add_argument("--splits", nargs="+", default=["train", "valid"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    s, D = a.sensor, a.sensor.upper()
    kernel = genmtf_matlab(GNYQ_TABLE.get(D) or [0.3] * 4, 4, 41)
    wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
    for split in a.splits:
        src = os.path.join(ROOT, "data", "PanCollection", D, f"{split}_{s}.h5")
        dst = os.path.join(ROOT, "data", "PanCollection", D, f"{split}_{s}_msfix.h5")
        if os.path.exists(dst) and not a.force:
            print(f"[{split}] {os.path.relpath(dst, ROOT)} 있음 — 건너뜀"); continue
        with h5py.File(src) as f:
            n = f["gt"].shape[0]; shapes = {k: f[k].shape for k in ("gt", "ms", "lms", "pan")}
            print(f"[{split}] {n} patches {shapes}")
            with h5py.File(dst + ".tmp", "w") as g:
                for k in ("gt", "pan"):
                    g.create_dataset(k, data=f[k][:], dtype=f[k].dtype)       # 그대로 복사
                ms_new = g.create_dataset("ms", shapes["ms"], dtype=f["ms"].dtype)
                lms_new = g.create_dataset("lms", shapes["lms"], dtype=f["lms"].dtype)
                changed = 0; maxd = 0.0
                for i in range(n):
                    gt = np.asarray(f["gt"][i], dtype=np.float64)
                    ms = mtf_down(gt, kernel)
                    d = np.abs(ms - np.asarray(f["ms"][i], dtype=np.float64)).mean(); changed += d > 1e-6; maxd = max(maxd, d)
                    ms_new[i] = ms
                    lms_new[i] = wald.interp23tap(ms.transpose(1, 2, 0), 4).transpose(2, 0, 1)
                    if (i + 1) % 2000 == 0 or i + 1 == n:
                        print(f"  {i + 1}/{n}  ms 가 바뀐 패치 {changed}", flush=True)
                g.attrs["repair"] = "ms = genMTF(QB) replicate filter(gt)[2::4,2::4]; lms = interp23tap(ms); KNOWN_ISSUES F-3"
                g.attrs["source"] = os.path.basename(src); g.attrs["built_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        os.replace(dst + ".tmp", dst)
        # lpan 은 pan 에서 만든 것이라 그대로 — feeder 가 <이름>_pan.h5 를 찾으므로 링크
        lp = dst.replace(".h5", "_pan.h5")
        if not os.path.exists(lp):
            os.symlink(os.path.basename(src).replace(".h5", "_pan.h5"), lp)
        print(f"[{split}] 저장 {os.path.relpath(dst, ROOT)}  (ms 변경 {changed}/{n} = {100 * changed / n:.0f}%, 최대 MAD {maxd:.2f} DN)")
        # 검증: 전 패치가 (2,2) 위상으로 MAD 0 인가
        with h5py.File(dst) as g:
            idx = np.linspace(0, n - 1, 50).astype(int); bad = 0
            for i in idx:
                gt = np.asarray(g["gt"][i], dtype=np.float64); ms = np.asarray(g["ms"][i], dtype=np.float64)
                bad += np.abs(mtf_down(gt, kernel) - ms).mean() > 1e-9
            print(f"[{split}] 검증: 표본 50 중 (2,2) 위상 불일치 {bad}")


if __name__ == "__main__":
    main()
