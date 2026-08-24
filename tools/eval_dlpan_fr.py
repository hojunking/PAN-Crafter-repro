"""full-resolution 무참조 지표 D_lambda / D_s / HQNR.

MATLAB 이 없어서 ../CANConv/tools/eval_fr.py (DLPan Demo_Full_Resolution.m 포팅,
CANNet 논문 대비 HQNR 0.2% 이내 검증됨) 의 함수를 재사용한다.

  python tools/eval_dlpan_fr.py --preset wv3 \
      --mat baseline_broken=work_dir/wv3_baseline/results/full_best_reduced.mat \
      --mat baseline_repaired=work_dir/wv3_baseline/results/full_frrepair.mat
"""
import os, sys, argparse
import numpy as np, h5py
from scipy.io import loadmat

CANCONV = os.environ.get("PANCRAFTER_CANCONV", "/home/knuvi/Desktop/song/CANConv")
DLPAN = os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")
sys.path.insert(0, CANCONV)
from tools.eval_fr import load_dlpan, d_lambda_k, d_s  # noqa: E402

SCALE = {"wv3": 2047.0, "qb": 2047.0, "gf2": 1023.0, "wv2": 2047.0}
H5 = {s: f"data/PanCollection/{s.upper()}/full_examples_h5/test_{s}_OrigScale_multiExm1.h5"
      for s in SCALE}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", required=True, choices=list(SCALE))
    ap.add_argument("--mat", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--ratio", type=int, default=4)
    ap.add_argument("--indices", default=None, help="예: '12-19'. 기본 전체 20장")
    ap.add_argument("--baseline", action="store_true", help="lms(EXP) 기준선 포함")
    a = ap.parse_args()

    bit = int(round(np.log2(SCALE[a.preset] + 1)))
    wald = load_dlpan(DLPAN)
    with h5py.File(H5[a.preset]) as f:
        lms = np.asarray(f["lms"], dtype=np.float64).transpose(0, 2, 3, 1)
        pan = np.asarray(f["pan"], dtype=np.float64)[:, 0]

    sel = list(range(len(lms)))
    if a.indices:
        lo, hi = a.indices.split("-"); sel = list(range(int(lo), int(hi) + 1))

    targets = []
    if a.baseline:
        targets.append(("lms (EXP, 모델 무관)", lms))
    for spec in a.mat:
        name, path = spec.split("=", 1)
        sr = loadmat(path)["sr"].astype(np.float64)
        if sr.shape[1] in (4, 8):
            sr = sr.transpose(0, 2, 3, 1)
        targets.append((name, sr))

    print(f"preset={a.preset}  N={len(sel)}장  block={a.block_size}  bit depth L={bit}")
    print(f"MTF/Q2n: DLPan-Toolbox 공식 구현 ({DLPAN})\n")
    hdr = f"{'':28s} {'D_lambda↓':>16} {'D_s↓':>16} {'HQNR↑':>16}"
    print(hdr); print("-" * len(hdr))
    for name, data in targets:
        rows = []
        for i in sel:
            fused = np.clip(data[i], 0.0, float(2**bit))
            dl = d_lambda_k(fused, lms[i], a.preset, a.ratio, a.block_size, wald)
            ds = d_s(fused, lms[i], pan[i], a.ratio, a.block_size, wald)
            rows.append((dl, ds, (1 - dl) * (1 - ds)))
            print(f"  {name} {len(rows)}/{len(sel)}", end="\r", file=sys.stderr)
        r = np.array(rows)
        print(" " * 60, end="\r", file=sys.stderr)
        print(f"{name:28s} " + " ".join(f"{r[:,j].mean():>9.4f}±{r[:,j].std():<5.3f}" for j in range(3)))
    print("\n* ../CANConv/tools/eval_fr.py 재사용. 논문 Table 은 20장 중 12-19 부분집합과 일치한다"
          "\n  (CANConv RUNBOOK 8절). 전체 20장 값은 실험 간 비교용으로만 쓸 것.")


if __name__ == "__main__":
    main()
