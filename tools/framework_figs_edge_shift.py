"""framework 그림용 — PAN 의 이동을 **edge 오버레이**로 보이게 한다.

실제 이동은 sub-pixel ~2 px 이라 영상으로는 안 보인다. 그래서 PAN 의 edge 만 뽑아
원본과 이동본을 두 색으로 겹친다 — **겹치면 흰색, 어긋나면 색이 갈라진다.**

  원본 PAN 의 edge   -> cyan  (0,1,1)
  이동한 PAN 의 edge -> red   (1,0,0)
  완전히 겹침        -> white

`framework_figs_export.py` 와 같은 규약(같은 npz·장면·128² 크롭·×4 nearest·512×512).
edge 는 128² 네이티브에서 계산한 뒤 ×4 nearest 로 키운다 — 블록이 곧 1 PAN 픽셀이라
이동량을 눈으로 셀 수 있다.

내는 것(묶음마다):
  EDGE_pan.png              PAN edge 만 (흑백)
  EDGE_shift_eps.png        P vs W(P, eps)   eps = (+1.6,-1.2), |eps| = 2.0 px   <- **실제값, 과장 없음**
  EDGE_shift_c0.png         P vs W(P, c0)    c0 = aligner 실측                    <- **실제값** (거의 흰색으로 보인다)
  EDGE_shift_c0_x<k>.png    P vs W(P, k*c0)  **과장 k 배** — 개념 설명용, 파일명에 배율 명시
  *_zoom.png                위 셋의 32² 부분 확대 (콜아웃용)

주의: `_x<k>` 는 **실제 영상이 그렇게 보인다는 뜻이 아니다.** 도식에 쓸 때 배율을 반드시 적을 것
(README 의 "과장하지 말 것" 과 같은 취지 — 여기서는 파일명·캡션으로 과장을 드러낸다).
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/knuvi/Desktop/song/PAN-Crafter"
sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")

from pa.warp import warp_pan                                # noqa: E402
from pa.losses import scharr                                # noqa: E402

OUT = os.path.join(ROOT, "results_log", "paper_figs", "framework_figs")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t0_rr.npz")
IDX = os.path.join(OUT, "index.csv")
R, N, UP = 2047.0, 128, 4
EPS = (1.6, -1.2)
ZOOM = 32                       # 콜아웃 확대 창 (네이티브 px)
THR = 0.18                      # 이 아래 gradient 는 버린다 (잔텍스처 억제)


def save(path, img, up=UP):
    a = np.repeat(np.repeat(img, up, 0), up, 1)
    plt.imsave(path, np.clip(a, 0, 1))


def edge_of(p_full, sl):
    """[256,256] DN -> 크롭의 정규화 edge [0,1]. 전체에서 Scharr 후 크롭(경계 인공물 회피)."""
    t = torch.from_numpy(p_full)[None, None].float()
    gx, gy = scharr(t)
    e = torch.sqrt(gx[0, 0] ** 2 + gy[0, 0] ** 2).numpy()[sl]
    e = e / (np.percentile(e, 99) + 1e-9)
    e = np.clip(e, 0, 1)
    return np.clip((e - THR) / (1 - THR), 0, 1)          # soft threshold


def overlay(e_ref, e_mov):
    """cyan = 원본, red = 이동본, 흰색 = 겹침."""
    rgb = np.zeros(e_ref.shape + (3,))
    rgb[..., 0] = e_mov
    rgb[..., 1] = e_ref
    rgb[..., 2] = e_ref
    return rgb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exaggerate", type=int, default=8, help="c0 과장 배율 (기본 8)")
    ap.add_argument("--only", default=None, help="한 묶음만 (예: 3_city)")
    a = ap.parse_args()

    if not os.path.exists(SRC):
        sys.exit(f"{SRC} 없음 — python tools/framework_figs_run_t0.py 를 먼저 돌린다")
    z = np.load(SRC)
    idx = {r["sample"]: r for r in csv.DictReader(open(IDX))}

    for name, r in idx.items():
        if a.only and name != a.only:
            continue
        s, y0, x0 = int(r["scene"]), int(r["crop_row"]), int(r["crop_col"])
        c0 = np.array([float(r["c0_dy"]), float(r["c0_dx"])], dtype=np.float32)
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        sl = (slice(y0, y0 + N), slice(x0, x0 + N))

        pan = z["pan"][s, 0]                                  # [256,256] DN
        P = torch.from_numpy(pan)[None, None].float()
        warp = lambda v: warp_pan(P, torch.tensor([v], dtype=torch.float32))[0, 0].numpy()

        e_ref = edge_of(pan, sl)
        cases = [("eps", np.array(EPS, np.float32), "real"),
                 ("c0", c0, "real"),
                 (f"c0_x{a.exaggerate}", c0 * a.exaggerate, f"EXAGGERATED x{a.exaggerate}")]

        save(os.path.join(d, "EDGE_pan.png"), np.repeat(e_ref[..., None], 3, 2))
        zs = (slice(N // 2 - ZOOM // 2, N // 2 + ZOOM // 2),) * 2
        for tag, vec, kind in cases:
            e_mov = edge_of(warp(tuple(vec)), sl)
            ov = overlay(e_ref, e_mov)
            save(os.path.join(d, f"EDGE_shift_{tag}.png"), ov)
            save(os.path.join(d, f"EDGE_shift_{tag}_zoom.png"), ov[zs], up=UP * (N // ZOOM))
            print(f"{name:12} {tag:8} ({vec[0]:+.3f},{vec[1]:+.3f})  |.|={np.linalg.norm(vec):.3f} px  [{kind}]")
        print()

    print("cyan = 원본 PAN edge · red = 이동한 PAN edge · white = 겹침")
    print(f"_x{a.exaggerate} 는 개념 설명용 과장이다 — 도식 캡션에 배율을 반드시 적을 것")
    print("->", OUT)


if __name__ == "__main__":
    main()
