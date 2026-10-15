"""PANPV WV2 5개 씬 정성 비교 그림 (teaser 형식).

왼쪽 열: 확대 영역의 LRMS(×4 bicubic) · PAN. 오른쪽: 방법별 원해상도 결과 패널 —
본 영역에 빨간 확대 상자, 반대쪽 모서리에 2배 확대 inset, 점선 연결.

영역 선택은 방법 출력을 보지 않는다:
  1) 감사(audit) 타일 중 3×3 로 이어진 묶음 가운데 PAN Sobel 크기 평균이 가장 큰 것
  2) 그 안에서 사분면 하나에 들어가는 1/4 크기 창 중 PAN Sobel 평균이 가장 큰 것
표시 stretch 는 영역의 원 MS 로 밴드별 1–99 분위를 정해 모든 방법·LRMS 에 같게 쓴다.
감사 타일은 씬 단독 학습·미세조정에 쓰이지 않은 공간 holdout 이다.

사용:
  python tools/panpv_qualitative.py --out results_log/<폴더>
"""
import argparse
import json
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "work_dir/PANPV_S1_WV2_20261004_R1/s1"
SCENES_DIR = ROOT / "data/private/PANPV_S1_WV2_20261003_v3/extracted/practical_5scene_v3_20261002/scenes"

# 데이터 개정 문서의 V01–V05 순서 (도시 → 도시·혼합 → 농촌 → 농촌·혼합 → 박무)
SCENES = [
    ("200014103764_01", "Urban"),
    ("200014103758_01", "Urban / lake / hills"),
    ("200014103802_01", "Rural plain"),
    ("200014103714_01", "Rural / new town"),
    ("200014103525_01", "Haze / thin cloud"),
]
METHODS = [
    ("BROVEY8", "Brovey", "8-band, equal weights"),
    ("ZS_WV3", "Zero-shot", "WV3 model as is"),
    ("SC_SCENE_EXACT20K", "Scene-trained", "from scratch, 20K"),
    ("FT_WV3_SCENE_EXACT20K", "Scene fine-tuned", "from WV3 model, 20K"),
]
RGB = (4, 2, 1)  # Coastal,B,G,Y,R,RE,NIR1,NIR2 → R,G,B
TILE_MS, TILE_PAN, NT = 64, 256, 3
DN_MAX = 2047.0


def sobel_mag(img):
    img = img.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def pick_region(scene_h5, rc):
    """3×3 로 이어진 감사 타일 묶음 중 PAN 에지 에너지 최대. 반환: 좌상단 MS (r, c)."""
    pan = scene_h5["native/pan"]
    energy = {}
    for r, c in rc:
        t = pan[0, 0, 4 * r:4 * r + TILE_PAN, 4 * c:4 * c + TILE_PAN]
        energy[(int(r), int(c))] = float(sobel_mag(t).mean())
    best, best_e = None, -1.0
    for (r, c) in energy:
        keys = [(r + TILE_MS * i, c + TILE_MS * j) for i in range(NT) for j in range(NT)]
        if all(k in energy for k in keys):
            e = float(np.mean([energy[k] for k in keys]))
            if e > best_e:
                best, best_e = (r, c), e
    return best, best_e


def pick_box(pan_region, size):
    """사분면 하나에 완전히 들어가는 size 창 중 PAN 에지 에너지 최대. 반환: (y, x, 사분면)."""
    g = sobel_mag(pan_region)
    H = pan_region.shape[0]
    half = H // 2
    best, best_e = None, -1.0
    for qy in (0, 1):
        for qx in (0, 1):
            for y in range(qy * half, qy * half + half - size + 1, 16):
                for x in range(qx * half, qx * half + half - size + 1, 16):
                    e = float(g[y:y + size, x:x + size].mean())
                    if e > best_e:
                        best, best_e = (y, x, (qy, qx)), e
    return best


def stitch(pred_h5, index, r0, c0, nt=NT):
    out = np.zeros((8, nt * TILE_PAN, nt * TILE_PAN), np.float32)
    ds = pred_h5["fr/prediction"]
    for i in range(nt):
        for j in range(nt):
            k = index[(r0 + TILE_MS * i, c0 + TILE_MS * j)]
            out[:, i * TILE_PAN:(i + 1) * TILE_PAN, j * TILE_PAN:(j + 1) * TILE_PAN] = ds[k]
    return out


def stretch_params(ms_region):
    lo = np.array([np.percentile(ms_region[b], 1) for b in RGB], np.float32)
    hi = np.array([np.percentile(ms_region[b], 99) for b in RGB], np.float32)
    return lo, hi


def to_rgb(cube, lo, hi):
    x = np.clip(cube, 0.0, DN_MAX)[list(RGB)].astype(np.float32)
    x = (x - lo[:, None, None]) / np.maximum(hi - lo, 1e-6)[:, None, None]
    return np.clip(x, 0, 1).transpose(1, 2, 0)


def pan_gray(p):
    lo, hi = np.percentile(p, 1), np.percentile(p, 99)
    return np.clip((p - lo) / max(hi - lo, 1e-6), 0, 1)


def bicubic_up(ms, ratio=4):
    t = torch.from_numpy(ms.astype(np.float32))[None]
    return F.interpolate(t, scale_factor=ratio, mode="bicubic", align_corners=False)[0].numpy()


def compose(img, box, quad, size):
    """img(H,W,3) 의 box 를 2배 확대해 box 반대 사분면 모서리에 붙인다."""
    y, x = box
    H, W = img.shape[:2]
    crop = img[y:y + size, x:x + size]
    big = np.kron(crop, np.ones((2, 2, 1), crop.dtype))  # nearest ×2, 보간 없음
    bs = 2 * size
    iy = 0 if quad[0] == 1 else H - bs
    ix = 0 if quad[1] == 1 else W - bs
    out = img.copy()
    out[iy:iy + bs, ix:ix + bs] = big
    return out, (iy, ix, bs)


def draw_panel(ax, img, box, size, inset, red="#d62728", lw=1.6, interp="nearest"):
    y, x = box
    iy, ix, bs = inset
    # 단일 그림은 1 영상 화소 = 1 출력 화소(리샘플링 없음). 모음 그림만 축소용 antialiased.
    ax.imshow(img, interpolation=interp)
    ax.add_patch(Rectangle((x - 0.5, y - 0.5), size, size, fill=False, ec=red, lw=lw))
    ax.add_patch(Rectangle((ix - 0.5, iy - 0.5), bs, bs, fill=False, ec=red, lw=lw))
    # 변위 방향에 따라 교차하지 않는 두 모서리를 잇는다
    dy, dx = np.sign(y - iy), np.sign(x - ix)
    if dy == dx:  # 좌상↔우하 배치: 우상·좌하 모서리
        pairs = [((x + size, y), (ix + bs, iy)), ((x, y + size), (ix, iy + bs))]
    else:         # 우상↔좌하 배치: 좌상·우하 모서리
        pairs = [((x, y), (ix, iy)), ((x + size, y + size), (ix + bs, iy + bs))]
    for (x0, y0), (x1, y1) in pairs:
        ax.plot([x0 - 0.5, x1 - 0.5], [y0 - 0.5, y1 - 0.5], ls=(0, (4, 3)), c=red, lw=lw * 0.8)
    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(img.shape[0] - 0.5, -0.5)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def side_panel(ax, img, label, red="#d62728"):
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1, interpolation="antialiased")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(red), s.set_linewidth(2.0)
    ax.text(0.04, 0.95, label, transform=ax.transAxes, ha="left", va="top", color="white",
            fontsize=12, bbox=dict(facecolor="black", alpha=0.65, edgecolor="none", pad=2.5))


_TILE_FR = {}


def region_fr(sid, key, r0, c0, nt):
    """영역 안 감사 타일들의 원 해상도 지표(HQNR · D_λ · D_s) 평균 — 씬 HQNR 과 같은 타일 평균 집계."""
    if (sid, key) not in _TILE_FR:
        rep = json.load(open(CAMPAIGN / f"evaluations/{sid}_{key}/report.json"))
        by_raw = {t["raw_index"]: t["metrics"] for t in rep["fr"]["per_tile"]}
        with h5py.File(CAMPAIGN / f"evaluations/{sid}_{key}/predictions.h5", "r") as ph:
            rc, raw = ph["canonical_ms_row_col"][:], ph["raw_index"][:]
        _TILE_FR[sid, key] = {(int(r), int(c)): by_raw[int(x)] for (r, c), x in zip(rc, raw)}
    tiles = [_TILE_FR[sid, key][(r0 + TILE_MS * i, c0 + TILE_MS * j)] for i in range(nt) for j in range(nt)]
    return {m: float(np.mean([t[m] for t in tiles])) for m in ("hqnr", "d_lambda", "d_s")}


def load_scene(sid, origin=None, nt=NT):
    """origin 이 없으면 pick_region 규칙으로 영역을 고른다. origin = 좌상단 MS (r, c)."""
    data = {}
    with h5py.File(SCENES_DIR / f"{sid}.h5", "r") as sh:
        if origin is None:
            with h5py.File(CAMPAIGN / f"evaluations/{sid}_ZS_WV3/predictions.h5", "r") as zh:
                rc = zh["canonical_ms_row_col"][:]
            (r0, c0), energy = pick_region(sh, rc)
        else:
            (r0, c0), energy = origin, None
        R = nt * TILE_PAN
        data["pan"] = sh["native/pan"][0, 0, 4 * r0:4 * r0 + R, 4 * c0:4 * c0 + R].astype(np.float32)
        data["ms"] = sh["native/ms"][0, :, r0:r0 + R // 4, c0:c0 + R // 4].astype(np.float32)
    preds = {}
    for key, _, _ in METHODS:
        with h5py.File(CAMPAIGN / f"evaluations/{sid}_{key}/predictions.h5", "r") as ph:
            rcm = ph["canonical_ms_row_col"][:]
            index = {(int(r), int(c)): k for k, (r, c) in enumerate(rcm)}
            preds[key] = stitch(ph, index, r0, c0, nt)
    data["preds"] = preds
    data["fr"] = {key: region_fr(sid, key, r0, c0, nt) for key, _, _ in METHODS}
    data["region"] = dict(scene_id=sid, ms_row=int(r0), ms_col=int(c0), size_pan=R,
                          pan_sobel_mean=energy, region_fr=data["fr"])
    return data


def sanity(data):
    """출력을 4×4 평균으로 내려 원 MS 와의 밴드 평균 상관 — 영역·좌표 대응 확인용."""
    ms = data["ms"]
    out = {}
    for key, p in data["preds"].items():
        d = p.reshape(8, ms.shape[1], 4, ms.shape[2], 4).mean(axis=(2, 4))
        out[key] = float(np.corrcoef(d.ravel(), ms.ravel())[0, 1])
    return out


def title(ax, name, sub, fs=15):
    """'Name (italic sub)' 한 줄 제목 — teaser 의 'Method (Venue)' 형식."""
    parts = [TextArea(f"{name} (", textprops=dict(fontsize=fs)),
             TextArea(sub, textprops=dict(fontsize=fs, style="italic")),
             TextArea(")", textprops=dict(fontsize=fs))]
    box = HPacker(children=parts, align="baseline", pad=0, sep=0)
    ax.add_artist(AnchoredOffsetbox(loc="lower center", child=box, pad=0, frameon=False,
                                    bbox_to_anchor=(0.5, 1.015), bbox_transform=ax.transAxes,
                                    borderpad=0))


def metric_label(ax, fr, best, dist):
    """패널 아래 원 해상도 지표 한 줄. dist 면 HQNR 옆에 D_λ · D_s(분광 · 공간 왜곡). 행 최고값은 굵게."""
    fs = 12 if dist else 14

    def part(txt, bold=False):
        return TextArea(txt, textprops=dict(fontsize=fs, fontweight="bold" if bold else "normal"))
    parts = [part("HQNR "), part(f"{fr['hqnr']:.4f}", fr["hqnr"] == best["hqnr"])]
    if dist:
        parts += [part("   "), part(r"$D_\lambda$ "), part(f"{fr['d_lambda']:.4f}", fr["d_lambda"] == best["d_lambda"]),
                  part("   "), part(r"$D_s$ "), part(f"{fr['d_s']:.4f}", fr["d_s"] == best["d_s"])]
    box = HPacker(children=parts, align="baseline", pad=0, sep=0)
    ax.add_artist(AnchoredOffsetbox(loc="upper center", child=box, pad=0, frameon=False,
                                    bbox_to_anchor=(0.5, -0.012), bbox_transform=ax.transAxes,
                                    borderpad=0))


def render_rows(rows, path, dpi=None, interp="nearest", dist=False):
    """dpi 가 없으면 결과 패널이 영상 화소와 1:1 이 되도록 dpi = 영상 크기 / 패널 크기.
    dist 면 HQNR 옆에 D_λ · D_s 를 같은 줄로 더한다."""
    S, L, g, T, m = 3.6, 1.75, 0.08, 0.42, 0.05
    Mh = 0.34  # 패널 아래 지표 줄 높이 (inch)
    if dpi is None:
        dpi = rows[0]["panels"][METHODS[0][0]].shape[1] / S
    nrow = len(rows)
    W = m * 2 + L + g + 4 * S + 3 * g
    H = m * 2 + T + nrow * (S + Mh) + (nrow - 1) * g
    fig = plt.figure(figsize=(W, H))
    lab_h = (S - g) / 2
    for ri, row in enumerate(rows):
        top = H - m - T - ri * (S + Mh + g)  # 행 윗변 (inch)
        bottom = top - S
        frs = [row["fr"][k] for k, _, _ in METHODS]
        best = dict(hqnr=max(f["hqnr"] for f in frs), d_lambda=min(f["d_lambda"] for f in frs),
                    d_s=min(f["d_s"] for f in frs))
        side = [("LRMS", row["lrms"]), ("PAN", row["pan"])]
        for si, (lab, im) in enumerate(side):
            y0 = top - (si + 1) * lab_h - si * g
            ax = fig.add_axes([m / W, y0 / H, L / W, lab_h / H])
            side_panel(ax, im, lab)
        for mi, (key, name, sub) in enumerate(METHODS):
            x0 = m + L + g + mi * (S + g)
            ax = fig.add_axes([x0 / W, bottom / H, S / W, S / H])
            draw_panel(ax, row["panels"][key], row["box"], row["size"], row["inset"], interp=interp)
            metric_label(ax, row["fr"][key], best, dist)
            if ri == 0:
                title(ax, name, sub)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def build_row(data, box=None):
    ms, pan = data["ms"], data["pan"]
    R = pan.shape[0]
    size = R // 4
    if box is None:
        by, bx, quad = pick_box(pan, size)
    else:  # 지정 상자 — 한 사분면 안에 있어야 inset 이 상자를 가리지 않는다
        by, bx = box
        quad = (int(by >= R // 2), int(bx >= R // 2))
        assert by // (R // 2) == (by + size - 1) // (R // 2) and bx // (R // 2) == (bx + size - 1) // (R // 2)
    lo, hi = stretch_params(ms)
    panels, inset = {}, None
    for key, _, _ in METHODS:
        rgb = to_rgb(data["preds"][key], lo, hi)
        panels[key], inset = compose(rgb, (by, bx), quad, size)
    lrms = to_rgb(bicubic_up(ms), lo, hi)[by:by + size, bx:bx + size]
    pan_crop = pan_gray(pan)[by:by + size, bx:bx + size]
    data["region"].update(box_pan_yx=[int(by), int(bx)], box_size_pan=size,
                          stretch_lo=lo.tolist(), stretch_hi=hi.tolist())
    return dict(panels=panels, box=(by, bx), size=size, inset=inset, lrms=lrms, pan=pan_crop,
                fr=data["fr"])


# 차량 비교 영역 — 감사 타일마다 vehicle_points 개수로 순위를 매긴 상위 30개 타일(rank_vehicle_tiles)의
# PAN 만 보고, 주차장 · 차량이 모인 도로가 뚜렷한 곳에 2×2 타일 영역과 확대 상자를 정했다(방법별 출력은
# 보지 않음). 함평 · 무안은 주차장이 없어 고속도로. (씬, 영역 좌상단 MS (r, c), 상자 좌상단 (y, x) PAN 화소)
VEHICLE_REGIONS = [
    ("200014103764_01", (2816, 7296), (304, 32)),    # 대형 노상 주차장
    ("200014103758_01", (2368, 4480), (352, 368)),   # 건물 앞 주차장 + 왼쪽 간선도로
    ("200014103802_01", (4864, 3328), (256, 330)),   # 고속도로 차량
    ("200014103714_01", (384, 1280), (300, 80)),     # 공터 주차장 · 화물차
    ("200014103525_01", (2240, 4736), (360, 128)),   # 아파트 단지 주차장
]
# 추가 묶음 — 같은 방법(PAN 만 보고)으로 고른 두 번째 영역. 첫 묶음과 겹치지 않음
VEHICLE_REGIONS_2 = [
    ("200014103764_01", (1216, 2624), (260, 44)),    # 간선도로 주행 차량
    ("200014103758_01", (2368, 4352), (360, 40)),    # 버스 · 승용차 주차장
    ("200014103802_01", (1600, 7808), (384, 256)),   # 마을 도로 버스 · 승용차
    ("200014103714_01", (5888, 4864), (384, 260)),   # 버스 · 화물차 차고지
    ("200014103525_01", (2688, 4608), (360, 30)),    # 고속도로 주행 차량
]
VEH_NT = 2  # 2×2 타일 = 512×512 PAN 화소


def vehicle_points(pan, ms):
    """PAN 에서 차량 크기의 작은 밝은/어두운 물체 중심을 1 로 표시한 지도.

    식생(MS NDVI > 0.2) 제외, 11×11 중앙값 배경이 평탄하고(국소 표준편차 < 중앙 밝기의 12%)
    어둡지 않은(> 중앙 밝기의 40%) 곳에서 배경과 중앙 밝기의 25%(최소 20 DN) 이상 차이 나는
    면적 6–50 화소 · 긴 변 12 화소 이하 성분.
    """
    from scipy.ndimage import median_filter
    H, W = pan.shape
    r, n = ms[4].astype(np.float32), ms[6].astype(np.float32)
    veg = cv2.resize(((n - r) / (n + r + 1e-6) > 0.2).astype(np.uint8), (W, H),
                     interpolation=cv2.INTER_NEAREST) > 0
    bg = median_filter(pan, size=11)
    mu = cv2.GaussianBlur(bg, (0, 0), 4)
    sd = np.sqrt(np.maximum(cv2.GaussianBlur(bg * bg, (0, 0), 4) - mu * mu, 0))
    lev = max(float(np.percentile(pan, 50)), 1.0)
    paved = (~veg) & (sd < 0.12 * lev) & (bg > 0.4 * lev)
    pts = np.zeros((H, W), np.float32)
    for sgn in (1, -1):
        m = ((sgn * (pan - bg)) > max(0.25 * lev, 20)).astype(np.uint8)
        n_, _, st, cen = cv2.connectedComponentsWithStats(m, 8)
        for k in range(1, n_):
            cx, cy = int(cen[k][0]), int(cen[k][1])
            if 6 <= st[k, 4] <= 50 and max(st[k, 2], st[k, 3]) <= 12 and paved[cy, cx]:
                pts[cy, cx] = 1.0
    return pts


def vehicle_closeups(out, regions, kinds, name, crop=64, up=5):
    """차량 영역 확대 상자의 가운데 crop×crop 를 up 배(화소 복제)로 — 씬 × (LRMS, PAN, 4 방법)."""
    cats = dict(SCENES)
    cols = [("LRMS", "x4 bicubic"), ("PAN", "input")] + [(m[1], None) for m in METHODS]
    n = len(regions)
    S, g, T, L = 2.0, 0.05, 0.35, 0.48
    W, H = L + len(cols) * S + (len(cols) - 1) * g + 0.05, T + n * S + (n - 1) * g + 0.05
    fig = plt.figure(figsize=(W, H))
    for i, (sid, origin, (by, bx)) in enumerate(regions):
        d = load_scene(sid, origin, VEH_NT)
        lo, hi = stretch_params(d["ms"])
        q = (by + (128 - crop) // 2, bx + (128 - crop) // 2)
        sl = (slice(q[0], q[0] + crop), slice(q[1], q[1] + crop))
        ims = [to_rgb(bicubic_up(d["ms"]), lo, hi)[sl], pan_gray(d["pan"])[sl]]
        ims += [to_rgb(d["preds"][k], lo, hi)[sl] for k, _, _ in METHODS]
        top = H - T - i * (S + g)
        for j, im in enumerate(ims):
            ax = fig.add_axes([(L + j * (S + g)) / W, (top - S) / H, S / W, S / H])
            im = np.kron(im, np.ones((up, up) + ((1,) if im.ndim == 3 else ()), im.dtype))
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                head, sub = cols[j]
                if sub:
                    title(ax, head, sub, fs=12)
                else:
                    ax.set_title(head, fontsize=12, pad=5)
            if j == 0:
                ax.set_ylabel(f"{cats[sid].replace(' / ', '/')}\n({kinds[i]})", fontsize=10)
    fig.savefig(out / name, dpi=crop * up / S, facecolor="white")  # 1 확대 화소 = 1 출력 화소
    plt.close(fig)


# 축소 해상도(학습 조건) vs 원 해상도 — 같은 감사 타일(같은 땅 넓이)을 두 해상도로
RR_FR_TILES = [("200014103764_01", (2880, 7296)), ("200014103525_01", (2752, 4608))]


def rr_vs_fr(out, S=2.0):
    """위 줄: 축소 해상도 정답(원 MS) · 방법별 축소 해상도 출력(64×64, 4배 화소 복제).
    아래 줄: 원 PAN · 방법별 원 해상도 출력(256×256, 1:1). 두 줄은 같은 땅 넓이."""
    cats = dict(SCENES)
    keys = [m[0] for m in METHODS]
    n = len(RR_FR_TILES)
    Lw, T, g = 0.5, 0.35, 0.05
    W = Lw + 5 * S + 4 * g + 0.05
    H = T + n * (2 * S + g) + (n - 1) * 0.25 + 0.05
    fig = plt.figure(figsize=(W, H))
    for i, (sid, (r, c)) in enumerate(RR_FR_TILES):
        with h5py.File(SCENES_DIR / f"{sid}.h5", "r") as sh:
            gt = sh["rr/target"][0, :, r:r + 64, c:c + 64].astype(np.float32)
            pan = sh["native/pan"][0, 0, 4 * r:4 * r + 256, 4 * c:4 * c + 256].astype(np.float32)
        lo, hi = stretch_params(gt)
        rr, fr = [to_rgb(gt, lo, hi)], [pan_gray(pan)]
        for key in keys:
            with h5py.File(CAMPAIGN / f"evaluations/{sid}_{key}/predictions.h5", "r") as ph:
                k = {(int(a), int(b)): j for j, (a, b) in enumerate(ph["canonical_ms_row_col"][:])}[(r, c)]
                rr.append(to_rgb(ph["rr/prediction"][k], lo, hi))
                fr.append(to_rgb(ph["fr/prediction"][k], lo, hi))
        heads = [("GT (RR target)", "PAN (FR input)")] + [(m[1], m[1]) for m in METHODS]
        top0 = H - T - i * (2 * S + g + 0.25)
        for row, (ims, lab) in enumerate(((rr, "Reduced res.\n(training scale)"), (fr, "Full res.\n(native)"))):
            top = top0 - row * (S + g)
            for j, im in enumerate(ims):
                ax = fig.add_axes([(Lw + j * (S + g)) / W, (top - S) / H, S / W, S / H])
                if im.shape[0] == 64:
                    im = np.kron(im, np.ones((4, 4, 1), im.dtype))
                ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([]), ax.set_yticks([])
                if row == 0 and i == 0:
                    ax.set_title(heads[j][0], fontsize=12, pad=4)
                if row == 1 and j == 0:
                    ax.text(0.03, 0.97, "PAN", transform=ax.transAxes, va="top", color="white", fontsize=10,
                            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=2))
                if j == 0:
                    ax.set_ylabel(f"{cats[sid].split(' /')[0]}\n{lab}", fontsize=9.5)
    fig.savefig(out / "rr_vs_fr.png", dpi=256 / S, facecolor="white")
    plt.close(fig)


def adapted_best_regions(min_margin=0.0031, nt=VEH_NT):
    """씬마다 2×2 감사 타일 영역 중 씬 적응(단독 학습 · 미세조정) HQNR 최고값이 제로샷 · Brovey 보다
    min_margin 이상 높은 곳 가운데 그 HQNR 이 가장 높은 영역. 확대 상자는 PAN 경계 규칙(pick_box)."""
    keys = [m[0] for m in METHODS]
    picks = []
    for sid, _ in SCENES:
        for key in keys:
            region_fr(sid, key, *next(iter(_tiles(sid))), 1)
        have = set(_TILE_FR[sid, keys[0]])
        best = None
        for r, c in sorted(have):
            g = [(r + TILE_MS * i, c + TILE_MS * j) for i in range(nt) for j in range(nt)]
            if not all(t in have for t in g):
                continue
            h = {k: float(np.mean([_TILE_FR[sid, k][t]["hqnr"] for t in g])) for k in keys}
            ad_key = max(("SC_SCENE_EXACT20K", "FT_WV3_SCENE_EXACT20K"), key=lambda k: h[k])
            margin = h[ad_key] - max(h["ZS_WV3"], h["BROVEY8"])
            if margin >= min_margin and (best is None or h[ad_key] > best[2]):
                best = ((r, c), ad_key, h[ad_key], margin)
        with h5py.File(SCENES_DIR / f"{sid}.h5", "r") as sh:
            (r, c) = best[0]
            R_ = nt * TILE_PAN
            pan = sh["native/pan"][0, 0, 4 * r:4 * r + R_, 4 * c:4 * c + R_].astype(np.float32)
        by, bx, _ = pick_box(pan, R_ // 4)
        picks.append(dict(sid=sid, origin=best[0], box=(by, bx), best_adapted=best[1],
                          best_adapted_hqnr=best[2], margin=best[3]))
    return picks


def _tiles(sid):
    with h5py.File(CAMPAIGN / f"evaluations/{sid}_ZS_WV3/predictions.h5", "r") as zh:
        return [tuple(int(v) for v in rc) for rc in zh["canonical_ms_row_col"][:]]


def adapted_best(out, dpi):
    cats = dict(SCENES)
    picks = adapted_best_regions()
    regions = [(p["sid"], p["origin"], p["box"]) for p in picks]
    kinds = ["scene-trained best" if p["best_adapted"].startswith("SC") else "fine-tuned best" for p in picks]
    rows, meta = [], []
    for i, p in enumerate(picks, 1):
        data = load_scene(p["sid"], p["origin"], VEH_NT)
        row = build_row(data, box=p["box"])
        data["region"].update(order=f"V{i:02d}", category=cats[p["sid"]], kind=kinds[i - 1],
                              selection={k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()})
        meta.append(data["region"])
        render_rows([row], out / f"ADP_V{i:02d}_{p['sid']}_adapted_best.png", dist=True)
        rows.append(row)
        print(p["sid"], p["origin"], p["box"], {k: round(v["hqnr"], 4) for k, v in data["fr"].items()}, flush=True)
    render_rows(rows, out / "all_scenes_adapted_best.png", max(dpi // 2, 100), interp="antialiased", dist=True)
    vehicle_closeups(out, regions, kinds, "adapted_best_closeups.png")
    (out / "adapted_best_regions.json").write_text(json.dumps(
        dict(rule="per scene: 2x2 audit tiles where max(scene-trained, fine-tuned) region HQNR exceeds "
                  "max(zero-shot, Brovey) by >= 0.0031, highest such HQNR; zoom box by PAN edge rule",
             regions=meta), indent=1))


def rank_vehicle_tiles(sid, top=30):
    """감사 타일을 vehicle_points 개수로 정렬한 상위 top 개 [(ms_row, ms_col, 개수)] — 영역 후보 목록."""
    with h5py.File(CAMPAIGN / f"evaluations/{sid}_ZS_WV3/predictions.h5", "r") as zh:
        rc = zh["canonical_ms_row_col"][:]
    out = []
    with h5py.File(SCENES_DIR / f"{sid}.h5", "r") as sh:
        for r, c in rc:
            pan = sh["native/pan"][0, 0, 4 * r:4 * r + TILE_PAN, 4 * c:4 * c + TILE_PAN].astype(np.float32)
            n = int(vehicle_points(pan, sh["native/ms"][0, :, r:r + TILE_MS, c:c + TILE_MS]).sum())
            out.append((int(r), int(c), n))
    return sorted(out, key=lambda t: -t[2])[:top]


VEHICLE_SETS = [
    ("", VEHICLE_REGIONS, ["parking lot", "parking lot", "highway", "parking lot", "parking lot"]),
    ("_2", VEHICLE_REGIONS_2, ["road traffic", "bus + car lot", "village road", "bus/truck depot",
                               "highway traffic"]),
]


def vehicles(out, dpi):
    cats = dict(SCENES)
    meta = {}
    for suffix, regions, kinds in VEHICLE_SETS:
        rows, meta[suffix or "_1"] = [], []
        for i, (sid, origin, box) in enumerate(regions, 1):
            data = load_scene(sid, origin, VEH_NT)
            row = build_row(data, box=box)
            data["region"].update(order=f"V{i:02d}", category=cats[sid], kind=kinds[i - 1],
                                  vehicle_candidates=int(vehicle_points(data["pan"], data["ms"]).sum()),
                                  ms_correlation=sanity(data))
            meta[suffix or "_1"].append(data["region"])
            render_rows([row], out / f"VEH{suffix.lstrip('_')}_V{i:02d}_{sid}_vehicles.png")
            rows.append(row)
            print(sid, origin, box, {k: round(v["hqnr"], 4) for k, v in data["fr"].items()}, flush=True)
        render_rows(rows, out / f"all_scenes_vehicles{suffix}.png", max(dpi // 2, 100), interp="antialiased")
        vehicle_closeups(out, regions, kinds, f"vehicle_closeups{suffix}.png")
    (out / "vehicle_regions.json").write_text(json.dumps(
        dict(rule="2x2 audit tiles and zoom box set by eye on PAN only (parking lots / roads with "
                  "vehicles), shortlisted by rank_vehicle_tiles; stretch as main figures; region_fr = "
                  "mean of per-tile FR metrics over the region tiles",
             sets=meta), indent=1))


def _power(img):
    """밴드 평균 강도의 Hann 창 파워 스펙트럼 (fftshift)."""
    img = img - img.mean()
    N = img.shape[0]
    win = np.outer(np.hanning(N), np.hanning(N))
    return np.abs(np.fft.fftshift(np.fft.fft2(img * win))) ** 2


def period4(out):
    """MS 화소 격자(4 PAN 화소) 주기 성분 진단 — 감사 타일 전수.

    ratio = 주기 4 위치(±N/4 축 위 4점) 파워 평균 / 같은 반경 고리(±2 bin) 파워 중앙값.
    격자 무늬가 없으면 1 근처. 같은 위치의 원 PAN 을 대조로 같이 잰다.
    """
    N = TILE_PAN
    c, k = N // 2, N // 4
    yy, xx = np.mgrid[:N, :N] - c
    ring = (np.hypot(yy, xx) > k - 2) & (np.hypot(yy, xx) < k + 2)
    pts = [(c, c + k), (c, c - k), (c + k, c), (c - k, c)]
    # 2화소 주기(체크무늬) — fftshift 뒤 축 위 Nyquist 는 index 0, 비교 고리는 반경 N/2−3 ~ N/2−1
    ring2 = (np.hypot(yy, xx) >= c - 3) & (np.hypot(yy, xx) <= c - 1)
    pts2 = [(c, 0), (0, c)]

    def ratio(P):
        return float(np.mean([P[y, x] for y, x in pts]) / np.median(P[ring]))

    def ratio2(P):
        return float(np.mean([P[y, x] for y, x in pts2]) / np.median(P[ring2]))

    stats, prof = {}, {}
    keys = ["PAN"] + [m[0] for m in METHODS]
    for sid, _ in SCENES:
        with h5py.File(CAMPAIGN / f"evaluations/{sid}_ZS_WV3/predictions.h5", "r") as zh:
            rc = zh["canonical_ms_row_col"][:]
        r_all = {k_: [] for k_ in keys}
        r2_all = {k_: [] for k_ in keys}
        p_sum = {k_: np.zeros(N // 2 - 1) for k_ in keys}
        with h5py.File(SCENES_DIR / f"{sid}.h5", "r") as sh:
            pan = sh["native/pan"]
            for r, cc in rc:
                P = _power(pan[0, 0, 4 * r:4 * r + N, 4 * cc:4 * cc + N].astype(np.float32))
                r_all["PAN"].append(ratio(P))
                r2_all["PAN"].append(ratio2(P))
                p_sum["PAN"] += 0.5 * (P[c, c + 1:] + P[c + 1:, c])
        for key, _, _ in METHODS:
            with h5py.File(CAMPAIGN / f"evaluations/{sid}_{key}/predictions.h5", "r") as ph:
                ds = ph["fr/prediction"]
                for i in range(len(rc)):
                    P = _power(np.clip(ds[i], 0, DN_MAX).mean(0))
                    r_all[key].append(ratio(P))
                    r2_all[key].append(ratio2(P))
                    p_sum[key] += 0.5 * (P[c, c + 1:] + P[c + 1:, c])
        stats[sid] = {k_: dict(median=float(np.median(v)), mean=float(np.mean(v)), tiles=len(v),
                               period2_median=float(np.median(r2_all[k_])))
                      for k_, v in r_all.items()}
        prof[sid] = {k_: (v / len(rc)).tolist() for k_, v in p_sum.items()}
        print(sid, " ".join(f"{k_}:{stats[sid][k_]['median']:.2f}" for k_ in keys))
    # 고주파 과다: 저주파(0.04–0.08) 정규화 축 프로파일의 0.30–0.45 구간 평균, PAN 대비 배수
    fr_ = np.arange(1, N // 2) / N
    lo_, hi_ = (fr_ >= 0.04) & (fr_ <= 0.08), (fr_ >= 0.30) & (fr_ <= 0.45)
    for sid in stats:
        rel = {k_: np.asarray(v)[hi_].mean() / np.asarray(v)[lo_].mean() for k_, v in prof[sid].items()}
        for k_ in rel:
            stats[sid][k_]["hf_030_045_vs_pan"] = float(rel[k_] / rel["PAN"])
    (out / "period4_spectrum.json").write_text(json.dumps(
        dict(definition="power at period-4 points (+-N/4 on axes) / median power on same-radius ring; "
                        "band-mean intensity, Hann window, every audit FR tile (256x256 PAN px). "
                        "period2_median = same ratio at the axis Nyquist (2-px period) against the outermost ring. hf_030_045_vs_pan = axis-profile power in 0.30-0.45 cycles/px after "
                        "normalising by 0.04-0.08 band, divided by the same for PAN",
             stats=stats), indent=1))
    # 축 방향 1D 파워 프로파일 (fy=0 행 · fx=0 열 평균), 씬별
    colors = {"PAN": "0.55", "BROVEY8": "#1f77b4", "ZS_WV3": "#d62728",
              "SC_SCENE_EXACT20K": "#2ca02c", "FT_WV3_SCENE_EXACT20K": "#9467bd"}
    names = {"PAN": "PAN (input)", **{m[0]: m[1] for m in METHODS}}
    f = np.arange(1, N // 2) / N
    fig, axes = plt.subplots(1, len(SCENES), figsize=(3.2 * len(SCENES), 3.0), sharey=False)
    lowf = (f >= 0.04) & (f <= 0.08)
    for ax, (sid, cat) in zip(axes, SCENES):
        # DN 척도(PAN vs MS)를 지우려고 저주파 구간 평균으로 정규화; PAN 은 맨 위에 점선
        # (8밴드 균등 가중 Brovey 의 밴드 평균은 정의상 PAN 과 같아 두 선이 겹친다)
        for k_ in keys[1:] + keys[:1]:
            p_ = np.asarray(prof[sid][k_])
            ax.semilogy(f, p_ / p_[lowf].mean(), c="k" if k_ == "PAN" else colors[k_],
                        lw=1.5, ls=(0, (1.5, 1.5)) if k_ == "PAN" else "-", label=names[k_],
                        zorder=3 if k_ == "PAN" else 2)
        ax.axvline(0.25, c="k", lw=0.8, ls="--")
        ax.set_title(f"{cat}", fontsize=12)
        ax.set_xlabel("spatial frequency (cycles / PAN px)")
        ax.set_xlim(0.05, 0.5)
    axes[0].set_ylabel("relative power (axis profile)")
    axes[-1].legend(fontsize=8.5, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "period4_spectrum.png", dpi=150, facecolor="white")
    plt.close(fig)


def curves(out):
    """적응 학습 곡선: 감사 구역 RR ERGAS(전수) · 고정 FR ROI HQNR(6 타일) vs update."""
    steps = [0, 500, 1000, 2000, 5000, 10000, 20000]
    xs = np.arange(len(steps))
    fig, axes = plt.subplots(2, len(SCENES), figsize=(3.2 * len(SCENES), 5.6))
    for j, (sid, cat) in enumerate(SCENES):
        ref = {}
        for key in ("ZS_WV3", "BROVEY8"):
            r = json.load(open(CAMPAIGN / f"evaluations/{sid}_{key}/report.json"))
            ref[key] = r["rr"]["metrics"]["ergas"]
        for meth, col, lab in (("SC_SCENE", "#2ca02c", "Scene-trained"),
                               ("FT_WV3_SCENE", "#9467bd", "Scene fine-tuned")):
            run = CAMPAIGN / f"runs/PV_S1_{sid}_{meth}_R01_SE261002"
            au = {a["step"]: a["diagnostic"] for a in map(json.loads, open(run / "audit_diagnostics.jsonl"))}
            # 단독 학습의 step 0 은 MS bicubic 과 같은 초기 상태라 그리지 않는다
            ss = [s for s in steps if s in au and not (meth == "SC_SCENE" and s == 0)]
            xi = [steps.index(s) for s in ss]
            axes[0, j].plot(xi, [au[s]["ERGAS"] for s in ss], "-o", c=col, ms=4, label=lab)
            axes[1, j].plot(xi, [au[s]["HQNR"] for s in ss], "-o", c=col, ms=4, label=lab)
        axes[0, j].axhline(ref["ZS_WV3"], c="#d62728", ls="--", lw=1.2, label="Zero-shot")
        axes[0, j].axhline(ref["BROVEY8"], c="#1f77b4", ls=":", lw=1.4, label="Brovey")
        axes[0, j].set_title(cat, fontsize=12)
        for i in (0, 1):
            axes[i, j].set_xticks(xs)
            axes[i, j].set_xticklabels(["0", ".5K", "1K", "2K", "5K", "10K", "20K"])
            axes[i, j].grid(alpha=0.3)
        axes[1, j].set_xlabel("updates")
    axes[0, 0].set_ylabel("RR ERGAS (audit area)")
    axes[1, 0].set_ylabel("FR HQNR (6 fixed ROI tiles)")
    axes[0, -1].legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "adaptation_curves.png", dpi=150, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--only", choices=["images", "spectrum", "curves", "vehicles", "rrfr", "adapted"],
                    default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"],
                         "mathtext.fontset": "stix"})
    if args.only in (None, "spectrum"):
        period4(out)
    if args.only in (None, "curves"):
        curves(out)
    if args.only in (None, "vehicles"):
        vehicles(out, args.dpi)
    if args.only in (None, "rrfr"):
        rr_vs_fr(out)
    if args.only in (None, "adapted"):
        adapted_best(out, args.dpi)
    if args.only not in (None, "images"):
        return
    rows, meta = [], []
    for i, (sid, cat) in enumerate(SCENES, 1):
        data = load_scene(sid)
        row = build_row(data)
        data["region"].update(order=f"V{i:02d}", category=cat, ms_correlation=sanity(data))
        meta.append(data["region"])
        render_rows([row], out / f"V{i:02d}_{sid}_qualitative.png")
        rows.append(row)
        print(sid, {k: round(v["hqnr"], 4) for k, v in data["fr"].items()})
    render_rows(rows, out / "all_scenes_qualitative.png", max(args.dpi // 2, 100), interp="antialiased")
    (out / "regions.json").write_text(json.dumps(
        dict(rule="audit 3x3 tiles with max mean PAN Sobel; zoom = quarter-size window inside one "
                  "quadrant with max PAN Sobel; stretch = per-band 1-99 pct of native MS in region",
             rgb_bands=list(RGB), regions=meta), indent=1))


if __name__ == "__main__":
    main()
