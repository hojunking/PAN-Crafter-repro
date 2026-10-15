"""전역 sub-pixel shift 추정 (계획 §5.1). GT 는 절대 보지 않는다 — 입력은 MTF↓PAN 과 native LRMS.

primary   : Scharr 크기 -> median/MAD 정규화 -> 상위 30% edge mask -> 정수 ZNCC 탐색
            -> 3x3 quadratic sub-pixel 보정
secondary : Census 5x5 + Hamming (정수 탐색 + quadratic)
게이트(§5.2): boundary hit 없음 · |primary-secondary| <= 0.25 · peak margin >= 0.05 · |δ| <= 1.0
하나라도 실패하면 accepted=False, delta_applied=(0,0).

부호: 결과 δ 는 aligned[y,x] = moving[y+dy, x+dx] 가 reference 와 맞는 양 (P <- M).
"""
import numpy as np
import cv2

GATES = dict(search_int=2, max_magnitude=1.0, min_peak_margin=0.05,
             max_primary_secondary_diff=0.25, edge_top_frac=0.30)


def scharr_mag(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    gx = cv2.Scharr(x, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(x, cv2.CV_32F, 0, 1)
    return np.sqrt(gx * gx + gy * gy)


def robust_norm(x):
    m = np.median(x)
    mad = np.median(np.abs(x - m)) + 1e-6
    return (x - m) / mad


def edge_mask(g, top_frac):
    return g >= np.quantile(g, 1.0 - top_frac)


def _overlap(ref, mov, mask, dy, dx, r):
    H, W = ref.shape
    a = ref[r:H - r, r:W - r]
    b = mov[r + dy:H - r + dy, r + dx:W - r + dx]
    m = mask[r:H - r, r:W - r]
    return a, b, m


def _zncc(a, b, m):
    a = a[m]; b = b[m]
    if a.size < 16:
        return -1.0
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def _quad(m, c, p):
    """1D 3점 quadratic 정점. c 가 최대(또는 최소)일 때 [-0.5, 0.5] 로 clip."""
    den = 2.0 * (m - 2.0 * c + p)
    if abs(den) < 1e-12:
        return 0.0
    return float(np.clip((m - p) / den, -0.5, 0.5))


def _search(score_fn, r, maximize=True):
    S = {}
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            S[(dy, dx)] = score_fn(dy, dx)
    key = max(S, key=S.get) if maximize else min(S, key=S.get)
    dy0, dx0 = key
    vals = sorted(S.values(), reverse=maximize)
    margin = abs(vals[0] - vals[1])
    hit = abs(dy0) == r or abs(dx0) == r
    if hit:
        return float(dy0), float(dx0), S[key], margin, True
    ry = _quad(S[(dy0 - 1, dx0)], S[key], S[(dy0 + 1, dx0)])
    rx = _quad(S[(dy0, dx0 - 1)], S[key], S[(dy0, dx0 + 1)])
    return dy0 + ry, dx0 + rx, S[key], margin, False


def census5(x):
    """5x5 census transform -> 24 bit 정수 맵 (경계 2px 은 0)."""
    x = np.asarray(x, dtype=np.float32)
    H, W = x.shape
    out = np.zeros((H, W), dtype=np.uint32)
    c = x[2:H - 2, 2:W - 2]
    bit = 0
    for oy in range(-2, 3):
        for ox in range(-2, 3):
            if oy == 0 and ox == 0:
                continue
            n = x[2 + oy:H - 2 + oy, 2 + ox:W - 2 + ox]
            out[2:H - 2, 2:W - 2] |= ((n > c).astype(np.uint32) << bit)
            bit += 1
    return out


def _hamming(a, b):
    v = np.bitwise_xor(a, b)
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    return (v * 0x01010101) >> 24


def estimate_shift(ref_lr, mov_lr, gates=GATES):
    """ref_lr: MTF↓PAN [h,w], mov_lr: native LRMS 밴드평균 [h,w]. 반환 dict (cache 열)."""
    r = int(gates["search_int"])
    g_ref = robust_norm(scharr_mag(ref_lr))
    g_mov = robust_norm(scharr_mag(mov_lr))
    mask = edge_mask(g_ref, gates["edge_top_frac"])

    def zncc_at(dy, dx):
        a, b, m = _overlap(g_ref, g_mov, mask, dy, dx, r)
        return _zncc(a, b, m)
    p_dy, p_dx, p_peak, p_margin, p_hit = _search(zncc_at, r, maximize=True)

    c_ref, c_mov = census5(ref_lr), census5(mov_lr)

    def ham_at(dy, dx):
        a, b, m = _overlap(c_ref, c_mov, mask, dy, dx, r)
        return float(_hamming(a[m], b[m]).mean()) if m.sum() >= 16 else 24.0
    s_dy, s_dx, _, _, s_hit = _search(ham_at, r, maximize=False)

    mag = float(np.hypot(p_dy, p_dx))
    ps_diff = float(np.hypot(p_dy - s_dy, p_dx - s_dx))
    accepted = (not p_hit) and (not s_hit) and ps_diff <= gates["max_primary_secondary_diff"] \
        and p_margin >= gates["min_peak_margin"] and mag <= gates["max_magnitude"]
    return dict(dy_lr_raw=p_dy, dx_lr_raw=p_dx, magnitude_raw=mag, peak_zncc=p_peak,
                peak_margin=p_margin, secondary_dy=s_dy, secondary_dx=s_dx,
                primary_secondary_diff=ps_diff, boundary_hit=bool(p_hit or s_hit),
                accepted=bool(accepted),
                dy_lr_applied=p_dy if accepted else 0.0, dx_lr_applied=p_dx if accepted else 0.0)
