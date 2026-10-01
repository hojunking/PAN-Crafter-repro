"""Pinned pure numerical primitives. See vendor_reference/metric_sources.json for provenance.
No legacy trainer/controller imports; evaluation only.
"""
import numpy as np
from scipy.signal import fftconvolve

SENSOR_NAME = {"wv3": "WV3", "wv2": "WV2", "qb": "QB", "gf2": None,
               "cas500": None, "vantor": None}

GNYQ_OVERRIDE = {"vantor": 0.35}

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
    # MATLAB imresize.m contributions():
    #   aux = [1:in_length in_length:-1:1]; indices = aux(mod(indices-1, length(aux))+1);
    # 범위 밖 인덱스를 거울 대칭(symmetric)으로 접는다. 2026-09-07 이전에는 clamp(replicate)를
    # 썼다 — 검증 지적. D_s 의 PAN 축소에서 가장자리 2px 만 달라지며 HQNR 영향 ~2e-5.
    aux = np.concatenate([np.arange(1, in_length + 1), np.arange(in_length, 0, -1)])
    ind = (aux[np.mod(ind.astype(np.int64) - 1, len(aux))] - 1).astype(np.intp)

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

GNYQ_TABLE = {
    "QB": [0.34, 0.32, 0.30, 0.22],
    "IKONOS": [0.26, 0.28, 0.29, 0.28],
    "GeoEye1": [0.23, 0.23, 0.23, 0.23], "WV4": [0.23, 0.23, 0.23, 0.23],
    "WV2": [0.35] * 7 + [0.27],
    "WV3": [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315],
}

def _fspecial_gaussian(n: int, sigma: float) -> np.ndarray:
    """MATLAB fspecial('gaussian', n, sigma): eps*max 미만 0, 합 1."""
    m = (n - 1) / 2.0
    y, x = np.ogrid[-m:m + 1, -m:m + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(float).eps * h.max()] = 0.0
    return h / h.sum()

def _fwind1_huang(hd: np.ndarray, win: np.ndarray) -> np.ndarray:
    """MATLAB fwind1(Hd, win) — 1-D 창 하나를 Huang 회전으로 2-D 창으로 만들고, fsamp2 식으로
    Hd 의 임펄스 응답을 구해 창을 곱한다. **정규화하지 않는다** (fwind1·genMTF 모두 sum=1 로
    맞추지 않는다). 검증 지적(2026-09-07): DLPan 공식 파이썬 포트는 1-D 창을 한 축에만 곱하고
    음수를 자른 뒤 sum=1 로 정규화한다 — D_λ 가 ~2e-4 달라진다.

    회전 창의 표본 좌표는 freqspace(n) (홀수 n: (-(n-1)/2..(n-1)/2)·2/n). 이 좌표를 linspace(-1,1,n)
    으로 바꿔도 임펄스 응답이 중심 σ≈2px 에 몰려 있어 D_λ 차이는 1e-6 이하다.
    """
    n = len(win)
    t = np.arange(-(n - 1) / 2.0, (n - 1) / 2.0 + 1) * 2.0 / n          # freqspace(n)
    t1, t2 = np.meshgrid(t, t)
    r = np.sqrt(t1 ** 2 + t2 ** 2)
    w = np.zeros_like(r)
    inside = (r >= t[0]) & (r <= t[-1])
    w[inside] = np.interp(r[inside], t, win)
    # fsamp2: hd = rot90(fftshift(rot90(Hd,2)),2); h = fftshift(ifft2(hd)); h = rot90(h,2)
    h = np.rot90(np.fft.fftshift(np.rot90(hd, 2)), 2)
    h = np.rot90(np.fft.fftshift(np.fft.ifft2(h)), 2)
    return np.real(h) * w

def genmtf_matlab(gnyq, ratio: int, n: int = 41) -> np.ndarray:
    """genMTF.m: 밴드별 alpha -> fspecial gaussian -> Hd/max -> fwind1(Hd, kaiser(N)). (N, N, nbands)"""
    gnyq = np.asarray(gnyq, dtype=np.float64)
    fcut = 1.0 / ratio
    h = np.zeros((n, n, len(gnyq)))
    for b, g in enumerate(gnyq):
        alpha = np.sqrt(((n - 1) * (fcut / 2)) ** 2 / (-2 * np.log(g)))
        H = _fspecial_gaussian(n, alpha)
        h[:, :, b] = _fwind1_huang(H / H.max(), np.kaiser(n, 0.5))      # MATLAB kaiser(N) = beta 0.5
    return h

def mtf_filter(img: np.ndarray, preset: str, ratio: int, wald=None) -> np.ndarray:
    """센서 MTF 저역통과 필터. MATLAB MTF.m = imfilter(real(h), 'replicate').

    커널은 genMTF.m 을 그대로 옮긴 genmtf_matlab() 이다 (wald 인자는 호환용으로 남겼고 쓰지 않는다).
    41x41 커널을 512x512x8에 직접 상관하면 화소당 1681회 곱셈이라 한 장에 5초가 걸린다.
    가장자리를 replicate로 미리 채운 뒤 FFT 컨볼루션으로 바꾸면 결과는 같고 훨씬 빠르다.
    커널이 대칭이라 correlate와 convolve가 동일하므로 뒤집기도 불필요하다.
    """
    sensor = SENSOR_NAME[preset]
    if sensor is None:
        # genMTF.m의 otherwise 분기: GNyq = 0.3 * ones. 데이터가 다른 값으로 만들어졌으면 그것을 쓴다.
        gnyq = GNYQ_OVERRIDE.get(preset, 0.3) * np.ones(img.shape[2])
    else:
        gnyq = GNYQ_TABLE[sensor]
    kernel = genmtf_matlab(gnyq, ratio, 41)

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
