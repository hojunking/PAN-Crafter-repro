"""Q2n (hypercomplex Q index) — DLPan-Toolbox MATLAB 원본의 충실한 Python 이식.

원본: DLPan-Toolbox/02-Test-toolbox-for-traditional-and-DL(Matlab)/Quality_Indices/
      {q2n, onions_quality, norm_blocco, onion_mult, onion_mult2D}.m
알고리즘: A. Garzelli, F. Nencini, "Hypercomplex quality assessment of multi/hyperspectral
          images", IEEE GRSL, 2009.

왜 직접 이식했는가:
    DLPan-Toolbox에는 공식 Python 포트도 있다
    (01-DL-toolbox(Pytorch)/UDL/pansharpening/common/evaluate.py).
    그런데 이 포트는 **Q2n > 1을 반환한다.** 검증하면 바로 드러난다.

        q2n(a, a)            -> 0.9957   (동일 입력이므로 정확히 1.0이어야 한다)
        WV3 RR 사전학습 Q8   -> 1.1161   (논문 0.920, 그리고 Q2n은 정의상 1을 넘을 수 없다)

    원인은 `norm_blocco`의 표준편차다. MATLAB `std2`는 N-1 정규화(ddof=1)를 쓰는데
    포트는 numpy 기본값 ddof=0을 쓴다. 블록 정규화 스케일이 어긋나면서
    hypercomplex 곱의 결과가 1을 넘어간다. (eps도 MATLAB eps 대신 1e-8을 쓴다.)

    아래 구현은 `q2n(a, a) == 1.0`과 논문 Q8 재현으로 검증되어 있다.
    자체 검사: python tools/q2n.py
"""

import math

import numpy as np

__all__ = ["q2n", "matlab_uint16"]


def matlab_uint16(x: np.ndarray) -> np.ndarray:
    """MATLAB uint16() 의미론: 반올림(half away from zero) 후 [0, 65535]로 포화.

    numpy의 astype(np.uint16)은 0쪽으로 절삭하고 음수를 wrap한다. FR의 msexp에는
    보간 링잉으로 음수가 있어(WV3 FR에서 -238) 그냥 캐스팅하면 65298로 wrap된다.
    """
    return np.clip(np.floor(np.asarray(x, dtype=np.float64) + 0.5), 0.0, 65535.0)


def _norm_blocco(x: np.ndarray):
    """norm_blocco.m — mean2/std2로 정규화. std2는 N-1 정규화다."""
    a = float(np.mean(x))
    c = float(np.std(x, ddof=1))
    if c == 0:
        c = float(np.finfo(np.float64).eps)
    return ((x - a) / c) + 1.0, a, c


def _conj(v: np.ndarray, axis: int) -> np.ndarray:
    """[v(1), -v(2:end)] — hypercomplex 켤레."""
    head = np.take(v, [0], axis=axis)
    tail = np.take(v, range(1, v.shape[axis]), axis=axis)
    return np.concatenate([head, -tail], axis=axis)


def _onion_mult(o1: np.ndarray, o2: np.ndarray) -> np.ndarray:
    """onion_mult.m — 1차원 Cayley-Dickson 곱."""
    n = o1.shape[0]
    if n <= 1:
        return o1 * o2

    half = n // 2
    a, b = o1[:half], _conj(o1[half:], 0)
    c, d = o2[:half], _conj(o2[half:], 0)

    if n == 2:
        return np.concatenate([a * c - d * b, a * d + c * b])

    ris1 = _onion_mult(a, c)
    ris2 = _onion_mult(d, _conj(b, 0))
    ris3 = _onion_mult(_conj(a, 0), d)
    ris4 = _onion_mult(c, b)
    return np.concatenate([ris1 - ris2, ris3 + ris4])


def _onion_mult2d(o1: np.ndarray, o2: np.ndarray) -> np.ndarray:
    """onion_mult2D.m — (H, W, N3) 화소별 Cayley-Dickson 곱."""
    n3 = o1.shape[2]
    if n3 <= 1:
        return o1 * o2

    half = n3 // 2
    a, b = o1[:, :, :half], _conj(o1[:, :, half:], 2)
    c, d = o2[:, :, :half], _conj(o2[:, :, half:], 2)

    if n3 == 2:
        return np.concatenate([a * c - d * b, a * d + c * b], axis=2)

    ris1 = _onion_mult2d(a, c)
    ris2 = _onion_mult2d(d, _conj(b, 2))
    ris3 = _onion_mult2d(_conj(a, 2), d)
    ris4 = _onion_mult2d(c, b)
    return np.concatenate([ris1 - ris2, ris3 + ris4], axis=2)


def _onions_quality(dat1: np.ndarray, dat2: np.ndarray, size1: int) -> np.ndarray:
    """onions_quality.m — 단일 블록의 밴드별 Q 성분."""
    dat1 = np.array(dat1, dtype=np.float64)
    dat2 = _conj(np.array(dat2, dtype=np.float64), 2)
    n3 = dat1.shape[2]
    size2 = size1

    for i in range(n3):
        a1, s, t = _norm_blocco(dat1[:, :, i])
        dat1[:, :, i] = a1
        if s == 0:
            if i == 0:
                dat2[:, :, i] = dat2[:, :, i] - s + 1
            else:
                dat2[:, :, i] = -(-dat2[:, :, i] - s + 1)
        else:
            if i == 0:
                dat2[:, :, i] = ((dat2[:, :, i] - s) / t) + 1
            else:
                dat2[:, :, i] = -(((-dat2[:, :, i] - s) / t) + 1)

    m1 = dat1.mean(axis=(0, 1))
    m2 = dat2.mean(axis=(0, 1))
    mod_q1m = math.sqrt(float((m1**2).sum()))
    mod_q2m = math.sqrt(float((m2**2).sum()))
    mod_q1 = np.sqrt((dat1**2).sum(axis=2))
    mod_q2 = np.sqrt((dat2**2).sum(axis=2))

    nn = size1 * size2
    ratio = nn / (nn - 1)
    termine2 = mod_q1m * mod_q2m
    termine4 = mod_q1m**2 + mod_q2m**2
    int1 = ratio * float(np.mean(mod_q1**2))
    int2 = ratio * float(np.mean(mod_q2**2))
    termine3 = int1 + int2 - ratio * (mod_q1m**2 + mod_q2m**2)
    mean_bias = 2 * termine2 / termine4 if termine4 != 0 else 0.0

    if termine3 == 0:
        q = np.zeros(n3)
        q[n3 - 1] = mean_bias
        return q

    cbm = 2 / termine3
    qu = _onion_mult2d(dat1, dat2)
    qm = _onion_mult(m1, m2)
    qv = ratio * qu.mean(axis=(0, 1))
    return (qv - ratio * qm) * mean_bias * cbm


def _pad_mirror(img: np.ndarray, est1: int, est2: int) -> np.ndarray:
    """q2n.m의 경계 보정 (블록 크기로 나누어떨어지지 않을 때)."""
    n1, n2, n3 = img.shape
    out = np.zeros((n1 + est1, n2 + est2, n3), dtype=np.float64)
    out[:n1, :n2, :] = img
    if est2 > 0:
        out[:n1, n2:n2 + est2, :] = img[:, n2 - 1:n2 - est2 - 1:-1, :]
    if est1 > 0:
        out[n1:n1 + est1, :, :] = out[n1 - 1:n1 - est1 - 1:-1, :, :]
    return out


def q2n(gt: np.ndarray, fused: np.ndarray, q_blocks_size: int = 32, q_shift: int = 32):
    """q2n.m — (Q2n_index, Q2n_index_map)을 돌려준다. 입력은 (H, W, C)."""
    gt = np.asarray(gt, dtype=np.float64)
    fused = np.asarray(fused, dtype=np.float64)
    if gt.shape != fused.shape:
        raise ValueError(f"shape 불일치: {gt.shape} vs {fused.shape}")

    n1, n2, n3 = gt.shape
    stepx = max(math.ceil(n1 / q_shift), 1)
    stepy = max(math.ceil(n2 / q_shift), 1)
    est1 = (stepx - 1) * q_shift + q_blocks_size - n1
    est2 = (stepy - 1) * q_shift + q_blocks_size - n2
    if est1 != 0 or est2 != 0:
        gt = _pad_mirror(gt, est1, est2)
        fused = _pad_mirror(fused, est1, est2)

    # MATLAB: I_F=uint16(I_F); I_GT=uint16(I_GT);
    gt = matlab_uint16(gt)
    fused = matlab_uint16(fused)

    n1, n2, n3 = gt.shape
    if math.ceil(math.log2(n3)) != math.log2(n3):  # 밴드 수를 2의 거듭제곱으로 zero-pad
        ndif = 2 ** math.ceil(math.log2(n3)) - n3
        pad = np.zeros((n1, n2, ndif))
        gt = np.concatenate([gt, pad], axis=2)
        fused = np.concatenate([fused, pad], axis=2)
    n3 = gt.shape[2]

    valori = np.zeros((stepx, stepy, n3))
    for j in range(stepx):
        for i in range(stepy):
            ys, xs = j * q_shift, i * q_shift
            valori[j, i, :] = _onions_quality(
                gt[ys:ys + q_blocks_size, xs:xs + q_blocks_size, :],
                fused[ys:ys + q_blocks_size, xs:xs + q_blocks_size, :],
                q_blocks_size)

    q2n_map = np.sqrt((valori**2).sum(axis=2))
    return float(q2n_map.mean()), q2n_map


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    failures = 0

    for bands in (4, 8):
        a = np.round(rng.random((64, 64, bands)) * 2047)
        idx, _ = q2n(a, a.copy())
        ok = abs(idx - 1.0) < 1e-9
        failures += not ok
        print(f"  q2n(a, a)  bands={bands}: {idx:.12f}  {'OK' if ok else 'FAIL (1.0이어야 한다)'}")

    a = np.round(rng.random((64, 64, 8)) * 2047)
    b = np.round(rng.random((64, 64, 8)) * 2047)
    idx, _ = q2n(a, b)
    ok = 0.0 <= idx <= 1.0
    failures += not ok
    print(f"  q2n(random, random): {idx:.6f}  {'OK (0..1 범위)' if ok else 'FAIL'}")

    # 블록 크기로 나누어떨어지지 않는 경우 (경계 보정 경로)
    a = np.round(rng.random((70, 70, 8)) * 2047)
    idx, _ = q2n(a, a.copy())
    ok = abs(idx - 1.0) < 1e-9
    failures += not ok
    print(f"  q2n(a, a)  70x70 (경계 보정): {idx:.12f}  {'OK' if ok else 'FAIL'}")

    print("전부 통과." if not failures else f"실패 {failures}건")
    raise SystemExit(1 if failures else 0)
