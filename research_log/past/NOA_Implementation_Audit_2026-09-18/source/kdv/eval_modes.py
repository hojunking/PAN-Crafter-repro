"""추론 정합 경로 모드 (계획 research_log/PAN_AllServers_StudentEval_AlignerAnalysis_CurrentMethod_Integrated_2026-09-18.md §4·§7.4).

같은 Student checkpoint를 네 가지 **추론** 경로로 돌린다. 학습은 하지 않는다 — optimizer·backward·파라미터 변경 없음.

    A_ON          c = A(P, M);  Z = M + F([W(P,c), M])          기존 경로
    A_BYPASS_RAW  Z = M + F([P, M])                             **aligner 0회 · warp 0회** (표시명 NOA)
    A_ZERO_WARP   Z = M + F([W(P,0), M])                        aligner 0회지만 sampler는 쓴다 (NOA와 다른 모드)
    A_CROP64_MED  c = median_u A(P^(u), M^(u)) (non-overlap 64 crop, 성분별 중앙값) → 전체 PAN 한 번 warp

`pa/model.py:33 PAModel.forward(pan, ms, lpan, aligner_enabled, delta_override)` 를 인자로만 조종한다 — 모델 파일·checkpoint·학습 경로를
바꾸지 않는다. NOA 는 `aligner_enabled=False` 만으로는 부족하다(그때도 `self.sampler` 가 True 면 warp 를 탄다) → `sampler` 를 **일시적으로**
False 로 두고 반드시 복원한다. 복원 여부와 호출 0회는 `CallCounter` 가 실제 호출을 세어 검증한다(계획 V02·V03·V06·V16).

주의: backbone 인자는 `(pan, lpan, ms, s)` 이고 PAModel 은 `(pan, ms, lpan)` 이다. ms 는 **LR MS** 를 그대로 넘긴다 — 확대는 backbone 안에서
한 번, residual base M 은 PAModel 이 한 번 더한다 (계획 §4.2 V04).
"""
import contextlib
import hashlib

import torch

import pa.model as pa_model

MODES = ("A_ON", "A_BYPASS_RAW", "A_ZERO_WARP", "A_CROP64_MED")
MODE_LABEL = {"A_ON": "A_ON", "A_BYPASS_RAW": "NOA", "A_ZERO_WARP": "ZERO", "A_CROP64_MED": "CROP64MED"}
CONSENSUS_CROP = 64                                        # 계획 §7.4: non-overlap 64 crop


class CallCounter:
    """aligner·warp 실제 호출 횟수를 센다 (계획 V02/V03). pa.model 이 import 한 이름을 감싼다 — 모델 인스턴스가 아니라 경로를 본다."""

    def __init__(self, model):
        self.model = model; self.aligner = 0; self.warp = 0; self._orig_warp = None; self._orig_fwd = None

    def __enter__(self):
        self._orig_warp = pa_model.warp_pan
        def _warp(pan, delta, _o=self._orig_warp):
            self.warp += 1; return _o(pan, delta)
        pa_model.warp_pan = _warp
        al = getattr(self.model, "aligner", None)
        if al is not None:
            self._orig_fwd = al.forward
            def _fwd(*a, _o=self._orig_fwd, **k):
                self.aligner += 1; return _o(*a, **k)
            al.forward = _fwd
        return self

    def __exit__(self, *exc):
        pa_model.warp_pan = self._orig_warp
        if self._orig_fwd is not None:
            self.model.aligner.forward = self._orig_fwd
        return False

    def as_dict(self):
        return dict(aligner_calls=int(self.aligner), warp_calls=int(self.warp))


def state_hash(model):
    """파라미터·버퍼 전체의 sha256 (앞 16자) — 평가 전후 불변 검사 (계획 V06)."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


@contextlib.contextmanager
def sampler_off(model):
    """NOA 전용: sampler 를 잠시 끄고 **반드시** 되돌린다 (계획 V16 — 평가 우회가 학습 경로에 남지 않는다)."""
    prev = bool(model.sampler)
    model.sampler = False
    try:
        yield
    finally:
        model.sampler = prev


def _crops(n, size):
    """[0, n) 를 size 로 겹치지 않게 자른 시작 좌표 (계획 §7.2: stride-4 좌표, 남는 가장자리는 버린다)."""
    return [i for i in range(0, n - size + 1, size)]


def consensus_delta(model, pan, ms_base, size=CONSENSUS_CROP):
    """non-overlap size×size crop 마다 Δ 를 예측해 **성분별 중앙값** 한 쌍을 만든다 (계획 §7.4).

    짝수 개수에서 가운데 두 값의 평균을 쓴다 — torch.median 은 작은 쪽만 돌려주므로 quantile(0.5) 로 계산한다.
    M 은 전체에서 한 번 만든 것을 같은 좌표로 crop 한다(crop 마다 재확대하지 않는다)."""
    B, _, H, W = pan.shape
    ys, xs = _crops(H, size), _crops(W, size)
    if not ys or not xs:
        raise ValueError(f"crop {size} 가 입력 {H}×{W} 보다 크다")
    ds = []
    for y in ys:
        for x in xs:
            ds.append(model.predict_delta(pan[..., y:y + size, x:x + size], ms_base[..., y:y + size, x:x + size]))
    D = torch.stack(ds, dim=0).float()                                   # [n_crop, B, 2]
    med = torch.quantile(D, 0.5, dim=0, interpolation="midpoint")        # 짝수면 가운데 두 값 평균
    return med, D


@torch.no_grad()
def forward_mode(model, pan, ms, lpan, mode, counter=None, return_extra=False):
    """모드별 추론. 반환 y 는 모델 출력 스케일([-1,1] 규약) 그대로 — DN 변환은 호출자가 기존 규약으로 한다.

    counter 를 주면 그 안에서 호출 횟수가 누적된다(호출자가 `with CallCounter(model) as c:` 로 감싼다)."""
    if mode not in MODES:
        raise ValueError(f"알 수 없는 모드 {mode!r} (지원 {MODES})")
    extra = {}
    if mode == "A_ON":
        out = model(pan, ms, lpan)
    elif mode == "A_ZERO_WARP":
        out = model(pan, ms, lpan, aligner_enabled=False)                # Δ=0 이지만 sampler 를 탄다
    elif mode == "A_BYPASS_RAW":
        with sampler_off(model):
            out = model(pan, ms, lpan, aligner_enabled=False)            # aligner·warp 둘 다 호출 0회
    else:                                                                # A_CROP64_MED
        ms_base = torch.nn.functional.interpolate(ms, scale_factor=4, mode="bicubic")
        med, D = consensus_delta(model, pan, ms_base)
        extra = dict(consensus_delta=[float(v) for v in med[0]], n_crops=int(D.shape[0]),
                     crop_delta_std=[float(v) for v in D[:, 0, :].std(dim=0, unbiased=True)] if D.shape[0] > 1 else [0.0, 0.0])
        out = model(pan, ms, lpan, delta_override=med)
    if return_extra:
        extra["delta"] = [float(v) for v in out["delta"][0]]
        return out["y"], extra
    return out["y"]


def expected_calls(mode, n_forward=1):
    """모드별 기대 호출 횟수 (계획 V02/V03). A_CROP64_MED 의 aligner 는 crop 수만큼이라 호출자가 채운다."""
    return {"A_ON": dict(aligner_calls=n_forward, warp_calls=n_forward),
            "A_BYPASS_RAW": dict(aligner_calls=0, warp_calls=0),
            "A_ZERO_WARP": dict(aligner_calls=0, warp_calls=n_forward),
            "A_CROP64_MED": dict(aligner_calls=None, warp_calls=n_forward)}[mode]
