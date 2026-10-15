#!/usr/bin/env python
"""Global alignment 단위 검사 T01~T10 (계획 §22.1). 어떤 50K run 도 이것을 통과한 뒤에 시작한다.

  python tools/test_alignment.py            # 전부
  python -m pytest tools/test_alignment.py  # 도 된다
"""
import os
import sys

import h5py
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from align.resample import (interp23tap, phase_shift_upsample, upsample_shift, warp_hr,   # noqa: E402
                            transform_delta, border_mask, augment_hr)
from align.estimator import estimate_shift                                                # noqa: E402
from align.model import AlignCfg, AlignedModel                                            # noqa: E402
from align.shiftnet import structural_input                                               # noqa: E402

R = os.path.join(ROOT, "data/PanCollection/WV3")
H5 = {"train": f"{R}/train_wv3.h5", "rr": f"{R}/reduced_examples_h5/test_wv3_multiExm1.h5",
      "fr": f"{R}/full_examples_h5_repaired/test_wv3_OrigScale_multiExm1.h5"}
torch.manual_seed(0)


def _zncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def _smooth(B, C, H, W, dtype=torch.float64):
    """저주파 합성 영상 — sub-pixel 보간 검사에 쓴다."""
    x = torch.randn(B, C, H // 8, W // 8, dtype=dtype)
    return torch.nn.functional.interpolate(x, size=(H, W), mode="bicubic", align_corners=False)


def test_T01_phase_mapping():
    lr = torch.zeros(1, 1, 8, 8, dtype=torch.float64); lr[0, 0, 3, 5] = 1.0
    hr = interp23tap(lr, 4)
    y, x = divmod(int(hr[0, 0].argmax()), hr.shape[-1])
    assert (y, x) == (4 * 3 + 2, 4 * 5 + 2), f"interp23tap 중심 {y, x}"
    assert abs(hr[0, 0, 14, 22].item() - 1.0) < 1e-9
    hb = phase_shift_upsample(lr, torch.zeros(1, 2, dtype=torch.float64), 0.0, phase=2.0)
    assert abs(hb[0, 0, 14, 22].item() - 1.0) < 1e-9, "phase-2 bicubic: 4j+2 가 LR j 를 정확히 참조"


def test_T02_provided_lms_reproduction():
    for split, n in (("train", 64), ("rr", 20), ("fr", 20)):
        with h5py.File(H5[split]) as f:
            ms = torch.tensor(f["ms"][:n]); lms = f["lms"][:n]
        up = interp23tap(ms, 4).numpy()
        z = np.mean([_zncc(up[i], lms[i]) for i in range(n)])
        mad = float(np.abs(up - lms).mean())
        assert z >= 0.9999 and mad < 1e-6, f"{split}: ZNCC {z:.6f} MAD {mad}"
        # 계획 원안(phase-2 bicubic)은 참고로만 찍는다 — gate 를 못 넘는다
        pb = phase_shift_upsample(ms, torch.zeros(n, 2, dtype=ms.dtype), 0.0).numpy()
        zb = np.mean([_zncc(pb[i], lms[i]) for i in range(n)])
        print(f"  [T02] {split}: interp23tap ZNCC {z:.6f} MAD {mad:.2e} | phase-2 bicubic ZNCC {zb:.6f} (참고)")


def test_T03_sign_convention():
    # warp_hr: out[y,x] = src[y+dy, x+dx] -> 임펄스가 (y0-dy, x0-dx) 로 간다
    A = torch.zeros(1, 1, 32, 32, dtype=torch.float64); A[0, 0, 16, 20] = 1
    B = warp_hr(A, torch.tensor([[3.0, -2.0]], dtype=torch.float64))
    y, x = divmod(int(B[0, 0].argmax()), 32)
    assert (y, x) == (13, 22), (y, x)
    # phase_shift_upsample α=1, 정수 LR shift: LR (j,i) -> HR (4(j-dy)+2, 4(i-dx)+2)
    lr = torch.zeros(1, 1, 12, 12, dtype=torch.float64); lr[0, 0, 6, 7] = 1
    hr = phase_shift_upsample(lr, torch.tensor([[1.0, -1.0]], dtype=torch.float64), 1.0)
    y, x = divmod(int(hr[0, 0].argmax()), 48)
    assert (y, x) == (4 * 5 + 2, 4 * 8 + 2), (y, x)
    hr2 = upsample_shift(lr, torch.tensor([[1.0, -1.0]], dtype=torch.float64), 1.0, kind="interp23tap")
    y, x = divmod(int(hr2[0, 0].argmax()), 48)
    assert (y, x) == (4 * 5 + 2, 4 * 8 + 2), (y, x)
    # estimator: mov[y,x] = ref[y-dy, x-dx] (내용이 +d 만큼 이동) -> δ = (dy,dx) 를 돌려줘야 한다
    # 질감 있는 128² (FR LR 크기). 3x3 quadratic 의 실측 오차는 p50 0.044 / p90 0.086 px 라
    # 허용은 0.12 로 둔다. 부호와 크기 순서가 맞는지가 이 검사의 목적이다.
    ref = torch.nn.functional.interpolate(torch.randn(1, 1, 32, 32, dtype=torch.float64),
                                          size=(128, 128), mode="bicubic", align_corners=False)[0, 0]
    for dy, dx in ((0.3, -0.2), (-0.45, 0.1), (0.0, 0.35)):
        mov = warp_hr(ref[None, None], torch.tensor([[-dy, -dx]], dtype=torch.float64))[0, 0]
        r = estimate_shift(ref.numpy(), mov.numpy())
        assert abs(r["dy_lr_raw"] - dy) < 0.12 and abs(r["dx_lr_raw"] - dx) < 0.12, (dy, dx, r)


def test_T04_inverse():
    """+Δ 후 -Δ 가 원위치로 돌아오는가. 정수 shift 는 정확히, sub-pixel 은 보간 blur 만큼만
    어긋난다 — 그 blur 의 크기 자체는 계획 §20 round-trip control(tools/align_diag.py)이 잰다."""
    A = _smooth(2, 3, 64, 64)
    d_int = torch.tensor([[3.0, -2.0], [-5.0, 1.0]], dtype=torch.float64)
    back = warp_hr(warp_hr(A, d_int), -d_int)
    m = border_mask(d_int / 4, 64, 64).bool().expand_as(A)
    assert (back - A).abs()[m].max() < 1e-9, "정수 왕복은 정확해야 한다"
    d = torch.tensor([[1.3, -0.7], [-2.2, 0.4]], dtype=torch.float64)
    back = warp_hr(warp_hr(A, d), -d)
    rel = ((back - A).abs()[m].max() / A.abs().max()).item()
    print(f"  [T04] sub-pixel 왕복 최대 상대오차 {rel:.4f} (bicubic 2회 blur)")
    assert rel < 0.1
    lr = _smooth(1, 8, 16, 16)
    dl = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    rt = warp_hr(upsample_shift(lr, dl, 1.0), -4 * dl)
    m = border_mask(dl, 64, 64).bool().expand_as(rt)
    base = interp23tap(lr)
    assert ((rt - base).abs()[m].max() / base.abs().max()).item() < 0.1
    # 부호 일관성: forward(+Δ) 뒤 inverse(-Δ) 가 forward(+Δ) 뒤 forward(+Δ) 보다 원본에 훨씬 가깝다
    wrong = warp_hr(upsample_shift(lr, dl, 1.0), +4 * dl)
    assert (rt - base).abs()[m].mean() < 0.2 * (wrong - base).abs()[m].mean()


def test_T05_augmentation():
    """impulse 쌍에 known shift 를 주고 flip/rot 뒤 변환된 delta 로 정렬하면 impulse 가 일치해야 한다."""
    H = W = 24
    for hf in (0, 1):
        for vf in (0, 1):
            for rot in range(4):
                ref = torch.zeros(H, W, dtype=torch.float64); ref[9, 14] = 1
                dy, dx = 2, -3                       # mov[y,x] = ref[y-dy, x-dx]
                mov = torch.roll(ref, shifts=(dy, dx), dims=(0, 1))
                def aug(x):
                    if hf: x = x.flip(1)
                    if vf: x = x.flip(0)
                    return torch.rot90(x, rot, (0, 1))
                ra, ma = aug(ref), aug(mov)
                d = transform_delta(torch.tensor([[float(dy), float(dx)]]), torch.tensor([hf]),
                                    torch.tensor([vf]), torch.tensor([rot]))[0]
                ndy, ndx = int(round(d[0].item())), int(round(d[1].item()))
                aligned = torch.roll(ma, shifts=(-ndy, -ndx), dims=(0, 1))   # aligned[y,x] = ma[y+dy, x+dx]
                assert torch.equal(aligned, ra), (hf, vf, rot, d)


def _model(cfg):
    from model.pancrafter_paper import PANCrafterPaper
    torch.manual_seed(1)
    bb = PANCrafterPaper(in_channels=1, out_channels=8, hidden_size=16, depth=[1, 1, 1], norm="ln",
                         in_mode="released", attn_locations=[], mode_modulation=True)
    with torch.no_grad():
        for p in bb.parameters():
            if p.abs().sum() == 0: p.normal_(0, 1e-3)
    return AlignedModel(bb, AlignCfg.from_dict(cfg)).double()


def _inputs(B=2, dtype=torch.float64):
    return (_smooth(B, 1, 64, 64, dtype), _smooth(B, 1, 16, 16, dtype), _smooth(B, 8, 16, 16, dtype))


def test_T06_alpha0_equals_P0():
    pan, lpan, ms = _inputs()
    p0 = _model(dict(delta_source="zero", alpha=0.0))
    c2 = _model(dict(delta_source="cache", alpha=0.0, output_frame="M", inverse_location="none"))
    c2.backbone.load_state_dict(p0.backbone.state_dict())
    d = torch.tensor([[0.4, -0.3], [-0.2, 0.5]], dtype=torch.float64)
    sw = torch.ones(2, dtype=torch.float64)
    with torch.no_grad():
        a = p0(pan, lpan, ms, sw, torch.zeros(2, 2, dtype=torch.float64))
        b = c2(pan, lpan, ms, sw, d)
    assert torch.equal(a, b), "α=0 경로가 P0 와 비트 동일해야 한다"
    assert torch.equal(upsample_shift(ms, d, 0.0), interp23tap(ms))


def test_T07_alpha1_uses_cache_delta():
    ms = _smooth(2, 8, 16, 16)
    d = torch.tensor([[0.4, -0.3], [-0.2, 0.5]], dtype=torch.float64)
    assert torch.equal(upsample_shift(ms, d, 1.0), warp_hr(interp23tap(ms), 4 * d))
    half = upsample_shift(ms, d, 0.5)
    assert torch.equal(half, warp_hr(interp23tap(ms), 2 * d))


def test_T08_pan_mode_no_inverse():
    pan, lpan, ms = _inputs()
    m = _model(dict(delta_source="cache", alpha=1.0, output_frame="P", inverse_location="loss_branch"))
    d = torch.tensor([[0.4, -0.3], [-0.2, 0.5]], dtype=torch.float64)
    v = m.build_views(pan, lpan, ms, d)
    res = m.residual(v["x11"], torch.zeros(2, dtype=torch.float64))    # PAN mode
    p_hat = v["lpan_hr"].repeat(1, 8, 1, 1) + res
    assert m.n_inverse_calls == 0 and p_hat.shape == (2, 8, 64, 64)


def test_T09_mode_duplication():
    d = torch.tensor([[0.4, -0.3], [-0.2, 0.5]], dtype=torch.float64)
    dual = torch.cat([d.detach(), d], 0)
    assert torch.equal(dual[:2], dual[2:])


def test_T10_gradient():
    pan, lpan, ms = _inputs()
    gt = _smooth(2, 8, 64, 64)
    # frozen: cache delta 에 grad 없음
    m = _model(dict(delta_source="cache", alpha=1.0, output_frame="M", inverse_location="final_output"))
    d = torch.tensor([[0.4, -0.3], [-0.2, 0.5]], dtype=torch.float64)
    v = m.build_views(pan, lpan, ms, d)
    fin = m.finalize_ms(v["ms_base_hr"], m.residual(v["x11"], torch.ones(2, dtype=torch.float64)), d)
    assert not fin["y_final"].requires_grad or not d.requires_grad
    # trainable: MS loss -> ShiftNet grad 있음 / PAN loss(detach) -> 없음
    t = _model(dict(delta_source="trainable", trainable_shift_net=True, alpha=1.0, output_frame="M",
                    inverse_location="final_output"))
    with torch.no_grad():
        t.shift_net.head.weight.normal_(0, 0.1)
    dp = t.predict_delta(lpan, ms)
    assert dp.requires_grad
    v = t.build_views(pan, lpan, ms, dp)
    fin = t.finalize_ms(v["ms_base_hr"], t.residual(v["x11"], torch.ones(2, dtype=torch.float64)), dp)
    (fin["y_loss"] - gt).abs().mean().backward()
    g = [p.grad for p in t.shift_net.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert g, "MS loss 가 ShiftNet 으로 흘러야 한다"
    t.zero_grad(set_to_none=True)
    dp2 = t.predict_delta(lpan, ms).detach()
    v = t.build_views(pan, lpan, ms, dp2)
    res = t.residual(v["x11"], torch.zeros(2, dtype=torch.float64))
    ((v["lpan_hr"].repeat(1, 8, 1, 1) + res) - pan.repeat(1, 8, 1, 1)).abs().mean().backward()
    assert all(p.grad is None for p in t.shift_net.parameters()), "PAN loss 는 ShiftNet 을 건드리면 안 된다"
    assert structural_input(lpan, ms).shape == (2, 2, 16, 16)


def test_T11_augmentation_phase_real_data():
    """2026-09-05 지적: LR 을 flip/rot 한 뒤 phase-2 로 올리면 HR 과 1px 어긋난다. 고친 경로
    (feeder 가 LR 원본 + 플래그, wrapper 가 upsample 뒤 HR 증강)에서는 ms_base_hr 이 feeder 가
    증강한 lms 와 **비트 수준으로** 같아야 한다 — 16 조합 전부."""
    import random
    from feeders.feeder_align import PanFeederAlign
    fd = PanFeederAlign(dataroot=H5["train"], crop=False, hflip=True, vflip=True, rot=True)
    m = _model(dict(delta_source="zero", alpha=0.0))
    seen = set()
    for trial in range(40):
        gt, lms, ms, lpan, pan, meta = fd[trial]
        hf, vf, r = int(meta[2]), int(meta[3]), int(meta[4])
        seen.add((hf, vf, r))
        v = m.build_views(pan[None].double(), lpan[None].double(), ms[None].double(),
                          torch.zeros(1, 2, dtype=torch.float64),
                          aug=(meta[None, 2], meta[None, 3], meta[None, 4]))
        mad = (v["ms_base_hr"][0] - lms.double()).abs().mean().item()
        assert mad < 1e-6, f"aug (hf,vf,rot)=({hf},{vf},{r}): ms_base_hr vs 증강 lms MAD {mad:.3e}"   # feeder 는 float32 (옛 경로는 ~2e-2)
        # 반대로 LR 을 먼저 증강했다면 (옛 경로) 1px 어긋난다 — 회귀 방지용 대조
        if r != 2:
            ms_np = ms.numpy()[:, ::-1, ::-1] if True else ms.numpy()
            ms_aug = np.ascontiguousarray(np.rot90(ms_np, r, (1, 2)))
            old = interp23tap(torch.tensor(ms_aug)[None].double())[0]
            assert (old - lms.double()).abs().mean().item() > 1e-3, "옛 경로가 어긋나지 않으면 이 검사는 무의미"   # 정규화 단위 (~2e-2)
    assert len(seen) >= 3, seen
    # 모든 rot 를 강제로 훑는다 (hflip/vflip 은 항상 켜져 있다)
    for r in range(4):
        gt, lms, ms, lpan, pan = (np.array(fd.gt[3]), np.array(fd.lms[3]), np.array(fd.ms[3]),
                                  np.array(fd.lpan[3]), np.array(fd.pan[3]))
        gt, lms, pan = fd._apply((gt, lms, pan), 1, 1, r)
        v = m.build_views(fd.np2tensor(pan)[None].double(), fd.np2tensor(lpan)[None].double(),
                          fd.np2tensor(ms)[None].double(), torch.zeros(1, 2, dtype=torch.float64),
                          aug=(torch.tensor([1]), torch.tensor([1]), torch.tensor([r])))
        assert (v["ms_base_hr"][0] - fd.np2tensor(lms).double()).abs().max().item() < 1e-6, r
        assert (v["lpan_hr"][0] - augment_hr(interp23tap(fd.np2tensor(lpan)[None].double()),
                                             torch.tensor([1]), torch.tensor([1]), torch.tensor([r]))[0]).abs().max() < 1e-12


def test_T12_warp_commutes_with_augmentation():
    """full-shift 경로: (원본 frame 에서 shift) 뒤 HR 증강 == HR 증강 뒤 (변환된 Δ 로 shift).
    inverse warp 를 증강된 출력 frame 에서 transform_delta(Δ) 로 거는 근거."""
    lr = _smooth(1, 8, 16, 16)
    d = torch.tensor([[0.37, -0.22]], dtype=torch.float64)
    for hf in (0, 1):
        for vf in (0, 1):
            for r in range(4):
                fl = (torch.tensor([hf]), torch.tensor([vf]), torch.tensor([r]))
                a = augment_hr(upsample_shift(lr, d, 1.0), *fl)
                b = warp_hr(augment_hr(interp23tap(lr), *fl), 4 * transform_delta(d, *fl))
                m = border_mask(d, 64, 64).bool().expand_as(a)
                assert (a - b)[m].abs().max() < 1e-9, (hf, vf, r, (a - b)[m].abs().max())
                # inverse 도 같은 frame 에서 -T(Δ) 로 돌아온다
                back = warp_hr(a, -4 * transform_delta(d, *fl))
                ref = augment_hr(interp23tap(lr), *fl)
                assert ((back - ref)[m].abs().max() / ref.abs().max()).item() < 0.1


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_T")]
    fails = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS {name}")
        except Exception as e:
            fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - fails}/{len(tests)} 통과")
    sys.exit(1 if fails else 0)
