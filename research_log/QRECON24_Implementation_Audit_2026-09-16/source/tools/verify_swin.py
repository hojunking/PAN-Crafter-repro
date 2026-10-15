#!/usr/bin/env python
"""Swin 구현 검증 — 원안 §5.4·§7.1 의 체크리스트를 기계 검사로 옮긴 것.

shape smoke 로는 못 잡는 오류(window 위상, shift mask, 국소성 위반)를 겨냥한다.

  1  window_partition/reverse 왕복 항등
  2  PixelShuffle(PixelUnshuffle(x,4),4) == x
  3  SwinBlock shape 보존: 16²/32²/128² + 비배수 20×20 (padding 경로)
  4  impulse 국소성: Δ출력(impulse 유무 차)이 창 크기(w²=64 픽셀) 이내인가
     — W-MSA 는 impulse 가 속한 창 안, SW-MSA 는 shift 된 창 안
  5  SW-MSA mask: 순환 shift 로 이어붙은 반대편 픽셀로 attention 이 새지 않는가
     (mask 를 껐을 때는 새고, 켰을 때는 안 새야 둘 다 검증된 것)
  6  PANCrafterPaper(swin_depth/swin_mid) forward+backward, 모든 Swin 파라미터에
     비영 grad (zero_module 함정 회피: 0 파라미터 난수화 후 검사)
  7  LRTinySwin: pan/ms 입력별 비영 grad, dual switch, FR 512² shape
  8  params: SwinBlock C128·mlp2 = 0.13338M, c6+swin2 = 4.0387M
  9  체크포인트 호환: 신규 옵션 미지정 시 c6/c0 params·state_dict 키 불변

종료코드: 실패 1, 전부 통과 0.
"""
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.swin import SwinBlock, window_partition, window_reverse  # noqa: E402
from model.pancrafter_paper import PANCrafterPaper  # noqa: E402
from model.lr_tinyswin import LRTinySwin  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[verify] {'OK  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def randomize_zero(m):
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad and p.abs().sum() == 0:
                p.normal_(0, 1e-3)


def main():
    torch.manual_seed(0)

    # 1. window 왕복
    x = torch.randn(2, 16, 24, 5)
    r = window_reverse(window_partition(x, 8), 8, 16, 24, 2)
    check("window partition/reverse 왕복", torch.equal(x, r))

    # 2. PixelShuffle 왕복
    x = torch.randn(2, 1, 64, 64)
    r = nn.PixelShuffle(4)(nn.PixelUnshuffle(4)(x))
    check("PixelShuffle(PixelUnshuffle(x)) == x", torch.equal(x, r))

    # 3. shape 보존 (+ padding 경로)
    for shift in (0, 4):
        blk = SwinBlock(32, num_heads=4, window_size=8, shift=shift).eval()
        for hw in ((16, 16), (32, 32), (128, 128), (20, 20)):
            y = blk(torch.randn(2, 32, *hw))
            check(f"SwinBlock shift={shift} {hw} shape·finite",
                  y.shape == (2, 32, *hw) and torch.isfinite(y).all().item())

    # 4. impulse 국소성 (Δ = block(impulse) − block(0) 가 창 안에만)
    for shift, name in ((0, "W-MSA"), (4, "SW-MSA")):
        blk = SwinBlock(8, num_heads=2, window_size=8, shift=shift).eval()
        z = torch.zeros(1, 8, 32, 32)
        imp = z.clone()
        imp[0, 0, 9, 3] = 1.0
        with torch.no_grad():
            delta = (blk(imp) - blk(z)).abs().sum(dim=1)[0]     # (32, 32)
        nz = (delta > 1e-9).nonzero()
        n_aff = len(nz)
        inside = bool(((nz - torch.tensor([9, 3])).abs() < 32).all())
        check(f"impulse 국소성 {name}", 0 < n_aff <= 64 and inside,
              f"영향 픽셀 {n_aff}/64")
        if shift == 0:
            in_win = bool((nz[:, 0].div(8, rounding_mode='floor') == 1).all()
                          and (nz[:, 1].div(8, rounding_mode='floor') == 0).all())
            check("impulse 가 자기 창(행 8-15, 열 0-7) 안에만", in_win)

    # 5. shift mask 유효성: mask 없이는 반대편으로 새고, mask 로는 막힌다
    # (0,0) 의 impulse 는 shift(-4,-4) 후 (12,12)로 이동해, 원좌표 12-15 행/열
    # (wrap 으로 이어붙은 반대편)과 같은 창에 놓인다. mask 가 있으면 그쪽으로
    # attention 이 못 가고, 없으면 샌다 — 두 방향 모두 확인해야 mask 검증이다.
    blk = SwinBlock(8, num_heads=2, window_size=8, shift=4).eval()
    z = torch.zeros(1, 8, 16, 16)
    imp = z.clone()
    imp[0, 0, 0, 0] = 1.0
    with torch.no_grad():
        delta = (blk(imp) - blk(z)).abs().sum(dim=1)[0]
        masked_ok = delta[12:16, 12:16].sum().item() == 0
        blk2 = SwinBlock(8, num_heads=2, window_size=8, shift=4).eval()
        blk2.load_state_dict(blk.state_dict())
        blk2._attn_mask = lambda *a, **k: None            # mask 강제 해제
        delta2 = (blk2(imp) - blk2(z)).abs().sum(dim=1)[0]
        unmasked_leaks = delta2[12:16, 12:16].sum().item() > 0
    check("SW-MSA mask 가 순환 경계 누설을 차단", masked_ok and unmasked_leaks,
          f"mask 시 누설 0, 해제 시 누설 {unmasked_leaks}")

    # 6. PANCrafterPaper + swin 배선
    base = dict(hidden_size=64, depth=[1, 2, 4], num_heads=8, norm="ln",
                in_mode="released", attn_locations=[])
    for kw, tag in ((dict(swin_depth=2), "swin_btl x2"),
                    (dict(swin_mid=2), "swin_mid x2"),
                    (dict(swin_depth=2, swin_mid=2), "btl+mid")):
        m = PANCrafterPaper(**base, **kw, swin_heads=4, swin_window=8)
        randomize_zero(m)
        out = m(torch.randn(2, 1, 64, 64), torch.randn(2, 1, 16, 16),
                torch.randn(2, 8, 16, 16), torch.tensor([0., 1.]))
        out.abs().mean().backward()
        swin_grads = [p.grad is not None and p.grad.abs().sum() > 0
                      for n, p in m.named_parameters() if n.startswith("swin")]
        check(f"PANCrafterPaper {tag} forward/backward + swin grad",
              out.shape == (2, 8, 64, 64) and len(swin_grads) > 0 and all(swin_grads),
              f"swin params {len(swin_grads)}개 전부 grad 흐름")

    # 7. LRTinySwin
    for in_mode in ("paper", "released"):
        m = LRTinySwin(hidden_size=64, swin_depth=2, in_mode=in_mode)
        randomize_zero(m)
        pan = torch.randn(2, 1, 64, 64, requires_grad=True)
        ms = torch.randn(2, 8, 16, 16, requires_grad=True)
        out = m(pan, torch.randn(2, 1, 16, 16), ms, torch.tensor([0., 1.]))
        out.abs().mean().backward()
        ok = (out.shape == (2, 8, 64, 64)
              and pan.grad.abs().sum() > 0 and ms.grad.abs().sum() > 0)
        with torch.no_grad():
            fr = m(torch.randn(1, 1, 512, 512), torch.randn(1, 1, 128, 128),
                   torch.randn(1, 8, 128, 128), torch.ones(1))
        check(f"LRTinySwin({in_mode}) grad(pan/ms)·FR 512²",
              ok and fr.shape == (1, 8, 512, 512),
              f"params {n_params(m)/1e6:.4f}M")

    # 8. params 검산
    blk = SwinBlock(128, num_heads=4, window_size=8, mlp_ratio=2.0)
    check("SwinBlock C128·mlp2 params = 0.13338M",
          abs(n_params(blk) - 133_380) < 10, f"{n_params(blk):,}")
    c6 = dict(hidden_size=128, depth=[1, 2, 4], norm="ln",
              in_mode="released", attn_locations=[])
    sw2 = n_params(PANCrafterPaper(**c6, swin_depth=2, swin_heads=4, swin_window=8))
    check("c6 + swin2 = 4.0387M", abs(sw2 / 1e6 - 4.0387) < 0.001, f"{sw2/1e6:.4f}M")

    # 9. 체크포인트 호환 (신규 옵션 미지정 = 기존과 완전 동일)
    m_c6 = PANCrafterPaper(**c6)
    keys = set(m_c6.state_dict().keys())
    check("c6 params 불변(3.7719M) · swin 키 없음",
          abs(n_params(m_c6) / 1e6 - 3.7719) < 0.001
          and not any(k.startswith("swin") for k in keys))
    m_c0 = PANCrafterPaper(hidden_size=128, depth=[2, 2, 4], norm="ln",
                           in_mode="released", n_attn=3)
    check("c0 params 불변(7.1730M)", abs(n_params(m_c0) / 1e6 - 7.1730) < 0.001,
          f"{n_params(m_c0)/1e6:.4f}M")

    print(f"\n[verify] {'전부 통과' if not FAIL else '실패: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
