#!/usr/bin/env python
"""아키텍처 사전 smoke 검증 — 체인이 2시간짜리 슬롯을 태우기 전에 분 단위로 잡는다.

계획서 §8.1 (research_log/2026-08-28_architecture-search-24h-plan.md) 의 점검 항목:
  1. config 로드 -> 모델 build -> params 실측
  2. 학습 형상 forward + backward (dual 이면 switch 0/1 혼합, mars: ms 면 1만)
     -> 출력 shape (B,8,64,64) · finite · grad finite
  3. FR 형상(512², WV3 full-res) no-grad forward -> shape · finite
     (PixelUnshuffle 배수 문제·해상도 의존 버그 검출)
  4. 실배치 OOM 검사: batch_size x MARs 복제(dual 이면 2배)로 forward+backward+
     AdamW step 1회 — 학습 시작 후에야 OOM 나는 사고(s1_A2 d444) 재발 방지
  5. peak VRAM · 대략적인 step 시간

zero_module 함정: 출력 conv 가 0 으로 초기화돼 있어 그대로면 상류 grad 가 전부 0
이라 backward 검사가 공허하다. 검사 전에 0 파라미터를 난수화한다 (CLAUDE.md 함정 목록).

사용:
  python tools/smoke_cases.py <config이름> [<config이름> ...]
종료코드: 하나라도 실패면 1 (체인은 이걸 보고 해당 case 를 학습 없이 FAILED 처리)
"""
import os
import sys
import time
import importlib

import torch
import torch.nn as nn
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class ConfigMissing(Exception):
    """config 파일 부재 — 빌드 실패와 달리 일시 사유다 (예: git pull 전).

    체인은 이 경우(exit 2)를 실패 원장에 남기지 않고 이번 패스만 건너뛴다.
    """


def load_cfg(name):
    for p in (os.path.join(ROOT, "config", f"{name}.yaml"),
              os.path.join(ROOT, "config", f"pancrafter_{name}.yaml")):
        if os.path.exists(p):
            return yaml.safe_load(open(p))
    raise ConfigMissing(f"config 없음: {name}")


def build(cfg):
    mod, cls = cfg["model"].rsplit(".", 1)
    M = getattr(importlib.import_module(mod), cls)
    return M(**cfg["model_args"])


def randomize_zero_params(m):
    """zero_module 로 0 초기화된 파라미터를 난수화 — backward 가 실제로 흐르게."""
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad and p.dtype.is_floating_point and p.abs().sum() == 0:
                p.normal_(0, 1e-3)


def check_trainer_extras(cfg):
    """신규 trainer 의 실제 실패 지점을 학습 전에 재현한다.

    kd: teacher 를 **실제로 로딩**(CPU) — 존재/키/uncertainty 호환을 전부 검증.
    mutual: peer 2벌 build — 구성 오류·대략적 2x 메모리 사실을 드러낸다.
    """
    tr = cfg.get("trainer")
    if tr == "kd":
        variant = (cfg.get("kd_args") or {}).get("variant", "k0")
        if variant == "k0":
            return ""
        ck = cfg.get("teacher_checkpoint")
        assert ck and os.path.isdir(os.path.join(ROOT, ck)), f"teacher checkpoint 없음: {ck}"
        from train_kd import load_teacher
        t, has_unc = load_teacher(os.path.join(ROOT, cfg["teacher_config"]),
                                  os.path.join(ROOT, ck),
                                  torch.device("cpu"), torch.float32)
        del t
        if variant in ("k2", "k3", "k4", "k5"):
            assert has_unc, f"{variant} 는 uncertainty teacher 필요 — {ck} 에 head 없음"
        return f" teacher={'unc' if has_unc else 'plain'}"
    if tr == "mutual":
        b = build(cfg)     # peer_b 구성 재현 (같은 model_args)
        del b
        return " 2-peer"
    return ""


def smoke_one(name, dev):
    cfg = load_cfg(name)
    extra_note = check_trainer_extras(cfg)
    m = build(cfg).to(dev)
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    # config 에 expect_params_m 이 있으면 실측과 대조한다 — 옵션 하나(예:
    # cm3a_pan_branch) 빠뜨려 딴 모델을 학습하는 사고를 학습 전에 잡는다.
    exp = cfg.get("expect_params_m")
    if exp is not None:
        assert abs(n_params - float(exp)) / float(exp) < 0.005, \
            f"params {n_params:.4f}M ≠ 기대 {exp}M — config 옵션 누락 의심"
    randomize_zero_params(m)
    dual = cfg.get("mars", "dual") != "ms"

    # 학습 형상: PAN 64², LPAN/MS 16² (feeder ms_size=16 기준)
    B = 4
    pan = torch.randn(B, 1, 64, 64, device=dev)
    lpan = torch.randn(B, 1, 16, 16, device=dev)
    ms = torch.randn(B, cfg.get("num_bands", 8), 16, 16, device=dev)
    switch = (torch.tensor([0., 0., 1., 1.], device=dev) if dual
              else torch.ones(B, device=dev))

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    out = m(pan, lpan, ms, switch)
    assert out.shape == (B, cfg.get("num_bands", 8), 64, 64), f"학습 출력 shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "학습 출력에 NaN/Inf"
    out.abs().mean().backward()
    bad = [n for n, p in m.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"grad NaN/Inf: {bad[:3]}"
    n_grad = sum(1 for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_grad > 0, "0 이 아닌 grad 가 없다 — backward 가 흐르지 않음"
    step_ms = (time.time() - t0) * 1000

    # FR 형상: WV3 full-res 는 PAN 512², LPAN/MS 128²
    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        out_fr = m(torch.randn(1, 1, 512, 512, device=dev),
                   torch.randn(1, 1, 128, 128, device=dev),
                   torch.randn(1, cfg.get("num_bands", 8), 128, 128, device=dev),
                   torch.ones(1, device=dev))
    assert out_fr.shape[-2:] == (512, 512), f"FR 출력 shape {tuple(out_fr.shape)}"
    assert torch.isfinite(out_fr).all(), "FR 출력에 NaN/Inf"

    # 실배치 OOM 검사: 학습과 같은 실효 배치(batch_size x MARs 복제)로
    # forward+backward+AdamW step 을 1회 돌려 peak VRAM 을 잰다.
    # s1_A2 가 d444·batch48 에서 학습 시작 후에야 OOM 난 사고의 재발 방지 —
    # GPU 가 유휴한 case 시작 직전에 미리 터뜨린다. optimizer state 까지 잡는다.
    peak_train = 0.0
    if dev.type == "cuda":
        m.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        Bt = int(cfg.get("batch_size", 48)) * (2 if dual else 1)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01)
        pan_t = torch.randn(Bt, 1, 64, 64, device=dev)
        lpan_t = torch.randn(Bt, 1, 16, 16, device=dev)
        ms_t = torch.randn(Bt, cfg.get("num_bands", 8), 16, 16, device=dev)
        sw_t = (torch.arange(Bt, device=dev) >= Bt // 2).float() if dual \
            else torch.ones(Bt, device=dev)
        out_t = m(pan_t, lpan_t, ms_t, sw_t)
        out_t.abs().mean().backward()
        opt.step()
        peak_train = torch.cuda.max_memory_allocated(dev) / 2**20
        del opt, pan_t, lpan_t, ms_t, sw_t, out_t

    peak = (torch.cuda.max_memory_allocated(dev) / 2**20) if dev.type == "cuda" else 0
    del m
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return n_params, step_ms, max(peak, peak_train), ("dual" if dual else "ms") + extra_note


def main():
    names = sys.argv[1:]
    if not names:
        print(__doc__)
        sys.exit(2)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    failed, missing = [], []
    print(f"[smoke] device={dev}  cases={len(names)}")
    for name in names:
        try:
            p, ms_t, peak, mars = smoke_one(name, dev)
            print(f"[smoke] OK   {name:26s} {p:7.4f}M  step≈{ms_t:6.0f}ms  "
                  f"peakVRAM {peak:6.0f}MB  mars={mars}")
        except ConfigMissing as e:
            print(f"[smoke] MISS {name:26s} {e}")
            missing.append(name)
        except Exception as e:
            print(f"[smoke] FAIL {name:26s} {type(e).__name__}: {e}")
            failed.append(name)
    if failed:
        print(f"[smoke] 실패 {len(failed)}: {' '.join(failed)}")
        sys.exit(1)
    if missing:
        print(f"[smoke] config 없음 {len(missing)}: {' '.join(missing)}")
        sys.exit(2)
    print("[smoke] 전부 통과")


if __name__ == "__main__":
    main()
