"""AlignedModel — backbone(PANCrafterPaper) 앞/뒤의 정렬 wrapper (계획 §8·§9).

case 는 config `alignment:` 의 5개 key 로만 갈린다 (계획 §16):
    delta_source        zero | cache | trainable
    alpha               0.0 ~ 1.0
    output_frame        M | P
    inverse_location    none | final_output | loss_branch
    trainable_shift_net bool
여기에 upsampler(interp23tap | bicubic_phase2) 가 붙는다 — resample.py 머리말 참고.

코어 U-Net 은 수정하지 않는다: backbone.forward(..., x_in=x11) 로 11ch 입력만 바꿔 넣는다.
11ch 순서는 배포/재구성 코드가 기대하는 (PAN, up(LPAN), PAN-up(LPAN), up(MS)) 그대로다.
"""
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from align.resample import upsample_shift, warp_hr, border_mask
from align.shiftnet import GlobalShiftNet, structural_input


@dataclass
class AlignCfg:
    upsampler: str = "interp23tap"
    phase: float = 2.0
    padding_mode: str = "border"
    delta_source: str = "zero"
    alpha: float = 0.0
    output_frame: str = "M"
    inverse_location: str = "none"
    trainable_shift_net: bool = False
    cache_dir: str = "outputs/global_shift_cache"
    max_global_shift_lr: float = 1.0
    shiftnet_pretrained: str = "outputs/global_shift_cache/shiftnet_pretrained.pt"
    lambda_shift: float = 0.1
    lambda_zero: float = 0.01
    shiftnet_lr: float = 1.0e-5
    shiftnet_wd: float = 1.0e-4
    scale: int = 4
    border_extra: int = 2

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        known = {k for k in cls.__dataclass_fields__}
        bad = set(d) - known
        assert not bad, f"alignment: 모르는 key {sorted(bad)}"
        c = cls(**d)
        c.validate()
        return c

    def validate(self):
        assert self.upsampler in ("interp23tap", "bicubic_phase2")
        assert self.delta_source in ("zero", "cache", "trainable")
        assert self.output_frame in ("M", "P")
        assert self.inverse_location in ("none", "final_output", "loss_branch")
        assert 0.0 <= float(self.alpha) <= 1.0
        if self.delta_source == "zero":
            assert float(self.alpha) == 0.0 and self.inverse_location == "none" and self.output_frame == "M", \
                "delta_source=zero 는 P0 — alpha 0 · inverse none · M frame"
        if self.inverse_location == "none":
            assert self.output_frame == "M", "inverse 없이 P frame 출력은 정의되지 않는다"
        if self.inverse_location == "final_output":
            assert self.output_frame == "M" and float(self.alpha) == 1.0, "round-trip 은 full shift + M 출력"
        if self.inverse_location == "loss_branch":
            assert self.output_frame == "P" and float(self.alpha) == 1.0, "dual-frame 은 full shift + P 출력"
        assert self.trainable_shift_net == (self.delta_source == "trainable"), \
            "trainable_shift_net 은 delta_source=trainable 과 함께만"

    @property
    def full_shift(self):
        return self.inverse_location != "none"

    def as_dict(self):
        return asdict(self)


class AlignedModel(nn.Module):
    def __init__(self, backbone, cfg: AlignCfg):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.shift_net = GlobalShiftNet(cfg.max_global_shift_lr) if cfg.trainable_shift_net else None
        self.n_inverse_calls = 0          # T08 진단용 — PAN mode 에서 0 이어야 한다

    # ---- shift ------------------------------------------------------------
    def predict_delta(self, lpan, ms):
        assert self.shift_net is not None
        return self.shift_net(structural_input(lpan, ms))

    # ---- views ------------------------------------------------------------
    def _up(self, lr, delta_lr, alpha):
        return upsample_shift(lr, delta_lr, alpha, kind=self.cfg.upsampler, scale=self.cfg.scale,
                              phase=self.cfg.phase, padding_mode=self.cfg.padding_mode)

    def build_views(self, pan, lpan, ms, delta_lr):
        """공통 입력 view. PAN 계열 3ch 은 shift 하지 않는다 (§4.4)."""
        B = ms.shape[0]
        zeros = torch.zeros(B, 2, dtype=ms.dtype, device=ms.device)
        lpan_hr = self._up(lpan, zeros, 0.0)
        pan_hf = pan - lpan_hr
        if self.cfg.full_shift:                           # C1/C3/C4: cond == base == P-frame MS
            ms_cond = self._up(ms, delta_lr, 1.0)
            ms_base = ms_cond
        elif self.cfg.delta_source == "zero":            # P0
            ms_cond = self._up(ms, zeros, 0.0)
            ms_base = ms_cond
        else:                                             # C2: cond 만 α·Δ, base 는 M-frame
            ms_cond = self._up(ms, delta_lr, float(self.cfg.alpha))
            ms_base = self._up(ms, zeros, 0.0)
        x11 = torch.cat((pan, lpan_hr, pan_hf, ms_cond), dim=1)
        return dict(x11=x11, ms_cond_hr=ms_cond, ms_base_hr=ms_base, lpan_hr=lpan_hr, pan_hf=pan_hf)

    def residual(self, x11, switch):
        return self.backbone(None, None, None, switch, x_in=x11)

    # ---- MS mode 마무리 ------------------------------------------------------
    def finalize_ms(self, ms_base_hr, res, delta_lr):
        """반환: y_final(배포 출력), y_loss(GT loss 뷰), mask(loss 용), y_pan, y_ms."""
        y = ms_base_hr + res
        H, W = y.shape[-2:]
        one = torch.ones(y.shape[0], 1, H, W, dtype=y.dtype, device=y.device)
        if self.cfg.inverse_location == "none":           # P0 / C2 : 전부 M-frame
            return dict(y_final=y, y_loss=y, mask=one, y_pan=None, y_ms=y)
        self.n_inverse_calls += 1
        y_ms = warp_hr(y, -self.cfg.scale * delta_lr, self.cfg.padding_mode)
        mask = border_mask(delta_lr.to(y.dtype), H, W, self.cfg.scale, self.cfg.border_extra)
        if self.cfg.inverse_location == "final_output":   # C1 / C4A : 최종 출력도 M-frame
            return dict(y_final=y_ms, y_loss=y_ms, mask=mask, y_pan=y, y_ms=y_ms)
        return dict(y_final=y, y_loss=y_ms, mask=mask, y_pan=y, y_ms=y_ms)   # C3 / C4B : P-frame 출력

    # ---- 단일 mode 편의 forward (평가용) ---------------------------------------
    @torch.no_grad()
    def infer_ms(self, pan, lpan, ms, delta_lr):
        v = self.build_views(pan, lpan, ms, delta_lr)
        sw = torch.ones(ms.shape[0], dtype=ms.dtype, device=ms.device)
        out = self.finalize_ms(v["ms_base_hr"], self.residual(v["x11"], sw), delta_lr)
        out.update(v)
        return out

    def forward(self, pan, lpan, ms, switch, delta_lr=None):
        """accelerate/smoke 호환용: MS 최종 출력만 돌려준다."""
        if delta_lr is None:
            delta_lr = torch.zeros(ms.shape[0], 2, dtype=ms.dtype, device=ms.device)
        v = self.build_views(pan, lpan, ms, delta_lr)
        res = self.residual(v["x11"], switch)
        return self.finalize_ms(v["ms_base_hr"], res, delta_lr)["y_final"]
