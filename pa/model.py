"""§2 PAModel: Ŷ = M + Fθ(concat(P̃, M)), M = bicubic↑S (B0 와 같은 F.interpolate), P̃ = W(P, Aφ(P, M)).
backbone 은 B0 의 PANCrafterPaper(in_mode paper, mode_modulation false, mars ms). checkpoint 키 backbone.* / aligner.*
KDV(s2 W112) 확장: aligner=None 이면 Δ=0, sampler=False 면 PAN 을 sampling 하지 않고 그대로 넣는다(A-ID = B0 와 같은 그래프).
delta_override 가 주어지면 sampler 설정과 무관하게 warp 한다(진단용). 기본 인자에서는 이전 동작과 같다."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pa.warp import warp_pan


class PAModel(nn.Module):
    def __init__(self, backbone, aligner=None, aligner_margin=0, sampler=True):
        """aligner_margin: aligner 입력만 고정 내부 crop (PO10 §5, A1–A3 는 0 = 전체 view). U-Net·잔차 base 는 전체 프레임."""
        super().__init__()
        self.backbone = backbone
        self.aligner = aligner
        self.aligner_margin = int(aligner_margin)
        self.sampler = bool(sampler)

    def _view(self, x):
        m = self.aligner_margin
        return x if m == 0 else x[..., m:-m, m:-m]

    def predict_delta(self, pan, ms_base):
        """Δ̂ [B,2] (FP32). aligner 가 없으면 0."""
        if self.aligner is None:
            return torch.zeros(pan.shape[0], 2, device=pan.device, dtype=torch.float32)
        with torch.autocast(device_type=pan.device.type, enabled=False):
            return self.aligner(self._view(pan.float()), self._view(ms_base.float()))

    def forward(self, pan, ms, lpan=None, aligner_enabled=True, delta_override=None):
        """반환 dict(y, ms_base, delta, pan_aligned). delta_override: 진단용(§11.1 zero/wrong-sign) — 학습에서는 None."""
        ms_base = F.interpolate(ms, scale_factor=4, mode="bicubic")
        with torch.autocast(device_type=pan.device.type, enabled=False):           # §4.3 aligner·warp 는 FP32
            if delta_override is not None:
                delta = delta_override.float()
                pan_aligned = warp_pan(pan, delta)
            else:
                delta = self.predict_delta(pan, ms_base) if aligner_enabled else torch.zeros(pan.shape[0], 2, device=pan.device, dtype=torch.float32)
                pan_aligned = warp_pan(pan, delta) if self.sampler else pan.float()   # Δ=0 이어도 sampling (§4.2); A-ID 만 sampler 부재
        pan_in = pan_aligned.to(pan.dtype)
        sw = torch.ones(pan.shape[0], device=pan.device, dtype=pan.dtype)
        res = self.backbone(pan_in, lpan if lpan is not None else pan_in, ms, sw)     # in_mode paper: cat(pan, ↑ms); lpan 미사용
        return dict(y=ms_base + res, ms_base=ms_base, delta=delta, pan_aligned=pan_aligned)
