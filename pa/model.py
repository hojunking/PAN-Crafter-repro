"""§2 PAModel: Ŷ = M + Fθ(concat(P̃, M)), M = bicubic↑S (B0 와 같은 F.interpolate), P̃ = W(P, Aφ(P, M)).
backbone 은 B0 의 PANCrafterPaper(in_mode paper, mode_modulation false, mars ms). checkpoint 키 backbone.* / aligner.*"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pa.warp import warp_pan


class PAModel(nn.Module):
    def __init__(self, backbone, aligner):
        super().__init__()
        self.backbone = backbone
        self.aligner = aligner

    def forward(self, pan, ms, lpan=None, aligner_enabled=True, delta_override=None):
        """반환 dict(y, ms_base, delta, pan_aligned). delta_override: 진단용(§11.1 zero/wrong-sign) — 학습에서는 None."""
        ms_base = F.interpolate(ms, scale_factor=4, mode="bicubic")
        with torch.autocast(device_type=pan.device.type, enabled=False):           # §4.3 aligner·warp 는 FP32
            if delta_override is not None:
                delta = delta_override.float()
            elif aligner_enabled:
                delta = self.aligner(pan.float(), ms_base.float())
            else:
                delta = torch.zeros(pan.shape[0], 2, device=pan.device, dtype=torch.float32)
            pan_aligned = warp_pan(pan, delta)                                      # Δ=0 이어도 항상 sampling (§4.2)
        pan_in = pan_aligned.to(pan.dtype)
        sw = torch.ones(pan.shape[0], device=pan.device, dtype=pan.dtype)
        res = self.backbone(pan_in, lpan if lpan is not None else pan_in, ms, sw)     # in_mode paper: cat(pan, ↑ms); lpan 미사용
        return dict(y=ms_base + res, ms_base=ms_base, delta=delta, pan_aligned=pan_aligned)
