"""Student/Teacher forward 계약 (plan §5, §16.3–16.4).

  M = bicubic↑S · Student: Δ_S = A_S^V(P_view, M) · P̃_S = W(P_view, Δ_S) · Ŷ_S = M + F_S([P̃_S, M])
  Teacher(no_grad): 자기 frozen aligner 로 P̃_T (A-FR 이고 같은 aligner·view 면 correction·P̃ 를 한 번 계산해 공유) · Ŷ_T = M + F_T([P̃_T, M])
A-FR: aligner·warp 는 no_grad (Student parameter 가 없다) — U-Net forward 는 graph 유지. A-ID: sampler 없음(P̃ = P_view).
PAModel.forward 와 같은 연산 순서/정밀도(aligner·warp FP32) 를 쓴다 — 평가 경로와 학습 경로가 수치적으로 같다."""
import torch
import torch.nn.functional as F

from pa.warp import warp_pan


def _correction_and_pan(model, pan_view, ms_base, live):
    with torch.autocast(device_type=pan_view.device.type, enabled=False):
        if model.aligner is None:
            delta = torch.zeros(pan_view.shape[0], 2, device=pan_view.device, dtype=torch.float32)
            pan_al = warp_pan(pan_view.float(), delta) if model.sampler else pan_view.float()
            return delta, pan_al
        if live:
            delta = model.predict_delta(pan_view, ms_base)
            return delta, warp_pan(pan_view.float(), delta)
        with torch.no_grad():
            delta = model.predict_delta(pan_view, ms_base)
            return delta, warp_pan(pan_view.float(), delta)


def _restore(backbone, pan_al, lpan, ms, dtype):
    sw = torch.ones(pan_al.shape[0], device=pan_al.device, dtype=dtype)
    p = pan_al.to(dtype)
    return backbone(p, lpan if lpan is not None else p, ms, sw)


def kdv_forward(student, teacher, pan_view, ms, lpan, *, share_correction, teacher_needed, aligner_live):
    """반환 dict(y, residual, delta, pan_aligned, ms_base, y_t, delta_t, pan_t). y_t 등은 teacher_needed=False 면 None."""
    ms_base = F.interpolate(ms, scale_factor=4, mode='bicubic')
    delta_s, pan_s = _correction_and_pan(student, pan_view, ms_base, live=(aligner_live and student.aligner is not None))
    res_s = _restore(student.backbone, pan_s, lpan, ms, pan_view.dtype)
    out = dict(y=ms_base + res_s, residual=res_s, delta=delta_s, pan_aligned=pan_s, ms_base=ms_base, y_t=None, delta_t=None, pan_t=None)
    if teacher_needed:
        if teacher is None:
            raise RuntimeError('teacher_needed but teacher is None')
        with torch.no_grad():
            if share_correction:
                delta_t, pan_t = delta_s.detach(), pan_s.detach()
            else:
                delta_t, pan_t = _correction_and_pan(teacher, pan_view, ms_base, live=False)
            res_t = _restore(teacher.backbone, pan_t, lpan, ms, pan_view.dtype)
            out.update(y_t=ms_base + res_t, delta_t=delta_t, pan_t=pan_t)
    return out
