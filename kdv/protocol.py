"""입력 프로토콜 (plan §5.4): I-A(native 만) · I-N(native/corrupt update 1:1, corrupt 는 P_ε = W(P, ε) 를 raw view 로) · I-NATIVE-TRANSFER(N donor, corruption 없음).
ε 은 pa.offset.sample_offsets(원판 면적 균등, 전용 CPU generator). 두 번째(learned) warp 는 forward 에서 순차 raster warp 로 수행한다 (W(W(P,ε),c)) — 합성 warp 로 바꾸지 않는다."""
import torch

from pa.offset import sample_offsets
from pa.warp import warp_pan

PROTOCOLS = ('I-A', 'I-N', 'I-NATIVE-TRANSFER', 'I-AEQ')   # I-AEQ (NF16 §4.2): 복원은 매 update native, 홀수 update 에 aligner 전용 offset 연습 (P_ε 는 U-Net 에 가지 않는다)


def is_corrupt_update(protocol, update_index):
    return protocol == 'I-N' and (int(update_index) % 2 == 1)


def prepare_view(pan, protocol, update_index, radius_hr, generator):
    """반환 (P_view, eps[B,2], corrupted). native 면 P_view = pan (같은 tensor), eps = 0."""
    if protocol not in PROTOCOLS:
        raise ValueError(protocol)
    B = pan.shape[0]
    if not is_corrupt_update(protocol, update_index):
        return pan, torch.zeros(B, 2, device=pan.device, dtype=torch.float32), False
    eps = sample_offsets(B, float(radius_hr), generator).to(pan.device)
    with torch.no_grad(), torch.autocast(device_type=pan.device.type, enabled=False):
        p_eps = warp_pan(pan.float(), eps)                   # 첫 raster warp (학습 대상 아님)
    return p_eps.to(pan.dtype), eps, True
