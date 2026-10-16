"""Read-only PAN-only epsilon response, shared by both Aligner architectures."""
import torch

from directshift48.method import DirectPANAligner
from pa.warp import warp_pan, warp_support_mask


@torch.no_grad()
def epsilon_response(aligner, pan, native_lms, epsilons):
    """Measure u(W(P,eps), LMS) + eps - u(P,LMS), not scene displacement GT.

    Each [dy,dx] epsilon is broadcast to the input batch. All shifted PAN views
    must retain original-observation support in crop8. No Teacher/GT, fitting,
    optimizer, model-mode changes or random sampling are performed.
    """
    if not isinstance(aligner, DirectPANAligner):
        raise TypeError('Explicit DirectShift-compatible Aligner required')
    if (not isinstance(epsilons, torch.Tensor) or epsilons.ndim != 2
            or epsilons.shape[0] < 1 or epsilons.shape[1] != 2
            or epsilons.dtype != torch.float32 or epsilons.device != pan.device
            or not bool(torch.isfinite(epsilons).all())
            or bool((epsilons.norm(dim=1) > 3.).any())):
        raise ValueError('Finite FP32 [N,2] current-PAN epsilons within radius3 required')
    native = aligner(pan, native_lms).detach()
    rows = []
    for shift in epsilons:
        epsilon = shift.expand(len(pan), -1)
        support = warp_support_mask(*pan.shape[-2:], epsilon)
        margin = aligner.view_margin
        if not bool(support[..., margin:-margin, margin:-margin].all()):
            raise ValueError('Synthetic epsilon support does not contain clean crop8')
        changed = aligner(warp_pan(pan, epsilon), native_lms).detach()
        residual = changed + epsilon - native
        rows.append({'epsilon_dy_dx': shift.cpu().tolist(), 'native_u': native.cpu().tolist(),
                     'shifted_u': changed.cpu().tolist(), 'residual_dy_dx': residual.cpu().tolist(),
                     'l1_axis_mean_px': float(residual.abs().mean()),
                     'l2_sample_mean_px': float(residual.norm(dim=1).mean())})
    return {'scope': 'PAN_ONLY_SYNTHETIC_EQUIVARIANCE_NOT_TRUE_DISPLACEMENT_ACCURACY',
            'target': 'u_shifted_plus_epsilon_equals_u_native',
            'units': 'current_PAN_pixels_no_x4', 'architecture': aligner.manifest(), 'rows': rows}
