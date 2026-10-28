"""Pinned Lanczos-4 for Stage1, epsilon injection, Teacher and Student.

Adapted from the independently gradchecked panda_warp_candidates windowed sinc.
No dynamic import of an experimental implementation enters this campaign.
"""
from copy import deepcopy
import torch
from .contract import WARP_MANIFEST


def warp_manifest():
    return deepcopy(WARP_MANIFEST)


def boundary_manifest(epsilon_radius=3.):
    """Registered R3/R4 injection fits the unchanged eight-pixel input view."""
    if epsilon_radius not in (3.,4.):
        raise ValueError("Registered epsilon radius 3 or 4 required")
    from directshift48.method import boundary_manifest as original_boundary
    value = original_boundary()
    value.pop('bicubic_conservative_tap_margin')
    value.update(epsilon_max_radius_pan_px=epsilon_radius, lanczos_conservative_tap_margin=4,
                 minimum_clean_input_margin=int(epsilon_radius)+4,
                 aligner_view_rule=f'4*ceil(({epsilon_radius:g}+4)/4)=8_before_any_spatial_statistics')
    return value


def mask_manifest():
    from donor30.masking import mask_manifest as original_mask
    value = original_mask()
    value.update(policy='DETACHED_LANCZOS8_SUPPORT_PER_SAMPLE_MEAN_V2',
                 support='all_floor_q_minus3_through_plus4_taps_inside_original_PAN',
                 coordinates='original_PAN_sampling_grid_before_reflection_extension')
    return value


def _validate(image, delta):
    if (not isinstance(image, torch.Tensor) or image.ndim != 4
            or image.dtype not in (torch.float32, torch.float64)
            or min(image.shape) < 1 or min(image.shape[-2:]) < 2):
        raise ValueError('Finite FP32/FP64 BCHW image with H,W>=2 required')
    if (not isinstance(delta, torch.Tensor) or delta.shape != (len(image), 2)
            or delta.dtype not in (torch.float32, torch.float64)
            or delta.device != image.device or not bool(torch.isfinite(delta).all())
            or not bool(torch.isfinite(image).all())):
        raise ValueError('Finite floating B2 current-PAN-pixel shift on image device required')
    return delta.to(image.dtype)


def _axis(image, shift, axis):
    integer = torch.floor(shift)
    fraction = shift-integer
    taps = torch.arange(-3, 5, device=image.device, dtype=image.dtype)
    z = fraction[:, None]-taps[None]
    weights = torch.sinc(z)*torch.sinc(z/4)*(z.abs()<4)
    weights = weights/weights.sum(1, keepdim=True)
    length = image.shape[axis]
    coordinate = torch.arange(length, device=image.device)[None]+integer.to(torch.int64)[:, None]
    result = torch.zeros_like(image)
    for i, offset in enumerate(range(-3, 5)):
        indices = torch.remainder(coordinate+offset, 2*(length-1))
        indices = torch.where(indices<length, indices, 2*(length-1)-indices)
        gather = (indices[:, None, None, :] if axis == -1 else indices[:, None, :, None]).expand_as(image)
        result = result+image.gather(axis,gather)*weights[:, i, None, None, None]
    return result


def warp_pan(image, delta):
    """W(P,d)[y,x]=P[y+dy,x+dx], normalized separable 8-tap Lanczos."""
    d = _validate(image, delta)
    return _axis(_axis(image,d[:,0],-2),d[:,1],-1)


def warp_support_mask(height, width, delta):
    """Conservative support: all eight taps on each axis read observed pixels.

    Endpoint taps are included even if zero at an exact integer displacement.
    ROI8 with the one-pixel Scharr stencil requires floor(delta) in [-4,3].
    This deliberately rejects unsupported full shifts; no clipping or adaptive
    ROI hides a larger Aligner displacement.
    """
    if (type(height) is not int or type(width) is not int or min(height,width)<2
            or not isinstance(delta,torch.Tensor) or delta.ndim!=2 or delta.shape[1]!=2
            or delta.dtype not in (torch.float32,torch.float64)
            or not bool(torch.isfinite(delta).all())):
        raise ValueError('Positive image dimensions and finite floating B2 shift required')
    with torch.no_grad():
        y=torch.arange(height,device=delta.device,dtype=delta.dtype)[None]
        x=torch.arange(width,device=delta.device,dtype=delta.dtype)[None]
        fy=torch.floor(y+delta[:,0,None]); fx=torch.floor(x+delta[:,1,None])
        oky=(fy-3>=0)&(fy+4<height); okx=(fx-3>=0)&(fx+4<width)
    return (oky[:,:,None]&okx[:,None,:])[:,None]
