"""Native PAN Gaussian LP and normalized PLH, separate from evaluation MTF."""
import numpy as np
import torch
from torch.nn import functional as F

SENSOR_PROFILES = {
    'WV3': {'bands': 8, 'max_dn': 2047, 'band_order': ['Coastal', 'B', 'G', 'Y', 'R', 'RE', 'NIR1', 'NIR2']},
    'GF2': {'bands': 4, 'max_dn': 1023, 'band_order': ['B', 'G', 'R', 'NIR']},
    'QB': {'bands': 4, 'max_dn': 2047, 'band_order': ['B', 'G', 'R', 'NIR']},
}
LP_RECIPE = {'id': 'PANDEP_GAUSS198_K41_REPLICATE_PHASE2_v1', 'sigma': 1.98,
             'kernel': 41, 'padding': 'replicate', 'decimation': '[2::4,2::4]',
             'ratio': 4, 'calculation_dtype': 'float64', 'cache_dtype': 'float32'}
AUGMENTATION = {'id': 'FIXED_HV_THEN_ROT4_v1', 'hflip': True, 'vflip': True,
                'rotations': [0, 1, 2, 3], 'crop': False}


def make_lpan(pan):
    """Exact fixed repair_lpan recipe: float64 calculation, float32 cache."""
    import cv2
    pan = np.asarray(pan, dtype=np.float64)
    if pan.ndim != 4 or pan.shape[1] != 1 or any(n % 4 for n in pan.shape[-2:]):
        raise ValueError('LP requires N1HW PAN with ratio-four dimensions')
    if not np.isfinite(pan).all():
        raise ValueError('Nonfinite native PAN')
    kernel = cv2.getGaussianKernel(41, 1.98)
    kernel = kernel @ kernel.T
    return np.stack([cv2.filter2D(p[0], -1, kernel, borderType=cv2.BORDER_REPLICATE)
                     [2::4, 2::4][None] for p in pan]).astype(np.float32)


def normalize_dn(array, sensor):
    maximum = SENSOR_PROFILES[sensor]['max_dn']
    if isinstance(array, torch.Tensor):
        return array.to(torch.float32) * (2.0 / maximum) - 1.0
    return np.asarray(array, dtype=np.float32) * np.float32(2.0 / maximum) - np.float32(1)


def up4(value):
    return F.interpolate(value, scale_factor=4, mode='bicubic', align_corners=False)


def plh_inputs(sensor_id, pan, ms, lpan):
    profile = SENSOR_PROFILES[sensor_id]
    if pan.ndim != 4 or ms.ndim != 4 or lpan.ndim != 4:
        raise ValueError('BCHW inputs required')
    b, channels, height, width = pan.shape
    if (channels != 1 or height % 4 or width % 4 or
        tuple(ms.shape) != (b, profile['bands'], height // 4, width // 4) or
        tuple(lpan.shape) != (b, 1, height // 4, width // 4)):
        raise ValueError('Sensor band count / ratio-four geometry mismatch')
    if len({pan.device, ms.device, lpan.device}) != 1 or len({pan.dtype, ms.dtype, lpan.dtype}) != 1:
        raise ValueError('Inputs must share dtype and device')
    base, low = up4(ms), up4(lpan)
    return torch.cat((pan, low, pan - low, base), dim=1), base


def fixed_view(array, rotation):
    if not isinstance(rotation, (int, np.integer)) or rotation not in range(4):
        raise ValueError('Rotation must be integer 0,1,2,3')
    return np.rot90(array[..., ::-1, ::-1], int(rotation), axes=(-2, -1)).copy()
