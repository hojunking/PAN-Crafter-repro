"""Core metrics from the user's 51_sharpness_qb_gf2.py (2026-11-06).

Identical full-image masks, RGB mean, Sobel nearest boundary, filters and
scene-wise arithmetic aggregation. This is a diagnostic, not an FR GT metric.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter, sobel

PROTOCOL = {
    'source': 'user-provided results_log/paper_figs/sources/51_sharpness_qb_gf2.py',
    'luminance': 'float32 arithmetic RGB mean /255',
    'PAN_edges': 'Sobel magnitude >= scene quantile .90, no border crop',
    'edge_pm2': 'mean maximum_filter(output Sobel magnitude,5) on PAN_edges',
    'HF': 'mean abs(L-Gaussian(L,1.5)), whole scene',
    'flat': 'PAN local std9 <= scene quantile .20',
    'flat_grain': 'mean output local std5 on flat',
    'aggregation': 'arithmetic mean of 20 scene means',
}


def luminance(rgb):
    return np.asarray(rgb, np.float32).mean(2) / 255


def gradient(x):
    return np.hypot(sobel(x, 1, mode='nearest'), sobel(x, 0, mode='nearest'))


def local_std(x, k):
    m = uniform_filter(x, k)
    return np.sqrt(np.maximum(uniform_filter(x*x, k)-m*m, 0))


def measure(rgb, pan_rgb):
    pan, lum = luminance(pan_rgb), luminance(rgb)
    if pan.shape != lum.shape or pan.shape != (512, 512):
        raise ValueError('Matched full FR512 images required')
    g = gradient(pan); edges = g >= np.quantile(g, .90)
    ps = local_std(pan, 9); flat = ps <= np.quantile(ps, .20)
    g_lum = gradient(lum)
    return dict(edge_pm2=float(maximum_filter(g_lum, 5)[edges].mean()),
                edge_exact=float(g_lum[edges].mean()),
                hf=float(np.abs(lum-gaussian_filter(lum, 1.5)).mean()),
                flat_grain=float(local_std(lum, 5)[flat].mean()))


def rgb_display(raw, peak):
    x = np.clip(np.asarray(raw, np.float32), 0, peak)[[2, 1, 0]]
    flat = x.reshape(3, -1); ordered = np.sort(flat, 1)
    lo = ordered[:, int(flat.shape[1]*.01), None]
    hi = ordered[:, int(flat.shape[1]*.99), None]
    rgb = np.clip((flat-lo)/np.maximum(hi-lo, 1e-6), 0, 1)
    return (rgb.reshape(x.shape).transpose(1, 2, 0)*255).round().astype(np.uint8)


def pan_display(raw):
    raw = np.asarray(raw, np.float64).squeeze()
    lo, hi = np.percentile(raw, [1, 99])
    gray = np.rint(np.clip((raw-lo)/max(hi-lo, 1e-12), 0, 1)*255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)
