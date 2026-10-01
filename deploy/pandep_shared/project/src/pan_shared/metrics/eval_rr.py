"""Pinned pure numerical primitives. See vendor_reference/metric_sources.json for provenance.
No legacy trainer/controller imports; evaluation only.
"""
import numpy as np

def sam(sr: np.ndarray, gt: np.ndarray) -> float:
    """Spectral Angle Mapper (degree). 0에 가까울수록 분광 보존이 좋다."""
    num = (sr * gt).sum(axis=2)
    den = np.linalg.norm(sr, axis=2) * np.linalg.norm(gt, axis=2)
    valid = den > 1e-8
    cos = np.clip(num[valid] / den[valid], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())

def ergas(sr: np.ndarray, gt: np.ndarray, ratio: int = 4) -> float:
    """Erreur Relative Globale Adimensionnelle de Synthese."""
    rmse_b = np.sqrt(((sr - gt) ** 2).mean(axis=(0, 1)))
    mu_b = gt.mean(axis=(0, 1))
    return float(100.0 / ratio * np.sqrt(np.mean((rmse_b / np.maximum(mu_b, 1e-8)) ** 2)))
