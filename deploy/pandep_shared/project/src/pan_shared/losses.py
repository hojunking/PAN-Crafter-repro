"""Plan v4's sole training objective; no clipping, auxiliary terms or /3."""
import torch


def mean_l1(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and GT must share BCHW geometry")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError("DIVERGED: nonfinite output/target")
    return (prediction - target).abs().mean()
