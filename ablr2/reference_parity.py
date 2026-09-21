"""Reference-scoped online AXIS16 checks of the actual four-view train cache.

This is a numerical integrity check, not a quality threshold. Constant or high
positive q remains valid. Only the fixed sample is re-evaluated, never all N.
"""
from contextlib import contextmanager
import random

import numpy as np
import torch

from fh12.calibration import AXIS16, axis16_q
from fh12.data import AUGMENTATION
from ablr2.common import check_deadline, object_sha

PARITY_SEED, PARITY_COUNT = 271828, 16
ATOL, RTOL = 3e-6, 2e-5


def parity_indices(count):
    if count < PARITY_COUNT:
        raise ValueError('Q-cache parity requires 16 distinct train IDs')
    return np.random.default_rng(PARITY_SEED).choice(count, PARITY_COUNT, replace=False)


@contextmanager
def isolated_teacher(model):
    """Do not perturb training RNG, nested modes, gradients or freeze flags."""
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    modes = [(module, module.training) for module in model.modules()]
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    try:
        model.eval().requires_grad_(False)
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])


def identity_for(manifest):
    return {key: manifest[key] for key in ('reference_id', 'teacher_run_id',
        'teacher_checkpoint_sha256', 'q_cache_sha256')} | {
        'source_content_sha256': manifest['source_identity']['content_sha256']}


def verify_q_cache(model, dataset, q, *, reference_identity, data_identity,
                   device='cuda', deadline=None, synthetic_test=False):
    count = dataset.base_count
    ids = parity_indices(count)
    if (not dataset.has_gt or len(dataset) != count or getattr(model, 'bands', None) not in (4, 8)
            or np.asarray(q).shape != (count, 4) or not np.isfinite(q).all()
            or (np.asarray(q) < 0).any() or (not synthetic_test and count < 3072)):
        raise ValueError('Invalid C4/C8 base-train q-cache parity inputs')
    measured = np.empty((PARITY_COUNT, 4), dtype=np.float32)
    with isolated_teacher(model):
        for view in range(4):
            check_deadline(deadline)
            rows = [dataset.get_view(int(index), view, augment=True) for index in ids]
            for index, row in zip(ids, rows):
                metadata = torch.as_tensor(row[-1]).cpu().tolist()
                if metadata != [int(index), view, 1, 1]:
                    raise ValueError('Q-cache online view/base-ID metadata mismatch')
            ms = torch.stack([row[2] for row in rows]).to(device)
            pan = torch.stack([row[4] for row in rows]).to(device)
            if ms.shape[1] != model.bands:
                raise ValueError('Online q-check sensor band count differs')
            online, _, _ = axis16_q(model, pan, ms)
            measured[:, view] = online.detach().cpu().numpy()
    cached = np.asarray(q)[ids].copy()
    if not np.isfinite(measured).all() or not np.allclose(measured, cached, atol=ATOL, rtol=RTOL):
        error = float(np.max(np.abs(measured-cached)))
        raise ValueError(f"Reference {reference_identity.get('reference_id')} q-cache/online AXIS16 parity failed: {error}")
    check_deadline(deadline)
    return dict(schema='ABLR2_QCACHE_ONLINE_PARITY_v1', status='MEASURED', passed=True,
        synthetic_test=bool(synthetic_test), reference_identity=reference_identity,
        data_identity=data_identity, base_count=count, seed=PARITY_SEED,
        indices=ids.tolist(), indices_sha256=object_sha(ids.tolist()), views=[0, 1, 2, 3],
        augmentation=AUGMENTATION, augmentation_sha256=object_sha(AUGMENTATION),
        AXIS16=[list(x) for x in AXIS16], units='HR_PAN_PIXELS',
        cached_q=cached.tolist(), online_q=measured.tolist(),
        max_abs_error=float(np.max(np.abs(measured-cached))), atol=ATOL, rtol=RTOL,
        note='Actual frozen Teacher and local fixed-HV/rotation views; no quality gate')


def validate_receipt(receipt, *, reference_identity, data_identity, q):
    """Reject stale, incomplete, synthetic, looser-tolerance or fabricated PASS-only data."""
    ids = parity_indices(len(q))
    expected = dict(schema='ABLR2_QCACHE_ONLINE_PARITY_v1', status='MEASURED', passed=True,
        synthetic_test=False, reference_identity=reference_identity, data_identity=data_identity,
        base_count=len(q), seed=PARITY_SEED, indices=ids.tolist(),
        indices_sha256=object_sha(ids.tolist()), views=[0, 1, 2, 3],
        augmentation=AUGMENTATION, augmentation_sha256=object_sha(AUGMENTATION),
        AXIS16=[list(x) for x in AXIS16], units='HR_PAN_PIXELS', atol=ATOL, rtol=RTOL)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('Q-cache parity receipt identity/protocol mismatch')
    cached, online = np.asarray(receipt.get('cached_q')), np.asarray(receipt.get('online_q'))
    if (cached.shape != (PARITY_COUNT, 4) or online.shape != cached.shape
            or not np.isfinite(cached).all() or not np.isfinite(online).all()
            or not np.array_equal(cached, np.asarray(q)[ids])
            or not np.allclose(cached, online, atol=ATOL, rtol=RTOL)
            or receipt.get('max_abs_error') != float(np.max(np.abs(cached-online)))):
        raise ValueError('Q-cache parity receipt measurements invalid')
