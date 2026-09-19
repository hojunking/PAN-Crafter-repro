"""Independent coordinate checks and bounded, read-only QB phase evidence.

A decimation phase index is an HR sample, not an entire LR sample. Neither
this fact nor a reconstruction residual establishes native physical PAN shift.
"""
import numpy as np

from qg40.common import check_deadline, object_sha


def phase_coordinate_checks():
    """P0 analytic ramp/impulse checks of the actual LP/MS decimation helpers."""
    from tools.repair_lpan import make_lpan
    from tools.repair_qb_ms import mtf_down
    size = 96
    yy, xx = np.indices((size, size), dtype=np.float64)
    ramp = yy + 2 * xx
    identity_kernel = np.zeros((41, 41, 4), np.float64)
    identity_kernel[20, 20] = 1
    expected = ramp[2::4, 2::4]
    ms = mtf_down(np.repeat(ramp[None], 4, axis=0), identity_kernel)
    if not np.allclose(ms, expected[None], atol=1e-10, rtol=0):
        raise ValueError('P0 MS decimation does not sample HR coordinates 2+4j')
    lp = make_lpan(ramp[None, None])[0, 0]
    # Symmetric normalized filtering preserves an affine ramp away from borders.
    interior = (slice(5, 19), slice(5, 19))
    if not np.allclose(lp[interior], expected[interior], atol=1e-10, rtol=0):
        raise ValueError('P0 Gaussian LP ramp phase is not HR [2::4,2::4]')
    impulse = np.zeros((size, size), np.float64)
    impulse[42, 46] = 1
    expected_impulse = np.zeros((4, 24, 24), np.float64)
    expected_impulse[:, 10, 11] = 1
    impulse_ms = mtf_down(np.repeat(impulse[None], 4, axis=0), identity_kernel)
    if not np.allclose(impulse_ms, expected_impulse, atol=1e-12, rtol=0):
        raise ValueError('P0 MS impulse sample position differs from analytic phase')
    coord = np.arange(-20, 21, dtype=np.float64)
    gaussian = np.exp(-coord ** 2 / (2 * 1.98 ** 2))
    gaussian /= gaussian.sum()
    full_expected = np.zeros_like(impulse)
    full_expected[22:63, 26:67] = gaussian[:, None] * gaussian[None, :]
    impulse_lp = make_lpan(impulse[None, None])[0, 0]
    if not np.allclose(impulse_lp, full_expected[2::4, 2::4], atol=1e-14, rtol=0):
        raise ValueError('P0 LP impulse differs from analytic sigma1.98/k41/phase2')
    return dict(schema='QG40_PHASE_COORDINATES_v1', status='PASS', complete=True,
                checks=['MS identity-filter ramp', 'LP affine ramp interior',
                        'MS sampled impulse', 'LP analytic Gaussian impulse'],
                ratio=4, phase=[2, 2], one_phase_index_hr_pixels=1,
                one_phase_index_lr_spacing=.25, one_lr_index_hr_pixels=4,
                interpretation='Coordinate-unit test, not measured native physical displacement',
                legacy_note='The historical LR1px/HR4px phrase cannot be inferred from phase(1,2) versus(2,2) alone')


def qb_phase_residuals(manifest, *, deadline_utc=None, max_samples=16, seed=5678):
    """P1 raw train/val phase residuals; never regenerate or shift an input."""
    import h5py
    from scipy.signal import fftconvolve
    from qg40.data import canonical_band_indices
    from tools.metrics.eval_fr import genmtf_matlab, GNYQ_TABLE
    if manifest['sensor'] != 'QB':
        return dict(status='NOT_APPLICABLE', sensor=manifest['sensor'])
    if max_samples < 1:
        raise ValueError('Positive bounded phase sample count required')
    order = canonical_band_indices(manifest['band_order'])
    kernel = genmtf_matlab(GNYQ_TABLE['QB'], 4, 41)
    phases = [(y, x) for y in range(4) for x in range(4)]
    result = dict(schema='QG40_QB_PHASE_RESIDUAL_v1', status='MEASURED', p0_gate=False,
                  dataset_manifest_sha256=object_sha(manifest), sampling_seed=seed,
                  units='raw DN mean absolute reconstruction residual', splits={},
                  interpretation='Bounded raw-MS versus filtered-GT diagnostic; not native PAN displacement GT')
    for split in ('train', 'val'):
        check_deadline(deadline_utc)
        proof = manifest['source_provenance']
        raw_path = proof[f'raw_{split}_path']
        with h5py.File(raw_path, 'r') as source:
            n = len(source['gt'])
            ids = np.sort(np.random.default_rng(seed).choice(n, min(n, max_samples), replace=False))
            rows = []
            for index in ids:
                check_deadline(deadline_utc)
                gt = np.asarray(source['gt'][int(index)], np.float64)[list(order)]
                actual = np.asarray(source['ms'][int(index)], np.float64)[list(order)]
                filtered = np.stack([fftconvolve(np.pad(gt[b], 20, mode='edge'),
                                      kernel[:, :, b][::-1, ::-1], mode='valid') for b in range(4)])
                errors = [float(np.abs(filtered[:, y::4, x::4] - actual).mean()) for y, x in phases]
                if not np.isfinite(errors).all():
                    raise ValueError('Nonfinite QB phase residual')
                best = int(np.argmin(errors))
                rows.append(dict(sample_id=int(index), mae_by_phase=errors,
                                 minimum_residual_phase=list(phases[best]),
                                 phase2_mae=errors[phases.index((2, 2))]))
        result['splits'][split] = dict(source_path=raw_path,
            declared_source_sha256=proof[f'raw_{split}_sha256'], sample_ids=ids.tolist(),
            population_count=n, full_population=len(ids) == n, phases=[list(p) for p in phases],
            rows=rows, read_only_audit=True)
    return result
