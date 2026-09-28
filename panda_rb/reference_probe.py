"""Fixed validation-only F1 e-conditional q diagnostics, not Student repeats."""
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from fh12.calibration import AXIS16, axis16_q
from fh12.data import build_dataset
from panda_rb.common import ROOT, atomic_json, object_sha, read_json, sha256, utcnow
from panda_rb.plan import campaign_dir, shift_grid
from panda_rb.evaluation import seal, validate_artifacts, save_csv
from panda_rb.stress import shifted_inputs, array_digest

PROBE_SEED = 9280001
MAX_PATCHES = 256


def probe_indices(count):
    if count < 1:
        raise ValueError('Empty validation reference probe')
    return np.sort(np.random.default_rng(PROBE_SEED).choice(count, min(MAX_PATCHES, count), replace=False))


def _correlation(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    result = float(spearmanr(a, b).statistic)
    return result if np.isfinite(result) else None


@torch.no_grad()
def reference_probe(binding_path, root=ROOT, device='cuda', batch_size=16):
    from panda_rb.binding import load_binding
    from reporting_extra.evaluation import _precision
    from panda_rb.common import source_identity
    teacher, _cfg, binding, data, _q = load_binding(binding_path, device=device)
    dataset = build_dataset(data, 'val', root=root)
    indices = probe_indices(len(dataset))
    grid = shift_grid(root)['shifts']
    # Predeclared radius1/radius4 x eight directions: some probe-disjoint,
    # but these are not automatically out-of-distribution training shifts.
    shifts = [x for x in grid if x['radius_hr'] in (1., 4.)]
    identity = dict(schema='PANDA_RB_REFERENCE_QE_v1', binding_sha256=object_sha(binding),
                    data_sha256=object_sha(data), validation_sha256=data['splits']['val']['sha256'],
                    seed=PROBE_SEED, indices=indices.tolist(), indices_sha256=object_sha(indices.tolist()),
                    shifts=shifts, source_identity=source_identity(root), support='full64 validation patch',
                    augmentation='none', test_inputs_used=False)
    directory = campaign_dir(root) / 'common/reference_probe'; path = directory / 'report.json'
    if path.exists():
        report = validate_artifacts(read_json(path), directory)
        if report['identity'] != identity:
            raise ValueError('Fixed validation probe identity changed')
        return report
    started = time.monotonic(); rows = []
    with _precision(identity['source_identity'], device):
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start + batch_size]
            items = [dataset.base(int(i)) for i in ids]
            gt, _lms, ms, lp, pan = [torch.stack([r[k] for r in items]).to(device) for k in range(5)]
            native = teacher(pan, ms, lp)['y'].float()
            e = (native - gt.float()).abs().mean((1, 2, 3))
            q, _qr, correction = axis16_q(teacher, pan, ms)
            texture = ((ms[..., 1:, :] - ms[..., :-1, :]).abs().mean((1, 2, 3))
                       + (ms[..., :, 1:] - ms[..., :, :-1]).abs().mean((1, 2, 3))) / 2
            base = {int(n): dict(scene_index=int(n), e=float(e[j]), q=float(q[j]),
                                 texture_lrms=float(texture[j]), correction_dy=float(correction[j, 0]),
                                 correction_dx=float(correction[j, 1])) for j, n in enumerate(ids)}
            for shift in shifts:
                epsilon = pan.new_tensor([shift['dy'], shift['dx']]).expand(len(ids), 2)
                shifted, shifted_lp = shifted_inputs(pan, lp, epsilon)
                out = teacher(shifted, ms, shifted_lp)
                shifted_e = (out['y'].float() - gt.float()).abs().mean((1, 2, 3))
                response = (out['delta'] + epsilon - correction).abs().sum(1)
                overlap = any(np.allclose([shift['dy'], shift['dx']], point, rtol=0, atol=1e-12) for point in AXIS16)
                for j, n in enumerate(ids):
                    row = dict(base[int(n)], shift_id=shift['id'], radius_hr=shift['radius_hr'],
                               q_probe_overlap=overlap, shifted_e=float(shifted_e[j]),
                               degradation=float(shifted_e[j] - e[j]), relative_response_l1_sum=float(response[j]),
                               prediction_digest=array_digest(out['y'][j].cpu().numpy()))
                    numeric = ('e', 'q', 'shifted_e', 'degradation', 'relative_response_l1_sum',
                               'texture_lrms', 'correction_dy', 'correction_dx')
                    if not all(np.isfinite(row[k]) for k in numeric):
                        row['status'] = 'nonfinite'
                        for key in numeric:
                            if not np.isfinite(row[key]): row[key] = None
                    else: row['status'] = 'ok'
                    rows.append(row)
    # Descriptive validation strata never feed into train weight maps.
    base_rows = {r['scene_index']: r for r in rows}
    ordered = sorted(base_rows, key=lambda i: (float('inf') if base_rows[i]['e'] is None else base_rows[i]['e'], i))
    strata = {index: min(4, rank * 5 // len(ordered)) for rank, index in enumerate(ordered)}
    for row in rows: row['e_quintile'] = strata[row['scene_index']]
    correlations = []; low_high = []
    for shift in shifts:
        for stratum in range(5):
            part = [r for r in rows if r['shift_id'] == shift['id'] and r['e_quintile'] == stratum]
            complete = all(r['status'] == 'ok' for r in part)
            correlations.append(dict(shift_id=shift['id'], e_quintile=stratum, n=len(part),
                 spearman_q_degradation=_correlation([r['q'] for r in part], [r['degradation'] for r in part]) if complete else None,
                 complete=complete))
            if complete and part:
                ordered_q = sorted(part, key=lambda r: (r['q'], r['scene_index']))
                count = max(1, len(ordered_q) // 5)
                groups = {}
                for label, group in (('low_q', ordered_q[:count]), ('high_q', ordered_q[-count:])):
                    groups[label] = dict(n=len(group), scene_indices=[r['scene_index'] for r in group],
                        q_mean=float(np.mean([r['q'] for r in group])),
                        e_mean=float(np.mean([r['e'] for r in group])),
                        degradation_mean=float(np.mean([r['degradation'] for r in group])),
                        texture_lrms_mean=float(np.mean([r['texture_lrms'] for r in group])),
                        correction_abs_max=float(max(max(abs(r['correction_dy']), abs(r['correction_dx'])) for r in group)))
            else: groups = None
            low_high.append(dict(shift_id=shift['id'], e_quintile=stratum, n=len(part),
                                 policy='bottom/top fifth q within fixed-e stratum; source-index tie break',
                                 groups=groups, complete=complete))
    save_csv(directory / 'per_patch_shift.csv', rows)
    report = seal(dict(complete=True, identity=identity, n_reference_patches=len(indices),
                       n_independent_students=0, n_independent_teachers=1, correlations=correlations,
                       conditional_q_low_high=low_high,
                       n_nonfinite_observations=sum(r['status'] != 'ok' for r in rows),
                       file_hashes={'per_patch_shift.csv': sha256(directory / 'per_patch_shift.csv')},
                       elapsed_seconds=time.monotonic() - started, completed_at_utc=utcnow(),
                       note='Fixed F1 descriptive validation probe; not calibration, causal proof, or Student-seed replication.'))
    atomic_json(path, report)
    return validate_artifacts(report, directory)
