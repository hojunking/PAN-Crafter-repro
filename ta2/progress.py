"""Fixed-checkpoint AXIS16 progress probes and scalar training-curve records.

These are diagnostic observations, not official FR quality or checkpoints to
select by. Canonical sample0..3 is fixed before results; no feeder/RNG advance,
augmentation, resampling of the MS reference, or official quality mask occurs.
"""
from __future__ import annotations

import math
import numbers

import torch
from torch.nn import functional as F

from .common import digest
from .diagnostics import PROBE_SEED, correction_statistics, measure_responses, response_grid, summarize_responses
from .evaluation import preserve_runtime, seal
from .model import state_hash
from .probes import OFFLINE, make_correction
from .reporting import make_record, response_records


def progress_steps(total_updates):
    if total_updates not in (50000, 100000):
        raise ValueError('Only registered50K/100K progress points are supported')
    scale = total_updates // 50000
    return tuple(scale * value for value in (1010, 25250, 50000))


@torch.no_grad()
def progress_protocol(model, bundle, cfg, checkpoint_step, device='cuda'):
    if checkpoint_step not in progress_steps(cfg['total_updates']):
        raise ValueError('Unregistered fixed progress point')
    offsets = [row for row in response_grid() if row['probe'] == 'AXIS16']
    model_sha = state_hash(model)
    response, native, inputs = [], {}, {}
    with preserve_runtime(model):
        correct = make_correction(model, bundle, cfg)
        for split in ('train', 'val', 'rr', 'fr'):
            if len(bundle.datasets[split]) < 4:
                raise ValueError('Fixed progress protocol requires canonical first4 samples of every split')
            samples, corrections = [], []
            for index in range(4):
                data = bundle.batch(split, [index], device=device)
                pan, lms = (data['pan'] + 1) * .5, (data['lms'] + 1) * .5
                expected = 256 if split == 'rr' else 512 if split == 'fr' else 64
                if pan.shape != (1, 1, expected, expected):
                    raise ValueError('Progress data geometry differs from canonical train64/RR256/FR512')
                if cfg['a_reference'] == 'BICUBIC_MS':
                    reference = (F.interpolate(data['ms'], scale_factor=4, mode='bicubic', align_corners=False) + 1) * .5
                else:
                    reference = lms
                if model.policy == 'TRAIN_GT_PROXY' and split == 'train':
                    reference = (data['gt'] + 1) * .5
                if not bool(torch.isfinite(pan).all()) or not bool(torch.isfinite(reference).all()):
                    raise FloatingPointError('Nonfinite progress observation')
                samples.append((index, pan, reference))
                corrections.append(correct(split, pan, reference, index, 'REESTIMATE_CURRENT_INPUT')[0].cpu().tolist())
            inputs[split] = dict(sample_ids=[0, 1, 2, 3], shape=list(samples[0][1].shape),
                                 A_reference='GT_TRAIN_PRIVILEGED_PROXY' if model.policy == 'TRAIN_GT_PROXY' and split == 'train' else cfg['a_reference'])
            fn = lambda p, r, i, mode: correct(split, p, r, i, mode)
            response.extend(measure_responses(fn, samples, split=split, grid=offsets,
                measurement_domain=f'CANONICAL_NATIVE_{samples[0][1].shape[-1]}')['rows'])
            if model.policy in OFFLINE:
                response.extend(measure_responses(fn, samples, split=split, grid=offsets,
                    mode='HOLD_NATIVE_CACHE', measurement_domain=f'CANONICAL_NATIVE_{samples[0][1].shape[-1]}')['rows'])
            native[split] = correction_statistics(corrections)
    if state_hash(model) != model_sha:
        raise ValueError('Progress probe altered model parameters/buffers')
    return seal(dict(schema='TA2_FIXED_PROGRESS_RESPONSE_v1', status='COMPLETE',
                     completed_step=int(checkpoint_step), config_sha256=digest(cfg),
                     data_sha256=digest(bundle.manifest), model_state_sha256=model_sha, probe_seed=PROBE_SEED,
                     grid_sha256=digest(offsets), sampling='canonical_first4_each_split_no_augmentation',
                     inputs=inputs, response_rows=response, response_summary=summarize_responses(response),
                     correction_summary=native, independent_training_seeds=1,
                     official_metric_masking=False, quality_selection_eligible=False,
                     gt_in_official_inference=False, native_stream_advanced=False,
                     note='Small fixed progress diagnostic; final protocol provides fuller coverage'))


def _leaves(value, prefix=''):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, (prefix+'/' if prefix else '')+str(key))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _leaves(child, prefix+'/'+str(index))
    elif value is None or (isinstance(value, numbers.Real) and not isinstance(value, bool)):
        if value is not None and not math.isfinite(value):
            raise ValueError('Nonfinite curve measurement: '+prefix)
        yield prefix, value


def progress_records(context, training_log, probe):
    """Use actual full-batch log values and measured fixed-probe corrections.

    Weighted loss values are already weighted: never multiply by lambda again.
    Missing A updates or undefined correlations are blank with an explicit N/A
    reason. Training mini-batch structure stats are identified separately from
    canonical probe geometry; they are not official quality measurements.
    """
    step = int(probe['completed_step'])
    if int(training_log['completed_step']) != step:
        raise ValueError('Training curve and progress probe refer to different updates')
    context = dict(context, completed_step=step, selector='FIXED_PROGRESS',
                   Probe_seed=probe['probe_seed'])
    result = []
    base = dict(Split_probe='train/effective48_actual_update', Step=step,
                Struct_loss=training_log.get('struct'),
                Epsilon_effective_count=training_log.get('epsilon_effective_count'),
                Effective_updates=step, Status='MEASURED',
                Reason='Actual update log; structural support is training-only; official FR metric unmasked')
    keys = ('total', 'rec', 'epsilon', 'struct', 'scale', 'weighted_epsilon', 'weighted_struct',
            'weighted_scale', 'elapsed_seconds', 'grad_norm', 'actual_parameter_update',
            'actual_a_group_update', 'learning_rates', 'epsilon_effective_count')
    for key in keys:
        if key not in training_log:
            continue
        for path, value in _leaves(training_log[key], key):
            result.append(make_record(context, 'CURVE', 'training_update/'+path,
                dict(base, Metric_name=path, Estimate=value,
                     Reason=base['Reason'] if value is not None else 'N/A: parameter group absent or measurement undefined')))
    for index, sample in enumerate(training_log.get('structural_statistics', [])):
        # Preserve exact reported per-microbatch values rather than inventing a
        # globally reweighted valid-band correlation from insufficient counts.
        for path, value in _leaves(sample):
            result.append(make_record(context, 'CURVE', f'training_structure/microbatch{index}/{path}',
                dict(base, Metric_name=f'structural_microbatch{index}/{path}', Estimate=value,
                     Low_texture_fraction=sample.get('low_texture_fraction'),
                     Reason='Actual microbatch target-valid rho/rho2/support; sample_weight separately recorded')))
    # response_records also renders correction means/norms as numeric rows.
    for row in response_records(context, probe):
        result.append(make_record(context, 'CURVE', 'progress/'+row['Record_ID'], row))
    return result
