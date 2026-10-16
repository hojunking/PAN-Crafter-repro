"""Panda40 deployment profiles: actual Direct/S2D A+T or A+S, never KD Teacher.

The existing native GPU profiler and MAC convention are reused unchanged. The
signature/store are separate because the historical store excludes S2D. This
module can be imported by the CPU-only sheet watcher without importing Torch.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from .reporting import atomic_json, canonical_json, digest, utc_now, _config

ROOT = Path(__file__).resolve().parents[1]


def is_s2d(cfg):
    return (cfg.get('aligner_kind') == 's2d' or
            cfg['baseline_resolved']['aligner_architecture']['name'] == 'S2DGlobalPANAligner')


def signature(cfg):
    _config(cfg)
    b = cfg['baseline_resolved']
    architecture = cfg['student_architecture'] if cfg['phase'] == 'stage2' else b['teacher_architecture']
    paths = ['panda40/cost.py', 'experiment_cost/measurement.py',
             'directshift48/method.py', 'directshift48/training.py',
             'panda_method/reconstruction.py', 'panda_method/edge_data.py',
             'panda_method/stages.py', 'pa/warp.py', 'model/pancrafter_paper.py',
             'model/pancrafter.py', 'fh12/model.py']
    if is_s2d(cfg):
        paths += ['s2dshift48/aligner.py', 's2dshift48/stage1.py']
    if cfg['phase'] == 'stage2':
        paths += ['panda_stage2_condq/method.py']
    return {'schema': 'PANDA40_NATIVE_COST_SIGNATURE_V1', 'server': cfg['server'],
        'dataset': cfg['dataset'], 'phase': cfg['phase'],
        'aligner': 'S2DGlobalPANAligner' if is_s2d(cfg) else 'DirectPANAligner',
        'aligner_architecture': deepcopy(b['aligner_architecture']),
        'reconstructor': deepcopy(architecture), 'layout': 'P0', 'base': 'BICUBIC_MS',
        'view_margin': b['aligner_architecture']['view_margin'],
        'input_shapes': {'rr': [1, 1, 256, 256], 'fr': [1, 1, 512, 512]},
        'source_registry_sha256': b.get('source_registry_sha256'),
        'lms_override_sha256': b.get('lms_override_sha256'),
        'batch': 1, 'precision': 'FP32', 'amp': False, 'tf32': False,
        'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}


def find_profile(cfg, root):
    key_sig = signature(cfg)
    key = digest(key_sig)
    path = Path(root)/'cost_profiles'/(key+'.json')
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    proof = value.pop('profile_sha256')
    if (digest(value) != proof or value['signature'] != key_sig or
            value['architecture_key'] != key or value['schema'] != 'PANDA40_COST_V1'):
        raise ValueError('Panda40 cost profile integrity or architecture mismatch')
    from experiment_cost.store import _validate_measurement
    _validate_measurement(value['measurement'], key_sig)
    return {**value, 'profile_sha256': proof, 'path': str(path.resolve())}


def build_deployment(cfg, sources):
    from panda_method.reconstruction import ReconstructionSpec
    from directshift48.training import build_direct_stage1
    b, seeds = cfg['baseline_resolved'], cfg['stream_seeds']
    spec = deepcopy(cfg['student_architecture'] if cfg['phase'] == 'stage2' else b['teacher_architecture'])
    spec.update(input_layout='P0', base_reference='BICUBIC_MS')
    kwargs = dict(teacher_spec=ReconstructionSpec(**spec),
        teacher_seed=seeds['S_init'] if cfg['phase'] == 'stage2' else seeds['T_init'],
        aligner_seed=seeds['A_init'], epsilon_seed=seeds['rr_epsilon'],
        aligner_view_margin=b['aligner_architecture']['view_margin'],
        roi_margin=b['absolute_roi_margin'], boundary_policy=b['boundary_policy'],
        lambda_abs=cfg.get('lambda_abs', 1.), lambda_eps=cfg.get('lambda_epsilon', .001))
    if is_s2d(cfg):
        from s2dshift48.stage1 import build_s2d_stage1
        graph = build_s2d_stage1(sources, mask_policy='UNMASKED', **kwargs)
    else:
        graph = build_direct_stage1(sources, **kwargs)
    if cfg['phase'] != 'stage2':
        return graph
    # Use the exact new Student deployment path, not the historical Stage 1
    # wrapper with different admission/bookkeeping. Attribute 'teacher' is
    # solely the legacy profiler's component name; no donor Teacher is loaded.
    from torch import nn
    from panda_stage2_condq.method import StudentInference
    class StudentDeployment(nn.Module):
        def __init__(self, fresh):
            super().__init__()
            self.aligner, self.teacher = fresh.aligner, fresh.teacher
            self.aligner.requires_grad_(False).eval()
            self.sensor = sources.sensor

        def native_prediction(self, batch):
            return StudentInference(self.aligner, self.teacher, self.sensor)(batch)
    return StudentDeployment(graph)


def ensure_cost(cfg, sources, *, retained_trainer, device, deadline_unix,
                own_gpu_pid=None, cost_root=None):
    """Measure before trainer GPU allocation; return immutable profile envelope.

    cost_root MUST be the campaign root used by publication. CUDA and an idle
    GPU are required; failures leave no invented numeric costs or COMPLETE file.
    """
    if cost_root is None:
        raise ValueError('Explicit campaign cost_root required')
    found = find_profile(cfg, cost_root)
    if found is not None:
        return found
    if deadline_unix is None or deadline_unix-time.time() < 120:
        raise TimeoutError('Not enough time for complete cost measurement')
    import gc
    import os
    import torch
    from donor30.cost import _assert_retained_cpu, _require_empty_cuda, _gpu_pids, _stage2_labels
    from experiment_cost.measurement import profile_stage
    from experiment_cost.store import _validate_measurement
    target = torch.device(device)
    if target.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Only isolated GPU measurements can fill production cost columns')
    _assert_retained_cpu(retained_trainer)
    _require_empty_cuda(target)
    pid = os.getpid() if own_gpu_pid is None else own_gpu_pid
    if type(pid) is not int or pid < 1:
        raise ValueError('Verified host PID required (Docker --pid=host)')
    def isolation():
        other = _gpu_pids()-{pid}
        if other:
            raise RuntimeError('Cost profiling refused concurrent CUDA jobs: '+str(sorted(other)))
    isolation()
    graph = None
    try:
        graph = build_deployment(cfg, sources).to(target)
        measured = profile_stage(graph, sources, device=str(target), deadline_unix=deadline_unix,
            gpu_isolation_assertion={'server': cfg['server'], 'other_compute_pids': [],
                'training_state_retained_in_memory': True, 'retained_train_device': 'cpu',
                'retained_optimizer_updates': retained_trainer.updates, 'profiler_host_pid': pid,
                'mode': 'FRESH_DEPLOYMENT_RETAINED_TRAINER_CPU', 'cuda_allocated_before_graph': 0})
        isolation()
        _assert_retained_cpu(retained_trainer)
        measured['deployment_binding'] = {'phase': cfg['phase'],
            'inference_graph': 'A+warp+S' if cfg['phase'] == 'stage2' else 'A+warp+T',
            'aligner_class': type(graph.aligner).__name__,
            'training_teacher_included': cfg['phase'] == 'donor',
            'excludes': ['training loss', 'KD calibration', 'training integrity bookkeeping'],
            'fresh_weights_not_selected_checkpoint': True}
        if cfg['phase'] == 'stage2':
            _stage2_labels(measured)
        for split in ('rr', 'fr'):
            fields = measured['observations'][split]['sheet_fields']
            fields['Cost_scope'] = fields['Cost_scope'].replace(
                'direct-CNN', type(graph.aligner).__name__)
        sig = signature(cfg)
        _validate_measurement(measured, sig)
        value = {'schema': 'PANDA40_COST_V1', 'signature': sig,
                 'architecture_key': digest(sig), 'measurement': measured, 'created_at': utc_now()}
        value['profile_sha256'] = digest(value)
        path = Path(cost_root)/'cost_profiles'/(value['architecture_key']+'.json')
        atomic_json(path, value)
        return find_profile(cfg, cost_root)
    finally:
        del graph
        gc.collect()
        torch.cuda.empty_cache()


def attach_cost(projected, root, *, require_cost=True):
    """Fill only verified deployment cost; preserve missing timing as missing."""
    from .reporting import INDEX
    from experiment_cost.publication import _validated_fields
    from experiment_cost.timing import timing_for_config
    missing, attached = [], []
    for i, row in enumerate(projected['rows']):
        try:
            note = json.loads(row[INDEX['Original description / notes']])
            cfg = note['extra_parameters']['configuration']
            profile = find_profile(cfg, root)
            if profile is None:
                missing.append({'run_id': cfg['run_id'], 'reason': 'NO_VERIFIED_PANDA40_COST'})
                continue
            split, fields = _validated_fields(profile['measurement'])
            for name, value in fields.items():
                row[INDEX[name]] = value
            timing = timing_for_config(cfg, row[INDEX['Selected_step']], root,
                                       history_roots=[], cost_root=root)
            for name, value in timing['sheet_fields'].items():
                row[INDEX[name]] = value
            evidence = {'profile_sha256': profile['profile_sha256'],
                'architecture_key': profile['architecture_key'], 'profile_path': profile['path'],
                'sheet_fields': fields, 'default_sheet_split': split,
                'all_domains': {s: profile['measurement']['observations'][s]['sheet_fields'] for s in ('rr', 'fr')},
                'deployment_binding': profile['measurement']['deployment_binding'],
                'observed_timing': timing, 'same_graph_not_same_weights': True}
            note['cost_measurement'] = evidence
            short = json.loads(projected['notes'][i]); short['cost_measurement'] = evidence
            row[INDEX['Original description / notes']] = canonical_json(note)
            projected['notes'][i] = canonical_json(short)
            if max(len(row[INDEX['Original description / notes']]), len(projected['notes'][i])) >= 45000:
                raise ValueError('COST_NOTE_TOO_LARGE')
            attached.append({'result_id': row[INDEX['Result_ID']], **evidence})
        except (ValueError, OSError, KeyError, TypeError) as exc:
            projected['conflicts'].append({'reason': 'PANDA40_COST_FAILED:'+str(exc),
                                         'result_id': row[INDEX['Result_ID']]})
    if require_cost:
        projected['conflicts'].extend(missing)
    projected['cost'] = {'bound_count': len(attached), 'bindings': attached,
                         'missing': missing, 'require_cost': require_cost}
