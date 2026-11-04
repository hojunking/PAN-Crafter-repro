"""Actual A+Student deployment, isolated batch-one RR/FR cost measurements."""
import json
import os
from pathlib import Path
import time

from directshift48.reporting import atomic_json, utc_now
from panda_wv3_main_rgb import plan
from ta2.evaluation import file_sha

SCHEMA = 'WV3_MAIN_RGB_DEPLOYMENT_COST_V1'


def graph_signature(cfg):
    plan.validate(cfg)
    if cfg['stage'] != 'stage2':
        raise ValueError('Only the deployed Stage2 Aligner+Student is profiled')
    # LR, loss coefficients, horizon and seed do not change this inference graph.
    keys = ('dataset', 'architecture', 'student_input', 'warp', 'scale', 'upsampler',
            'precision', 'source_registry_sha256', 'inference_tta')
    from panda_wv3_main_rgb.release import verify
    return dict({k: cfg[k] for k in keys}, source_release=verify()['source_sha256'])


def deployment(cfg, state):
    import torch
    from torch import nn
    from panda_wv3_ablation.model import Stage2, build_aligner
    from panda_tune30.reconstruction import StudentReconstructionSpec, build_reconstructor

    class Deployment(nn.Module):
        # Execute the same native inference methods as the evaluated main model.
        corrected = Stage2.corrected
        reconstruct = Stage2.reconstruct
        native_prediction = Stage2.native_prediction

        def __init__(self):
            super().__init__()
            self.cfg = cfg
            self.aligner = build_aligner(cfg['stream_seeds']['A_init'])
            spec = StudentReconstructionSpec(8,112,(1,2,3),'P0','BICUBIC_MS',2)
            self.student = build_reconstructor(spec, seed=cfg['stream_seeds']['S_init'],
                role='S', sampling_variant='UBIL', input_variant='CORRECTED_DIFF')
            for role in ('aligner', 'student'):
                prefix = role + '.'
                selected = {k[len(prefix):]: v for k,v in state.items() if k.startswith(prefix)}
                getattr(self, role).load_state_dict(selected, strict=True)
            self.requires_grad_(False).eval()

        @property
        def teacher(self):
            # Compatibility with the legacy profiler's component naming only.
            return self.student

    return Deployment()


def measure(root, cfg, step, *, device='cuda:0'):
    import torch
    from donor30.cost import _gpu_pids, _require_empty_cuda, _stage2_labels
    from experiment_cost.measurement import profile_stage
    from panda_tune30.cost.profiling import account_student_lanczos4
    from panda_wv3_main_rgb.assets import sources
    from panda_wv3_main_rgb.release import verify
    from panda_wv3_main_rgb.evaluation import verify_record
    verify(environment=True)
    root = Path(root)
    if _gpu_pids() - {os.getpid()}:
        raise RuntimeError('Pause training before measuring isolated deployment cost')
    _require_empty_cuda(device)
    path = root/'runs'/cfg['run_id']/f'evaluation_{step:07d}.json'
    record = json.loads(path.read_text()); verify_record(record, cfg)
    checkpoint = plan.local_path(record['checkpoint']['path'])
    if file_sha(checkpoint) != record['checkpoint']['file_sha256']:
        raise ValueError('Profile checkpoint SHA mismatch')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['config'] != cfg or saved['updates'] != step or saved['model_sha256'] != record['model_sha256']:
        raise ValueError('Profile checkpoint is not the requested evaluated model')
    graph = deployment(cfg, saved['model']).to(device)
    source, _ = sources()
    measured = profile_stage(graph, source, device=device, warmup=10, repeats=50,
        deadline_unix=time.time()+300,
        gpu_isolation_assertion={'profiler_host_pid':os.getpid(), 'other_compute_pids':[],
            'mode':'STANDALONE_ISOLATED_NO_TRAINER_GPU', 'server':cfg['server']})
    if _gpu_pids() - {os.getpid()}:
        raise RuntimeError('Concurrent GPU computation invalidates the profile')
    _stage2_labels(measured)
    account_student_lanczos4(measured)
    measured.update(weight_basis='ACTUAL_EVALUATED_CHECKPOINT',
        weight_dependence='Measured checkpoint below; shared as an architecture cost across identical graphs, not per-checkpoint timing.',
        memory_definition='Peak allocated MiB (2^20 bytes), A+S+resident inputs+output/workspace; no training Teacher/optimizer; not reserved memory or process RSS.')
    for split, obs in measured['observations'].items():
        accounting = obs['accounting']
        obs['sheet_fields']['FLOPs(G)'] = accounting['arithmetic_flops_covered_estimate']/1e9
        obs['sheet_fields']['Cost_scope'] = (
            f'{split.upper()} {obs["pan_shape"][-2:]} batch1 FP32, RTX GPU specified in cost evidence; '
            'A + Lanczos4 + corrected-PAN/difference + interp23 + W112 D123 UBIL Student; '
            'synchronized wall latency median (10 warmup/50 repeats), resident inputs, excludes data/H2D/metrics/Teacher; '
            'Mem(MB) denotes peak allocated MiB; FLOPs(G) = 2*MAC / 1e9 PARTIAL Conv/MatMul/resampling estimate; '
            'omits normalization/activation/weight-generation/indexing arithmetic; actual checkpoint graph shared across recipes')
        accounting['publication_convention'] = '1 MAC = 2 arithmetic FLOPs; partial estimate, not a full operator count'
    measured['sheet_cost'] = measured['observations']['rr']['sheet_fields']
    result = {'schema':SCHEMA, 'created_at':utc_now(), 'graph':graph_signature(cfg),
        'checkpoint':record['checkpoint'], 'run_id':cfg['run_id'], 'step':step,
        'evaluation_sha256':file_sha(path), 'measurement':measured}
    result['profile_sha256'] = plan.digest(result)
    atomic_json(root/'reporting/cost.json', result)
    return result


def load(root, cfg):
    path = Path(root)/'reporting/cost.json'
    value = json.loads(path.read_text())
    if (value.get('schema') != SCHEMA or value.get('profile_sha256') !=
            plan.digest({k:v for k,v in value.items() if k!='profile_sha256'})
            or value.get('graph') != graph_signature(cfg)
            or value['measurement']['status'] != 'COMPLETE'):
        raise ValueError('Missing or incompatible deployment cost evidence')
    return value
