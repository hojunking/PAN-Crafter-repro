"""S1-only finite donor -> five-scene PANPV campaign, with explicit revision.

No legacy controller, stopped experiment, input H5 or original donor is changed.
The original unavailable s2 donor is replaced only under the 2026-10-04 user
authorization, before looking at any WV2 model-comparison result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import os
from pathlib import Path
import shutil
import socket

from .common import atomic_json, canonical_sha, file_sha, load_donor, load_json


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if load_json(path) != value:
            raise ValueError('Immutable pipeline artifact differs: ' + str(path))
    else:
        atomic_json(path, value)


def source_hashes(repo):
    paths = sorted((repo / 'panpv').glob('*.py'))
    paths += [repo / 'panpv/start_s1.sh', repo / 'panpv/Dockerfile']
    paths += sorted((repo / 'deploy/pandep_shared/project/src/pan_shared').rglob('*.py'))
    return {str(p.relative_to(repo)): file_sha(p) for p in paths}


def configure_torch():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')


def adaptation_preflight(scene, donor, output, device='cuda', micro_batch=12):
    """Disposable weights: FT0 parity and matched effective48 accumulation."""
    import torch
    from .training import masked_l1_sum
    from .v3_data import RRPatchV3
    from deploy.pandep_shared.project.src.pan_shared.model import SingleSensorPLHUNet
    output = Path(output)
    if output.exists():
        previous = load_json(output)
        if (previous.get('status') != 'PASS' or previous.get('donor_sha256') != donor['sha256']
                or previous.get('micro_batch') != micro_batch or previous.get('scene_sha256') != scene.sha256):
            raise ValueError('Preflight identity mismatch')
        return previous
    configure_torch()
    if micro_batch != 12:
        raise ValueError('This revision preregisters micro12 x4; no automatic tuning')
    ds = RRPatchV3(scene, 'train')
    if ds.provisional:
        raise ValueError('Verified normalized scene required')
    items = [ds[i % len(ds)] for i in range(48)]
    batch = {key: torch.stack([item[key] for item in items]).to(device)
             for key in ('pan', 'ms', 'lpan', 'gt', 'mask')}
    a = load_donor(donor['path'], donor['sha256']).to(device).eval()
    b = load_donor(donor['path'], donor['sha256']).to(device).eval()
    with torch.no_grad():
        sample = [batch[key][:2] for key in ('pan', 'ms', 'lpan')]
        ft0_error = float((a('WV3', *sample) - b('WV3', *sample)).abs().max())
        sc = SingleSensorPLHUNet('WV3', width=104, depth=(1, 2, 2), seed=261002).to(device).eval()
        sc0_error = float((sc('WV3', *sample) - torch.nn.functional.interpolate(
            sample[1], scale_factor=4, mode='bicubic', align_corners=False)).abs().max())
    del b, sc
    if ft0_error != 0 or sc0_error != 0:
        raise AssertionError('FT0=ZS / SC0=MS-up identity failed')
    gradients, losses = [], []
    denominator = int(batch['mask'].sum()) * 8
    for micro in (12, 24):
        a.zero_grad(set_to_none=True)
        loss_value = 0.0
        for start in range(0, 48, micro):
            prediction = a('WV3', *(batch[k][start:start+micro] for k in ('pan', 'ms', 'lpan')))
            numerator, _ = masked_l1_sum(prediction, batch['gt'][start:start+micro], batch['mask'][start:start+micro])
            loss = numerator / denominator
            loss.backward()
            loss_value += float(loss.detach())
        state = {}
        for name, parameter in a.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError('Disconnected/nonfinite FT gradient: ' + name)
            state[name] = parameter.grad.detach().cpu().clone()
        gradients.append(state)
        losses.append(loss_value)
    max_error = max(float((gradients[0][k] - gradients[1][k]).abs().max()) for k in gradients[0])
    gradient_pass = all(torch.allclose(gradients[0][k], gradients[1][k], rtol=1e-3, atol=1e-5)
                        for k in gradients[0])
    if not gradient_pass or abs(losses[0]-losses[1]) > 1e-5:
        raise AssertionError('effective48 accumulation parity failed')
    before = {k: v.detach().clone() for k, v in a.named_parameters()}
    optimizer = torch.optim.AdamW(a.parameters(), lr=1e-5, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    optimizer.step()
    if not any(not torch.equal(before[k], v) for k, v in a.named_parameters()):
        raise AssertionError('Disposable FT update did not change parameters')
    if not all(torch.isfinite(v).all() for v in a.parameters()):
        raise AssertionError('Nonfinite disposable FT update')
    result = dict(status='PASS', donor_sha256=donor['sha256'], scene_sha256=scene.sha256,
                  micro_batch=12, effective_batch=48, ft0_zs_parity=True, ft0_max_abs=ft0_error,
                  sc0_ms_up_max_abs=sc0_error, microbatch_verified=True,
                  microbatch_comparison=[12, 24], gradient_max_abs=max_error,
                  gradient_tolerance={'rtol': 1e-3, 'atol': 1e-5}, losses=losses,
                  one_disposable_ft_update=True, smoke_weights_saved=False,
                  device=device, amp=False, tf32=False, precision='FP32',
                  gpu_name=torch.cuda.get_device_name() if device == 'cuda' else None)
    atomic_json(output, result)
    del a, optimizer, batch, gradients, before
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return result


def run_pipeline(args):
    from .v3_data import SceneV3
    from .pretrain import run_pretrain
    from .campaign_eval import freeze_protocol
    from .campaign import run_campaign
    repo, work = args.repo_root.resolve(strict=True), args.work_root.resolve()
    preparation = load_json(args.preparation)
    if preparation['state'] != 'NEW_DATA_VERIFIED_FORMAL_CAMPAIGN_NOT_STARTED':
        raise ValueError('Complete v3 input verification is required')
    for record in preparation.get('evidence', {}).values():
        if file_sha(record['path']) != record['sha256']:
            raise ValueError('Prior input-verification evidence changed')
    input_root = Path(preparation['input_root']).resolve(strict=True)
    if (work == repo or not work.is_relative_to(repo / 'work_dir')
            or work.is_relative_to(input_root) or input_root.is_relative_to(work)):
        raise ValueError('Use a new isolated work_dir campaign root')
    if args.device != 'cuda':
        raise ValueError('Production pipeline requires the measured s1 CUDA runtime')
    expected_host = 'knuvi-hojun-local'
    if os.environ.get('PANPV_HOSTNAME', socket.gethostname()) != expected_host:
        raise ValueError('This pipeline is confined to s1 ' + expected_host)
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / '.pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    configure_torch()
    try:
        reserve = 10 if (work / 'registry.json').exists() else 65
        if shutil.disk_usage(work).free < reserve * 1024**3:
            raise RuntimeError(f'Insufficient output reserve: need {reserve} GiB for this tile-scope campaign')
        identity = dict(schema='PANPV_PIPELINE_REVISION_v1', campaign_id='PANPV_S1_WV2_20261004_R1',
                        execution_server='s1', target_sensor='WV2', donor_sensor='WV3',
                        authorization='2026-10-04 user: find similar local donor, otherwise train new donor and proceed',
                        donor_revision='FRESH_WV3_ONLY_W104_D122_PLH_NOALIGN_NOKD_EXACT50K',
                        old_unavailable_donor_sha256='6ad7cff8fc60542ab687a405f7b55430926942f008a65345de99fa69fa8b3846',
                        donor_seed=271001, adaptation_seed=261002,
                        scope='RR_SPATIAL_HOLDOUT_AND_FIXED_NATIVE_FR_TILES_NOT_FULL_SCENE',
                        preparation_path=str(args.preparation.resolve()), preparation_sha256=file_sha(args.preparation),
                        source_sha256=source_hashes(repo), micro_batch=12, effective_batch=48,
                        runtime_image=os.environ.get('PANPV_IMAGE_ID', 'NOT_SUPPLIED'),
                        device=args.device, input_root=str(input_root), work_root=str(work))
        immutable_json(work / 'pipeline_manifest.json', identity)
        def status(stage, **fields):
            atomic_json(work / 'pipeline_status.json', dict(stage=stage,
                        updated_utc=datetime.now(timezone.utc).isoformat(), **fields))
            print('PANPV PIPELINE:', stage, fields, flush=True)
        scenes, protocols = [], {}
        for entry in preparation['scenes']:
            relative = 'scenes/' + entry['scene_id'] + '.h5'
            with SceneV3(input_root / relative, expected_sha256=entry['sha256']) as scene:
                summary = scene.summary()
                if summary['patch_index_sha256'] != entry['patch_index_sha256']:
                    raise ValueError('Frozen split changed')
                scenes.append(dict(scene_id=entry['scene_id'], file=relative,
                                   sha256=entry['sha256'], patch_index_sha256=entry['patch_index_sha256']))
                protocols[entry['scene_id']] = freeze_protocol(scene)
                status('FREEZING_PROTOCOL', scenes_frozen=len(protocols))
        protocol_path = work / 'protocol.json'
        immutable_json(protocol_path, dict(scenes=protocols))
        status('PRETRAINING_DONOR')
        donor = run_pretrain(repo, work / 'donor_pretrain', device=args.device, micro_batch=12)
        if not donor or donor.get('status') in ('PAUSED', 'STOPPED_BY_OPERATOR', 'PAUSED_SAFE'):
            status('PAUSED_DONOR_PRETRAINING')
            return 130
        if donor.get('status') != 'READY':
            raise RuntimeError('Donor did not finish safely: ' + str(donor.get('status')))
        status('DONOR_READY', donor_sha256=donor['sha256'])
        if source_hashes(repo) != identity['source_sha256']:
            raise RuntimeError('Source changed while donor was training; do not silently chain a different implementation')
        first = scenes[0]
        with SceneV3(input_root / first['file'], expected_sha256=first['sha256']) as scene:
            adaptation_preflight(scene, donor, work / 'preflight/adaptation.json', args.device)
        registry = dict(schema='PANPV_CAMPAIGN_v1', campaign_id=identity['campaign_id'],
                        work_root=str(work), input_root=str(input_root), execution_server='s1',
                        target_sensor='WV2', seed=261002, device=args.device, micro_batch=12,
                        scenes=scenes, donor={k: donor[k] for k in ('path', 'sha256', 'provenance_path', 'provenance_sha256')},
                        protocol=dict(path=str(protocol_path), sha256=file_sha(protocol_path)),
                        source_root=str(repo), source_sha256=identity['source_sha256'],
                        preflight=dict(path=str(work / 'preflight/adaptation.json'),
                                       sha256=file_sha(work / 'preflight/adaptation.json')))
        registry_path = work / 'registry.json'
        immutable_json(registry_path, registry)
        status('WV2_CAMPAIGN_RUNNING', registry=str(registry_path))
        result = run_campaign(registry_path)
        status('WV2_CAMPAIGN_RETURNED', campaign_status_path=str(work / 'campaign_status.json'),
               return_code=result)
        return result
    except Exception as error:
        atomic_json(work / 'pipeline_status.json', dict(stage='FAILED', error=f'{type(error).__name__}: {error}',
                    updated_utc=datetime.now(timezone.utc).isoformat()))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    return run_pipeline(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
