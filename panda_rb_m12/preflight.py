"""Actual-runtime gates are automatic, mandatory and independent of scores."""
import os
from pathlib import Path
import shutil
import subprocess
from panda_rb_m12.common import ROOT, atomic_json, read, sha256, object_sha, server_dir, source_identity, utcnow, docker_required
from panda_rb_m12.plan import validate_plan, SEEDS, STEP1, MODES, campaign_dir, weights_dir, training_runs, run_dir


def resource_admission(server):
    from panda_rb.deployment import DEFAULT_IMAGE
    if os.environ.get('PANDA_RB_IMAGE_ID') != DEFAULT_IMAGE:
        raise RuntimeError('Use the exact pinned M12 Docker image')
    if os.environ.get('PANDA_RB_SERVER') != server:
        raise ValueError('M12 container server identity differs from requested cohort')
    result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True, timeout=20).strip()
    pids = [int(value.strip()) for value in result.splitlines() if value.strip()]
    others = [pid for pid in pids if pid != os.getpid()]
    if others:
        raise RuntimeError('WAITING_FOR_GPU_OWNER: ' + ','.join(map(str, others)))
    return dict(own_pid=os.getpid(), other_compute_pids=[], processes_stopped=False)


def storage_admission(server, root=ROOT):
    from panda_rb_m12.training import required_storage_bytes
    estimate = required_storage_bytes(40)
    retained = 0; remaining = 0; seen = set()
    for case in training_runs(server, root):
        allocated = 0
        for path in run_dir(case['run_id'], root).rglob('*'):
            if not path.is_file():
                continue
            stat = path.stat(); inode = (stat.st_dev, stat.st_ino)
            if inode in seen:
                continue
            seen.add(inode); allocated += stat.st_size
        retained += allocated
        wd = run_dir(case['run_id'], root)
        complete = ((wd / 'checkpoints/selection_manifest.json').is_file()
                    and (wd / 'native/metrics.json').is_file()
                    and (case['case_id'] not in STEP1 or all(
                        (wd / 'stress' / mode / 'completion.json').is_file() for mode in MODES)))
        if complete:
            # Corrupt completion bytes are not a reason to grant free capacity.
            from panda_rb_m12.evaluation import selected_checkpoints, validate_artifacts
            selected_checkpoints(read(wd / 'checkpoints/selection_manifest.json'))
            validate_artifacts(read(wd / 'native/metrics.json'), wd)
            if case['case_id'] in STEP1:
                for mode in MODES:
                    validate_artifacts(read(wd / 'stress' / mode / 'completion.json'), wd)
        else:
            # A retry can accumulate arbitrarily many retained resume bundles.
            # Bytes already allocated do NOT reduce the future normal budget.
            remaining += estimate['estimated_per_run_bytes']
    needed = remaining + 5*1024**3
    target = server_dir(server, root)
    while not target.exists():
        target = target.parent
    free = shutil.disk_usage(target).free
    if free < needed:
        raise OSError(f'M12 needs {needed} available bytes, found {free}; no old files are deleted')
    return dict(**estimate, available_bytes=free, estimated_remaining_bytes=needed, retained_bytes=retained,
                incomplete_run_policy='full conservative run budget; retries never credit future capacity')


def recover_old_evidence(server, root=ROOT):
    """Inventory only. Missing old/remote results never cause new Student training."""
    from panda_rb.plan import training_runs as old_runs, run_dir as old_dir
    root = Path(root); old = []
    for case in old_runs(server, root):
        wd = old_dir(case['run_id'], root)
        paths = [wd / 'native/metrics.json'] + [wd / 'stress' / mode / 'completion.json'
                    for mode in ('A_ON', 'A_ZERO_INFERENCE_ONLY')]
        old.append(dict(run_id=case['run_id'], status='FOUND' if all(p.is_file() for p in paths) else 'MISSING_OLD_EVIDENCE',
                        artifacts={str(p.relative_to(root)): sha256(p) for p in paths if p.is_file()}))
    slots = []
    for period, case in [(3, 14)] + [(5, c) for c in range(10, 16)]:
        matches = sorted(p for p in (root / 'work_dir').glob(f'ABLR2*WV3*P{period:02d}*C{case:02d}*') if p.is_dir())
        slots.append(dict(period=period, case=f'C{case:02d}', status='FOUND' if matches else 'OLD_SLOT_NOT_FOUND',
                          paths=[str(p.relative_to(root)) for p in matches], replacement_training_admitted=False))
    result = dict(server=server, old_b01=old, ablr2=slots, remote_missing_blocks_local=False,
                  original_evidence_modified=False, new_student_updates=0, checked_at_utc=utcnow())
    atomic_json(server_dir(server, root) / 'old_evidence_inventory.json', result)
    return result


def prepare(server, binding, root=ROOT, device='cuda'):
    import numpy  # pinned MKL preload order
    import torch
    from panda_rb_m12.binding import bind_existing, validate_binding
    from panda_rb_m12.weights import prepare_weights
    from panda_rb_m12.training import training_smoke
    from panda_rb_m12.evaluation import evaluator_identity
    from panda_rb_m12.stress import warp_convention_test
    from panda_rb_m12.reference_probe import response_probe
    docker_required()
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('M12 admission needs actual CUDA, not a CPU-only test')
    resources = resource_admission(server)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    contract = validate_plan(root)
    capacity = storage_admission(server, root)
    evaluator = evaluator_identity(root)
    if not Path(binding).is_file():
        bind_existing(root=root, destination=binding)
    _, bound, data, q = validate_binding(binding)
    if q.shape != (9714, 4) or any(data['splits'][s]['count'] != 20 for s in ('rr', 'fr')):
        raise ValueError('Expected full WV3 train × four views and RR20/FR20')
    maps = prepare_weights(binding, root=root)
    identity = dict(protocol=contract['protocol_identity'], binding_common_sha256=bound['common_sha256'],
        weights=maps, source_identity=source_identity(root), evaluator=evaluator,
        image_id=os.environ['PANDA_RB_IMAGE_ID'], device_name=torch.cuda.get_device_name(device))
    path = server_dir(server, root) / 'preflight.json'; previous = read(path)
    if previous and previous.get('identity') != identity:
        raise ValueError('Frozen M12 source/runtime/binding changed; no silent cohort resume')
    recover_old_evidence(server, root)
    if previous.get('passed'):
        response_probe(binding, server, root=root, device=device)
        return previous
    smoke = training_smoke(binding, weights_dir(root), seed=SEEDS[server][0], device=device, root=root)
    if not smoke.get('actual_batch48') or not smoke.get('full_state_resume', {}).get('passed'):
        raise ValueError('Actual batch48 CUDA parity and full-state resume are mandatory')
    probe = response_probe(binding, server, root=root, device=device)
    report = dict(passed=True, identity=identity, plan=contract, storage=capacity, resources=resources, smoke=smoke,
                  warp_convention=warp_convention_test(device), response_probe=probe['payload_sha256'],
                  no_score_based_admission=True, registered_updates=0, checked_at_utc=utcnow())
    atomic_json(path, report)
    return report
